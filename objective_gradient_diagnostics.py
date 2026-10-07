"""Read-only decomposition of shared-LoRA gradients by training objective."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from models import iter_tri_lora, shared_route_parameters


def shared_parameters_by_layer(model) -> dict[int, list[torch.nn.Parameter]]:
    """Return the shared-route tensors of each Transformer block.

    In dense_private mode the shared route is the dense weight itself, so the
    decomposition follows that tensor instead of the absent shared adapter.
    Tensors held fixed are skipped: autograd has no gradient to report for them.
    """
    grouped = defaultdict(list)
    for _, module in iter_tri_lora(model):
        if module.layer_index is None:
            continue
        grouped[module.layer_index].extend(shared_route_parameters(module))
    return {
        layer: parameters
        for layer, parameters in sorted(grouped.items())
        if parameters
    }


def _objective_vectors(
    loss: torch.Tensor,
    parameters_by_layer: dict[int, list[torch.nn.Parameter]],
) -> dict[int, torch.Tensor]:
    ordered = [
        parameter
        for layer in sorted(parameters_by_layer)
        for parameter in parameters_by_layer[layer]
    ]
    gradients = torch.autograd.grad(
        loss, ordered, retain_graph=True, allow_unused=True, materialize_grads=False
    )
    vectors = {}
    offset = 0
    for layer, parameters in parameters_by_layer.items():
        parts = []
        for parameter in parameters:
            gradient = gradients[offset]
            offset += 1
            parts.append(
                torch.zeros(parameter.numel(), dtype=torch.float32)
                if gradient is None
                else gradient.detach().float().flatten().cpu()
            )
        vectors[layer] = torch.cat(parts)
    return vectors


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.norm() <= 1e-20 or right.norm() <= 1e-20:
        return float("nan")
    return float(F.cosine_similarity(left, right, dim=0))


def objective_shared_gradient_metrics(
    model,
    objective_losses: dict[str, torch.Tensor | None],
    *,
    selected_layers: set[int] | None = None,
) -> dict[str, float]:
    """Measure each objective's exact gradient on shared A/B parameters.

    Losses must already contain the coefficients used by training. This uses
    ``autograd.grad`` without writing ``parameter.grad``, so the subsequent
    optimizer update is unchanged.
    """
    parameters_by_layer = shared_parameters_by_layer(model)
    if selected_layers is not None:
        parameters_by_layer = {
            layer: parameters
            for layer, parameters in parameters_by_layer.items()
            if layer in selected_layers
        }
    if not parameters_by_layer:
        return {}
    valid_losses = {
        name: loss
        for name, loss in objective_losses.items()
        if loss is not None and loss.requires_grad
    }
    vectors = {
        name: _objective_vectors(loss, parameters_by_layer)
        for name, loss in valid_losses.items()
    }
    metrics = {}
    for layer in parameters_by_layer:
        prefix = f"layer_{layer:02d}"
        for name, by_layer in vectors.items():
            metrics[f"{prefix}/{name}_norm"] = float(by_layer[layer].norm())
        diffusion = vectors.get("diffusion", {}).get(layer)
        if diffusion is not None:
            diffusion_norm = float(diffusion.norm())
            for auxiliary in ("jepa", "sigreg", "hsic"):
                vector = vectors.get(auxiliary, {}).get(layer)
                if vector is None:
                    continue
                metrics[f"{prefix}/{auxiliary}_to_diffusion_norm_ratio"] = (
                    float(vector.norm()) / max(diffusion_norm, 1e-20)
                )
                metrics[f"{prefix}/diffusion_{auxiliary}_cosine"] = _cosine(
                    diffusion, vector
                )
        jepa = vectors.get("jepa", {}).get(layer)
        sigreg = vectors.get("sigreg", {}).get(layer)
        hsic = vectors.get("hsic", {}).get(layer)
        if jepa is not None and sigreg is not None:
            metrics[f"{prefix}/jepa_sigreg_cosine"] = _cosine(jepa, sigreg)
        if jepa is not None and hsic is not None:
            metrics[f"{prefix}/jepa_hsic_cosine"] = _cosine(jepa, hsic)
    return metrics
