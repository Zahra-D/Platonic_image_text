"""Latent JEPA objective for the shared branch of strict no-base Tri-LoRA."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


_MODULEWISE_MODULE_SUFFIX = {
    "qkv": "attn.qkv",
    "out_proj": "attn.out_proj",
    "mlp.0": "mlp.0",
    "mlp.3": "mlp.3",
}


def _predictor_key(module: str, layer: int) -> str:
    """Stable ModuleDict key for a native adapter and Transformer layer."""
    return f"{module.replace('.', '_')}_l{layer:02d}"


def _native_module_name(layer: int, module: str) -> str:
    try:
        return f"blocks.{layer}.{_MODULEWISE_MODULE_SUFFIX[module]}"
    except KeyError as error:
        raise ValueError(f"Unknown modulewise JEPA module: {module}") from error


def _normalized_mse(
    prediction: torch.Tensor, target: torch.Tensor, loss_type: str
) -> torch.Tensor:
    prediction = prediction.float()
    target = target.float()
    if loss_type == "mse":
        return F.mse_loss(prediction, target)
    if loss_type == "normalized_mse":
        scale = math.sqrt(prediction.size(-1))
        return F.mse_loss(
            F.normalize(prediction, p=2, dim=-1, eps=1e-6) * scale,
            F.normalize(target, p=2, dim=-1, eps=1e-6) * scale,
        )
    if loss_type == "cosine":
        return 1.0 - F.cosine_similarity(prediction, target, dim=-1, eps=1e-6).mean()
    raise ValueError("JEPA loss_type must be mse, normalized_mse, or cosine")


def modulewise_shared_jepa_loss(
    model,
    masked_shared_native: dict[str, torch.Tensor],
    clean_shared_native: dict[str, torch.Tensor],
    prediction_mask: torch.Tensor,
    *,
    layers: list[int] | tuple[int, ...],
    modules: list[str] | tuple[str, ...],
    mode: str,
    loss_type: str,
) -> tuple[torch.Tensor, dict[str, float], dict[str, float]]:
    """EMA-JEPA on selected native shared adapter updates.

    Each selected module at each selected layer receives a direct JEPA
    gradient.  ``mode='average'`` predicts the normalized mean clean update
    across layers from *every* selected student layer; ``mode='layerwise'``
    predicts the same-layer clean update.

    The student map decides how far the gradient travels.  Passing the
    recorder's ordinary native map (recomputed from a detached adapter input)
    keeps this loss layer-local: it reaches only that layer's shared A/B and
    its predictor.  Passing the recorder's differentiable native map instead
    retains the real forward graph, so the trainer can route the gradient
    end-to-end into every selected lower shared adapter; in that case the
    trainer, not this function, is responsible for gating which parameters
    receive it.
    """
    selected = tuple(sorted(set(layers)))
    selected_modules = tuple(dict.fromkeys(modules))
    unknown_modules = set(selected_modules).difference(_MODULEWISE_MODULE_SUFFIX)
    if not selected or not selected_modules or mode not in {"average", "layerwise"}:
        raise ValueError("modulewise JEPA needs selected layers and average or layerwise mode")
    if unknown_modules:
        raise ValueError(f"Unknown modulewise JEPA modules: {sorted(unknown_modules)}")
    if prediction_mask.ndim != 2 or not prediction_mask.any():
        raise ValueError("modulewise JEPA prediction_mask must select at least one token")

    losses: list[torch.Tensor] = []
    diagnostics: dict[str, float] = {}
    cosines: dict[str, float] = {}
    for module in selected_modules:
        names = {layer: _native_module_name(layer, module) for layer in selected}
        missing = [name for name in names.values() if name not in masked_shared_native or name not in clean_shared_native]
        if missing:
            raise RuntimeError(f"Modulewise JEPA recorder missing native updates: {missing}")
        if mode == "average":
            clean_parts = [
                F.normalize(clean_shared_native[names[layer]].detach().float(), p=2, dim=-1, eps=1e-6)
                for layer in selected
            ]
            target = F.normalize(torch.stack(clean_parts, dim=0).mean(dim=0), p=2, dim=-1, eps=1e-6)
            for layer in selected:
                student = masked_shared_native[names[layer]]
                prediction = model.predict_modulewise_jepa(module, layer, student)
                loss = _normalized_mse(prediction[prediction_mask], target[prediction_mask], loss_type)
                key = _predictor_key(module, layer)
                losses.append(loss)
                diagnostics[key] = loss.detach().item()
                cosines[key] = F.cosine_similarity(
                    prediction[prediction_mask].detach().float(), target[prediction_mask].float(), dim=-1, eps=1e-8
                ).mean().item()
            continue
        for layer in selected:
            name = names[layer]
            student = masked_shared_native[name]
            target = clean_shared_native[name].detach()
            if student.shape != target.shape:
                raise RuntimeError(
                    f"Modulewise JEPA shape mismatch for {name}: "
                    f"{tuple(student.shape)} vs {tuple(target.shape)}"
                )
            prediction = model.predict_modulewise_jepa(module, layer, student)
            loss = _normalized_mse(prediction[prediction_mask], target[prediction_mask], loss_type)
            key = _predictor_key(module, layer)
            losses.append(loss)
            diagnostics[key] = loss.detach().item()
            cosines[key] = F.cosine_similarity(
                prediction[prediction_mask].detach().float(), target[prediction_mask].float(), dim=-1, eps=1e-8
            ).mean().item()
    # Every selected (module, layer) pair has equal weight.
    return torch.stack(losses).mean(), diagnostics, cosines


def modulewise_data2vec_loss(
    masked_shared_native: dict[str, torch.Tensor],
    clean_shared_native: dict[str, torch.Tensor],
    prediction_mask: torch.Tensor,
    *,
    layers: list[int] | tuple[int, ...],
    modules: list[str] | tuple[str, ...],
    mode: str,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Direct no-predictor data2vec-style loss on final native shared updates.

    The student is the final selected layer.  In average mode its target is
    the mean of parameter-free LayerNorm teacher updates from all selected
    layers; no-average uses only the final teacher layer.  The supplied student
    map must retain the actual forward graph.  Parameter routing is handled by
    the trainer, not by detaching this representation.
    """
    if mode not in {"data2vec_average", "data2vec_no_average"}:
        raise ValueError(f"Invalid data2vec modulewise mode: {mode}")
    if beta <= 0:
        raise ValueError("data2vec SmoothL1 beta must be positive")
    selected = tuple(sorted(set(layers)))
    selected_modules = tuple(dict.fromkeys(modules))
    if not selected or not selected_modules or not prediction_mask.any():
        raise ValueError("data2vec modulewise JEPA needs layers, modules, and masked positions")
    final_layer = selected[-1]
    losses: list[torch.Tensor] = []
    diagnostics: dict[str, float] = {}
    for module in selected_modules:
        names = {layer: _native_module_name(layer, module) for layer in selected}
        missing = [name for name in names.values() if name not in masked_shared_native or name not in clean_shared_native]
        if missing:
            raise RuntimeError(f"data2vec recorder missing native updates: {missing}")
        teacher_layers = selected if mode == "data2vec_average" else (final_layer,)
        normalized_teacher = [
            F.layer_norm(clean_shared_native[names[layer]].detach().float(), (clean_shared_native[names[layer]].size(-1),))
            for layer in teacher_layers
        ]
        target = torch.stack(normalized_teacher, dim=0).mean(dim=0)
        student = masked_shared_native[names[final_layer]]
        loss = F.smooth_l1_loss(student[prediction_mask].float(), target[prediction_mask], beta=beta)
        key = _predictor_key(module, final_layer)
        losses.append(loss)
        diagnostics[key] = loss.detach().item()
    return torch.stack(losses).mean(), diagnostics


def _centered_rbf_hsic(shared: torch.Tensor, private: torch.Tensor) -> torch.Tensor:
    """Biased RBF-HSIC with detached median-distance bandwidths."""
    count = shared.size(0)
    if count < 3:
        return shared.new_zeros(())
    shared = F.normalize(shared.float(), p=2, dim=-1, eps=1e-6)
    private = F.normalize(private.float(), p=2, dim=-1, eps=1e-6)

    def rbf(values: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(values, values).square()
        non_diagonal = distances[~torch.eye(count, device=values.device, dtype=torch.bool)]
        bandwidth_squared = non_diagonal.detach().median().clamp_min(1e-6)
        return torch.exp(-distances / (2.0 * bandwidth_squared))

    shared_kernel = rbf(shared)
    private_kernel = rbf(private)
    centered_shared = (
        shared_kernel - shared_kernel.mean(dim=0, keepdim=True)
        - shared_kernel.mean(dim=1, keepdim=True) + shared_kernel.mean()
    )
    centered_private = (
        private_kernel - private_kernel.mean(dim=0, keepdim=True)
        - private_kernel.mean(dim=1, keepdim=True) + private_kernel.mean()
    )
    return (centered_shared * centered_private).sum() / float((count - 1) ** 2)


def modulewise_private_hsic_loss(
    masked_shared_native: dict[str, torch.Tensor],
    masked_private_native: dict[str, torch.Tensor],
    prediction_mask: torch.Tensor,
    *,
    layers: list[int] | tuple[int, ...],
    modules: list[str] | tuple[str, ...],
    max_tokens: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Average HSIC(shared update, text-private update) for each selected module.

    Sampling is deterministic from ``seed`` and happens after selecting only
    corrupted content positions.  This bounds the quadratic kernel calculation
    without favoring a fixed prefix of batch rows or token positions.
    """
    if max_tokens < 3:
        raise ValueError("HSIC max_tokens must be at least 3")
    selected = tuple(sorted(set(layers)))
    selected_modules = tuple(dict.fromkeys(modules))
    unknown_modules = set(selected_modules).difference(_MODULEWISE_MODULE_SUFFIX)
    if not selected_modules:
        raise ValueError("HSIC requires at least one native module")
    if unknown_modules:
        raise ValueError(f"Unknown modulewise HSIC modules: {sorted(unknown_modules)}")
    positions = prediction_mask.nonzero(as_tuple=False)
    if positions.size(0) < 3:
        raise ValueError("HSIC requires at least three masked positions")
    if positions.size(0) > max_tokens:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        choice = torch.randperm(positions.size(0), generator=generator)[:max_tokens]
        positions = positions[choice.to(positions.device)]
    row, token = positions.unbind(dim=1)
    losses: list[torch.Tensor] = []
    diagnostics: dict[str, float] = {}
    for layer in selected:
        for module in selected_modules:
            name = _native_module_name(layer, module)
            if name not in masked_shared_native or name not in masked_private_native:
                raise RuntimeError(f"HSIC recorder missing native updates for {name}")
            shared = masked_shared_native[name][row, token]
            private = masked_private_native[name][row, token]
            if shared.shape != private.shape:
                raise RuntimeError(f"HSIC shape mismatch for {name}")
            loss = _centered_rbf_hsic(shared, private)
            losses.append(loss)
            diagnostics[_predictor_key(module, layer)] = loss.detach().item()
    return torch.stack(losses).mean(), diagnostics


def shared_latent_jepa_loss(
    model,
    masked_shared_by_layer: dict[int, torch.Tensor],
    clean_shared_by_layer: dict[int, torch.Tensor],
    prediction_mask: torch.Tensor,
    *,
    layers: list[int] | tuple[int, ...] | set[int] | None = None,
    loss_type: str = "mse",
) -> tuple[torch.Tensor, dict[int, float], dict[int, float]]:
    """Predict clean shared token latents at exactly the corrupted positions.

    Both dictionaries contain ``[batch, sequence, d_model]`` tensors.  The
    target is explicitly detached, and the shared activation recorder itself
    detaches each adapter's input.  Consequently this loss updates only the
    current shared LoRA branches and their JEPA predictors; it cannot update
    private branches, embeddings, or earlier layers through the recorded path.
    """
    available = set(range(len(model.blocks)))
    if set(masked_shared_by_layer) != available or set(clean_shared_by_layer) != available:
        raise RuntimeError(
            "Shared JEPA expected token representations for every Transformer block; "
            f"masked={sorted(masked_shared_by_layer)}, clean={sorted(clean_shared_by_layer)}"
        )
    selected = available if layers is None else set(layers)
    if not selected or not selected.issubset(available):
        raise ValueError(f"Invalid shared JEPA layers: {sorted(selected)}")
    if loss_type not in {"mse", "normalized_mse", "cosine"}:
        raise ValueError("Shared JEPA loss_type must be mse, normalized_mse, or cosine")
    if prediction_mask.ndim != 2 or not prediction_mask.any():
        raise ValueError("Shared JEPA prediction_mask must select at least one token")

    layer_losses = {}
    layer_cosines = {}
    differentiable_losses = []
    for layer in sorted(selected):
        masked_representation = masked_shared_by_layer[layer]
        clean_target = clean_shared_by_layer[layer].detach()
        if masked_representation.shape != clean_target.shape:
            raise RuntimeError(
                f"Shared JEPA shape mismatch in layer {layer}: "
                f"{tuple(masked_representation.shape)} vs {tuple(clean_target.shape)}"
            )
        prediction = model.predict_shared_jepa(layer, masked_representation)
        selected_prediction = prediction[prediction_mask]
        selected_target = clean_target[prediction_mask]
        selected_prediction = selected_prediction.float()
        selected_target = selected_target.float()
        if loss_type == "mse":
            loss = F.mse_loss(selected_prediction, selected_target)
        elif loss_type == "normalized_mse":
            # This equals 2 * (1 - cosine) while remaining an MSE. Scaling
            # unit vectors by sqrt(d) prevents the value shrinking as 1/d.
            scale = math.sqrt(selected_prediction.size(-1))
            prediction_for_loss = F.normalize(
                selected_prediction, p=2, dim=-1, eps=1e-6
            ) * scale
            target_for_loss = F.normalize(
                selected_target, p=2, dim=-1, eps=1e-6
            ) * scale
            loss = F.mse_loss(prediction_for_loss, target_for_loss)
        else:
            loss = 1.0 - F.cosine_similarity(
                selected_prediction, selected_target, dim=-1, eps=1e-6
            ).mean()
        differentiable_losses.append(loss)
        layer_losses[layer] = loss.detach().item()
        layer_cosines[layer] = F.cosine_similarity(
            selected_prediction.detach().float(), selected_target.float(), dim=-1, eps=1e-8
        ).mean().item()
    return torch.stack(differentiable_losses).mean(), layer_losses, layer_cosines

def data2vec_hidden_loss(
    model,
    student_hidden_by_layer: dict[int, torch.Tensor],
    teacher_hidden_by_layer: dict[int, torch.Tensor],
    prediction_mask: torch.Tensor,
    *,
    top_k: int,
    beta: float,
    mode: str = "average",
    layers: list[int] | None = None,
    target_layer: int | None = None,
    pool_mask: torch.Tensor | None = None,
    token_weight: float = 1.0,
    global_weight: float = 0.0,
    variance_weight: float = 0.0,
    variance_target: float = 1.0,
    normalize_targets: bool = True,
    stop_gradient: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """data2vec objective on Transformer hidden states, averaged or layerwise.

    Both modes normalize every teacher state with a parameter-free LayerNorm
    before it becomes a target, exactly as data2vec does (the paper's ablations
    show this is what keeps the objective from collapsing to a constant), and
    both take Smooth-L1 at masked positions only.  They differ in what predicts
    what:

    ``average``
        One target, the mean of the selected teacher blocks, predicted from the
        student's own final block through a single prediction head.  This is the
        original recipe.
    ``layerwise``
        One target per selected block, each predicted from the student's state
        at that same block through that block's own prediction head, and the
        losses averaged.  Nothing is mixed across depth, so a block is never
        asked to match a shallower block's code.

    ``fixed_target``
        Every selected block predicts *one* teacher block -- ``target_layer``,
        the block whose representation is empirically the best -- through its
        own prediction head.  This separates two things ``layerwise`` confounds:
        whether its advantage comes from giving each block its own head and
        student, or from matching each block to a target at its own depth.

    ``layers`` selects the blocks explicitly; without it the last ``top_k``
    blocks are used.  This acts on whole hidden states rather than one LoRA
    branch's update, so it applies to a dense model.

    ``normalize_targets`` and ``stop_gradient`` are what distinguish data2vec
    from LeJEPA.  data2vec needs both -- the parameter-free LayerNorm keeps the
    target scale-free, and the stop-gradient onto an EMA teacher is what rules
    out the constant solution.  LeJEPA argues neither is needed if the
    embedding distribution is instead constrained to an isotropic Gaussian, so
    the teacherless path switches both off and the caller adds a SIGReg term.
    """
    if mode not in {"average", "layerwise", "fixed_target"}:
        raise ValueError(f"Unknown data2vec hidden mode: {mode!r}")
    if top_k < 1:
        raise ValueError("data2vec top_k must be at least one block")
    if beta <= 0:
        raise ValueError("data2vec SmoothL1 beta must be positive")
    if token_weight < 0 or global_weight < 0 or variance_weight < 0:
        raise ValueError("data2vec token, global, and variance weights must be non-negative")
    if token_weight == 0 and global_weight == 0 and variance_weight == 0:
        raise ValueError("data2vec needs at least one non-zero loss weight")
    if variance_target <= 0:
        raise ValueError("data2vec variance target must be positive")
    if prediction_mask.ndim != 2 or not prediction_mask.any():
        raise ValueError("data2vec needs at least one masked position")
    if pool_mask is not None:
        if pool_mask.shape != prediction_mask.shape:
            raise ValueError("data2vec pool mask must match the prediction mask shape")
        if not pool_mask.any(dim=1).all():
            raise ValueError("every data2vec sequence needs at least one pooled token")
    available = sorted(teacher_hidden_by_layer)
    if layers:
        missing = [layer for layer in layers if layer not in teacher_hidden_by_layer]
        if missing:
            raise ValueError(f"data2vec layers {missing} are not produced by this model")
        selected = sorted(layers)
    else:
        selected = available[-top_k:]
    if not selected:
        raise RuntimeError("teacher hidden states are missing")
    width = teacher_hidden_by_layer[selected[0]].size(-1)

    def normalized_target(layer: int) -> torch.Tensor:
        target = teacher_hidden_by_layer[layer]
        if stop_gradient:
            target = target.detach()
        target = target.float()
        return F.layer_norm(target, (width,)) if normalize_targets else target

    def masked_mean(values: torch.Tensor) -> torch.Tensor:
        mask = pool_mask if pool_mask is not None else prediction_mask
        weights = mask.to(values.dtype).unsqueeze(-1)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    per_layer_cosine: dict[int, float] = {}
    global_losses: list[torch.Tensor] = []
    variance_losses: list[torch.Tensor] = []
    global_cosines: list[float] = []

    def global_terms(student: torch.Tensor, target: torch.Tensor, layer: int | None) -> None:
        if global_weight == 0 and variance_weight == 0:
            return
        student_global = masked_mean(student)
        target_global = masked_mean(target)
        prediction_global = model.predict_data2vec(student_global, layer=layer)
        if global_weight:
            global_losses.append(F.smooth_l1_loss(
                prediction_global.float(), target_global.float(), beta=beta
            ))
        if variance_weight:
            # A batch-level VICReg-style floor prevents the scratch model from
            # satisfying the EMA target with one nearly constant scene code.
            # Normalize per example first so the floor cannot be met merely by
            # increasing hidden-state magnitude.
            normalized_global = F.layer_norm(student_global.float(), (width,))
            feature_std = torch.sqrt(normalized_global.var(dim=0, unbiased=False) + 1e-4)
            variance_losses.append(F.relu(variance_target - feature_std).mean())
        with torch.no_grad():
            global_cosines.append(F.cosine_similarity(
                prediction_global.detach().float(), target_global.float(), dim=-1, eps=1e-8
            ).mean().item())

    if mode == "average":
        target = torch.stack([normalized_target(layer) for layer in selected], dim=0).mean(dim=0)
        student = student_hidden_by_layer[max(student_hidden_by_layer)]
        prediction = model.predict_data2vec(student)
        token_loss = F.smooth_l1_loss(
            prediction[prediction_mask].float(), target[prediction_mask], beta=beta
        )
        global_terms(student, target, None)
        targets_for_spread = [target]
    else:
        fixed = mode == "fixed_target"
        if fixed:
            if target_layer is None:
                raise ValueError("fixed_target mode needs data2vec target_layer")
            if target_layer not in teacher_hidden_by_layer:
                raise ValueError(f"data2vec target_layer {target_layer} is not produced by this model")
        losses, targets_for_spread = [], []
        for layer in selected:
            target = normalized_target(target_layer if fixed else layer)
            prediction = model.predict_data2vec(student_hidden_by_layer[layer], layer=layer)
            losses.append(F.smooth_l1_loss(
                prediction[prediction_mask].float(), target[prediction_mask], beta=beta
            ))
            global_terms(student_hidden_by_layer[layer], target, layer)
            targets_for_spread.append(target)
            with torch.no_grad():
                per_layer_cosine[layer] = F.cosine_similarity(
                    prediction[prediction_mask].detach().float(), target[prediction_mask],
                    dim=-1, eps=1e-8,
                ).mean().item()
        token_loss = torch.stack(losses).mean()
        target = targets_for_spread[-1]
        prediction = model.predict_data2vec(
            student_hidden_by_layer[selected[-1]], layer=selected[-1]
        )
    global_loss = (
        torch.stack(global_losses).mean() if global_losses else token_loss.new_zeros(())
    )
    variance_loss = (
        torch.stack(variance_losses).mean() if variance_losses else token_loss.new_zeros(())
    )
    loss = (
        token_weight * token_loss
        + global_weight * global_loss
        + variance_weight * variance_loss
    )
    with torch.no_grad():
        cosine = (
            sum(per_layer_cosine.values()) / len(per_layer_cosine) if per_layer_cosine
            else F.cosine_similarity(
                prediction[prediction_mask].detach().float(), target[prediction_mask],
                dim=-1, eps=1e-8,
            ).mean().item()
        )
        # Target spread across positions: a collapsing teacher drives this to 0.
        # Layerwise reports the least spread of any target, so one collapsing
        # block cannot hide behind the others.
        spread = min(
            float(F.normalize(one[prediction_mask], dim=-1).std(dim=0).mean())
            for one in targets_for_spread
        )
    details = {
        "layers": float(len(selected)),
        "token_loss": token_loss.detach().item(),
        "global_loss": global_loss.detach().item(),
        "variance_loss": variance_loss.detach().item(),
        "global_cosine": sum(global_cosines) / len(global_cosines) if global_cosines else 0.0,
        "prediction_target_cosine": cosine,
        "target_position_spread": spread,
    }
    details.update({f"cosine_l{layer}": value for layer, value in per_layer_cosine.items()})
    return loss, details
