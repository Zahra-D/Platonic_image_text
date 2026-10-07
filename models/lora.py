"""Standard and modality-routed Tri-LoRA layers for the multimodal model."""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


TEXT_ROUTE_ID = 0
IMAGE_ROUTE_ID = 1
# Apply the image-private branch while suppressing the shared branch for that
# token.  This is used by asymmetric text-to-image fine-tuning: shared LoRA is
# computed from text tokens only, while image tokens contribute image-private
# queries/keys/values and receive text information through joint attention.
IMAGE_PRIVATE_ONLY_ROUTE_ID = 2
TEXT_PRIVATE_ONLY_ROUTE_ID = 3
_ROUTE_IDS: ContextVar[torch.Tensor | None] = ContextVar("tri_lora_route_ids", default=None)
_SHARED_RECORDER: ContextVar["SharedActivationRecorder | None"] = ContextVar(
    "tri_lora_shared_recorder", default=None
)
_SHARED_REPLACEMENTS: ContextVar[dict[str, torch.Tensor] | None] = ContextVar(
    "tri_lora_shared_replacements", default=None
)


@contextmanager
def lora_modality_context(route_ids: torch.Tensor | None) -> Iterator[None]:
    token = _ROUTE_IDS.set(route_ids)
    try:
        yield
    finally:
        _ROUTE_IDS.reset(token)


class SharedActivationRecorder:
    """Aggregate every shared LoRA activation into one fixed-width representation.

    Adapter outputs have different widths (for example qkv and MLP expansion
    layers), so each pooled activation is adaptively reduced to ``d_model``
    before averaging.  Keeping every activation is essential: retaining only
    the last one makes the adversarial loss update only the final adapter.
    """

    def __init__(self, attention_mask: torch.Tensor | None, representation_dim: int,
                 private_branch: str = "text") -> None:
        if private_branch not in {"text", "image"}:
            raise ValueError("private_branch must be 'text' or 'image'")
        self.attention_mask = attention_mask
        self.representation_dim = representation_dim
        # Which private branch the native private map records.  Single-modality
        # training must record its own modality's private adapter, otherwise
        # HSIC would decorrelate the shared branch from an unused one.
        self.private_branch = private_branch
        self.representations: list[torch.Tensor] = []
        self.layer_representations: dict[int, list[torch.Tensor]] = {}
        self.layer_token_representations: dict[int, list[torch.Tensor]] = {}
        self.native_by_module: dict[str, torch.Tensor] = {}
        # The ordinary native map is computed from a detached adapter input so
        # module-local objectives cannot leak into earlier Transformer paths.
        # The second map retains the actual forward update for the optional
        # data2vec-style end-to-end objective, whose parameter gradients are
        # explicitly gated by the trainer.
        self.native_differentiable_by_module: dict[str, torch.Tensor] = {}
        # Native private updates are retained only when requested by a caller
        # such as the modulewise JEPA/HSIC objective.  Unlike the aggregate
        # readout above, these tensors preserve the exact output width of the
        # individual adapter linear map.
        self.native_private_by_module: dict[str, torch.Tensor] = {}

    @property
    def representation(self) -> torch.Tensor | None:
        if not self.representations:
            return None
        return torch.stack(self.representations, dim=0).mean(dim=0)

    def record(
        self,
        shared: torch.Tensor,
        layer_index: int | None = None,
        module_name: str | None = None,
        private: torch.Tensor | None = None,
        differentiable_shared: torch.Tensor | None = None,
    ) -> None:
        if shared.ndim != 3:
            return
        if module_name is not None:
            self.native_by_module[module_name] = shared
            if differentiable_shared is not None:
                if differentiable_shared.shape != shared.shape:
                    raise RuntimeError(
                        f"Differentiable recorder shape {tuple(differentiable_shared.shape)} does not match "
                        f"shared shape {tuple(shared.shape)} for {module_name}"
                    )
                self.native_differentiable_by_module[module_name] = differentiable_shared
            if private is not None:
                if private.shape != shared.shape:
                    raise RuntimeError(
                        f"Private recorder shape {tuple(private.shape)} does not match "
                        f"shared shape {tuple(shared.shape)} for {module_name}"
                    )
                self.native_private_by_module[module_name] = private
        token_representation = shared
        if token_representation.shape[-1] != self.representation_dim:
            batch, length, width = token_representation.shape
            token_representation = F.adaptive_avg_pool1d(
                token_representation.reshape(batch * length, 1, width),
                self.representation_dim,
            ).reshape(batch, length, self.representation_dim)
        if self.attention_mask is None:
            pooled = token_representation.mean(dim=1)
        else:
            weights = self.attention_mask.to(token_representation.dtype).unsqueeze(-1)
            pooled = (token_representation * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        self.representations.append(pooled)
        if layer_index is not None:
            self.layer_representations.setdefault(layer_index, []).append(pooled)
            self.layer_token_representations.setdefault(layer_index, []).append(
                token_representation
            )

    @property
    def per_layer_representation(self) -> dict[int, torch.Tensor]:
        """Average the shared adapter readouts within each Transformer block."""
        return {
            layer: torch.stack(representations, dim=0).mean(dim=0)
            for layer, representations in self.layer_representations.items()
        }

    @property
    def per_layer_token_representation(self) -> dict[int, torch.Tensor]:
        """Average the four shared-adapter token readouts in each block."""
        return {
            layer: torch.stack(representations, dim=0).mean(dim=0)
            for layer, representations in self.layer_token_representations.items()
        }


@contextmanager
def shared_activation_context(
    attention_mask: torch.Tensor | None, representation_dim: int,
    private_branch: str = "text",
) -> Iterator[SharedActivationRecorder]:
    recorder = SharedActivationRecorder(attention_mask, representation_dim, private_branch)
    token = _SHARED_RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _SHARED_RECORDER.reset(token)


@contextmanager
def shared_replacement_context(
    replacements: dict[str, torch.Tensor] | None,
) -> Iterator[None]:
    """Replace target shared deltas with token-aligned source shared deltas."""
    token = _SHARED_REPLACEMENTS.set(replacements)
    try:
        yield
    finally:
        _SHARED_REPLACEMENTS.reset(token)


class LoRALinear(nn.Module):
    """Conventional shared LoRA retained for checkpoint/API compatibility."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return self.base(x) + update * self.scale


class TriLoRALinear(nn.Module):
    """Shared/private low-rank affine branches, optionally with a dense base."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        name: str,
        delete_base_weights: bool = True,
        shared_branch: bool = True,
        private_rank: int | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError(f"Tri-LoRA rank and alpha must be positive for {name}")
        self.base = base
        self.name = name
        pieces = name.split(".")
        self.layer_index = (
            int(pieces[1])
            if len(pieces) >= 3 and pieces[0] == "blocks" and pieces[1].isdigit()
            else None
        )
        self.rank = rank
        self.alpha = alpha
        # ``shared_branch=False`` keeps the dense base as the shared path and
        # adds only modality-private adapters of the requested rank, scaled by
        # alpha/private_rank as in ordinary LoRA.
        self.shared_branch_enabled = shared_branch
        self.effective_rank = min(rank, base.in_features, base.out_features)
        if not shared_branch:
            requested = private_rank if private_rank is not None else max(1, self.effective_rank // 3)
            self.rank_text = self.rank_image = max(1, min(requested, base.in_features, base.out_features))
            self.rank_shared = 0
            self.nominal_shared_rank = self.rank_text
            self.scaling = alpha / self.rank_text
        elif self.effective_rank == 1:
            self.rank_shared = self.rank_text = self.rank_image = 1
            self.nominal_shared_rank = max(1, (2 * rank) // 3)
            self.scaling = alpha / self.nominal_shared_rank
        else:
            self.rank_shared = max(1, (2 * self.effective_rank) // 3)
            derived = max(1, self.effective_rank - self.rank_shared)
            self.rank_text = self.rank_image = derived
            self.nominal_shared_rank = max(1, (2 * rank) // 3)
            self.scaling = alpha / self.nominal_shared_rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        branches = [("text", self.rank_text), ("image", self.rank_image)]
        if shared_branch:
            branches.insert(0, ("shared", self.rank_shared))
        for branch, branch_rank in branches:
            a = nn.Parameter(torch.empty(branch_rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
            b = nn.Parameter(torch.zeros(base.out_features, branch_rank, device=base.weight.device, dtype=base.weight.dtype))
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            setattr(self, f"{branch}_A", a)
            setattr(self, f"{branch}_B", b)
        if delete_base_weights:
            # Keep an affine offset without retaining any frozen dense base
            # parameter. Initializing it from nn.Linear's ordinary random
            # bias also breaks the all-zero activation/gradient deadlock that
            # occurs when two zero-B adapter-only linears are composed.
            if base.bias is None:
                initial_bias = torch.zeros(
                    base.out_features, device=base.weight.device, dtype=base.weight.dtype
                )
            else:
                initial_bias = base.bias.detach().clone()
            self.shared_bias = nn.Parameter(initial_bias)
            self.base.weight = None
            self.base.bias = None
        else:
            self.register_parameter("shared_bias", None)

    def _delta(self, x: torch.Tensor, branch: str) -> torch.Tensor:
        a = getattr(self, f"{branch}_A")
        b = getattr(self, f"{branch}_B")
        return F.linear(F.linear(self.dropout(x), a), b) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense_shared = not self.shared_branch_enabled
        shared = (
            self.base(x) if dense_shared else self._delta(x, "shared")
        )
        recorder = _SHARED_RECORDER.get()
        if recorder is not None:
            # Recompute from a detached input so the adversarial gradient updates
            # the selected adapter branches but cannot leak backward into earlier
            # private branches.  The text-private output is also recorded for
            # modulewise HSIC.  Text-only training is its first supported use.
            recorder.record(
                self.base(x.detach()) if dense_shared else self._delta(x.detach(), "shared"),
                self.layer_index,
                self.name,
                private=self._delta(x.detach(), recorder.private_branch),
                differentiable_shared=shared,
            )
        replacements = _SHARED_REPLACEMENTS.get()
        if replacements is not None and self.name in replacements:
            replacement = replacements[self.name]
            if replacement.shape != shared.shape:
                raise RuntimeError(
                    f"Shared replacement shape {tuple(replacement.shape)} does not match "
                    f"{self.name} output {tuple(shared.shape)}"
                )
            shared = replacement
        route_ids = _ROUTE_IDS.get()
        if route_ids is None:
            shared_mask = None
        elif route_ids.ndim == 1 and route_ids.numel() == x.shape[0]:
            # Retain compatibility with older homogeneous batches/checkpoints.
            route_ids = route_ids[:, None]
        elif route_ids.shape != x.shape[:-1]:
            raise RuntimeError(
                f"Tri-LoRA route ids must be [batch] or match the token axes for {self.name}; "
                f"got {tuple(route_ids.shape)} for input {tuple(x.shape)}"
            )
        if route_ids is not None:
            shared_mask = _expand_route_mask(
                route_ids.ne(IMAGE_PRIVATE_ONLY_ROUTE_ID)
                & route_ids.ne(TEXT_PRIVATE_ONLY_ROUTE_ID),
                shared,
            )
            shared = shared * shared_mask
        if dense_shared:
            # `shared` already is the dense write, biases included.
            output = shared
        elif self.base.weight is None:
            output = shared + self.shared_bias
        else:
            output = self.base(x) + shared
        if route_ids is None:
            return output
        text_mask = _expand_route_mask(
            route_ids.eq(TEXT_ROUTE_ID) | route_ids.eq(TEXT_PRIVATE_ONLY_ROUTE_ID),
            output,
        )
        image_mask = _expand_route_mask(
            route_ids.eq(IMAGE_ROUTE_ID) | route_ids.eq(IMAGE_PRIVATE_ONLY_ROUTE_ID),
            output,
        )
        if text_mask.any():
            output = output + self._delta(x, "text") * text_mask
        if image_mask.any():
            output = output + self._delta(x, "image") * image_mask
        return output


def _expand_route_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return mask.to(device=target.device, dtype=target.dtype).reshape(
        list(mask.shape) + [1] * (target.ndim - mask.ndim)
    )


def condition_target_route_ids(
    route_ids: torch.Tensor,
    modality_ids: torch.Tensor,
    target_modality_id: int,
) -> torch.Tensor:
    """Keep condition tokens full-routed and make target tokens private-only."""
    if target_modality_id not in {1, 2}:
        raise ValueError("target_modality_id must be 1 (text) or 2 (image)")
    result = route_ids.clone()
    target = modality_ids.eq(target_modality_id)
    private_only_id = (
        TEXT_PRIVATE_ONLY_ROUTE_ID if target_modality_id == 1
        else IMAGE_PRIVATE_ONLY_ROUTE_ID
    )
    result[target] = private_only_id
    return result


def _matches(name: str, targets: list[str]) -> bool:
    return any(name == target or name.endswith(f".{target}") for target in targets)


def _adapter_injection_excluded(name: str) -> bool:
    """Keep auxiliary heads/bridges dense even when names match backbone targets."""
    return name.startswith((
        "shared_translation_bridges.",
        "shared_jepa_predictors.",
        "modality_discriminator.",
    ))


def inject_lora(model: nn.Module, target_modules: list[str], rank: int, alpha: float, dropout: float) -> list[str]:
    replaced = []

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if _adapter_injection_excluded(full_name):
                continue
            if isinstance(child, nn.Linear) and _matches(full_name, target_modules):
                setattr(parent, child_name, LoRALinear(child, rank, alpha, dropout))
                replaced.append(full_name)
            else:
                visit(child, full_name)

    visit(model)
    if not replaced:
        raise ValueError(f"No Linear modules matched LoRA targets: {target_modules}")
    return replaced


def inject_tri_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int,
    alpha: float,
    dropout: float,
    delete_base_weights: bool = True,
    shared_branch: bool = True,
    private_rank: int | None = None,
) -> list[str]:
    replaced = []

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if _adapter_injection_excluded(full_name):
                continue
            if isinstance(child, nn.Linear) and _matches(full_name, target_modules):
                setattr(parent, child_name, TriLoRALinear(
                    child, rank, alpha, dropout, full_name, delete_base_weights,
                    shared_branch=shared_branch, private_rank=private_rank,
                ))
                replaced.append(full_name)
            else:
                visit(child, full_name)

    visit(model)
    if not replaced:
        raise ValueError(f"No Linear modules matched Tri-LoRA targets: {target_modules}")
    return replaced


def shared_route_parameters(module) -> list[nn.Parameter]:
    """Trainable tensors of a Tri-LoRA module's shared route.

    Tri-LoRA carries the shared route in its own adapter; dense_private keeps
    it in the wrapped dense weight.  Frozen tensors are omitted so callers can
    hand the result straight to autograd.
    """
    if getattr(module, "shared_branch_enabled", True):
        candidates = (module.shared_A, module.shared_B)
    else:
        candidates = (module.base.weight,)
    return [
        parameter for parameter in candidates
        if parameter is not None and parameter.requires_grad
    ]


def iter_tri_lora(model: nn.Module) -> Iterator[tuple[str, TriLoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, TriLoRALinear):
            yield name, module


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    markers = (".lora_A", ".lora_B", ".shared_A", ".shared_B", ".shared_bias", ".text_A", ".text_B", ".image_A", ".image_B")
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if any(marker in name for marker in markers)
        or name.startswith("shared_translation_bridges.")
    }


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
