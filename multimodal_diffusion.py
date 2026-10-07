"""Masking, loss, and conditional image generation for multimodal D3PM."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from models import condition_target_route_ids


def span_mask(
    selected: torch.Tensor,
    t: torch.Tensor,
    min_length: int,
    max_length: int,
) -> torch.Tensor:
    """Mask contiguous windows instead of independently chosen tokens.

    Each row gets as many windows as it takes to reach its target count
    ``round(t * eligible)`` in expectation, every window's size is drawn uniformly from
    ``[min_length, max_length]``, and starts are drawn uniformly over the row's
    eligible positions.  Windows may overlap and are clipped at the eligible
    span's edges, so the realized fraction sits slightly below the target --
    the same behavior as wav2vec 2.0's span masking.

    A window hides a whole phrase rather than scattered words, so the model
    cannot fill the gap from its immediate lexical neighbours and has to use
    the surrounding context, which is the property JEPA-style objectives want.
    """
    if not 1 <= min_length <= max_length:
        raise ValueError("mask window sizes must satisfy 1 <= min <= max")
    device = selected.device
    batch, length = selected.shape
    eligible = selected.sum(dim=1)
    target = (t * eligible.float()).round().long().clamp_min(1)
    windows = (
        target.float() / ((min_length + max_length) / 2)
    ).round().long().clamp_min(1)
    count = int(windows.max().item())
    # Draw starts over each row's eligible positions, then map that rank back
    # to a column index: a stable descending sort lists eligible columns first,
    # in their original order.
    order = selected.long().argsort(dim=1, descending=True, stable=True)
    ranks = (
        torch.rand(batch, count, device=device) * eligible.clamp_min(1).unsqueeze(1).float()
    ).long().clamp_max((eligible.clamp_min(1) - 1).unsqueeze(1))
    starts = order.gather(1, ranks)
    sizes = torch.randint(min_length, max_length + 1, (batch, count), device=device)
    columns = torch.arange(length, device=device)[None, None, :]
    active = (
        torch.arange(count, device=device)[None, :] < windows.unsqueeze(1)
    ).unsqueeze(-1)
    inside = columns.ge(starts.unsqueeze(-1)) & columns.lt((starts + sizes).unsqueeze(-1))
    return (inside & active).any(dim=1) & selected


def block_mask_2d(
    selected: torch.Tensor,
    t: torch.Tensor,
    grid: tuple[int, int],
    scale_range: tuple[float, float],
    aspect_range: tuple[float, float],
) -> torch.Tensor:
    """Mask rectangular blocks of an image token grid, as I-JEPA does.

    Image tokens arrive row-major, so the eligible positions of a row map onto
    a ``grid = (height, width)`` lattice.  Each block draws an area fraction
    from ``scale_range`` and an aspect ratio from ``aspect_range``, and its
    top-left corner uniformly; enough blocks are drawn to reach ``t`` of the
    grid in expectation, and blocks may overlap.

    A rectangle removes a contiguous region of the image rather than scattered
    codes, so neighbouring tokens -- which in a VQ grid are highly redundant --
    cannot give the answer away.  That is the 2D analogue of window masking on
    text, and the reason to prefer it here.
    """
    height, width = grid
    low, high = scale_range
    thin, wide = aspect_range
    if not 0 < low <= high <= 1:
        raise ValueError("block scale range must satisfy 0 < min <= max <= 1")
    if not 0 < thin <= wide:
        raise ValueError("block aspect range must satisfy 0 < min <= max")
    eligible = selected.sum(dim=1)
    cells = height * width
    if int(eligible.max().item()) != cells or int(eligible.min().item()) != cells:
        raise ValueError(
            f"2D block masking needs exactly {cells} eligible tokens per row, "
            f"got {int(eligible.min().item())}-{int(eligible.max().item())}"
        )
    device = selected.device
    batch = selected.size(0)
    count = max(1, int(round(float(t.mean().item()) / ((low + high) / 2))))
    scales = torch.empty(batch, count, device=device).uniform_(low, high)
    aspects = torch.empty(batch, count, device=device).uniform_(thin, wide)
    areas = scales * cells
    block_h = (areas / aspects).sqrt().round().long().clamp(1, height)
    block_w = (areas * aspects).sqrt().round().long().clamp(1, width)
    top = (torch.rand(batch, count, device=device) * (height - block_h + 1).float()).long()
    left = (torch.rand(batch, count, device=device) * (width - block_w + 1).float()).long()
    rows = torch.arange(height, device=device)[None, None, :, None]
    columns = torch.arange(width, device=device)[None, None, None, :]
    inside = (
        rows.ge(top[:, :, None, None]) & rows.lt((top + block_h)[:, :, None, None])
        & columns.ge(left[:, :, None, None]) & columns.lt((left + block_w)[:, :, None, None])
    )
    grid_mask = inside.any(dim=1).reshape(batch, cells)
    # Map grid cells back to sequence positions: a stable descending sort lists
    # the eligible columns first, in their original row-major order.
    order = selected.long().argsort(dim=1, descending=True, stable=True)[:, :cells]
    masked = torch.zeros_like(selected)
    masked.scatter_(1, order, grid_mask)
    return masked & selected


def corrupt_batch(
    input_ids: torch.Tensor,
    eligible_mask: torch.Tensor,
    modality_ids: torch.Tensor,
    mask_token_id: int,
    eps: float = 1e-3,
    objective: str = "both",
    fixed_t: float | None = None,
    full_mask_probability: float = 0.0,
    bert_replacement: bool = False,
    random_token_ranges: dict[int, tuple[int, int]] | None = None,
    mask_span_min: int | None = None,
    mask_span_max: int | None = None,
    mask_block_grid: tuple[int, int] | None = None,
    mask_block_scale: tuple[float, float] = (0.10, 0.25),
    mask_block_aspect: tuple[float, float] = (0.75, 1.5),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask eligible tokens and return (corrupted inputs, prediction mask, t).

    ``bert_replacement`` swaps this repo's "every selected token becomes
    [MASK]" corruption for BERT's, which data2vec inherits: of the selected
    positions, 80% become [MASK], 10% become a random token of the same
    modality, and 10% are left untouched.  The returned prediction mask still
    covers all selected positions, so the loss is taken over the full 15% as in
    BERT -- only what the student *sees* changes.  ``random_token_ranges`` gives
    the half-open id range to draw from per modality id, which keeps a text
    position from being replaced by an image code.

    ``mask_span_min``/``mask_span_max`` switch position selection from
    independent tokens to contiguous windows of that size range; see
    :func:`span_mask`.  ``mask_block_grid`` instead treats the eligible tokens
    as an image lattice and masks rectangles on it; see :func:`block_mask_2d`.
    """
    if objective not in {"both", "image", "text"}:
        raise ValueError("objective must be both, image, or text")
    if bert_replacement and not random_token_ranges:
        raise ValueError("BERT replacement needs random_token_ranges per modality")
    selected = eligible_mask.clone()
    if objective == "image":
        selected &= modality_ids.eq(2)
    elif objective == "text":
        selected &= modality_ids.eq(1)

    batch = input_ids.size(0)
    if not 0.0 <= full_mask_probability <= 1.0:
        raise ValueError("full_mask_probability must be in [0, 1]")
    if fixed_t is None:
        t = torch.rand(batch, device=input_ids.device) * (1.0 - eps) + eps
        if full_mask_probability:
            force_full = torch.rand(batch, device=input_ids.device) < full_mask_probability
            t = t.masked_fill(force_full, 1.0)
    else:
        t = torch.full((batch,), fixed_t, device=input_ids.device)
    if mask_block_grid is not None:
        if mask_span_min is not None or mask_span_max is not None:
            raise ValueError("2D block masking and 1D window masking are exclusive")
        masked = block_mask_2d(
            selected, t, mask_block_grid, mask_block_scale, mask_block_aspect
        )
    elif mask_span_min is not None or mask_span_max is not None:
        if mask_span_min is None or mask_span_max is None:
            raise ValueError("window masking needs both mask_span_min and mask_span_max")
        masked = span_mask(selected, t, mask_span_min, mask_span_max)
    else:
        masked = (torch.rand_like(input_ids, dtype=torch.float32) < t[:, None]) & selected
    # Every applicable row must contribute a target, including at very small t.
    for row in range(batch):
        if selected[row].any() and not masked[row].any():
            candidates = selected[row].nonzero(as_tuple=False).flatten()
            choice = candidates[torch.randint(len(candidates), (1,), device=input_ids.device)]
            masked[row, choice] = True
    if not bert_replacement:
        corrupted = input_ids.masked_fill(masked, mask_token_id)
        return corrupted, masked, t
    # BERT's 80/10/10 split over the selected positions.
    roll = torch.rand_like(input_ids, dtype=torch.float32)
    corrupted = input_ids.masked_fill(masked & roll.lt(0.8), mask_token_id)
    randomized = masked & roll.ge(0.8) & roll.lt(0.9)
    if randomized.any():
        replacements = input_ids.clone()
        for modality_id, (low, high) in random_token_ranges.items():
            if high <= low:
                raise ValueError(f"empty random token range for modality {modality_id}")
            draw = torch.randint(low, high, input_ids.shape, device=input_ids.device)
            replacements = torch.where(modality_ids.eq(modality_id), draw, replacements)
        corrupted = torch.where(randomized, replacements, corrupted)
    return corrupted, masked, t


def masked_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masked: torch.Tensor,
    t: torch.Tensor,
    weight_by_t: bool,
) -> torch.Tensor:
    per_token = F.cross_entropy(logits.transpose(1, 2).float(), targets, reduction="none")
    weights = masked.float()
    if weight_by_t:
        weights = weights / t[:, None].clamp_min(1e-6)
    denominator = masked.sum().clamp_min(1)
    return (per_token * weights).sum() / denominator


@torch.no_grad()
def generate_masked_modality(
    model,
    batch: dict[str, torch.Tensor],
    modality_id: int,
    token_start: int,
    token_count: int,
    mask_token_id: int,
    num_steps: int,
    temperature: float = 1.0,
    reveal_order: str = "confidence",
    amp_dtype=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Iteratively fill one fully masked modality and return row confidences.

    ``route_ids`` are consumed exactly as supplied by the caller.  This allows
    unpaired translation to disable the source-private adapter while retaining
    the target-private adapter.  Confidence is the mean probability assigned
    when each generated token was committed.
    """
    if reveal_order not in {"confidence", "random"}:
        raise ValueError("reveal_order must be confidence or random")
    if modality_id not in {1, 2}:
        raise ValueError("modality_id must be 1 (text) or 2 (image)")
    if token_count < 1:
        raise ValueError("token_count must be positive")
    ids = batch["input_ids"].clone()
    target_positions = batch["eligible_mask"] & batch["modality_ids"].eq(modality_id)
    ids[target_positions] = mask_token_id
    remaining = target_positions.clone()
    confidence_sum = torch.zeros(ids.size(0), device=ids.device, dtype=torch.float32)
    confidence_count = torch.zeros_like(confidence_sum)
    was_training = model.training
    model.eval()

    route_ids = batch.get("route_ids")
    if getattr(model, "asymmetric_condition_target", False) and route_ids is not None:
        route_ids = condition_target_route_ids(route_ids, batch["modality_ids"], modality_id)

    for step in range(max(1, num_steps)):
        device_type = ids.device.type
        autocast = (
            torch.autocast(device_type=device_type, dtype=amp_dtype)
            if amp_dtype is not None else nullcontext()
        )
        with autocast:
            logits = model(
                ids,
                batch["attention_mask"],
                batch["position_ids"],
                batch["modality_ids"],
                route_ids,
            )[..., token_start : token_start + token_count]
        if temperature <= 0:
            sampled_local = logits.argmax(dim=-1)
            confidence = logits.softmax(dim=-1).amax(dim=-1)
        else:
            probs = (logits / temperature).softmax(dim=-1)
            sampled_local = torch.multinomial(probs.reshape(-1, token_count), 1).view(ids.shape)
            confidence = probs.gather(-1, sampled_local.unsqueeze(-1)).squeeze(-1)
        sampled = sampled_local + token_start

        for row in range(ids.size(0)):
            candidates = remaining[row].nonzero(as_tuple=False).flatten()
            if not len(candidates):
                continue
            steps_left = max(1, num_steps - step)
            reveal_count = min(len(candidates), math.ceil(len(candidates) / steps_left))
            if reveal_order == "confidence":
                scores = confidence[row, candidates]
                chosen = candidates[scores.topk(reveal_count).indices]
            else:
                chosen = candidates[torch.randperm(len(candidates), device=ids.device)[:reveal_count]]
            ids[row, chosen] = sampled[row, chosen]
            confidence_sum[row] += confidence[row, chosen].float().sum()
            confidence_count[row] += len(chosen)
            remaining[row, chosen] = False
        if not remaining.any():
            break
    model.train(was_training)
    return ids, confidence_sum / confidence_count.clamp_min(1)


@torch.no_grad()
def generate_shared_translation(
    model,
    condition: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    token_start: int,
    token_count: int,
    mask_token_id: int,
    num_steps: int,
    temperature: float = 1.0,
    reveal_order: str = "confidence",
    amp_dtype=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a target through condition-shared K/V translation bridges."""
    if reveal_order not in {"confidence", "random"}:
        raise ValueError("reveal_order must be confidence or random")
    ids = target["input_ids"].clone()
    target_positions = target["eligible_mask"]
    ids[target_positions] = mask_token_id
    remaining = target_positions.clone()
    confidence_sum = torch.zeros(ids.size(0), device=ids.device, dtype=torch.float32)
    confidence_count = torch.zeros_like(confidence_sum)
    was_training = model.training
    model.eval()
    for step in range(max(1, num_steps)):
        autocast = (
            torch.autocast(device_type=ids.device.type, dtype=amp_dtype)
            if amp_dtype is not None else nullcontext()
        )
        with autocast:
            logits = model.forward_shared_translation(
                condition["input_ids"], condition["attention_mask"],
                condition["position_ids"], condition["modality_ids"], condition["route_ids"],
                ids, target["attention_mask"], target["position_ids"],
                target["modality_ids"], target["route_ids"],
                condition_kv_mask=condition.get("eligible_mask"),
            )[..., token_start : token_start + token_count]
        if temperature <= 0:
            sampled_local = logits.argmax(dim=-1)
            confidence = logits.softmax(dim=-1).amax(dim=-1)
        else:
            probabilities = (logits / temperature).softmax(dim=-1)
            sampled_local = torch.multinomial(
                probabilities.reshape(-1, token_count), 1
            ).view(ids.shape)
            confidence = probabilities.gather(-1, sampled_local.unsqueeze(-1)).squeeze(-1)
        sampled = sampled_local + token_start
        for row in range(ids.size(0)):
            candidates = remaining[row].nonzero(as_tuple=False).flatten()
            if not len(candidates):
                continue
            steps_left = max(1, num_steps - step)
            reveal_count = min(len(candidates), math.ceil(len(candidates) / steps_left))
            if reveal_order == "confidence":
                chosen = candidates[confidence[row, candidates].topk(reveal_count).indices]
            else:
                chosen = candidates[
                    torch.randperm(len(candidates), device=ids.device)[:reveal_count]
                ]
            ids[row, chosen] = sampled[row, chosen]
            confidence_sum[row] += confidence[row, chosen].float().sum()
            confidence_count[row] += len(chosen)
            remaining[row, chosen] = False
        if not remaining.any():
            break
    model.train(was_training)
    return ids, confidence_sum / confidence_count.clamp_min(1)


@torch.no_grad()
def generate_conditioned_images(
    model,
    batch: dict[str, torch.Tensor],
    image_offset: int,
    num_image_codes: int,
    mask_token_id: int,
    num_steps: int,
    temperature: float = 1.0,
    reveal_order: str = "confidence",
) -> torch.Tensor:
    """Fill only image positions; text tokens remain fixed conditioning context."""
    ids, _ = generate_masked_modality(
        model, batch, 2, image_offset, num_image_codes, mask_token_id,
        num_steps, temperature, reveal_order,
    )
    image_positions = batch["eligible_mask"] & batch["modality_ids"].eq(2)
    total_positions = image_positions.sum(dim=1)

    flat = ids[image_positions].view(ids.size(0), int(total_positions[0].item()))
    return flat - image_offset
