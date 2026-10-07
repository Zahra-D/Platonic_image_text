"""Unpaired cycle/back-translation losses for the Tri-LoRA training stage.

Real text and image rows remain deliberately mismatched.  A target modality is
generated with the real source routed through shared LoRA only; the generated
pseudo-pair then reconstructs the original source at one random mask level.
The target-private branch is enabled only on the modality being generated or
reconstructed.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from multimodal_diffusion import (
    corrupt_batch,
    generate_masked_modality,
    generate_shared_translation,
    masked_loss,
)


TEXT_MODALITY = 1
IMAGE_MODALITY = 2
PRIVATE_ROUTE = {TEXT_MODALITY: 0, IMAGE_MODALITY: 1}
TENSOR_FIELDS = ("input_ids", "position_ids", "modality_ids", "eligible_mask")


def take_rows(batch: dict[str, torch.Tensor], count: int) -> dict[str, torch.Tensor]:
    """Take the same leading rows from every batch field."""
    return {name: value[:count] for name, value in batch.items()}


def _pack_rows(rows, pad_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = len(rows)
    max_length = max(row["input_ids"].numel() for row in rows)
    result = {
        "input_ids": torch.full((batch_size, max_length), pad_id, dtype=torch.long, device=device),
        "attention_mask": torch.zeros((batch_size, max_length), dtype=torch.bool, device=device),
        "position_ids": torch.zeros((batch_size, max_length), dtype=torch.long, device=device),
        "modality_ids": torch.zeros((batch_size, max_length), dtype=torch.long, device=device),
        "eligible_mask": torch.zeros((batch_size, max_length), dtype=torch.bool, device=device),
        "route_ids": torch.full((batch_size, max_length), -1, dtype=torch.long, device=device),
        "pair_indices": torch.arange(batch_size, dtype=torch.long, device=device),
    }
    for row_index, row in enumerate(rows):
        length = row["input_ids"].numel()
        result["attention_mask"][row_index, :length] = True
        for field in TENSOR_FIELDS:
            result[field][row_index, :length] = row[field]
    return result


def combine_unpaired_batches(
    text_batch: dict[str, torch.Tensor],
    image_batch: dict[str, torch.Tensor],
    pad_id: int,
) -> dict[str, torch.Tensor]:
    """Join independently sampled pure-modality rows without creating supervision."""
    if text_batch["input_ids"].size(0) != image_batch["input_ids"].size(0):
        raise ValueError("Text and image pseudo-pair batches must have equal row counts")
    if torch.any(text_batch["pair_indices"].eq(image_batch["pair_indices"])):
        raise ValueError("Back-translation received an actual paired row; expected strict derangement")
    rows = []
    for row_index in range(text_batch["input_ids"].size(0)):
        parts = []
        for batch in (text_batch, image_batch):
            valid = batch["attention_mask"][row_index]
            parts.append({field: batch[field][row_index, valid] for field in TENSOR_FIELDS})
        rows.append({field: torch.cat([part[field] for part in parts]) for field in TENSOR_FIELDS})
    result = _pack_rows(rows, pad_id, text_batch["input_ids"].device)
    result["pair_indices"] = text_batch["pair_indices"].clone()
    return result


def extract_modality(
    batch: dict[str, torch.Tensor], modality_id: int, pad_id: int
) -> dict[str, torch.Tensor]:
    """Extract one clean modality segment from mixed pseudo-pair rows."""
    rows = []
    for row_index in range(batch["input_ids"].size(0)):
        selected = batch["attention_mask"][row_index] & batch["modality_ids"][row_index].eq(modality_id)
        rows.append({field: batch[field][row_index, selected] for field in TENSOR_FIELDS})
    result = _pack_rows(rows, pad_id, batch["input_ids"].device)
    result["route_ids"][result["modality_ids"].eq(TEXT_MODALITY)] = PRIVATE_ROUTE[TEXT_MODALITY]
    result["route_ids"][result["modality_ids"].eq(IMAGE_MODALITY)] = PRIVATE_ROUTE[IMAGE_MODALITY]
    return result


def translation_routes(batch: dict[str, torch.Tensor], target_modality: int) -> torch.Tensor:
    """Use shared-only source routing and shared+private target routing."""
    routes = torch.full_like(batch["route_ids"], -1)
    routes[batch["modality_ids"].eq(target_modality)] = PRIVATE_ROUTE[target_modality]
    return routes


def shared_only_routes(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = dict(batch)
    result["route_ids"] = torch.full_like(batch["route_ids"], -1)
    return result


def _autocast(batch, amp_dtype):
    return (
        torch.autocast(device_type=batch["input_ids"].device.type, dtype=amp_dtype)
        if amp_dtype is not None else nullcontext()
    )


def _model_shared(model, batch, amp_dtype, teacher: bool) -> torch.Tensor:
    batch = shared_only_routes(batch)
    if teacher:
        was_training = model.training
        model.eval()
        with torch.no_grad(), _autocast(batch, amp_dtype):
            _, shared = model(
                batch["input_ids"], batch["attention_mask"], batch["position_ids"],
                batch["modality_ids"], batch["route_ids"], return_shared=True,
            )
        model.train(was_training)
        return shared.detach()
    with _autocast(batch, amp_dtype):
        _, shared = model(
            batch["input_ids"], batch["attention_mask"], batch["position_ids"],
            batch["modality_ids"], batch["route_ids"], return_shared=True,
        )
    return shared


def _direction_loss(
    model,
    paired_template,
    original_source,
    generated_modality: int,
    token_start: int,
    token_count: int,
    mask_token_id: int,
    pad_id: int,
    generation_steps: int,
    temperature: float,
    reveal_order: str,
    minimum_confidence: float,
    cycle_weight: float,
    alignment_weight: float,
    alignment_loss_type: str,
    contrastive_temperature: float,
    eps: float,
    weight_by_t: bool,
    amp_dtype,
):
    use_shared_translation = bool(getattr(model, "shared_translation_layers", ()))
    target_template = extract_modality(paired_template, generated_modality, pad_id)
    if use_shared_translation:
        generated_target_ids, confidence = generate_shared_translation(
            model, original_source, target_template, token_start, token_count,
            mask_token_id, generation_steps, temperature, reveal_order, amp_dtype,
        )
        generated_target = dict(target_template)
        generated_target["input_ids"] = generated_target_ids
    else:
        generation_batch = dict(paired_template)
        generation_batch["route_ids"] = translation_routes(paired_template, generated_modality)
        generated_ids, confidence = generate_masked_modality(
            model, generation_batch, generated_modality, token_start, token_count,
            mask_token_id, generation_steps, temperature, reveal_order, amp_dtype,
        )
        pseudo_pair = dict(paired_template)
        pseudo_pair["input_ids"] = generated_ids
        generated_target = extract_modality(pseudo_pair, generated_modality, pad_id)
    accepted = confidence.ge(minimum_confidence)
    metrics = {
        "confidence": confidence.mean().item(),
        "acceptance": accepted.float().mean().item(),
    }
    if not accepted.any():
        return None, metrics

    source_modality = TEXT_MODALITY if generated_modality == IMAGE_MODALITY else IMAGE_MODALITY
    total = None

    if cycle_weight > 0:
        cycle_batch = dict(original_source if use_shared_translation else pseudo_pair)
        if not use_shared_translation:
            cycle_batch["route_ids"] = translation_routes(cycle_batch, source_modality)
        corrupted, masked, t = corrupt_batch(
            cycle_batch["input_ids"], cycle_batch["eligible_mask"], cycle_batch["modality_ids"],
            mask_token_id, eps, "text" if source_modality == TEXT_MODALITY else "image",
        )
        masked &= accepted[:, None]
        with _autocast(cycle_batch, amp_dtype):
            if use_shared_translation:
                logits = model.forward_shared_translation(
                    generated_target["input_ids"], generated_target["attention_mask"],
                    generated_target["position_ids"], generated_target["modality_ids"],
                    generated_target["route_ids"], corrupted, cycle_batch["attention_mask"],
                    cycle_batch["position_ids"], cycle_batch["modality_ids"],
                    cycle_batch["route_ids"],
                    condition_kv_mask=generated_target.get("eligible_mask"),
                )
            else:
                logits = model(
                    corrupted, cycle_batch["attention_mask"], cycle_batch["position_ids"],
                    cycle_batch["modality_ids"], cycle_batch["route_ids"],
                )
            cycle = masked_loss(logits, cycle_batch["input_ids"], masked, t, weight_by_t)
        total = cycle * cycle_weight
        metrics["cycle_loss"] = cycle.detach().item()
        metrics["cycle_accuracy"] = (
            logits.detach().argmax(dim=-1)[masked].eq(cycle_batch["input_ids"][masked]).float().mean().item()
        )

    if alignment_weight > 0:
        source_shared = _model_shared(model, original_source, amp_dtype, teacher=True)
        target_shared = _model_shared(model, generated_target, amp_dtype, teacher=False)
        source_selected = F.normalize(source_shared[accepted].float(), dim=-1)
        target_selected = F.normalize(target_shared[accepted].float(), dim=-1)
        if alignment_loss_type == "contrastive" and source_selected.size(0) > 1:
            similarity = target_selected @ source_selected.transpose(0, 1)
            similarity = similarity / contrastive_temperature
            labels = torch.arange(similarity.size(0), device=similarity.device)
            alignment = F.cross_entropy(similarity, labels)
            metrics["alignment_top1"] = (
                similarity.detach().argmax(dim=-1).eq(labels).float().mean().item()
            )
        else:
            alignment = (1.0 - (target_selected * source_selected).sum(dim=-1)).mean()
        term = alignment * alignment_weight
        total = term if total is None else total + term
        metrics["alignment_loss"] = alignment.detach().item()

    return total, metrics


def unpaired_backtranslation_losses(
    model,
    text_batch: dict[str, torch.Tensor],
    image_batch: dict[str, torch.Tensor],
    tokenizer,
    image_offset: int,
    num_image_codes: int,
    batch_size: int,
    generation_steps: int,
    temperature: float,
    reveal_order: str,
    minimum_confidence: float,
    cycle_weight: float,
    alignment_weight: float,
    eps: float,
    weight_by_t: bool,
    amp_dtype=None,
    alignment_loss_type: str = "cosine",
    contrastive_temperature: float = 0.07,
):
    """Return differentiable losses for both unpaired cycle directions."""
    count = min(batch_size, text_batch["input_ids"].size(0), image_batch["input_ids"].size(0))
    if count < 1:
        return {}
    if alignment_loss_type not in {"cosine", "contrastive"}:
        raise ValueError("alignment_loss_type must be cosine or contrastive")
    if contrastive_temperature <= 0:
        raise ValueError("contrastive_temperature must be positive")
    text_batch = take_rows(text_batch, count)
    image_batch = take_rows(image_batch, count)
    paired = combine_unpaired_batches(text_batch, image_batch, tokenizer.pad_id)
    lexical_start = len(tokenizer.SPECIAL_TOKENS)
    specifications = (
        (
            "text_to_image", text_batch, IMAGE_MODALITY,
            image_offset, num_image_codes,
        ),
        (
            "image_to_text", image_batch, TEXT_MODALITY,
            lexical_start, len(tokenizer) - lexical_start,
        ),
    )
    results = {}
    for name, source, generated_modality, token_start, token_count in specifications:
        loss, metrics = _direction_loss(
            model, paired, source, generated_modality, token_start, token_count,
            tokenizer.mask_id, tokenizer.pad_id, generation_steps, temperature,
            reveal_order, minimum_confidence, cycle_weight, alignment_weight,
            alignment_loss_type, contrastive_temperature,
            eps, weight_by_t, amp_dtype,
        )
        results[name] = {"loss": loss, **metrics}
    return results
