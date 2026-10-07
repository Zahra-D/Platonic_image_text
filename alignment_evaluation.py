"""Paired conditional validation for the multimodal masked-diffusion model."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from models import condition_target_route_ids
from unpaired_backtranslation import extract_modality


MODALITY_ID = {"text": 1, "image": 2}


def _conditioned_batch(
    batch: dict[str, torch.Tensor], objective: str, context: str
) -> dict[str, torch.Tensor]:
    """Rebuild a paired batch with matched, shuffled, or absent context.

    The target always comes from the original row.  Shuffling therefore tests
    the value of the correspondence rather than changing the prediction target.
    """
    if objective not in MODALITY_ID:
        raise ValueError("objective must be text or image")
    if context not in {"matched", "shuffled", "null"}:
        raise ValueError("context must be matched, shuffled, or null")

    target_id = MODALITY_ID[objective]
    context_id = MODALITY_ID["image" if objective == "text" else "text"]
    batch_size = batch["input_ids"].size(0)
    if context == "shuffled" and batch_size < 2:
        raise ValueError("shuffled context requires a batch of at least two pairs")

    fields = ("input_ids", "position_ids", "modality_ids", "eligible_mask")
    segments: dict[int, list[dict[str, torch.Tensor]]] = {target_id: [], context_id: []}
    for row in range(batch_size):
        valid = batch["attention_mask"][row]
        for modality_id in (target_id, context_id):
            select = valid & batch["modality_ids"][row].eq(modality_id)
            segments[modality_id].append({field: batch[field][row, select] for field in fields})

    rows = []
    for row in range(batch_size):
        parts = []
        if context != "null":
            source = row if context == "matched" else (row + 1) % batch_size
            parts.append(segments[context_id][source])
        parts.append(segments[target_id][row])
        rows.append({field: torch.cat([part[field] for part in parts]) for field in fields})

    max_length = max(row["input_ids"].numel() for row in rows)
    device = batch["input_ids"].device
    result = {
        "input_ids": torch.zeros((batch_size, max_length), dtype=torch.long, device=device),
        "attention_mask": torch.zeros((batch_size, max_length), dtype=torch.bool, device=device),
        "position_ids": torch.zeros((batch_size, max_length), dtype=torch.long, device=device),
        "modality_ids": torch.zeros((batch_size, max_length), dtype=torch.long, device=device),
        "eligible_mask": torch.zeros((batch_size, max_length), dtype=torch.bool, device=device),
        "route_ids": torch.full((batch_size, max_length), -1, dtype=torch.long, device=device),
    }
    # Padding token values are never attended.  Preserve the real pad id when possible.
    if (~batch["attention_mask"]).any():
        result["input_ids"].fill_(int(batch["input_ids"][~batch["attention_mask"]][0]))
    for index, row in enumerate(rows):
        length = row["input_ids"].numel()
        result["attention_mask"][index, :length] = True
        for field in fields:
            result[field][index, :length] = row[field]
    result["route_ids"][result["modality_ids"].eq(MODALITY_ID["text"])] = 0
    result["route_ids"][result["modality_ids"].eq(MODALITY_ID["image"])] = 1
    return result


def _target_mask_spec(batch: dict[str, torch.Tensor], objective: str, ratio: float, seed: int):
    """Sample masks in target-local coordinates so controls use identical targets."""
    if not 0 < ratio <= 1:
        raise ValueError("mask ratio must be in (0, 1]")
    target_id = MODALITY_ID[objective]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    specifications = []
    for row in range(batch["input_ids"].size(0)):
        count = int((batch["eligible_mask"][row] & batch["modality_ids"][row].eq(target_id)).sum())
        selected = torch.rand(count, generator=generator).lt(ratio) if ratio < 1 else torch.ones(count, dtype=torch.bool)
        if count and not selected.any():
            selected[torch.randint(count, (1,), generator=generator)] = True
        specifications.append(selected)
    return specifications


def _apply_target_mask(batch, objective, specifications, mask_token_id):
    target_id = MODALITY_ID[objective]
    masked = torch.zeros_like(batch["eligible_mask"])
    for row, specification in enumerate(specifications):
        positions = (batch["eligible_mask"][row] & batch["modality_ids"][row].eq(target_id)).nonzero().flatten()
        if len(positions) != len(specification):
            raise ValueError("Target changed between paired-validation controls")
        masked[row, positions] = specification.to(masked.device)
    return batch["input_ids"].masked_fill(masked, mask_token_id), masked


@torch.no_grad()
def evaluate_paired_conditioning(
    model,
    loader,
    mask_token_id: int,
    device: str,
    amp_dtype,
    mask_ratios=(0.5, 0.75, 1.0),
    control_ratios=(0.75, 1.0),
    seed: int = 13,
    directions=("text_to_image", "image_to_text"),
    asymmetric_condition_target: bool = False,
) -> dict[str, float]:
    """Measure requested conditional NLLs and matched-context advantages."""
    direction_to_objective = {
        "text_to_image": "image",
        "image_to_text": "text",
    }
    unknown = set(directions) - set(direction_to_objective)
    if unknown:
        raise ValueError(f"Unknown paired-validation directions: {sorted(unknown)}")
    was_training = model.training
    model.eval()
    totals = defaultdict(lambda: {"loss": 0.0, "correct": 0, "tokens": 0})
    control_ratios = {float(value) for value in control_ratios}
    use_shared_translation = bool(getattr(model, "shared_translation_layers", ()))

    for batch_index, cpu_batch in enumerate(loader):
        batch = {key: value.to(device, non_blocking=True) for key, value in cpu_batch.items()}
        for direction in directions:
            objective = direction_to_objective[direction]
            matched = _conditioned_batch(batch, objective, "matched")
            target_id = MODALITY_ID[objective]
            context_id = MODALITY_ID["image" if objective == "text" else "text"]
            translation_target = (
                extract_modality(matched, target_id, 0)
                if use_shared_translation else None
            )
            for ratio_index, ratio_value in enumerate(mask_ratios):
                ratio = float(ratio_value)
                specifications = _target_mask_spec(
                    matched, objective, ratio, seed + batch_index * 1009 + ratio_index * 17 + MODALITY_ID[objective]
                )
                if use_shared_translation:
                    translation_corrupted, translation_masked = _apply_target_mask(
                        translation_target, objective, specifications, mask_token_id
                    )
                contexts = ("matched", "shuffled", "null") if ratio in control_ratios else ("matched",)
                for context in contexts:
                    conditioned = matched if context == "matched" else _conditioned_batch(batch, objective, context)
                    if asymmetric_condition_target and not use_shared_translation:
                        conditioned = dict(conditioned)
                        conditioned["route_ids"] = condition_target_route_ids(
                            conditioned["route_ids"], conditioned["modality_ids"], MODALITY_ID[objective]
                        )
                    if use_shared_translation:
                        condition_source = matched if context == "null" else conditioned
                        translation_condition = extract_modality(condition_source, context_id, 0)
                        corrupted, masked = translation_corrupted, translation_masked
                        scoring_batch = translation_target
                    else:
                        corrupted, masked = _apply_target_mask(conditioned, objective, specifications, mask_token_id)
                        scoring_batch = conditioned
                    with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
                        if use_shared_translation:
                            logits = model.forward_shared_translation(
                                translation_condition["input_ids"],
                                translation_condition["attention_mask"],
                                translation_condition["position_ids"],
                                translation_condition["modality_ids"],
                                translation_condition["route_ids"],
                                corrupted,
                                translation_target["attention_mask"],
                                translation_target["position_ids"],
                                translation_target["modality_ids"],
                                translation_target["route_ids"],
                                disable_bridge=context == "null",
                                condition_kv_mask=translation_condition["eligible_mask"],
                            )
                        else:
                            logits = model(
                                corrupted,
                                conditioned["attention_mask"],
                                conditioned["position_ids"],
                                conditioned["modality_ids"],
                                conditioned["route_ids"],
                            )
                    token_losses = F.cross_entropy(
                        logits.transpose(1, 2).float(), scoring_batch["input_ids"], reduction="none"
                    )
                    key = (direction, ratio, context)
                    totals[key]["loss"] += float(token_losses[masked].sum())
                    totals[key]["correct"] += int(
                        logits.argmax(dim=-1)[masked].eq(scoring_batch["input_ids"][masked]).sum()
                    )
                    totals[key]["tokens"] += int(masked.sum())

    metrics = {}
    for (direction, ratio, context), values in totals.items():
        suffix = f"t{ratio:g}/{context}"
        count = max(1, values["tokens"])
        metrics[f"val/paired/{direction}/{suffix}_loss"] = values["loss"] / count
        metrics[f"val/paired/{direction}/{suffix}_accuracy"] = values["correct"] / count
    for direction in directions:
        for ratio in control_ratios:
            prefix = f"val/paired/{direction}/t{ratio:g}"
            matched = metrics.get(f"{prefix}/matched_loss")
            if matched is None:
                continue
            metrics[f"{prefix}/shuffle_gap"] = metrics[f"{prefix}/shuffled_loss"] - matched
            metrics[f"{prefix}/context_gain"] = metrics[f"{prefix}/null_loss"] - matched
    model.train(was_training)
    return metrics
