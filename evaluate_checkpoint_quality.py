"""Evaluate marginal and paired conditional quality for multimodal checkpoints.

The marginal metrics give text-only and image-only denoising quality at a fixed
75% masking ratio.  The paired metrics then test whether a model benefits from
the *correct* other modality rather than merely modeling its own marginal.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from alignment_evaluation import _conditioned_batch, evaluate_paired_conditioning
from data import ClevrMultimodalDataset, ClevrTextTokenizer, MultimodalCollator
from multimodal_diffusion import corrupt_batch
from train_multimodal import build_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--caption-field", default="caption_human")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def parse_checkpoint(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError(f"Expected NAME=CHECKPOINT, got {specification!r}")
    name, path = specification.split("=", 1)
    return name, Path(path)


def checkpoint_args(payload) -> Namespace:
    values = dict(payload["args"])
    # Support checkpoints written before later schedule fields were added.
    values.setdefault("lora_architecture", "tri")
    values.setdefault("delete_base_weights", False)
    values.setdefault("modality_adversarial", False)
    values.setdefault("modality_discriminator_hidden", None)
    return Namespace(**values)


@torch.no_grad()
def marginal_metrics(model, loader, tokenizer, device, seed: int):
    model.eval()
    totals = {
        "text": {"loss_sum": 0.0, "correct": 0, "tokens": 0},
        "image": {"loss_sum": 0.0, "correct": 0, "tokens": 0},
    }
    for batch_index, cpu_batch in enumerate(loader):
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        for objective in ("text", "image"):
            conditioned = _conditioned_batch(batch, objective, "null")
            # Identical deterministic masks for every compared checkpoint.
            torch.manual_seed(seed + batch_index * 17 + (0 if objective == "text" else 1))
            corrupted, masked, _ = corrupt_batch(
                conditioned["input_ids"], conditioned["eligible_mask"],
                conditioned["modality_ids"], tokenizer.mask_id, objective=objective,
                fixed_t=0.75,
            )
            logits = model(
                corrupted, conditioned["attention_mask"], conditioned["position_ids"],
                conditioned["modality_ids"], conditioned["route_ids"],
            )
            token_loss = F.cross_entropy(
                logits.transpose(1, 2).float(), conditioned["input_ids"], reduction="none"
            )
            totals[objective]["loss_sum"] += float(token_loss[masked].sum())
            totals[objective]["correct"] += int(
                logits.argmax(dim=-1)[masked].eq(conditioned["input_ids"][masked]).sum()
            )
            totals[objective]["tokens"] += int(masked.sum())
    return {
        name: {
            "mask_ratio": 0.75,
            "loss": values["loss_sum"] / max(1, values["tokens"]),
            "accuracy": values["correct"] / max(1, values["tokens"]),
            "tokens": values["tokens"],
        }
        for name, values in totals.items()
    }


def main():
    cli = parse_args()
    specifications = [parse_checkpoint(item) for item in cli.checkpoint]
    first = torch.load(specifications[0][1], map_location="cpu", weights_only=False)
    tokenizer = ClevrTextTokenizer(first["text_vocabulary"])
    first_args = checkpoint_args(first)
    dataset = ClevrMultimodalDataset(
        cli.dataset_root, cli.token_cache, "paired",
        pair_manifest=cli.manifest, caption_field=cli.caption_field,
    )
    dataset = Subset(dataset, range(min(cli.num_samples, len(dataset))))
    collator = MultimodalCollator(tokenizer, first_args.num_image_codes, first_args.max_text_length)
    loader = DataLoader(dataset, cli.batch_size, shuffle=False, num_workers=0, collate_fn=collator)

    results = {
        "dataset_root": cli.dataset_root,
        "manifest": cli.manifest,
        "num_samples": len(dataset),
        "marginal_mask_ratio": 0.75,
        "paired_mask_ratios": [0.75, 1.0],
        "models": {},
    }
    for model_index, (name, path) in enumerate(specifications):
        payload = first if model_index == 0 else torch.load(path, map_location="cpu", weights_only=False)
        if payload["text_vocabulary"] != first["text_vocabulary"]:
            raise ValueError(f"Vocabulary differs for {name}")
        model_args = checkpoint_args(payload)
        model, _ = build_model(model_args, len(tokenizer))
        model.load_state_dict(payload["model"])
        model.to(cli.device).eval()
        torch.manual_seed(cli.seed)
        results["models"][name] = {
            "checkpoint": str(path),
            "epoch": payload.get("epoch"),
            "step": payload.get("step"),
            "marginal": marginal_metrics(model, loader, tokenizer, cli.device, cli.seed),
            "paired": evaluate_paired_conditioning(
                model, loader, tokenizer.mask_id, cli.device, None,
                mask_ratios=(0.75, 1.0), control_ratios=(0.75, 1.0), seed=cli.seed,
            ),
        }
        print(name, json.dumps(results["models"][name]["marginal"], sort_keys=True))
        del model, payload

    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
