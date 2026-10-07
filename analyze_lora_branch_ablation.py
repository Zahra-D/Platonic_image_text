"""Read-only paired validation with shared/private Tri-LoRA branch ablations."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from alignment_evaluation import evaluate_paired_conditioning
from analyze_paired_representations import checkpoint_args
from data import ClevrMultimodalDataset, ClevrTextTokenizer, MultimodalCollator
from models import TriLoRALinear
from train_multimodal import build_model


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--caption-field", default="caption_human")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", default="outputs/paired_representation_analysis/branch_ablation.json")
    return parser.parse_args()


@contextmanager
def branch_mode(model, mode):
    if mode not in {"full", "shared_only", "private_only", "no_lora_branches"}:
        raise ValueError(mode)
    saved = []
    for module in model.modules():
        if not isinstance(module, TriLoRALinear):
            continue
        disabled = []
        if mode in {"shared_only", "no_lora_branches"}:
            disabled.extend(("text_B", "image_B"))
        if mode in {"private_only", "no_lora_branches"}:
            disabled.append("shared_B")
        for name in disabled:
            parameter = getattr(module, name)
            saved.append((parameter, parameter.detach().clone()))
            parameter.data.zero_()
    try:
        yield
    finally:
        for parameter, value in saved:
            parameter.data.copy_(value)


def compact(metrics):
    result = {}
    for direction in ("text_to_image", "image_to_text"):
        prefix = f"val/paired/{direction}/t1"
        for metric in ("matched_loss", "shuffled_loss", "null_loss", "shuffle_gap", "context_gain", "matched_accuracy"):
            result[f"{direction}/{metric}"] = metrics[f"{prefix}/{metric}"]
    result["mean_matched_loss"] = sum(result[f"{d}/matched_loss"] for d in ("text_to_image", "image_to_text")) / 2
    return result


def main():
    args = arguments()
    specs = [value.split("=", 1) for value in args.checkpoint]
    first = torch.load(specs[0][1], map_location="cpu", weights_only=False)
    tokenizer = ClevrTextTokenizer(first["text_vocabulary"])
    first_args = checkpoint_args(first)
    dataset = ClevrMultimodalDataset(
        args.split_dir, args.token_cache, "paired",
        pair_manifest=args.pair_manifest, caption_field=args.caption_field,
    )
    dataset = Subset(dataset, range(min(args.num_samples, len(dataset))))
    collator = MultimodalCollator(tokenizer, first_args.num_image_codes, first_args.max_text_length)
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=0, collate_fn=collator)
    results = {"dataset": args.pair_manifest, "num_samples": len(dataset), "models": {}}

    for index, (name, path) in enumerate(specs):
        payload = first if index == 0 else torch.load(path, map_location="cpu", weights_only=False)
        model_args = checkpoint_args(payload)
        model, _ = build_model(model_args, len(tokenizer))
        model.load_state_dict(payload["model"])
        model.to(args.device)
        is_tri = any(isinstance(module, TriLoRALinear) for module in model.modules())
        modes = ("full", "shared_only", "private_only", "no_lora_branches") if is_tri else ("full",)
        model_results = {}
        for mode in modes:
            with branch_mode(model, mode):
                metrics = evaluate_paired_conditioning(
                    model, loader, tokenizer.mask_id, args.device, None,
                    mask_ratios=(1.0,), control_ratios=(1.0,), seed=args.seed,
                )
            model_results[mode] = compact(metrics)
        results["models"][name] = {
            "checkpoint": {"path": path, "epoch": payload.get("epoch"), "step": payload.get("step")},
            **model_results,
        }
        del model, payload

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
