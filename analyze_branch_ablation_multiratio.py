"""No-base Tri-LoRA branch ablation at one or more marginal mask ratios.

The diagnostic keeps embeddings, norms, residual paths, and heads active.  It
zeros the disabled LoRA B matrices in every injected layer, restores them after
each measurement, and never modifies the checkpoint on disk.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from alignment_evaluation import _conditioned_batch
from analyze_paired_representations import checkpoint_args
from data import ClevrMultimodalDataset, ClevrTextTokenizer, MultimodalCollator
from multimodal_diffusion import corrupt_batch
from models import TriLoRALinear
from train_multimodal import build_model


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--caption-field", default="caption_human")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mask-ratio", action="append", type=float, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


@contextmanager
def branch_mode(model, mode):
    if mode not in {"full", "shared_only", "private_only"}:
        raise ValueError(mode)
    saved = []
    for module in model.modules():
        if not isinstance(module, TriLoRALinear):
            continue
        disabled = []
        if mode == "shared_only":
            disabled.extend(("text_B", "image_B"))
        elif mode == "private_only":
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


@torch.no_grad()
def marginal_metrics(model, loader, tokenizer, device, seed, mask_ratio):
    totals = {
        "text": {"loss_sum": 0.0, "correct": 0, "tokens": 0},
        "image": {"loss_sum": 0.0, "correct": 0, "tokens": 0},
    }
    model.eval()
    for batch_index, cpu_batch in enumerate(loader):
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        for objective in ("text", "image"):
            conditioned = _conditioned_batch(batch, objective, "null")
            torch.manual_seed(seed + batch_index * 17 + (0 if objective == "text" else 1))
            corrupted, masked, _ = corrupt_batch(
                conditioned["input_ids"], conditioned["eligible_mask"],
                conditioned["modality_ids"], tokenizer.mask_id,
                objective=objective, fixed_t=mask_ratio,
            )
            logits = model(
                corrupted, conditioned["attention_mask"], conditioned["position_ids"],
                conditioned["modality_ids"], conditioned["route_ids"],
            )
            losses = F.cross_entropy(
                logits.transpose(1, 2).float(), conditioned["input_ids"], reduction="none"
            )
            totals[objective]["loss_sum"] += float(losses[masked].sum())
            totals[objective]["correct"] += int(
                logits.argmax(dim=-1)[masked].eq(conditioned["input_ids"][masked]).sum()
            )
            totals[objective]["tokens"] += int(masked.sum())
    return {
        name: {
            "mask_ratio": mask_ratio,
            "loss": values["loss_sum"] / max(1, values["tokens"]),
            "accuracy": values["correct"] / max(1, values["tokens"]),
            "tokens": values["tokens"],
        }
        for name, values in totals.items()
    }


def main():
    args = arguments()
    ratios = sorted(set(args.mask_ratio))
    if any(not 0 < ratio <= 1 for ratio in ratios):
        raise ValueError("Mask ratios must be in (0, 1]")
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
    results = {
        "dataset": args.pair_manifest,
        "num_samples": len(dataset),
        "mask_ratios": ratios,
        "mode_definitions": {
            "full": "normal model forward; shared and active modality-private LoRA enabled",
            "shared_only": "all text-private and image-private LoRA B matrices zeroed in every layer",
            "private_only": "all shared LoRA B matrices zeroed in every layer",
            "dense_full": "dense reference has no LoRA branches; normal full forward",
        },
        "models": {},
    }
    for index, (name, path) in enumerate(specs):
        payload = first if index == 0 else torch.load(path, map_location="cpu", weights_only=False)
        if payload["text_vocabulary"] != first["text_vocabulary"]:
            raise ValueError(f"Tokenizer vocabulary differs for {name}")
        model_args = checkpoint_args(payload)
        model, _ = build_model(model_args, len(tokenizer))
        model.load_state_dict(payload["model"])
        model.to(args.device).eval()
        is_tri = any(isinstance(module, TriLoRALinear) for module in model.modules())
        if is_tri and not model_args.delete_base_weights:
            raise ValueError(f"{name} is not a no-base checkpoint")
        modes = ("full", "shared_only", "private_only") if is_tri else ("full",)
        model_results = {}
        for ratio in ratios:
            ratio_results = {}
            for mode in modes:
                with branch_mode(model, mode):
                    ratio_results[mode] = marginal_metrics(
                        model, loader, tokenizer, args.device, args.seed, ratio
                    )
                print(name, ratio, mode, json.dumps(ratio_results[mode], sort_keys=True), flush=True)
            model_results[f"{ratio:g}"] = ratio_results
        results["models"][name] = {
            "checkpoint": {"path": path, "epoch": payload.get("epoch"), "step": payload.get("step")},
            "train_mode": model_args.train_mode,
            "delete_base_weights": bool(model_args.delete_base_weights),
            "ratios": model_results,
        }
        del model, payload
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
