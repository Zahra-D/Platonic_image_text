"""Measure text/image marginal loss under Tri-LoRA branch ablations.

The base-aware modes preserve the frozen dense transformation. The pure modes
also zero each injected Linear's base weight, so they answer the stricter
question of what the shared or private LoRA transformations can do alone.
This is a read-only diagnostic: every parameter is restored after each mode.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from analyze_paired_representations import checkpoint_args
from data import ClevrMultimodalDataset, ClevrTextTokenizer, MultimodalCollator
from evaluate_checkpoint_quality import marginal_metrics
from models import TriLoRALinear
from train_multimodal import build_model


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--token-cache", required=True)
    p.add_argument("--caption-field", default="caption_human")
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--output", required=True)
    return p.parse_args()


@contextmanager
def branch_mode(model, mode):
    allowed = {
        "full", "base_shared", "base_private", "base_only",
        "pure_shared", "pure_private",
    }
    if mode not in allowed:
        raise ValueError(mode)
    saved = []
    disable_shared = mode in {"base_private", "base_only", "pure_private"}
    disable_private = mode in {"base_shared", "base_only", "pure_shared"}
    disable_base = mode in {"pure_shared", "pure_private"}
    for module in model.modules():
        if not isinstance(module, TriLoRALinear):
            continue
        names = []
        if disable_shared:
            names.append("shared_B")
        if disable_private:
            names.extend(("text_B", "image_B"))
        for name in names:
            parameter = getattr(module, name)
            saved.append((parameter, parameter.detach().clone()))
            parameter.data.zero_()
        if disable_base and module.base.weight is not None:
            parameter = module.base.weight
            saved.append((parameter, parameter.detach().clone()))
            parameter.data.zero_()
    try:
        yield
    finally:
        for parameter, value in saved:
            parameter.data.copy_(value)


def main():
    cli = arguments()
    specs = [item.split("=", 1) for item in cli.checkpoint]
    first = torch.load(specs[0][1], map_location="cpu", weights_only=False)
    tokenizer = ClevrTextTokenizer(first["text_vocabulary"])
    first_args = checkpoint_args(first)
    dataset = ClevrMultimodalDataset(
        cli.dataset_root, cli.token_cache, "paired",
        pair_manifest=cli.manifest, caption_field=cli.caption_field,
    )
    dataset = Subset(dataset, range(min(cli.num_samples, len(dataset))))
    collator = MultimodalCollator(tokenizer, first_args.num_image_codes, first_args.max_text_length)
    loader = DataLoader(dataset, cli.batch_size, shuffle=False, num_workers=0, collate_fn=collator)
    output = {
        "dataset": cli.manifest,
        "num_samples": len(dataset),
        "mask_ratio": 0.75,
        "mode_definitions": {
            "full": "W0 + shared LoRA + active modality-private LoRA",
            "base_shared": "W0 + shared LoRA; both private branches disabled",
            "base_private": "W0 + active modality-private LoRA; shared disabled",
            "base_only": "W0 only; all LoRA branches disabled",
            "pure_shared": "shared LoRA only inside injected Linear layers; W0 and private disabled",
            "pure_private": "active modality-private LoRA only; W0 and shared disabled",
        },
        "models": {},
    }
    for index, (name, path) in enumerate(specs):
        payload = first if index == 0 else torch.load(path, map_location="cpu", weights_only=False)
        model_args = checkpoint_args(payload)
        model, _ = build_model(model_args, len(tokenizer))
        model.load_state_dict(payload["model"])
        model.to(cli.device).eval()
        modes = ("full",)
        if any(isinstance(module, TriLoRALinear) for module in model.modules()):
            modes = ("full", "base_shared", "base_private", "base_only", "pure_shared", "pure_private")
        measured = {}
        for mode in modes:
            with branch_mode(model, mode):
                measured[mode] = marginal_metrics(model, loader, tokenizer, cli.device, cli.seed)
            print(name, mode, json.dumps(measured[mode], sort_keys=True), flush=True)
        output["models"][name] = {
            "checkpoint": path,
            "epoch": payload.get("epoch"),
            "step": payload.get("step"),
            "modes": measured,
        }
        del model, payload
    target = Path(cli.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
