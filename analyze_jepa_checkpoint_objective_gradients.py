"""Measure diffusion/JEPA/SIGReg gradient influence in a saved JEPA checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from analyze_layer_gradient_conflict import one_modality
from analyze_paired_representations import checkpoint_args
from data import ClevrMultimodalDataset, ClevrTextTokenizer, MultimodalCollator
from multimodal_diffusion import corrupt_batch, masked_loss
from objective_gradient_diagnostics import objective_shared_gradient_metrics
from shared_jepa import shared_latent_jepa_loss
from sigreg import sigreg_loss
from train_multimodal import build_model


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--caption-field", default="caption_human")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--jepa-layers", type=int, nargs="+", default=None)
    parser.add_argument("--sigreg-layers", type=int, nargs="+", default=None)
    parser.add_argument("--jepa-loss", choices=["mse", "normalized_mse", "cosine"], default="mse")
    parser.add_argument("--jepa-weight", type=float, default=None)
    parser.add_argument("--sigreg-weight", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def mean_metrics(rows):
    keys = sorted(set().union(*(row.keys() for row in rows)))
    return {
        key: sum(row[key] for row in rows if key in row and math.isfinite(row[key]))
        / max(1, sum(key in row and math.isfinite(row[key]) for row in rows))
        for key in keys
    }


def main():
    args = arguments()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = checkpoint_args(payload)
    if not getattr(model_args, "shared_jepa", False):
        raise ValueError("Checkpoint was not constructed with shared JEPA")
    tokenizer = ClevrTextTokenizer(payload["text_vocabulary"])
    dataset = ClevrMultimodalDataset(
        args.split_dir, args.token_cache, "paired", pair_manifest=args.pair_manifest,
        caption_field=args.caption_field,
    )
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[:args.num_samples].tolist()
    loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
        num_workers=0,
        collate_fn=MultimodalCollator(
            tokenizer, model_args.num_image_codes, model_args.max_text_length
        ),
    )
    model, _ = build_model(model_args, len(tokenizer))
    model.load_state_dict(payload["model"])
    model.to(args.device)
    rows = defaultdict(list)
    for batch_index, paired in enumerate(loader):
        for modality_index, modality in enumerate(("text", "image")):
            torch.manual_seed(args.seed + batch_index * 10 + modality_index)
            batch = one_modality(paired, modality, args.device)
            corrupted, mask, t = corrupt_batch(
                batch["input_ids"], batch["eligible_mask"], batch["modality_ids"],
                tokenizer.mask_id, objective=modality, fixed_t=args.mask_ratio,
            )
            model.train()
            output = model(
                corrupted, batch["attention_mask"], batch["position_ids"],
                batch["modality_ids"], batch["route_ids"], return_shared=True,
                return_shared_by_layer=True, return_shared_tokens_by_layer=True,
            )
            task = masked_loss(
                output[0], batch["input_ids"], mask, t,
                not getattr(model_args, "unweighting", False),
            )
            model.eval()
            with torch.no_grad():
                clean = model(
                    batch["input_ids"], batch["attention_mask"], batch["position_ids"],
                    batch["modality_ids"], batch["route_ids"], return_shared=True,
                    return_shared_tokens_by_layer=True,
                )
            model.train()
            jepa, layer_jepa, layer_cosine = shared_latent_jepa_loss(
                model, output[3], clean[3], mask,
                layers=args.jepa_layers, loss_type=args.jepa_loss,
            )
            selected_sigreg_layers = (
                set(output[2]) if args.sigreg_layers is None else set(args.sigreg_layers)
            )
            layer_sigreg = {
                layer: sigreg_loss(
                    representation,
                    num_slices=getattr(model_args, "sigreg_num_slices", 256),
                    num_points=getattr(model_args, "sigreg_num_points", 17),
                    t_max=getattr(model_args, "sigreg_t_max", 3.0),
                    seed=args.seed + batch_index + layer,
                )
                for layer, representation in output[2].items()
                if layer in selected_sigreg_layers
            }
            sigreg = torch.stack(list(layer_sigreg.values())).mean()
            jepa_weight = (
                getattr(model_args, "shared_jepa_weight", 0.1)
                if args.jepa_weight is None else args.jepa_weight
            )
            sigreg_weight = (
                getattr(model_args, "sigreg_weight", 0.01)
                if args.sigreg_weight is None else args.sigreg_weight
            )
            weighted_jepa = jepa_weight * jepa
            weighted_sigreg = sigreg_weight * sigreg
            gradient_metrics = objective_shared_gradient_metrics(
                model,
                {"diffusion": task, "jepa": weighted_jepa, "sigreg": weighted_sigreg},
                selected_layers=(set(args.jepa_layers) if args.jepa_layers else None),
            )
            scalar_metrics = {
                "diffusion_loss": task.detach().item(),
                "jepa_loss": jepa.detach().item(),
                "weighted_jepa": weighted_jepa.detach().item(),
                "sigreg_loss": sigreg.detach().item(),
                "weighted_sigreg": weighted_sigreg.detach().item(),
                **{f"layer_{layer:02d}/jepa_loss": value for layer, value in layer_jepa.items()},
                **{f"layer_{layer:02d}/jepa_cosine": value for layer, value in layer_cosine.items()},
                **{f"layer_{layer:02d}/sigreg_loss": value.detach().item() for layer, value in layer_sigreg.items()},
                **gradient_metrics,
            }
            rows[modality].append(scalar_metrics)
            del (
                output, clean, task, jepa, sigreg, weighted_jepa, weighted_sigreg,
                layer_jepa, layer_cosine, layer_sigreg, gradient_metrics,
            )
            model.zero_grad(set_to_none=True)
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            print(
                f"completed {modality} batch {batch_index + 1}/{len(loader)}",
                flush=True,
            )
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_step": payload.get("step"),
        "num_samples": min(args.num_samples, len(dataset)),
        "batch_size": args.batch_size,
        "mask_ratio": args.mask_ratio,
        "jepa_layers": args.jepa_layers or "all",
        "sigreg_layers": args.sigreg_layers or "all",
        "jepa_loss": args.jepa_loss,
        "jepa_weight": (
            getattr(model_args, "shared_jepa_weight", 0.1)
            if args.jepa_weight is None else args.jepa_weight
        ),
        "sigreg_weight": (
            getattr(model_args, "sigreg_weight", 0.01)
            if args.sigreg_weight is None else args.sigreg_weight
        ),
        "metrics": {modality: mean_metrics(values) for modality, values in rows.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
