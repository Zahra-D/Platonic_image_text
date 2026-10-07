#!/usr/bin/env python3
"""Frozen semantic probes on individual native shared/private LoRA updates.

``evaluate_jepa_modality.py`` probes the per-layer average of all four shared
adapter deltas, trained on clean text and tested on masked text.  That answers
a robustness question and dilutes any single module.  This evaluator instead
probes each native adapter update on its own -- exactly the feature that the
retrieval and counterfactual evaluations read -- on clean text for both the
probe's training and test sets, so it asks only: is the information linearly
decodable from this module's update at all?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import MultimodalCollator
from evaluate_jepa_modality import GROUPS, fit_semantic_probes, rows_and_labels, score_semantic_probes
from evaluate_shared_private_retrieval import load_model


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--caption-field", default="caption_human")
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--modules", nargs="+", default=["attn.out_proj", "mlp.3"])
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--val-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def captions(path: str, field: str, count: int) -> list[str]:
    rows = []
    with open(path) as handle:
        for line in handle:
            if len(rows) == count:
                break
            rows.append(json.loads(line)[field])
    if len(rows) != count:
        raise ValueError(f"Requested {count} rows from {path} but found {len(rows)}")
    return rows


@torch.no_grad()
def encode(model, tokenizer, model_args, texts, names, batch_size, device):
    collator = MultimodalCollator(tokenizer, model_args.num_image_codes, model_args.max_text_length)
    values = {(name, branch): [] for name in names for branch in ("shared", "private")}
    for start in range(0, len(texts), batch_size):
        batch = collator([
            {"kind": "text", "text": text, "pair_index": start + offset}
            for offset, text in enumerate(texts[start:start + batch_size])
        ])
        batch = {key: value.to(device) for key, value in batch.items()}
        _, _, shared_native, private_native = model(
            batch["input_ids"], batch["attention_mask"], batch["position_ids"],
            batch["modality_ids"], batch["route_ids"], return_shared=True,
            return_shared_private_native_by_module=True,
        )
        weights = batch["eligible_mask"].to(torch.float32).unsqueeze(-1)
        for name in names:
            for branch, native in (("shared", shared_native), ("private", private_native)):
                pooled = (native[name].float() * weights).sum(1) / weights.sum(1).clamp_min(1)
                values[(name, branch)].append(pooled.cpu())
    return {key: torch.cat(parts).numpy() for key, parts in values.items()}


def geometry(features: np.ndarray) -> dict[str, float]:
    x = F.normalize(torch.from_numpy(features).float(), dim=1)
    similarity = x @ x.T
    off_diagonal = similarity[~torch.eye(len(x), dtype=torch.bool)]
    spectrum = torch.linalg.svdvals(x - x.mean(0, keepdim=True)).square()
    weights = spectrum / spectrum.sum()
    return {
        "mean_offdiag_cosine": float(off_diagonal.mean()),
        "effective_rank": float(torch.exp(-(weights * weights.clamp_min(1e-12).log()).sum())),
    }


def main():
    args = arguments()
    device = torch.device(args.device)
    names = [f"blocks.{layer}.{module}" for layer in args.layers for module in args.modules]
    train_y = rows_and_labels(args.train_manifest, args.train_samples)
    val_y = rows_and_labels(args.val_manifest, args.val_samples)
    train_text = captions(args.train_manifest, args.caption_field, args.train_samples)
    val_text = captions(args.val_manifest, args.caption_field, args.val_samples)

    results = {"protocol": {
        "train_manifest": args.train_manifest, "val_manifest": args.val_manifest,
        "caption_field": args.caption_field, "train_samples": args.train_samples,
        "val_samples": args.val_samples, "modules": names, "seed": args.seed,
        "feature": "clean native adapter update, content-token mean, clean probe train and test",
    }, "checkpoints": {}}

    # Random features with the same shape set the floor for every score.
    rng = np.random.default_rng(args.seed)
    random_train = rng.standard_normal((args.train_samples, 384)).astype(np.float32)
    random_val = rng.standard_normal((args.val_samples, 384)).astype(np.float32)
    results["random_control"] = score_semantic_probes(
        fit_semantic_probes(random_train, train_y, args.seed), random_val, val_y
    )
    print(f"random control attribute mean={results['random_control']['attribute_group_mean']:.3f}", flush=True)

    for spec in args.checkpoint:
        label, path = spec.split("=", 1)
        model, tokenizer, model_args = load_model(path, device)
        train_features = encode(model, tokenizer, model_args, train_text, names, args.batch_size, device)
        val_features = encode(model, tokenizer, model_args, val_text, names, args.batch_size, device)
        entry = {"checkpoint": str(Path(path).resolve()), "modules": {}}
        for name in names:
            for branch in ("shared", "private"):
                key = (name, branch)
                scores = score_semantic_probes(
                    fit_semantic_probes(train_features[key], train_y, args.seed), val_features[key], val_y
                )
                scores["geometry"] = geometry(val_features[key])
                entry["modules"].setdefault(name, {})[branch] = scores
                print(
                    f"{label} {name} {branch}: count={scores['object_count']['balanced_accuracy']:.3f} "
                    f"attr_mean={scores['attribute_group_mean']:.3f} "
                    f"eff_rank={scores['geometry']['effective_rank']:.1f}", flush=True,
                )
        results["checkpoints"][label] = entry
        del model
        torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
