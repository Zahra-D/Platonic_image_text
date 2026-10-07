"""How many paired anchor examples does it take to align text and image reps?

Read-only diagnostic. For each checkpoint's most relevant representation (the
pooled shared Tri-LoRA branch for Tri-LoRA checkpoints, the pooled private
branch as a contrast baseline, and the whole final-layer representation for
the dense checkpoint, which has no branch split), this measures how well an
*orthogonal Procrustes* transform fit from only k paired (text, image) anchor
examples aligns the rest of the text distribution onto the image distribution.

This is the same technique used for cross-lingual word-embedding alignment
(Conneau et al., MUSE): center both anchor sets, take the SVD of their cross-
covariance, and the closest rotation (+ isotropic scale) is read off the
singular vectors. It answers "how easy is it to align" as a sample-efficiency
question -- a representation that only needs a handful of anchors to reach
good cross-modal retrieval already has closely-corresponding geometry; one
that needs hundreds does not.

Evaluation always happens on a held-out set that is *never* used as an anchor
at any k, fixed once per model/level so every point on the curve is scored on
the same examples -- only the anchor count changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from analyze_paired_representations import checkpoint_args, checkpoint_specification, representation_metrics
from data import ClevrMultimodalDataset, ClevrTextTokenizer, MultimodalCollator
from extract_layer_alignment_embeddings import encode_modalities
from train_multimodal import build_model

ANCHOR_COUNTS = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 320, 400]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--label", action="append", default=[], metavar="NAME=Display label")
    parser.add_argument("--run-id", action="append", default=[], metavar="NAME=W&B run id")
    parser.add_argument("--state", action="append", default=[], metavar="NAME=Run state")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--caption-field", default="caption_human")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--held-out-size", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output", default="outputs/paired_representation_analysis/procrustes_alignment_efficiency.json")
    return parser.parse_args()


def name_value_map(values: list[str]) -> dict[str, str]:
    return {name: payload for name, _, payload in (v.partition("=") for v in values)}


def fit_procrustes(x_anchor: np.ndarray, y_anchor: np.ndarray):
    x_mean = x_anchor.mean(axis=0)
    y_mean = y_anchor.mean(axis=0)
    xc = x_anchor - x_mean
    yc = y_anchor - y_mean
    u, s, vt = np.linalg.svd(xc.T @ yc)
    r = u @ vt
    denom = float((xc ** 2).sum())
    scale = float(s.sum() / denom) if denom > 1e-12 else 1.0
    return r, scale, x_mean, y_mean


def apply_procrustes(x: np.ndarray, r: np.ndarray, scale: float, x_mean: np.ndarray, y_mean: np.ndarray) -> np.ndarray:
    return (x - x_mean) @ r * scale + y_mean


def retrieval_and_cosine(text_aligned: np.ndarray, image: np.ndarray, ks=(1, 5, 10)) -> dict:
    tn = text_aligned / np.linalg.norm(text_aligned, axis=1, keepdims=True).clip(min=1e-8)
    im = image / np.linalg.norm(image, axis=1, keepdims=True).clip(min=1e-8)
    sim = tn @ im.T
    order = np.argsort(-sim, axis=1)
    target = np.arange(sim.shape[0])
    ranks = np.array([np.nonzero(order[i] == target[i])[0][0] for i in range(len(target))])
    metrics = {f"r@{k}": float(np.mean(ranks < k)) for k in ks}
    metrics["mean_cosine"] = float((tn * im).sum(axis=1).mean())
    return metrics


def sweep_model_level(text: np.ndarray, image: np.ndarray, held_out_size: int, repeats: int, seed: int) -> dict:
    n = len(text)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    held_idx = perm[:held_out_size]
    pool_idx = perm[held_out_size:]

    unaligned = retrieval_and_cosine(text[held_idx], image[held_idx])
    diagnostics = representation_metrics(
        torch.from_numpy(text), torch.from_numpy(image), permutations=20, seed=seed,
    )["collapse_diagnostics"]

    curve = []
    for k in ANCHOR_COUNTS:
        if k > len(pool_idx):
            break
        trial_metrics = []
        for trial in range(repeats):
            trial_rng = np.random.default_rng(seed + trial)
            anchor_idx = trial_rng.choice(pool_idx, size=k, replace=False)
            r, scale, x_mean, y_mean = fit_procrustes(text[anchor_idx], image[anchor_idx])
            aligned = apply_procrustes(text[held_idx], r, scale, x_mean, y_mean)
            trial_metrics.append(retrieval_and_cosine(aligned, image[held_idx]))
        point = {"k": k}
        for key in trial_metrics[0]:
            vals = [m[key] for m in trial_metrics]
            point[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        curve.append(point)

    return {
        "held_out_size": int(held_out_size),
        "chance_r@1": 1.0 / held_out_size,
        "unaligned": unaligned,
        "diagnostics": diagnostics,
        "curve": curve,
    }


def anchors_needed_for_fraction(curve: list[dict], target_value: float, key: str = "r@1") -> int | None:
    for point in curve:
        if point[key]["mean"] >= target_value:
            return point["k"]
    return None


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    labels = name_value_map(args.label)
    run_ids = name_value_map(args.run_id)
    states = name_value_map(args.state)

    specifications = [checkpoint_specification(value) for value in args.checkpoint]
    first_payload = torch.load(specifications[0][1], map_location="cpu", weights_only=False)
    tokenizer = ClevrTextTokenizer(first_payload["text_vocabulary"])
    first_args = checkpoint_args(first_payload)
    dataset = ClevrMultimodalDataset(
        args.split_dir, args.token_cache, "paired",
        pair_manifest=args.pair_manifest, caption_field=args.caption_field,
    )
    dataset = Subset(dataset, range(min(args.num_samples, len(dataset))))
    collator = MultimodalCollator(tokenizer, first_args.num_image_codes, first_args.max_text_length)
    loader = DataLoader(
        dataset, args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collator, pin_memory=args.device.startswith("cuda"),
    )

    output = {
        "dataset": str(args.pair_manifest),
        "num_samples": len(dataset),
        "held_out_size": args.held_out_size,
        "repeats": args.repeats,
        "anchor_counts": ANCHOR_COUNTS,
        "models": {},
    }

    for model_index, (name, path) in enumerate(specifications):
        print(f"[{model_index + 1}/{len(specifications)}] {name} <- {path}")
        payload = first_payload if model_index == 0 else torch.load(path, map_location="cpu", weights_only=False)
        if payload["text_vocabulary"] != first_payload["text_vocabulary"]:
            raise ValueError(f"Tokenizer vocabulary differs for {name}")
        model_args = checkpoint_args(payload)
        model, _ = build_model(model_args, len(tokenizer))
        model.load_state_dict(payload["model"])
        model.to(args.device)

        representations = encode_modalities(model, loader, args.device)
        levels = {}
        if "shared_lora_aggregate" in representations:
            for level, key in (("shared_lora_aggregate", "shared"), ("private_lora_aggregate", "private")):
                text_np = representations[level]["text"].numpy().astype(np.float64)
                image_np = representations[level]["image"].numpy().astype(np.float64)
                levels[key] = sweep_model_level(text_np, image_np, args.held_out_size, args.repeats, args.seed)
        else:
            text_np = representations["final"]["text"].numpy().astype(np.float64)
            image_np = representations["final"]["image"].numpy().astype(np.float64)
            levels["final"] = sweep_model_level(text_np, image_np, args.held_out_size, args.repeats, args.seed)

        output["models"][name] = {
            "label": labels.get(name, name),
            "run_id": run_ids.get(name, ""),
            "state": states.get(name, "unknown"),
            "primary_level": "shared" if "shared" in levels else "final",
            "levels": levels,
        }
        del model, payload
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1e6:.2f} MB)")

    for name, model_out in output["models"].items():
        for key, level_out in model_out["levels"].items():
            k50 = anchors_needed_for_fraction(level_out["curve"], 0.5)
            k80 = anchors_needed_for_fraction(level_out["curve"], 0.8)
            print(f"{name:32s} {key:8s} unaligned r@1={level_out['unaligned']['r@1']:.3f}  "
                  f"anchors for r@1>=0.5: {k50}  r@1>=0.8: {k80}")


if __name__ == "__main__":
    main()
