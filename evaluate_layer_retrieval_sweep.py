#!/usr/bin/env python3
"""Cross-pattern retrieval at every Transformer layer, dense or Tri-LoRA.

For each block l it scores three whole features, all of which exist in every
model so dense and Tri-LoRA are compared like for like:

* ``block{l}``: residual-stream hidden state after block l;
* ``attn{l}``: full attention sublayer output written into the residual stream
  (``attn.out_proj``; for Tri-LoRA shared + private + bias);
* ``mlp{l}``: full MLP sublayer output written into the residual stream
  (``mlp.3``; for Tri-LoRA shared + private + bias).

plus ``embedding``: the input token embeddings before any block.

Pooling (content-token mean, L2 normalization), the held-out gallery, and the
metric definitions are those of ``evaluate_shared_private_retrieval``.  The
metrics are computed with a vectorized implementation that is checked against
that evaluator's ``query_statistics`` on every run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import MultimodalCollator
from evaluate_shared_private_retrieval import load_model, query_statistics


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--gallery", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


@torch.no_grad()
def encode_all_layers(model, tokenizer, model_args, rows, batch_size, device):
    collator = MultimodalCollator(tokenizer, model_args.num_image_codes, model_args.max_text_length)
    captured, hooks = {}, []

    def keep(name):
        def hook(_module, _inputs, output):
            captured[name] = output
        return hook

    hooks.append(model.token_embed.register_forward_hook(keep("embedding")))
    for index, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(keep(f"block{index}")))
        hooks.append(block.attn.out_proj.register_forward_hook(keep(f"attn{index}")))
        hooks.append(block.mlp[3].register_forward_hook(keep(f"mlp{index}")))
    values: dict[str, list[torch.Tensor]] = {}
    try:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            batch = collator([
                {"kind": "text", "text": row["caption"], "pair_index": row["index"]} for row in chunk
            ])
            batch = {name: value.to(device) for name, value in batch.items()}
            captured.clear()
            model(batch["input_ids"], batch["attention_mask"], batch["position_ids"],
                  batch["modality_ids"], batch["route_ids"])
            weights = batch["eligible_mask"].to(torch.float32).unsqueeze(-1)
            for name, tensor in captured.items():
                pooled = (tensor.float() * weights).sum(1) / weights.sum(1).clamp_min(1)
                values.setdefault(name, []).append(F.normalize(pooled, dim=1).cpu())
    finally:
        for hook in hooks:
            hook.remove()
    return {name: torch.cat(parts).numpy() for name, parts in values.items()}


def per_query_cross_pattern(features: np.ndarray, world: np.ndarray, pattern: np.ndarray, device) -> np.ndarray:
    """[R@1, R@5, R@10, reciprocal rank] per query; same rule as query_statistics."""
    x = torch.from_numpy(features).to(device)
    similarity = x @ x.T
    similarity.fill_diagonal_(-float("inf"))
    order = similarity.argsort(dim=1, descending=True)
    w = torch.from_numpy(world).to(device)
    p = torch.from_numpy(pattern).to(device)
    positive = (w[order] == w[:, None]) & (p[order] != p[:, None])
    rank = positive.float().argmax(dim=1) + 1  # every query has 6 positives
    rank = rank.float()
    return torch.stack([rank <= 1, rank <= 5, rank <= 10, 1.0 / rank], dim=1).float().cpu().numpy()


def world_bootstrap(values: np.ndarray, world_indices: np.ndarray, repetitions: int, seed: int) -> dict:
    worlds = np.unique(world_indices)
    per_world = np.stack([values[world_indices == world].mean(axis=0) for world in worlds])
    counts = np.asarray([(world_indices == world).sum() for world in worlds], dtype=np.float64)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(worlds), size=(repetitions, len(worlds)))
    draws = (per_world[picks] * counts[picks][..., None]).sum(1) / counts[picks].sum(1)[:, None]
    low, high = np.quantile(draws, (0.025, 0.975), axis=0)
    point = values.mean(axis=0)
    return {
        key: {"value": float(point[i]), "ci95": [float(low[i]), float(high[i])]}
        for i, key in enumerate(("R@1", "R@5", "R@10", "MRR"))
    }


def main():
    args = arguments()
    rows = [json.loads(line) for line in Path(args.gallery).read_text().splitlines()]
    world = np.asarray([row["world_id"] for row in rows])
    world_indices = np.asarray([row["world_index"] for row in rows])
    pattern = np.asarray([row["pattern_id"] for row in rows])
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in args.checkpoint:
        label, path = spec.split("=", 1)
        destination = output_dir / f"{label}.json"
        if destination.exists():
            print(f"skip {label}", flush=True)
            continue
        model, tokenizer, model_args = load_model(path, device)
        features = encode_all_layers(model, tokenizer, model_args, rows, args.batch_size, device)
        # Guard the vectorized metric against the reference implementation.
        # CPU and GPU float32 matrix products round differently, which can
        # swap near-tied neighbors (cosines equal to ~1e-6) for a handful of
        # queries; anything beyond that tolerance is a real disagreement.
        reference = query_statistics(features["block4"], world, pattern)["cross_pattern_semantic"]
        fast = per_query_cross_pattern(features["block4"], world, pattern, device)
        flipped = int((~np.isclose(reference, fast, atol=1e-6).all(axis=1)).sum())
        mean_gap = float(np.abs(reference.mean(axis=0) - fast.mean(axis=0)).max())
        if flipped > 0.01 * len(rows) or mean_gap > 2e-3:
            raise RuntimeError(
                f"vectorized retrieval metric disagrees with query_statistics for {label}: "
                f"{flipped} queries differ, largest mean gap {mean_gap:.4f}"
            )
        metrics = {
            name: world_bootstrap(per_query_cross_pattern(value, world, pattern, device),
                                  world_indices, args.bootstrap, args.seed)
            for name, value in features.items()
        }
        destination.write_text(json.dumps({
            "protocol": {
                "checkpoint": str(Path(path).resolve()), "train_mode": model_args.train_mode,
                "gallery": str(Path(args.gallery).resolve()), "num_captions": len(rows),
                "task": "cross-pattern same-world retrieval", "bootstrap": args.bootstrap, "seed": args.seed,
            },
            "metrics": metrics,
        }, indent=2) + "\n")
        summary = " ".join(f"b{l}={metrics[f'block{l}']['R@10']['value']:.3f}" for l in range(len(model.blocks)))
        print(f"{label} [{model_args.train_mode}] R@10 blocks: {summary}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
