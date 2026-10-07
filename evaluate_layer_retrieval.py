#!/usr/bin/env python3
"""Cross-pattern semantic retrieval on whole-layer features, dense or LoRA.

``evaluate_shared_private_retrieval`` reads one LoRA branch, so it cannot run
on a dense model, and its numbers are not comparable to a dense layer.  This
evaluator reuses that evaluation's exact held-out gallery (read from an
existing ``variants.jsonl`` rather than regenerated), its pooling, and its
metric code, and scores features that exist in every model:

* ``mlp3_update`` (dense only): ``W x`` of ``blocks.4.mlp.3`` without its bias,
  the dense counterpart of a LoRA branch's bias-free native update.
* ``mlp3_output``: the full ``blocks.4.mlp.3`` output.  For Tri-LoRA this is
  shared + private + bias, i.e. everything that layer writes.
* ``block4_output``: the residual-stream hidden state after Transformer block 4.
* ``token_embedding_mean``: the input token embeddings, before any context.
  An order-free bag-of-embeddings reference for what needs no Transformer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import MultimodalCollator
from evaluate_shared_private_retrieval import load_model, query_statistics, summarize


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gallery", required=True, help="variants.jsonl from an existing retrieval result.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


@torch.no_grad()
def encode_dense(model, tokenizer, model_args, rows, batch_size, device):
    collator = MultimodalCollator(tokenizer, model_args.num_image_codes, model_args.max_text_length)
    linear = model.blocks[4].mlp[3]
    dense = isinstance(linear, torch.nn.Linear)
    captured = {}
    hooks = [
        linear.register_forward_hook(lambda _m, inputs, output: captured.update(mlp3_input=inputs[0], mlp3_output=output)),
        model.blocks[4].register_forward_hook(lambda _m, _i, output: captured.__setitem__("block4_output", output)),
        model.token_embed.register_forward_hook(lambda _m, _i, output: captured.__setitem__("token_embedding", output)),
    ]
    values = {"mlp3_output": [], "block4_output": [], "token_embedding_mean": []}
    if dense:
        values["mlp3_update"] = []
    try:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            batch = collator([
                {"kind": "text", "text": row["caption"], "pair_index": row["index"]} for row in chunk
            ])
            batch = {name: value.to(device) for name, value in batch.items()}
            model(batch["input_ids"], batch["attention_mask"], batch["position_ids"],
                  batch["modality_ids"], batch["route_ids"])
            maps = {
                "mlp3_output": captured["mlp3_output"],
                "block4_output": captured["block4_output"],
                "token_embedding_mean": captured["token_embedding"],
            }
            if dense:
                maps["mlp3_update"] = F.linear(captured["mlp3_input"], linear.weight)
            weights = batch["eligible_mask"].to(torch.float32).unsqueeze(-1)
            for name, native in maps.items():
                pooled = (native.float() * weights).sum(1) / weights.sum(1).clamp_min(1)
                values[name].append(F.normalize(pooled, dim=1).cpu())
    finally:
        for hook in hooks:
            hook.remove()
    return {name: torch.cat(parts).numpy() for name, parts in values.items()}


def main():
    args = arguments()
    gallery = Path(args.gallery)
    rows = [json.loads(line) for line in gallery.read_text().splitlines()]
    device = torch.device(args.device)
    model, tokenizer, model_args = load_model(args.checkpoint, device)
    features = encode_dense(model, tokenizer, model_args, rows, args.batch_size, device)
    world = np.asarray([row["world_id"] for row in rows])
    world_indices = np.asarray([row["world_index"] for row in rows])
    pattern = np.asarray([row["pattern_id"] for row in rows])
    result = {
        "protocol": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "gallery": str(gallery.resolve()),
            "gallery_sha256": hashlib.sha256(gallery.read_bytes()).hexdigest(),
            "num_captions": len(rows),
            "train_mode": model_args.train_mode,
            "features": {
                "mlp3_update": "dense only: blocks.4.mlp.3 weight times input, no bias",
                "mlp3_output": "full blocks.4.mlp.3 output (Tri-LoRA: shared + private + bias)",
                "block4_output": "residual hidden state after block 4",
                "token_embedding_mean": "input token embeddings before any Transformer block",
                "pooling": "content-token mean, then L2 normalization",
            },
            "bootstrap": args.bootstrap, "seed": args.seed,
        },
        "metrics": {
            name: summarize(query_statistics(value, world, pattern), world_indices, args.bootstrap, args.seed)
            for name, value in features.items()
        },
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({**{name: torch.from_numpy(value) for name, value in features.items()}, "metadata": rows},
               output / "features.pt")
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, metric in result["metrics"].items():
        cross = metric["cross_pattern_semantic"]
        print(f"{name}: R@1={cross['R@1']['value']:.3f} R@5={cross['R@5']['value']:.3f} "
              f"R@10={cross['R@10']['value']:.3f} MRR={cross['MRR']['value']:.3f}")


if __name__ == "__main__":
    main()
