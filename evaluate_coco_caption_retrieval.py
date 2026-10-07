#!/usr/bin/env python3
"""Evaluate frozen JEPA trunk layers by held-out COCO caption retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from analyze_paired_representations import checkpoint_args
from data import ClevrTextTokenizer, MultimodalCollator
from evaluate_shared_private_retrieval import load_model
from train_multimodal import build_model


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--retrieval-manifest", required=True)
    parser.add_argument("--max-images", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--ema-teacher", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_rows(path: Path, limit: int, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if limit and limit < len(rows):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(rows), generator=generator)[:limit].sort().values.tolist()
        rows = [rows[index] for index in indices]
    if any(not row.get("positives") for row in rows):
        raise ValueError("Every retrieval row must contain at least one held-out positive")
    return rows


def initialize(args: argparse.Namespace, device: torch.device):
    if args.random_init and args.ema_teacher:
        raise ValueError("--random-init and --ema-teacher are mutually exclusive")
    if not args.random_init:
        model, tokenizer, model_args = load_model(args.checkpoint, device)
        if args.ema_teacher:
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            state = payload.get("shared_jepa_ema_teacher")
            if state is None:
                raise ValueError("Checkpoint has no shared_jepa_ema_teacher state")
            model.load_state_dict(state, strict=True)
            model.eval()
        return model, tokenizer, model_args
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = checkpoint_args(payload)
    tokenizer = ClevrTextTokenizer(payload["text_vocabulary"])
    torch.manual_seed(args.seed)
    model, _ = build_model(model_args, len(tokenizer))
    return model.to(device).eval(), tokenizer, model_args


@torch.no_grad()
def encode(model, tokenizer, model_args, captions: list[str], batch_size: int, device: torch.device):
    collator = MultimodalCollator(tokenizer, model_args.num_image_codes, model_args.max_text_length)
    names = ["input"] + [f"L{layer}" for layer in range(model_args.n_layers)] + ["final_norm"]
    features = {name: [] for name in names}
    unknown = total = 0
    for start in range(0, len(captions), batch_size):
        chunk = captions[start : start + batch_size]
        examples = [{"kind": "text", "text": text, "pair_index": start + i} for i, text in enumerate(chunk)]
        batch = {key: value.to(device) for key, value in collator(examples).items()}
        unknown += int((batch["input_ids"] == tokenizer.unk_id).sum())
        total += int(batch["eligible_mask"].sum())
        embedded = model.input_embeddings(batch["input_ids"], batch["position_ids"], batch["modality_ids"])
        _, hidden = model(
            batch["input_ids"], batch["attention_mask"], batch["position_ids"],
            batch["modality_ids"], batch["route_ids"], return_hidden_by_layer=True,
        )
        maps = {"input": embedded, **{f"L{layer}": value for layer, value in hidden.items()}}
        maps["final_norm"] = model.norm_out(hidden[model_args.n_layers - 1])
        weights = batch["eligible_mask"].float().unsqueeze(-1)
        for name, values in maps.items():
            pooled = (values.float() * weights).sum(1) / weights.sum(1).clamp_min(1)
            features[name].append(F.normalize(pooled, dim=-1).cpu())
    return {name: torch.cat(parts) for name, parts in features.items()}, unknown / max(total, 1)


def retrieval_metrics(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    candidate_owners: torch.Tensor,
    chunk_size: int,
) -> dict:
    ranks = []
    owner_device = candidate_owners.to(queries.device)
    candidates = candidates.to(queries.device)
    for start in range(0, len(queries), chunk_size):
        query = queries[start : start + chunk_size].to(candidates.device)
        similarity = query @ candidates.T
        owners = torch.arange(start, start + len(query), device=candidates.device).unsqueeze(1)
        positive = owner_device.unsqueeze(0) == owners
        best_positive = similarity.masked_fill(~positive, -torch.inf).max(dim=1).values
        rank = 1 + (similarity > best_positive.unsqueeze(1)).sum(dim=1)
        ranks.append(rank.cpu())
    rank = torch.cat(ranks).float()
    return {
        "R@1": float((rank <= 1).float().mean()),
        "R@5": float((rank <= 5).float().mean()),
        "R@10": float((rank <= 10).float().mean()),
        "MRR": float((1.0 / rank).mean()),
        "mean_rank": float(rank.mean()),
        "median_rank": float(rank.median()),
        "queries": int(len(rank)),
        "candidates": int(len(candidates)),
        "positives_per_query": float(len(candidates) / len(rank)),
    }


def main() -> None:
    args = arguments()
    manifest = Path(args.retrieval_manifest)
    rows = load_rows(manifest, args.max_images, args.seed)
    query_captions = [row["query"] for row in rows]
    candidate_captions = []
    candidate_owners = []
    for owner, row in enumerate(rows):
        for positive in row["positives"]:
            candidate_captions.append(positive["caption"])
            candidate_owners.append(owner)
    device = torch.device(args.device)
    model, tokenizer, model_args = initialize(args, device)
    all_features, unknown_rate = encode(
        model, tokenizer, model_args, query_captions + candidate_captions, args.batch_size, device
    )
    owner_tensor = torch.tensor(candidate_owners, dtype=torch.long)
    metrics = {}
    for name, values in all_features.items():
        metrics[name] = retrieval_metrics(
            values[: len(rows)], values[len(rows) :], owner_tensor, args.query_chunk_size
        )
        score = metrics[name]
        print(
            f"{name}: R@1={score['R@1']:.4f} R@5={score['R@5']:.4f} "
            f"R@10={score['R@10']:.4f} MRR={score['MRR']:.4f}",
            flush=True,
        )
    result = {
        "protocol": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "random_init": args.random_init,
            "ema_teacher": args.ema_teacher,
            "retrieval_manifest": str(manifest.resolve()),
            "retrieval_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "sample_seed": args.seed,
            "images": len(rows),
            "heldout_candidates": len(candidate_captions),
            "query_definition": "one primary caption per image",
            "positive_definition": "all captions withheld from the primary manifest for that image",
            "pooling": "mean of content-token states followed by L2 normalization",
            "unknown_content_token_rate": unknown_rate,
        },
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
