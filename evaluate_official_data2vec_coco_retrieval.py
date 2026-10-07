#!/usr/bin/env python3
"""Evaluate the official data2vec-text checkpoint on held-out COCO captions.

The released model is a fairseq RoBERTa checkpoint.  This script maps its
encoder tensors explicitly into the equivalent Hugging Face RoBERTa module,
audits every loaded/ignored tensor, and evaluates every hidden layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, RobertaConfig, RobertaModel


class FairseqRobertaTokenizer:
    """Reproduce fairseq's GPT2BPE -> Dictionary integer remapping."""

    bos_token_id = 0
    pad_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    mask_token_id = 50264

    def __init__(self, vocab_json: str, merges: str, dictionary: str):
        self.bpe = GPT2TokenizerFast(vocab_file=vocab_json, merges_file=merges)
        symbols = [line.rsplit(" ", 1)[0] for line in Path(dictionary).read_text().splitlines() if line]
        self.dictionary = {symbol: index + 4 for index, symbol in enumerate(symbols)}
        if len(symbols) != 50260 or max(self.dictionary.values()) != 50263:
            raise RuntimeError("Unexpected fairseq RoBERTa dictionary layout")

    def __len__(self) -> int:
        return 50265

    def __call__(self, texts: list[str], *, padding: bool, truncation: bool,
                 max_length: int, return_special_tokens_mask: bool, return_tensors: str):
        if not (padding and truncation and return_special_tokens_mask and return_tensors == "pt"):
            raise ValueError("This evaluator requires padded, truncated PyTorch batches")
        sequences = []
        for text in texts:
            raw_ids = self.bpe.encode(text, add_special_tokens=False)[: max_length - 2]
            mapped = [self.dictionary.get(str(token), self.unk_token_id) for token in raw_ids]
            sequences.append([self.bos_token_id, *mapped, self.eos_token_id])
        width = max(map(len, sequences))
        input_ids = torch.full((len(sequences), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
        special_tokens_mask = torch.ones((len(sequences), width), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            length = len(sequence)
            input_ids[row, :length] = torch.tensor(sequence)
            attention_mask[row, :length] = 1
            special_tokens_mask[row, 1 : length - 1] = 0
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask,
        }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab-json", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--retrieval-manifest", required=True)
    parser.add_argument("--max-images", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--random-init", action="store_true", help="Same architecture/tokenizer without checkpoint weights")
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
        raise ValueError("Every retrieval row must have at least one held-out caption")
    return rows


def fairseq_to_hf_key(key: str) -> str | None:
    prefix = "encoder.sentence_encoder."
    if not key.startswith(prefix):
        return None
    key = key[len(prefix) :]
    exact = {
        "embed_tokens.weight": "embeddings.word_embeddings.weight",
        "embed_positions.weight": "embeddings.position_embeddings.weight",
        "layernorm_embedding.weight": "embeddings.LayerNorm.weight",
        "layernorm_embedding.bias": "embeddings.LayerNorm.bias",
    }
    if key in exact:
        return exact[key]
    if not key.startswith("layers."):
        return None
    fields = key.split(".")
    layer = fields[1]
    suffix = ".".join(fields[2:])
    replacements = {
        "self_attn.q_proj": "attention.self.query",
        "self_attn.k_proj": "attention.self.key",
        "self_attn.v_proj": "attention.self.value",
        "self_attn.out_proj": "attention.output.dense",
        "self_attn_layer_norm": "attention.output.LayerNorm",
        "fc1": "intermediate.dense",
        "fc2": "output.dense",
        "final_layer_norm": "output.LayerNorm",
    }
    for source, target in replacements.items():
        if suffix == source or suffix.startswith(source + "."):
            suffix = target + suffix[len(source) :]
            return f"encoder.layer.{layer}.{suffix}"
    return None


def build_model(checkpoint: Path, device: torch.device, random_init: bool = False):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = payload["model"]
    cfg = payload["cfg"]["model"]
    config = RobertaConfig(
        vocab_size=source["encoder.sentence_encoder.embed_tokens.weight"].shape[0],
        hidden_size=cfg["encoder_embed_dim"],
        num_hidden_layers=cfg["encoder_layers"],
        num_attention_heads=cfg["encoder_attention_heads"],
        intermediate_size=cfg["encoder_ffn_embed_dim"],
        hidden_act=cfg["activation_fn"],
        hidden_dropout_prob=cfg["dropout"],
        attention_probs_dropout_prob=cfg["attention_dropout"],
        max_position_embeddings=source["encoder.sentence_encoder.embed_positions.weight"].shape[0],
        type_vocab_size=1,
        layer_norm_eps=1e-5,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )
    model = RobertaModel(config, add_pooling_layer=False)
    if random_init:
        return model.to(device).eval(), config, {
            "source_model_entries": len(source),
            "mapped_encoder_tensors": 0,
            "random_init": True,
        }
    converted = {}
    ignored = []
    for key, value in source.items():
        target = fairseq_to_hf_key(key)
        if target is None:
            ignored.append(key)
        else:
            converted[target] = value
    missing, unexpected = model.load_state_dict(converted, strict=False)
    allowed_missing = {"embeddings.token_type_embeddings.weight"}
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(f"Invalid conversion: missing={missing}, unexpected={unexpected}")
    with torch.no_grad():
        model.embeddings.token_type_embeddings.weight.zero_()
    expected_ignored = {
        "encoder.sentence_encoder.version",
        "encoder.regression_head.0.weight",
        "encoder.regression_head.0.bias",
        "encoder.regression_head.2.weight",
        "encoder.regression_head.2.bias",
        "encoder._ema",
    }
    if set(ignored) != expected_ignored:
        raise RuntimeError(f"Unexpected ignored tensors: {sorted(ignored)}")
    audit = {
        "source_model_entries": len(source),
        "mapped_encoder_tensors": len(converted),
        "ignored_pretraining_entries": sorted(ignored),
        "missing_hf_tensor_initialized_to_zero": sorted(missing),
    }
    return model.to(device).eval(), config, audit


@torch.inference_mode()
def encode(model, tokenizer, captions: list[str], batch_size: int, device: torch.device):
    names = ["input"] + [f"L{layer}" for layer in range(model.config.num_hidden_layers)]
    features = {name: [] for name in names}
    unknown = content_total = 0
    for start in range(0, len(captions), batch_size):
        batch = tokenizer(
            captions[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=512,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        special = batch.pop("special_tokens_mask")
        batch = {key: value.to(device) for key, value in batch.items()}
        content_mask = batch["attention_mask"].bool() & ~special.to(device).bool()
        unknown += int(((batch["input_ids"] == tokenizer.unk_token_id) & content_mask).sum())
        content_total += int(content_mask.sum())
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(**batch, output_hidden_states=True, return_dict=True)
        weights = content_mask.float().unsqueeze(-1)
        for name, states in zip(names, output.hidden_states):
            pooled = (states.float() * weights).sum(1) / weights.sum(1).clamp_min(1)
            features[name].append(F.normalize(pooled, dim=-1).to(dtype=torch.float16, device="cpu"))
        if start % (batch_size * 20) == 0:
            print(f"encoded {min(start + batch_size, len(captions))}/{len(captions)} captions", flush=True)
    return {name: torch.cat(parts) for name, parts in features.items()}, unknown / max(content_total, 1)


def retrieval_metrics(queries, candidates, candidate_owners, chunk_size: int, device: torch.device) -> dict:
    queries = queries.float().to(device)
    candidates = candidates.float().to(device)
    owners = candidate_owners.to(device)
    ranks = []
    for start in range(0, len(queries), chunk_size):
        query = queries[start : start + chunk_size]
        similarity = query @ candidates.T
        query_owners = torch.arange(start, start + len(query), device=device).unsqueeze(1)
        positive = owners.unsqueeze(0) == query_owners
        best_positive = similarity.masked_fill(~positive, -torch.inf).max(dim=1).values
        ranks.append((1 + (similarity > best_positive.unsqueeze(1)).sum(dim=1)).cpu())
    rank = torch.cat(ranks).float()
    return {
        "R@1": float((rank <= 1).float().mean()),
        "R@5": float((rank <= 5).float().mean()),
        "R@10": float((rank <= 10).float().mean()),
        "MRR": float((1.0 / rank).mean()),
        "mean_rank": float(rank.mean()),
        "median_rank": float(rank.median()),
        "queries": len(queries),
        "candidates": len(candidates),
        # Averaged across queries, total positives equals total candidates, so
        # random R@1 is 1 / number of image owners (0.02% for 5,000 images).
        "chance_R@1": float(1.0 / len(queries)),
    }


def main() -> None:
    args = arguments()
    torch.manual_seed(args.seed)
    manifest = Path(args.retrieval_manifest)
    rows = load_rows(manifest, args.max_images, args.seed)
    queries = [row["query"] for row in rows]
    candidates, owners = [], []
    for owner, row in enumerate(rows):
        for positive in row["positives"]:
            candidates.append(positive["caption"])
            owners.append(owner)
    device = torch.device(args.device)
    tokenizer = FairseqRobertaTokenizer(args.vocab_json, args.merges, args.dictionary)
    if (tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id) != (0, 1, 2, 3):
        raise RuntimeError("Official RoBERTa special-token IDs were not reproduced")
    model, config, audit = build_model(Path(args.checkpoint), device, args.random_init)
    all_features, unknown_rate = encode(model, tokenizer, queries + candidates, args.batch_size, device)
    owner_tensor = torch.tensor(owners, dtype=torch.long)
    metrics = {}
    for name, values in all_features.items():
        metrics[name] = retrieval_metrics(
            values[: len(rows)], values[len(rows) :], owner_tensor, args.query_chunk_size, device
        )
        score = metrics[name]
        print(
            f"{name}: R@1={score['R@1']:.4f} R@5={score['R@5']:.4f} "
            f"R@10={score['R@10']:.4f} MRR={score['MRR']:.4f}",
            flush=True,
        )
    best = max(metrics, key=lambda name: metrics[name]["MRR"])
    result = {
        "protocol": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            "random_init": args.random_init,
            "retrieval_manifest": str(manifest.resolve()),
            "retrieval_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "sample_seed": args.seed,
            "images": len(rows),
            "heldout_candidates": len(candidates),
            "query_definition": "one primary caption per image",
            "positive_definition": "all other captions for the same COCO image",
            "pooling": "mean content-token state followed by L2 normalization",
            "ranking": "global ranking against every held-out candidate caption",
            "unknown_content_token_rate": unknown_rate,
            "tokenizer_size": len(tokenizer),
            "model_config": config.to_dict(),
            "conversion_audit": audit,
        },
        "best_layer_by_MRR": best,
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"best layer: {best}; wrote {output}", flush=True)


if __name__ == "__main__":
    main()
