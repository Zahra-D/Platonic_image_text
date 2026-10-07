#!/usr/bin/env python3
"""Held-out semantic-versus-template retrieval for Tri-LoRA text updates.

The input corpus has only one stored caption per world.  This program creates
truthful, deterministic alternative captions from validation worlds, then
compares the *native* ``blocks.4.mlp.3`` shared and text-private LoRA updates.
It never trains a model and never changes a checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from analyze_paired_representations import checkpoint_args
from data import ClevrTextTokenizer, MultimodalCollator
from train_multimodal import build_model


PATTERNS = (
    "inventory_then_relations",
    "relations_then_inventory",
    "reversed_inventory_then_relations",
    "reversed_relations_then_inventory",
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path to evaluate.")
    parser.add_argument("--manifest", required=True, help="Held-out validation JSONL manifest.")
    parser.add_argument(
        "--caption-generator",
        default="/home/zd25e122/clevr-dataset-gen_clone/image_generation/generate_human_captions.py",
        help="Local controlled-language generator used to make truthful variants.",
    )
    parser.add_argument("--num-worlds", type=int, default=256)
    parser.add_argument("--variants-per-pattern", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_caption_generator(path: str):
    spec = importlib.util.spec_from_file_location("heldout_caption_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load caption generator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def relation_sentences(record: dict, generator, rng: random.Random) -> list[str]:
    objects = record["world"]["objects"]
    by_id = {obj["id"]: obj for obj in objects}
    sentences = []
    for relation in record["world"].get("relations", []):
        subject_id = relation.get("subject_id")
        anchor_id = relation.get("anchor_id")
        if subject_id is None:
            subject_id = objects[relation["subject"]]["id"]
        if anchor_id is None:
            anchor_id = objects[relation["anchor"]]["id"]
        sentences.append(generator.relation_sentence(
            relation["relation"], by_id[subject_id], by_id[anchor_id], objects, rng,
        ))
    return sentences


def controlled_caption(record: dict, generator, pattern_id: int, seed: int) -> str:
    """Render the same world with independent lexical RNG and a known structure."""
    rng = random.Random(seed)
    inventory = generator.inventory_sentences(record["world"]["objects"], rng)
    relations = relation_sentences(record, generator, rng)
    if pattern_id == 0:
        sentences = inventory + relations
    elif pattern_id == 1:
        sentences = relations + inventory
    elif pattern_id == 2:
        sentences = list(reversed(inventory)) + relations
    elif pattern_id == 3:
        sentences = list(reversed(relations)) + list(reversed(inventory))
    else:
        raise ValueError(f"Unknown pattern_id={pattern_id}")
    return " ".join(sentences)


def world_signature(world: dict) -> str:
    """Stable semantics-only audit field, independent of wording and object order."""
    objects = sorted(
        (obj["id"], obj["size"], obj["color"], obj["material"], obj["shape"])
        for obj in world["objects"]
    )
    relations = sorted(
        (rel["relation"], rel.get("subject_id", rel.get("subject")),
         rel.get("anchor_id", rel.get("anchor")))
        for rel in world.get("relations", [])
    )
    return json.dumps({"objects": objects, "relations": relations}, separators=(",", ":"))


def build_variants(args) -> list[dict]:
    generator = load_caption_generator(args.caption_generator)
    worlds = []
    with open(args.manifest) as handle:
        for index, line in enumerate(handle):
            if len(worlds) >= args.num_worlds:
                break
            row = json.loads(line)
            if "world" not in row:
                raise ValueError(f"Manifest row {index} has no world field")
            worlds.append(row)
    if len(worlds) != args.num_worlds:
        raise ValueError(f"Requested {args.num_worlds} worlds, manifest has {len(worlds)}")
    rows = []
    for world_index, row in enumerate(worlds):
        semantic_id = row.get("world_id", row.get("id", world_index))
        signature = world_signature(row["world"])
        for pattern_id, pattern in enumerate(PATTERNS):
            for variant_id in range(args.variants_per_pattern):
                # Do not use Python's randomized hash: this seed is stable across machines.
                seed = args.seed + 10_000_019 * world_index + 101 * pattern_id + variant_id
                rows.append({
                    "index": len(rows), "world_id": semantic_id, "world_index": world_index,
                    "pattern_id": pattern_id, "pattern": pattern, "variant_id": variant_id,
                    "world_signature": signature,
                    "caption": controlled_caption(row, generator, pattern_id, seed),
                })
    captions = [row["caption"] for row in rows]
    if len(captions) != len(set(captions)):
        # Identical captions make exact surface retrieval artificially easy.  It is a
        # generator collision, not a model result, so fail transparently rather than hide it.
        duplicates = len(captions) - len(set(captions))
        raise RuntimeError(f"Generated {duplicates} duplicate captions; increase lexical variants or revise generator")
    return rows


def load_model(checkpoint: str, device: torch.device):
    # "TRUNK:<path>" evaluates only the shared dense trunk of a dense_private
    # checkpoint: the model is rebuilt as a plain dense model from the base
    # weights and every private LoRA tensor is dropped, so no readout,
    # routing or "best layer" can involve the private adapters.
    trunk_only = str(checkpoint).startswith("TRUNK:")
    if trunk_only:
        checkpoint = str(checkpoint)[len("TRUNK:"):]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = checkpoint_args(payload)
    tokenizer = ClevrTextTokenizer(payload["text_vocabulary"])
    state = payload["model"]
    if trunk_only:
        if getattr(args, "train_mode", "dense") != "dense_private":
            raise ValueError(f"TRUNK: needs a dense_private checkpoint, got {args.train_mode!r}")
        args.train_mode = "dense"
        private = ("text_A", "text_B", "image_A", "image_B")
        state = {
            re.sub(r"\.base\.(weight|bias)$", r".\1", key): value
            for key, value in state.items() if not key.endswith(private)
        }
    model, _ = build_model(args, len(tokenizer))
    incompatible = model.load_state_dict(state, strict=False)
    # Older modulewise checkpoints serialized a predictor naming scheme that
    # changed later.  Predictors are deliberately not used by this evaluator;
    # every backbone and LoRA tensor must nevertheless load exactly.
    ignored_prefixes = ("modulewise_jepa_predictors.", "shared_jepa_predictors.")
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith(ignored_prefixes)]
    missing = [key for key in incompatible.missing_keys if not key.startswith(ignored_prefixes)]
    if unexpected or missing:
        raise RuntimeError(
            "Checkpoint/model mismatch outside unused JEPA predictors: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.to(device).eval()
    return model, tokenizer, args


@torch.no_grad()
def encode_updates(model, tokenizer, model_args, rows, batch_size, device):
    collator = MultimodalCollator(tokenizer, model_args.num_image_codes, model_args.max_text_length)
    values = {"shared": [], "private": []}
    target = "blocks.4.mlp.3"
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        batch = collator([
            {"kind": "text", "text": row["caption"], "pair_index": row["index"]}
            for row in chunk
        ])
        batch = {name: value.to(device) for name, value in batch.items()}
        _, _, shared_native, private_native = model(
            batch["input_ids"], batch["attention_mask"], batch["position_ids"],
            batch["modality_ids"], batch["route_ids"], return_shared=True,
            return_shared_private_native_by_module=True,
        )
        if target not in shared_native or target not in private_native:
            raise KeyError(
                f"Checkpoint has no {target!r} native updates; available shared modules: "
                f"{sorted(shared_native)}"
            )
        weights = batch["eligible_mask"].to(torch.float32).unsqueeze(-1)
        for name, native in (("shared", shared_native[target]), ("private", private_native[target])):
            pooled = (native.float() * weights).sum(1) / weights.sum(1).clamp_min(1)
            values[name].append(F.normalize(pooled, dim=1).cpu())
    return {name: torch.cat(parts).numpy() for name, parts in values.items()}


def query_statistics(features: np.ndarray, world: np.ndarray, pattern: np.ndarray):
    similarity = features @ features.T
    np.fill_diagonal(similarity, -np.inf)
    count = len(features)
    semantic = np.zeros((count, 4), dtype=np.float64)  # r1, r5, r10, reciprocal best rank
    same_pattern = np.zeros_like(semantic)
    template = np.zeros_like(semantic)
    pattern_nearest = np.zeros(count, dtype=np.float64)
    cross_top = np.full(count, -1, dtype=np.int64)
    template_top = np.full(count, -1, dtype=np.int64)
    for query in range(count):
        order = np.argsort(-similarity[query])
        cross = (world[order] == world[query]) & (pattern[order] != pattern[query])
        same = (world[order] == world[query]) & (pattern[order] == pattern[query])
        templ = (world[order] != world[query]) & (pattern[order] == pattern[query])
        for destination, positives in ((semantic[query], cross), (same_pattern[query], same), (template[query], templ)):
            rank = int(np.flatnonzero(positives)[0]) + 1
            destination[:] = (rank <= 1, rank <= 5, rank <= 10, 1.0 / rank)
        cross_top[query] = order[0] if len(order) else -1
        template_top[query] = order[np.flatnonzero(templ)[0]]
        without_world = order[world[order] != world[query]]
        pattern_nearest[query] = float(pattern[without_world[0]] == pattern[query])
    return {
        "cross_pattern_semantic": semantic,
        "same_pattern_semantic": same_pattern,
        "template": template,
        "nearest_other_world_same_pattern": pattern_nearest,
        "cross_top": cross_top,
        "template_top": template_top,
    }


def bootstrap_world_ci(query_values: np.ndarray, world_indices: np.ndarray, repetitions: int, seed: int):
    """Paired world bootstrap; each sampled world brings all of its caption queries."""
    by_world = [np.flatnonzero(world_indices == idx) for idx in np.unique(world_indices)]
    rng = np.random.default_rng(seed)
    point = query_values.mean(axis=0)
    draws = np.empty((repetitions, query_values.shape[1]), dtype=np.float64)
    for repetition in range(repetitions):
        picked = rng.integers(0, len(by_world), len(by_world))
        indices = np.concatenate([by_world[index] for index in picked])
        draws[repetition] = query_values[indices].mean(axis=0)
    return {
        "value": [float(value) for value in point],
        "ci95": [[float(value) for value in row] for row in np.quantile(draws, (0.025, 0.975), axis=0)],
    }


def summarize(stats, world_indices, bootstrap, seed):
    keys = ("R@1", "R@5", "R@10", "MRR")
    result = {}
    for name in ("cross_pattern_semantic", "same_pattern_semantic", "template"):
        packed = bootstrap_world_ci(stats[name], world_indices, bootstrap, seed)
        result[name] = {
            key: {"value": packed["value"][index], "ci95": [packed["ci95"][0][index], packed["ci95"][1][index]]}
            for index, key in enumerate(keys)
        }
    vector = stats["nearest_other_world_same_pattern"][:, None]
    packed = bootstrap_world_ci(vector, world_indices, bootstrap, seed)
    result["nearest_other_world_same_pattern"] = {
        "value": packed["value"][0], "ci95": [packed["ci95"][0][0], packed["ci95"][1][0]],
        "description": "Pattern match of nearest different-world neighbor; chance is 0.25 for four balanced patterns.",
    }
    result["semantic_minus_template_mrr"] = (
        result["cross_pattern_semantic"]["MRR"]["value"] - result["template"]["MRR"]["value"]
    )
    return result


def examples(rows, features, stats, seed, count=12):
    rng = random.Random(seed)
    selected = rng.sample(range(len(rows)), min(count, len(rows)))
    report = []
    similarity = features @ features.T
    for query in selected:
        cross_candidates = np.flatnonzero(np.array([row["pattern_id"] for row in rows]) != rows[query]["pattern_id"])
        cross_candidates = cross_candidates[cross_candidates != query]
        best_cross = cross_candidates[np.argmax(similarity[query, cross_candidates])]
        best_template = int(stats["template_top"][query])
        report.append({
            "query": rows[query],
            "nearest_cross_pattern": {"cosine": float(similarity[query, best_cross]), "row": rows[int(best_cross)]},
            "nearest_template_candidate": {"cosine": float(similarity[query, best_template]), "row": rows[best_template]},
        })
    return report


def markdown_report(result: dict) -> str:
    lines = [
        "# Held-out shared/private semantic-template retrieval", "",
        "This report uses clean validation-only captions. Features are the native pooled, L2-normalized "
        "`blocks.4.mlp.3` shared or text-private LoRA updates; no hidden-state proxy, training data, or classifier is used.", "",
        "## Result", "",
        "| Route | Cross-pattern semantic R@1 | R@5 | R@10 | MRR | Same-pattern semantic MRR | Template MRR | Other-world nearest pattern match | Semantic − template MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for route in ("shared", "private", "random_control"):
        metrics = result["metrics"][route]
        cross = metrics["cross_pattern_semantic"]
        same = metrics["same_pattern_semantic"]
        template = metrics["template"]
        near = metrics["nearest_other_world_same_pattern"]
        lines.append(
            f"| {route} | {cross['R@1']['value']:.3f} | {cross['R@5']['value']:.3f} | {cross['R@10']['value']:.3f} | "
            f"{cross['MRR']['value']:.3f} | {same['MRR']['value']:.3f} | {template['MRR']['value']:.3f} | "
            f"{near['value']:.3f} | {metrics['semantic_minus_template_mrr']:.3f} |"
        )
    lines += [
        "", "Confidence intervals and deterministic neighbor examples are in `results.json`.", "",
        "Interpretation: shared specialization is supported when its cross-pattern semantic values exceed private "
        "with their paired world-bootstrap intervals. Private template preference is exploratory: private semantic "
        "retrieval is not an error because private LoRAs also learn the text-denoising task.", "",
        "The pattern-match chance level is 0.25 because there are four equally frequent patterns. Template MRR uses "
        "different worlds with the query's pattern as positives, after excluding every same-world caption.",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = arguments()
    output = Path(args.output)
    # The tmux launcher creates ``launcher.log`` before it starts Python.  That
    # diagnostic file is not a partial result and must not block the evaluator.
    existing = [] if not output.exists() else [path for path in output.iterdir() if path.name != "launcher.log"]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output already has result files: {output}; use --overwrite to replace result files")
    output.mkdir(parents=True, exist_ok=True)
    rows = build_variants(args)
    (output / "variants.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    device = torch.device(args.device)
    model, tokenizer, model_args = load_model(args.checkpoint, device)
    features = encode_updates(model, tokenizer, model_args, rows, args.batch_size, device)
    world = np.asarray([row["world_id"] for row in rows])
    world_indices = np.asarray([row["world_index"] for row in rows])
    pattern = np.asarray([row["pattern_id"] for row in rows])
    rng = np.random.default_rng(args.seed)
    random_features = rng.normal(size=features["shared"].shape).astype(np.float32)
    random_features /= np.linalg.norm(random_features, axis=1, keepdims=True).clip(1e-12)
    all_features = {**features, "random_control": random_features}
    all_stats = {name: query_statistics(value, world, pattern) for name, value in all_features.items()}
    result = {
        "protocol": {
            "checkpoint": str(Path(args.checkpoint).resolve()), "manifest": str(Path(args.manifest).resolve()),
            "num_worlds": args.num_worlds, "patterns": list(PATTERNS),
            "variants_per_pattern": args.variants_per_pattern, "num_captions": len(rows),
            "feature": "clean native blocks.4.mlp.3 update; content-token mean; L2 normalization",
            "bootstrap": args.bootstrap, "seed": args.seed,
        },
        "metrics": {name: summarize(stats, world_indices, args.bootstrap, args.seed) for name, stats in all_stats.items()},
        "examples": {name: examples(rows, value, all_stats[name], args.seed) for name, value in features.items()},
    }
    torch.save({"shared": torch.from_numpy(features["shared"]), "private": torch.from_numpy(features["private"]), "metadata": rows}, output / "features.pt")
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "REPORT.md").write_text(markdown_report(result))
    print(json.dumps(result["metrics"], indent=2))
    print(f"Wrote {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
