#!/usr/bin/env python3
"""Does a frozen representation encode the scene, or follow dataset statistics?

This is a checkpoint-only evaluation.  It does not optimize a model or a probe
and it does not make a new training or validation corpus.  It reuses the exact
minimal pairs from ``evaluate_semantic_dprime.py``:

* ``query`` and ``positive`` describe the same held-out world with different
  wording;
* ``swap`` uses exactly the positive's words, but exchanges one attribute
  between two objects.

The positive and swap consequently have identical token multisets and identical
per-attribute multisets.  They differ only in attribute binding.  Separately,
the training manifest supplies an empirical object prior

    p_train(color, shape, material, size).

For every pair we determine whether this prior prefers the true world or the
counterfactual swap.  The decisive group is ``prior_conflicts``: the swapped
world is statistically more typical than the true held-out world.  A
sample-specific representation should still prefer the true positive; a
dataset-statistics shortcut should reverse its preference.

The primary feature is predeclared as ``L7.residual``.  All other layers and
sublayers are exploratory and are reported so that a failure cannot be hidden
by selecting the best feature after looking at the result.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import binding_swap_captions as C
from evaluate_decomposed_representations import encode
from evaluate_hard_retrieval import attribute_swaps, relation_swaps
from evaluate_shared_private_retrieval import load_model


ATTRIBUTES = ("color", "shape", "material", "size")
SWAP_TYPES = set(ATTRIBUTES)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument(
        "--train-manifest",
        default="/home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_2m/train_text_only_human.jsonl",
        help="Existing training manifest, read only to estimate the empirical object prior.",
    )
    p.add_argument(
        "--manifest",
        default="/home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_1m/val_text_only_human.jsonl",
    )
    p.add_argument(
        "--caption-generator",
        default="/home/zd25e122/clevr-dataset-gen_clone/image_generation/generate_human_captions.py",
    )
    p.add_argument(
        "--reuse-dprime-dir", default="outputs/semantic_dprime_eval",
        help="Existing d-prime artifacts. The regenerated in-memory items must match these exactly.",
    )
    p.add_argument("--num-worlds", type=int, default=2000)
    p.add_argument("--max-train-scenes", type=int, default=None,
                   help="Optional debugging limit; the final experiment should use the full manifest.")
    p.add_argument("--max-tokens", type=int, default=190)
    p.add_argument("--sublayer-layers", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--prior-alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def object_key(obj):
    return tuple(obj[name] for name in ATTRIBUTES)


def estimate_prior(path, max_scenes=None):
    counts = Counter()
    scenes = objects = 0
    values = {name: set() for name in ATTRIBUTES}
    with open(path) as handle:
        for line in handle:
            if max_scenes is not None and scenes >= max_scenes:
                break
            world = json.loads(line)["world"]
            scenes += 1
            for obj in world["objects"]:
                key = object_key(obj)
                counts[key] += 1
                objects += 1
                for name, value in zip(ATTRIBUTES, key):
                    values[name].add(value)
    support_size = math.prod(len(values[name]) for name in ATTRIBUTES)
    return counts, objects, support_size, scenes


def scene_log_prior(world, counts, total, support_size, alpha):
    denominator = total + alpha * support_size
    return sum(math.log((counts[object_key(obj)] + alpha) / denominator)
               for obj in world["objects"])


def build_exact_dprime_items(args, tokenizer):
    """Rebuild the established items in memory, retaining the two worlds."""
    generator = C.load_generator(args.caption_generator)
    worlds = []
    with open(args.manifest) as handle:
        for line in handle:
            world = json.loads(line)["world"]
            if not C.has_twins(world) and len(world["objects"]) >= 3:
                worlds.append(world)
            if len(worlds) == args.num_worlds:
                break

    texts, text_index, items = [], {}, []

    def add(text):
        if text not in text_index:
            text_index[text] = len(texts)
            texts.append(text)
        return text_index[text]

    for position, world in enumerate(worlds):
        rng = random.Random(args.seed * 1_000_003 + position)
        objects, relations = len(world["objects"]), len(world.get("relations", []))
        plan = C.make_plan(generator, rng.randrange(2**31), objects, relations)
        other = C.make_plan(generator, rng.randrange(2**31), objects, relations)
        order = lambda k: rng.sample(range(k), k)
        pattern = rng.randrange(2)
        query = C.render(generator, world, plan, order(objects), order(relations), pattern)
        positive = C.render(generator, world, other, order(objects), order(relations), rng.randrange(2))

        twin = None
        for candidate in worlds[position + 1:] + worlds[:position]:
            if (len(candidate["objects"]), len(candidate.get("relations", []))) == (objects, relations) \
                    and C.scene_signature(candidate) != C.scene_signature(world):
                twin = C.render(generator, candidate, plan, list(range(objects)),
                                list(range(relations)), pattern)
                break

        pool = attribute_swaps(world) + relation_swaps(world)
        rng.shuffle(pool)
        swap_world = swap = None
        for candidate in pool:
            if C.scene_signature(candidate) != C.scene_signature(world):
                swap_world = candidate
                swap = C.render(generator, candidate, other, order(objects),
                                order(relations), rng.randrange(2))
                break
        if twin is None or swap is None or positive == query:
            continue
        if max(len(tokenizer.tokenize(t)) for t in (query, positive, twin, swap)) > args.max_tokens:
            continue
        if Counter(tokenizer.tokenize(positive)) != Counter(tokenizer.tokenize(swap)):
            continue

        # Identify the intervention type by which per-attribute object multiset
        # changed its binding. Relation-only swaps leave every object untouched
        # and are outside the object-joint prior tested here.
        changed = []
        original_objects = sorted(object_key(o) for o in world["objects"])
        swapped_objects = sorted(object_key(o) for o in swap_world["objects"])
        original_by_id = {obj["id"]: obj for obj in world["objects"]}
        swapped_by_id = {obj["id"]: obj for obj in swap_world["objects"]}
        for name in ATTRIBUTES:
            if any(original_by_id[obj_id][name] != swapped_by_id[obj_id][name]
                   for obj_id in original_by_id):
                changed.append(name)
        swap_type = changed[0] if len(changed) == 1 else (
            "relation" if original_objects == swapped_objects else "+".join(changed)
        )
        items.append({
            "world": position,
            "query": add(query), "positive": add(positive), "twin": add(twin), "swap": add(swap),
            "positive_world": world, "swap_world": swap_world, "swap_type": swap_type,
        })
    return texts, items


def audit_existing_artifacts(texts, items, directory):
    directory = Path(directory)
    stored_texts = [json.loads(line)["caption"] for line in (directory / "texts.jsonl").read_text().splitlines()]
    stored_items = [json.loads(line) for line in (directory / "items.jsonl").read_text().splitlines()]
    compact = [{"world": item["world"], "query": item["query"], "paraphrase": item["positive"],
                "twin": item["twin"], "swap": item["swap"]} for item in items]
    if texts != stored_texts or compact != stored_items:
        raise RuntimeError(
            "In-memory items do not exactly match the existing d-prime artifacts. "
            "Use the same --num-worlds, --seed, manifests, and tokenizer."
        )


def average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(a, b):
    a, b = average_ranks(a), average_ranks(b)
    a -= a.mean(); b -= b.mean()
    return float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-12))


def group_point(model_delta, prior_delta, mask):
    delta = model_delta[mask]
    wins = np.where(np.abs(delta) <= 1e-9, 0.5, (delta > 0).astype(float))
    follows = np.where(
        np.abs(delta) <= 1e-9, 0.5,
        (np.sign(delta) == np.sign(prior_delta[mask])).astype(float),
    )
    return {
        "items": int(mask.sum()),
        "true_preference": float(wins.mean()),
        "mean_cosine_margin": float(delta.mean()),
        "prior_follow_rate": float(follows.mean()),
    }


def score_feature(features, selected, repetitions, seed):
    query = np.asarray([item["query"] for item in selected])
    positive = np.asarray([item["positive"] for item in selected])
    swap = np.asarray([item["swap"] for item in selected])
    prior = np.asarray([item["prior_delta"] for item in selected])
    worlds = np.asarray([item["world"] for item in selected])
    delta = (features[query] * features[positive]).sum(1) - (features[query] * features[swap]).sum(1)
    masks = {
        "all": np.ones(len(selected), dtype=bool),
        "prior_supports_true": prior > 0,
        "prior_conflicts": prior < 0,
    }
    for kind in ATTRIBUTES:
        masks[f"swap_{kind}"] = np.asarray([item["swap_type"] == kind for item in selected])
    result = {name: group_point(delta, prior, mask) for name, mask in masks.items() if mask.any()}
    result["prior_margin_spearman"] = spearman(delta, prior)
    result["support_minus_conflict_preference"] = (
        result["prior_supports_true"]["true_preference"]
        - result["prior_conflicts"]["true_preference"]
    )

    unique = np.unique(worlds)
    by_world = {world: np.flatnonzero(worlds == world) for world in unique}
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in masks}
    draws["prior_margin_spearman"] = []
    draws["support_minus_conflict_preference"] = []
    for _ in range(repetitions):
        picked = np.concatenate([by_world[world] for world in rng.choice(unique, len(unique))])
        picked_prior, picked_delta = prior[picked], delta[picked]
        support, conflict = picked_prior > 0, picked_prior < 0
        for name, mask in masks.items():
            local = mask[picked]
            if local.any():
                draws[name].append(group_point(picked_delta, picked_prior, local)["true_preference"])
        draws["prior_margin_spearman"].append(spearman(picked_delta, picked_prior))
        draws["support_minus_conflict_preference"].append(
            group_point(picked_delta, picked_prior, support)["true_preference"]
            - group_point(picked_delta, picked_prior, conflict)["true_preference"]
        )
    for name, values in draws.items():
        if not values:
            continue
        ci = [float(x) for x in np.quantile(values, (0.025, 0.975))]
        if name in result and isinstance(result[name], dict):
            result[name]["true_preference_ci95"] = ci
        else:
            result[name] = {"value": float(result[name]), "ci95": ci}
    return result


def main():
    args = arguments()
    if args.prior_alpha <= 0:
        raise SystemExit("--prior-alpha must be positive")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = [spec.split("=", 1) for spec in args.checkpoint]
    _, tokenizer, _ = load_model(specs[0][1], torch.device("cpu"))
    texts, items = build_exact_dprime_items(args, tokenizer)
    audit_existing_artifacts(texts, items, args.reuse_dprime_dir)
    counts, total, support_size, train_scenes = estimate_prior(
        args.train_manifest, args.max_train_scenes
    )
    selected = []
    for item in items:
        if item["swap_type"] not in SWAP_TYPES:
            continue
        item = dict(item)
        item["prior_delta"] = (
            scene_log_prior(item["positive_world"], counts, total, support_size, args.prior_alpha)
            - scene_log_prior(item["swap_world"], counts, total, support_size, args.prior_alpha)
        )
        if abs(item["prior_delta"]) > 1e-12:
            selected.append(item)
    supports = sum(item["prior_delta"] > 0 for item in selected)
    conflicts = sum(item["prior_delta"] < 0 for item in selected)
    print(f"reused {len(items)} d-prime items exactly; attribute swaps {len(selected)}; "
          f"prior supports true {supports}, conflicts {conflicts}", flush=True)

    construction = {
        "protocol": "existing d-prime minimal pairs split by empirical training prior",
        "trained_anything": False,
        "generated_new_dataset": False,
        "train_scenes_read": train_scenes,
        "train_objects": total,
        "prior_support_size": support_size,
        "items": len(selected),
        "prior_supports_true": supports,
        "prior_conflicts": conflicts,
        "primary_feature": "L7.residual",
        "prior": "sum_object log p_train(color,shape,material,size), Laplace smoothed",
    }
    (out / "construction.json").write_text(json.dumps(construction, indent=2) + "\n")

    device = torch.device(args.device)
    for label, path in specs:
        destination = out / f"{label}.json"
        model, tokenizer, model_args = load_model(path, device)
        features, worst = encode(model, tokenizer, model_args, texts, args.batch_size, device,
                                 set(args.sublayer_layers), check=False)
        if worst > 1e-3:
            raise RuntimeError(f"{label}: decomposition error {worst:.2e}")
        metrics = {name: score_feature(value, selected, args.bootstrap, args.seed)
                   for name, value in features.items()}
        result = {
            "protocol": {**construction, "model": str(path), "train_mode": model_args.train_mode},
            "metrics": metrics,
        }
        destination.write_text(json.dumps(result, indent=2) + "\n")
        primary = metrics["L7.residual"]
        print(
            f"{label}: L7 true preference support "
            f"{primary['prior_supports_true']['true_preference']:.3f}, conflict "
            f"{primary['prior_conflicts']['true_preference']:.3f}, "
            f"gap {primary['support_minus_conflict_preference']['value']:+.3f}, "
            f"prior-margin rho {primary['prior_margin_spearman']['value']:+.3f}",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
