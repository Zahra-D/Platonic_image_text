#!/usr/bin/env python3
"""Direct semantic-versus-exact-template conflict evaluation.

For every held-out world, compare two candidates against a query:
1. the same world rendered in a different canonical wording template; and
2. a maximally dissimilar held-out world rendered with the *same* template.

If a representation is semantic, candidate 1 should be closer.  If it mainly
tracks wording template, candidate 2 should be closer.  No model is trained.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_shared_private_retrieval import encode_updates, load_model


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--num-worlds", type=int, default=256)
    parser.add_argument(
        "--contrast-mode", choices=("counterfactual", "heldout_max_dissimilar"), default="counterfactual",
        help="Use a slot-changing synthetic counterfactual, or the most dissimilar real held-out world.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def world_atoms(world: dict) -> set[str]:
    atoms = {f"count:{len(world['objects'])}"}
    for object_ in world["objects"]:
        for key in ("size", "color", "material", "shape"):
            atoms.add(f"{key}:{object_[key]}")
    for relation in world.get("relations", []):
        atoms.add(f"relation:{relation['relation']}")
    return atoms


def slot_atoms(world: dict) -> set[str]:
    """Attribute atoms retain object position, so changed slots cannot hide in a set."""
    atoms = {f"count:{len(world['objects'])}"}
    for index, object_ in enumerate(world["objects"]):
        for key in ("size", "color", "material", "shape"):
            atoms.add(f"object:{index}:{key}:{object_[key]}")
    for index, relation in enumerate(world.get("relations", [])):
        atoms.add(f"relation:{index}:{relation}")
    return atoms


def object_clauses(world: dict) -> str:
    return " ".join(
        f"Object {index + 1} is a {obj['size']} {obj['color']} {obj['material']} {obj['shape']}."
        for index, obj in enumerate(world["objects"])
    )


def relation_clauses(world: dict) -> str:
    relation_words = {
        "left": "to the left of", "right": "to the right of",
        "front": "in front of", "behind": "behind",
    }
    clauses = []
    for relation in world.get("relations", []):
        subject = relation.get("subject", relation.get("subject_id"))
        anchor = relation.get("anchor", relation.get("anchor_id"))
        # Text-only worlds use zero-based indices; rendered manifests' ids are
        # one-based.  The exact identity is immaterial here, but its wording
        # must be deterministic and faithful to each record.
        if "subject" in relation:
            subject += 1
        if "anchor" in relation:
            anchor += 1
        clauses.append(
            f"Object {subject} is {relation_words[relation['relation']]} object {anchor}."
        )
    return " ".join(clauses) if clauses else "No spatial relation is stated."


def render(world: dict, template_id: int) -> str:
    count = len(world["objects"])
    objects = object_clauses(world)
    relations = relation_clauses(world)
    if template_id == 0:
        return f"Scene summary: there are {count} objects. {objects} Spatial facts: {relations}"
    if template_id == 1:
        return f"Description of a scene. {objects} The scene contains {count} objects. Spatial description: {relations}"
    raise ValueError(f"Unknown template {template_id}")


def counterfactual_world(world: dict) -> dict:
    """Change each scene slot; change cardinality whenever the range allows it."""
    result = copy.deepcopy(world)
    colors = ("gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow")
    shapes = ("cube", "sphere", "cylinder")
    color_index = {value: index for index, value in enumerate(colors)}
    shape_index = {value: index for index, value in enumerate(shapes)}
    for object_ in result["objects"]:
        object_["size"] = "large" if object_["size"] == "small" else "small"
        object_["material"] = "rubber" if object_["material"] == "metal" else "metal"
        object_["color"] = colors[(color_index[object_["color"]] + 3) % len(colors)]
        object_["shape"] = shapes[(shape_index[object_["shape"]] + 1) % len(shapes)]
    if len(result["objects"]) < 10:
        seed = result["objects"][0]
        added = dict(seed)
        added["id"] = max(obj.get("id", index) for index, obj in enumerate(result["objects"])) + 1
        added["size"] = "large" if seed["size"] == "small" else "small"
        added["material"] = "rubber" if seed["material"] == "metal" else "metal"
        added["color"] = colors[(color_index[seed["color"]] + 5) % len(colors)]
        added["shape"] = shapes[(shape_index[seed["shape"]] + 2) % len(shapes)]
        result["objects"].append(added)
    else:
        result["objects"].pop()
    count = len(result["objects"])
    opposite = {"left": "right", "right": "left", "front": "behind", "behind": "front"}
    relations = []
    for relation in result.get("relations", []):
        if "subject" in relation and (relation["subject"] >= count or relation["anchor"] >= count):
            continue
        changed = dict(relation)
        changed["relation"] = opposite[relation["relation"]]
        relations.append(changed)
    if not relations and count >= 2:
        relations.append({"subject": 0, "anchor": 1, "relation": "left"})
    result["relations"] = relations
    return result


def load_worlds(manifest: str, count: int):
    rows = []
    with open(manifest) as handle:
        for line in handle:
            if len(rows) == count:
                break
            row = json.loads(line)
            rows.append(row)
    if len(rows) != count:
        raise ValueError(f"Requested {count} worlds but found {len(rows)}")
    return rows


def contrast_indices(worlds):
    atoms = [world_atoms(row["world"]) for row in worlds]
    result, overlaps = [], []
    for index, source in enumerate(atoms):
        candidates = []
        for other, target in enumerate(atoms):
            if other == index:
                continue
            overlap = len(source & target) / len(source | target)
            # Maximal semantic difference under the same fixed, held-out pool.
            candidates.append((overlap, other))
        overlap, other = min(candidates)
        result.append(other)
        overlaps.append(overlap)
    return result, overlaps


def bootstrap(values, repetitions, seed):
    rng = np.random.default_rng(seed)
    point = values.mean(axis=0)
    draws = np.empty((repetitions, values.shape[1]))
    for index in range(repetitions):
        picked = rng.integers(0, len(values), len(values))
        draws[index] = values[picked].mean(axis=0)
    limits = np.quantile(draws, (0.025, 0.975), axis=0)
    return {"value": point.tolist(), "ci95": limits.tolist()}


def route_metrics(features, count, bootstrap_repetitions, seed):
    # Rows are [source/t0, source/t1, counterfactual/t0, counterfactual/t1]
    # for every source world.
    indices = np.arange(count)
    q0, q1 = 4 * indices, 4 * indices + 1
    same0, same1 = q1, q0
    contrast0, contrast1 = 4 * indices + 2, 4 * indices + 3
    query = np.concatenate([q0, q1])
    semantic = np.concatenate([same0, same1])
    template = np.concatenate([contrast0, contrast1])
    semantic_similarity = (features[query] * features[semantic]).sum(axis=1)
    template_similarity = (features[query] * features[template]).sum(axis=1)
    # Average the two template directions before the paired world bootstrap.
    semantic_similarity = semantic_similarity.reshape(2, count).mean(0)
    template_similarity = template_similarity.reshape(2, count).mean(0)
    gap = semantic_similarity - template_similarity
    values = np.stack((semantic_similarity, template_similarity, gap, gap > 0, gap < 0), axis=1)
    packed = bootstrap(values, bootstrap_repetitions, seed)
    names = ("same_semantics_different_template_cosine", "same_template_different_semantics_cosine",
             "semantic_minus_template_cosine", "semantic_preference_rate", "template_preference_rate")
    return {
        name: {"value": float(packed["value"][index]),
               "ci95": [float(packed["ci95"][0][index]), float(packed["ci95"][1][index])]} 
        for index, name in enumerate(names)
    }


def report(result):
    lines = [
        "# Exact-template semantic conflict evaluation", "",
        "Each query is contrasted with a same-world/different-template caption and a slot-changed "
        "counterfactual/same-template caption. A semantic branch should prefer the former; a wording-template "
        "branch should prefer the latter.", "",
        "| Route | Same semantics, different template cosine | Same template, different semantics cosine | Semantic − template gap | Semantic preference | Template preference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for route in ("shared", "private"):
        metric = result["metrics"][route]
        lines.append(
            f"| {route} | {metric['same_semantics_different_template_cosine']['value']:.3f} | "
            f"{metric['same_template_different_semantics_cosine']['value']:.3f} | "
            f"{metric['semantic_minus_template_cosine']['value']:.3f} | "
            f"{metric['semantic_preference_rate']['value']:.1%} | {metric['template_preference_rate']['value']:.1%} |"
        )
    lines += ["", "The two canonical templates have exactly fixed non-slot wording. The same-template candidate is a "
              "counterfactual scene with changed count where possible, plus changed size, color, material, shape, "
              "and relation slots. It is a text-only control, not a rendered scene."]
    return "\n".join(lines) + "\n"


def main():
    args = arguments()
    output = Path(args.output)
    existing = [] if not output.exists() else [path for path in output.iterdir() if path.name != "launcher.log"]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output already has result files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    worlds = load_worlds(args.manifest, args.num_worlds)
    if args.contrast_mode == "counterfactual":
        contrast_worlds = [counterfactual_world(row["world"]) for row in worlds]
        overlaps = [len(slot_atoms(row["world"]) & slot_atoms(contrast)) /
                    len(slot_atoms(row["world"]) | slot_atoms(contrast))
                    for row, contrast in zip(worlds, contrast_worlds)]
    else:
        contrasts, overlaps = contrast_indices(worlds)
        contrast_worlds = [worlds[index]["world"] for index in contrasts]
    variants = []
    for index, row in enumerate(worlds):
        for source_kind, world in (("source", row["world"]), ("counterfactual", contrast_worlds[index])):
            for template_id in (0, 1):
                variants.append({
                    "index": len(variants), "world_index": index,
                    "world_id": row.get("world_id", row.get("id", index)),
                    "source_kind": source_kind, "template_id": template_id,
                    "caption": render(world, template_id), "contrast_atom_jaccard": overlaps[index],
                })
    if len({row["caption"] for row in variants}) != len(variants):
        raise RuntimeError("Canonical captions collided; refusing ambiguous exact-template evaluation")
    (output / "variants.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in variants))
    model, tokenizer, model_args = load_model(args.checkpoint, torch.device(args.device))
    features = encode_updates(model, tokenizer, model_args, variants, args.batch_size, torch.device(args.device))
    result = {
        "protocol": {
            "checkpoint": str(Path(args.checkpoint).resolve()), "manifest": str(Path(args.manifest).resolve()),
            "num_worlds": args.num_worlds, "templates": 2, "contrast_mode": args.contrast_mode,
            "feature": "clean native blocks.4.mlp.3 update",
            "mean_contrast_atom_jaccard": float(np.mean(overlaps)), "max_contrast_atom_jaccard": float(np.max(overlaps)),
            "bootstrap": args.bootstrap, "seed": args.seed,
        },
        "metrics": {route: route_metrics(value, args.num_worlds, args.bootstrap, args.seed)
                    for route, value in features.items()},
    }
    torch.save({**{route: torch.from_numpy(value) for route, value in features.items()}, "metadata": variants}, output / "features.pt")
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "REPORT.md").write_text(report(result))
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
