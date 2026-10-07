#!/usr/bin/env python3
"""Binding discrimination for image models, from natural minimal pairs.

The text side has `d_bind`: two captions with identical words that differ only
in which object has which attribute. Images had no equivalent, because building
one seemed to require re-rendering counterfactual scenes.

It does not. The rendered corpus already contains them. Two scenes are a
*content-matched pair* when their per-attribute multisets are identical -- same
colours, same shapes, same materials, same sizes -- but the attributes are
attached to different objects. A representation that encodes only *which
attributes are present* gives such a pair identical features, so it scores
exactly 50%; anything above that requires knowing which attributes co-occur on
the same object.

Protocol, mirroring `evaluate_binding_swap.py`:

1. Train a linear probe on held-out scenes to read the binding-dependent
   conjunction facts out of the pooled image representation.
2. For each content-matched pair, take the facts that differ between the two
   scenes and ask whether the probe puts each one on the right scene.
3. Score the fraction ranked correctly, bootstrapped over pairs.

Reported alongside: the probe's ordinary balanced accuracy on unmatched scenes
(is the probe any good at all) and a bootstrap interval, which the scene-probe
number in `evaluate_cross_modal_structure.py` lacks.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import binding_swap_captions as C
from data import MultimodalCollator
from evaluate_binding_swap import fit_probe, label_matrix
from evaluate_shared_private_retrieval import load_model

ATTRIBUTES = ("color", "shape", "material", "size")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--manifest", default="/home/zd25e122/clevr-dataset-gen_clone/output/platonic_clevr_v1_5M_train_gpu_visible/val_image_only.jsonl")
    p.add_argument("--image-cache", default="outputs/image_only_1_2m_token_cache/val_tokens.pt")
    p.add_argument("--probe-train", type=int, default=12000)
    p.add_argument("--min-label-count", type=int, default=50)
    p.add_argument("--per-layer", action="store_true",
                   help="Score the residual stream after the embedding and every block, "
                        "not only the last one.")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260924)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def content_signature(world):
    """Per-attribute multisets: equal signatures mean identical scene content."""
    objects = world["objects"]
    return (len(objects),) + tuple(tuple(sorted(o[a] for o in objects)) for a in ATTRIBUTES)


def binding_signature(world):
    return tuple(sorted(tuple(o[a] for a in ATTRIBUTES) for o in world["objects"]))


def build_pairs(worlds, rng):
    """Content-matched scene pairs whose attribute-to-object assignment differs."""
    groups = collections.defaultdict(list)
    for index, world in enumerate(worlds):
        groups[content_signature(world)].append(index)
    pairs = []
    for members in groups.values():
        if len(members) < 2:
            continue
        by_binding = collections.defaultdict(list)
        for index in members:
            by_binding[binding_signature(worlds[index])].append(index)
        keys = sorted(by_binding, key=lambda k: str(k))
        if len(keys) < 2:
            continue
        for first in range(len(keys)):
            for second in range(first + 1, len(keys)):
                pairs.append((by_binding[keys[first]][0], by_binding[keys[second]][0]))
    rng.shuffle(pairs)
    return pairs


@torch.no_grad()
def encode(model, tokenizer, model_args, image_tokens, batch_size, device, per_layer=False):
    """Pooled, L2-normalized image representation.

    With ``per_layer`` the residual stream is read after the embedding and
    after every block, so binding can be traced through depth the way the text
    side traces it; otherwise only the last block is returned.
    """
    collator = MultimodalCollator(tokenizer, model_args.num_image_codes, model_args.max_text_length)
    captured, hooks = {}, []
    if per_layer:
        hooks.append(model.blocks[0].register_forward_pre_hook(
            lambda _m, inputs: captured.__setitem__("embedding", inputs[0].detach())))
        for index, block in enumerate(model.blocks):
            hooks.append(block.register_forward_hook(
                lambda _m, _i, output, index=index: captured.__setitem__(f"L{index}", output.detach())))
            # The two sublayer writes, so the image side can be read the same way
            # the text side is: residual stream, attention output, MLP output.
            hooks.append(block.attn.out_proj.register_forward_hook(
                lambda _m, _i, output, index=index:
                    captured.__setitem__(f"L{index}.attn_out", output.detach())))
            hooks.append(block.mlp[3].register_forward_hook(
                lambda _m, _i, output, index=index:
                    captured.__setitem__(f"L{index}.mlp_out", output.detach())))
    else:
        hooks.append(model.blocks[-1].register_forward_hook(
            lambda _m, _i, output: captured.__setitem__("last", output.detach())))
    values: dict[str, list[torch.Tensor]] = {}
    try:
        for start in range(0, image_tokens.shape[0], batch_size):
            chunk = image_tokens[start:start + batch_size]
            batch = collator([{"kind": "image", "image_tokens": grid, "pair_index": start + k}
                              for k, grid in enumerate(chunk)])
            batch = {name: value.to(device) for name, value in batch.items()}
            captured.clear()
            model(batch["input_ids"], batch["attention_mask"], batch["position_ids"],
                  batch["modality_ids"], batch["route_ids"])
            weights = batch["eligible_mask"].double().unsqueeze(-1)
            for name, tensor in captured.items():
                pooled = (tensor.double() * weights).sum(1) / weights.sum(1).clamp_min(1)
                values.setdefault(name, []).append(F.normalize(pooled, dim=1).float().cpu())
    finally:
        for hook in hooks:
            hook.remove()
    return {name: torch.cat(parts).numpy() for name, parts in values.items()}


def main():
    args = arguments()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    cache = torch.load(args.image_cache, map_location="cpu", weights_only=False)
    tokens = (cache["tokens"] if isinstance(cache, dict) else cache).long()
    worlds = []
    with open(args.manifest) as handle:
        for row, line in enumerate(handle):
            if row >= tokens.shape[0]:
                break
            worlds.append(json.loads(line)["world"])
    tokens = tokens[:len(worlds)]

    labels = C.all_conjunction_labels()
    matrix = label_matrix(worlds, labels)
    keep = (matrix.sum(0) >= args.min_label_count) & ((1 - matrix).sum(0) >= args.min_label_count)
    labels = [label for label, k in zip(labels, keep) if k]
    matrix = matrix[:, keep]
    position = {label: k for k, label in enumerate(labels)}

    split = args.probe_train
    import random as _random
    pairs = [(a, b) for a, b in build_pairs(worlds, _random.Random(args.seed)) if a >= split and b >= split]
    items = []
    for a, b in pairs:
        facts_a = C.conjunction_labels(worlds[a]) & set(labels)
        facts_b = C.conjunction_labels(worlds[b]) & set(labels)
        flipped = sorted(facts_a ^ facts_b, key=position.get)
        if flipped:
            items.append((a, b, np.asarray([position[f] for f in flipped]),
                          np.asarray([1.0 if f in facts_a else -1.0 for f in flipped])))
    print(f"scenes {len(worlds)}, labels kept {len(labels)}, content-matched test pairs {len(items)}", flush=True)
    (out / "construction.json").write_text(json.dumps(
        {"scenes": len(worlds), "labels": len(labels), "pairs": len(items),
         "probe_train_scenes": split, "seed": args.seed}, indent=2) + "\n")

    for spec in args.checkpoint:
        label, path = spec.split("=", 1)
        destination = out / f"{label}.json"
        if destination.exists():
            print(f"{label}: exists, skipping", flush=True)
            continue
        model, tokenizer, model_args = load_model(path, device)
        features = encode(model, tokenizer, model_args, tokens, args.batch_size, device,
                          per_layer=args.per_layer)

        def measure(vectors):
            """Binding accuracy and clean probe accuracy for one feature."""
            logits_of = fit_probe(vectors[:split], matrix[:split], device)
            test_logits = logits_of(vectors[split:])
            predicted = test_logits > 0
            held = matrix[split:]
            balanced = []
            for column in range(held.shape[1]):
                positives, negatives = held[:, column] == 1, held[:, column] == 0
                if positives.sum() and negatives.sum():
                    balanced.append(0.5 * (predicted[positives, column].mean()
                                           + (~predicted[negatives, column]).mean()))
            scores = np.empty(len(items))
            for n, (a, b, flips, signs) in enumerate(items):
                difference = (test_logits[a - split, flips] - test_logits[b - split, flips]) * signs
                scores[n] = np.where(np.abs(difference) <= 1e-9, 0.5, (difference > 0).astype(float)).mean()
            return scores, float(np.mean(balanced))

        if args.per_layer:
            order = ["embedding"]
            for i in range(len(model.blocks)):
                order += [f"L{i}", f"L{i}.attn_out", f"L{i}.mlp_out"]
        else:
            order = ["last"]
        order = [name for name in order if name in features]
        per_layer = {}
        for name in order:
            scores, balanced = measure(features[name])
            draws = np.asarray([scores[rng.integers(0, len(scores), len(scores))].mean()
                                for _ in range(args.bootstrap)])
            low, high = np.quantile(draws, (0.025, 0.975))
            per_layer[name] = {"binding_accuracy": {"value": float(scores.mean()),
                                                    "ci95": [float(low), float(high)]},
                               "clean_probe_balanced_accuracy": balanced,
                               "per_pair": [float(x) for x in scores]}
        # The headline stays the deepest block, so results written before the
        # per-layer option remain directly comparable.
        headline = f"L{len(model.blocks) - 1}" if args.per_layer else "last"
        deepest = per_layer[headline]
        result = {"protocol": {"model": str(path), "pairs": len(items), "probe_train_scenes": split,
                               "per_layer": bool(args.per_layer), "headline_feature": headline},
                  "binding_accuracy": deepest["binding_accuracy"],
                  "clean_probe_balanced_accuracy": deepest["clean_probe_balanced_accuracy"],
                  # Per-pair scores, so two models can be compared on the same
                  # items: the paired difference removes the between-item
                  # variance that dominates these intervals.
                  "per_pair": deepest["per_pair"],
                  "layers": per_layer}
        destination.write_text(json.dumps(result, indent=2) + "\n")
        head = deepest["binding_accuracy"]
        print(f"{label}: binding {100*head['value']:.1f}% "
              f"[{100*head['ci95'][0]:.1f}, {100*head['ci95'][1]:.1f}] "
              f"| clean probe {100*deepest['clean_probe_balanced_accuracy']:.1f}%"
              + ("  | per layer " + " ".join(f"{n}={100*per_layer[n]['binding_accuracy']['value']:.1f}"
                                             for n in order) if args.per_layer else ""), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
