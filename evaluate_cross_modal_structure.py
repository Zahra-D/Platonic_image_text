#!/usr/bin/env python3
"""Semantic content per model, and whether text and image models agree.

Two measurements, both on the same set of scenes, so a text model and an image
model can be compared on identical content:

**Per model: how much of the scene is in the representation.**
  ``probe_accuracy`` -- a linear probe (the binding-swap probe's fitter) trained
  to read the scene's binding-dependent conjunction facts out of the pooled
  representation, scored as balanced accuracy on held-out scenes.
  ``rsa_scene`` -- Spearman correlation between representation cosine distance
  and scene-graph distance (symmetric difference of the two scenes' fact sets).
  Unlike a binary swap test this is graded and has no ceiling: it asks whether
  the geometry is *organized* by scene content.

**Across modalities: do independently trained models agree?**
  For the same scenes, the text model encodes a rendered caption and the image
  model encodes the VQ tokens.  The two live in unrelated 384-dimensional
  spaces, so only rotation-invariant comparisons are meaningful:
  ``cka`` -- linear centered kernel alignment between the two representations.
  ``rsa_cross`` -- Spearman between their pairwise-distance structures.
  Plus each side's covariance effective rank and mean pairwise cosine, because
  "the same distribution" also means a comparable shape and spread.

This is the platonic question for this project: two models that never shared a
parameter, trained on different modalities of the same scenes, should converge
on the same structure to the extent that either has learned the scenes.
"""

from __future__ import annotations

from clevr_paths import CLEVR_DATA_ROOT, CLEVR_GEN_ROOT
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import binding_swap_captions as C
from data import MultimodalCollator
from evaluate_binding_swap import fit_probe, label_matrix
from evaluate_shared_private_retrieval import load_model


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text-checkpoint", action="append", default=[], metavar="LABEL=PATH")
    p.add_argument("--image-checkpoint", action="append", default=[], metavar="LABEL=PATH")
    p.add_argument("--image-manifest", default=CLEVR_DATA_ROOT + "/platonic_clevr_v1_5M_train_gpu_visible/val_image_only.jsonl")
    p.add_argument("--image-cache", default="outputs/image_only_1_2m_token_cache/val_tokens.pt")
    p.add_argument("--caption-generator", default=CLEVR_GEN_ROOT + "/generate_human_captions.py")
    p.add_argument("--num-scenes", type=int, default=4000)
    p.add_argument("--probe-train", type=int, default=3000)
    p.add_argument("--rsa-pairs", type=int, default=200000)
    p.add_argument("--min-label-count", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--token-readouts", action="store_true",
                   help="Also read BOS, EOS, the modality token and the all-token mean of every block.")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def load_scenes(args):
    """Worlds, their cached image tokens, and one rendered caption each."""
    generator = C.load_generator(args.caption_generator)
    cache = torch.load(args.image_cache, map_location="cpu", weights_only=False)["tokens"]
    worlds, rows = [], []
    with open(args.image_manifest) as handle:
        for row, line in enumerate(handle):
            if row >= cache.shape[0] or len(worlds) == args.num_scenes:
                break
            worlds.append(json.loads(line)["world"])
            rows.append(row)
    captions = []
    for position, world in enumerate(worlds):
        rng = random.Random(args.seed * 1_000_003 + position)
        objects, relations = len(world["objects"]), len(world.get("relations", []))
        plan = C.make_plan(generator, rng.randrange(2**31), objects, relations)
        captions.append(C.render(
            generator, world, plan, rng.sample(range(objects), objects),
            rng.sample(range(relations), relations), rng.randrange(2),
        ))
    # Token caches are stored as uint16, which torch cannot index directly;
    # numpy can, and the gathered slice is small enough to widen afterwards.
    gathered = torch.from_numpy(cache.numpy()[rows].astype("int64"))
    return worlds, gathered, captions


@torch.no_grad()
def encode_features(model, tokenizer, model_args, device, batch_size, *, captions=None, image_tokens=None,
                    token_readouts=False):
    """Pooled, L2-normalized residual stream after the embedding and each block."""
    collator = MultimodalCollator(tokenizer, model_args.num_image_codes, model_args.max_text_length)
    captured, hooks = {}, []
    hooks.append(model.blocks[0].register_forward_pre_hook(
        lambda _m, inputs: captured.__setitem__("embedding", inputs[0].detach().clone())))
    for index, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(
            lambda _m, _i, output, index=index: captured.__setitem__(f"L{index}", output.detach().clone())))
        # The sublayer writes, before they are added into the residual stream.
        hooks.append(block.attn.out_proj.register_forward_hook(
            lambda _m, _i, output, index=index:
                captured.__setitem__(f"L{index}.attn_out", output.detach().clone())))
        hooks.append(block.mlp[3].register_forward_hook(
            lambda _m, _i, output, index=index:
                captured.__setitem__(f"L{index}.mlp_out", output.detach().clone())))
    count = len(captions) if captions is not None else image_tokens.shape[0]
    values: dict[str, list[torch.Tensor]] = {}
    try:
        for start in range(0, count, batch_size):
            if captions is not None:
                examples = [{"kind": "text", "text": text, "pair_index": start + k}
                            for k, text in enumerate(captions[start:start + batch_size])]
            else:
                examples = [{"kind": "image", "image_tokens": tokens, "pair_index": start + k}
                            for k, tokens in enumerate(image_tokens[start:start + batch_size])]
            batch = {name: value.to(device) for name, value in collator(examples).items()}
            captured.clear()
            model(batch["input_ids"], batch["attention_mask"], batch["position_ids"],
                  batch["modality_ids"], batch["route_ids"])
            weights = batch["eligible_mask"].double().unsqueeze(-1)
            for name, tensor in captured.items():
                pooled = (tensor.double() * weights).sum(1) / weights.sum(1).clamp_min(1)
                values.setdefault(name, []).append(F.normalize(pooled, dim=1).float().cpu())
            if token_readouts:
                # Special-token readouts of every block output: [TEXT]/[IMAGE] at 0,
                # BOS at 1, EOS at the last attended position, and the mean over all
                # attended tokens. Skipped at the embedding, where BOS/[mod] are constant.
                attended = batch["attention_mask"]
                last = attended.long().sum(1) - 1
                rows = torch.arange(attended.size(0), device=attended.device)
                every = attended.double().unsqueeze(-1)
                for name, tensor in captured.items():
                    if not (name.startswith("L") and "." not in name):
                        continue
                    t = tensor.double()
                    picks = {"mod": t[:, 0], "bos": t[:, 1], "eos": t[rows, last],
                             "all": (t * every).sum(1) / every.sum(1).clamp_min(1)}
                    for key, vector in picks.items():
                        values.setdefault(f"{name}@{key}", []).append(F.normalize(vector, dim=1).float().cpu())
    finally:
        for hook in hooks:
            hook.remove()
    return {name: torch.cat(parts).numpy() for name, parts in values.items()}


def scene_distance(labels_matrix, pairs):
    """Symmetric difference of the two scenes' fact sets."""
    left, right = labels_matrix[pairs[:, 0]], labels_matrix[pairs[:, 1]]
    return np.abs(left - right).sum(1)


def spearman(a, b):
    rank = lambda x: np.argsort(np.argsort(x)).astype(np.float64)
    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / max(np.sqrt((ra * ra).sum() * (rb * rb).sum()), 1e-12))


def linear_cka(x, y):
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    cross = np.linalg.norm(y.T @ x, ord="fro") ** 2
    return float(cross / max(np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro"), 1e-12))


def effective_rank(x):
    x = x - x.mean(0, keepdims=True)
    spectrum = np.linalg.svd(x, compute_uv=False) ** 2
    weights = spectrum / max(spectrum.sum(), 1e-12)
    return float(np.exp(-(weights * np.log(np.clip(weights, 1e-12, None))).sum()))


def semantic_metrics(features, labels_matrix, pairs, args, device):
    split = args.probe_train
    train_y, test_y = labels_matrix[:split], labels_matrix[split:]
    out = {}
    for name, value in features.items():
        logits = fit_probe(value[:split], train_y, device)(value[split:])
        predicted = logits > 0
        balanced = []
        for column in range(test_y.shape[1]):
            positives, negatives = test_y[:, column] == 1, test_y[:, column] == 0
            if positives.sum() and negatives.sum():
                balanced.append(0.5 * (predicted[positives, column].mean()
                                       + (~predicted[negatives, column]).mean()))
        cosine = (value[pairs[:, 0]] * value[pairs[:, 1]]).sum(1)
        out[name] = {
            "probe_accuracy": float(np.mean(balanced)),
            "rsa_scene": spearman(1.0 - cosine, scene_distance(labels_matrix, pairs)),
            "mean_pairwise_cosine": float(cosine.mean()),
            "effective_rank": effective_rank(value),
        }
    return out


def main():
    args = arguments()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    specs = ([("text", s.split("=", 1)) for s in args.text_checkpoint]
             + [("image", s.split("=", 1)) for s in args.image_checkpoint])
    if not specs:
        raise SystemExit("give at least one --text-checkpoint or --image-checkpoint")

    _, tokenizer, _ = load_model(specs[0][1][1], torch.device("cpu"))
    worlds, image_tokens, captions = load_scenes(args)
    labels = C.all_conjunction_labels()
    matrix = label_matrix(worlds, labels)
    keep = (matrix.sum(0) >= args.min_label_count) & ((1 - matrix).sum(0) >= args.min_label_count)
    matrix = matrix[:, keep]
    rng = np.random.default_rng(args.seed)
    pairs = rng.integers(0, len(worlds), size=(args.rsa_pairs, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    print(f"scenes: {len(worlds)}, labels kept: {matrix.shape[1]}, rsa pairs: {len(pairs)}", flush=True)

    store = {}
    for modality, (label, path) in specs:
        destination = out / f"{label}.json"
        model, tokenizer, model_args = load_model(path, device)
        features = encode_features(
            model, tokenizer, model_args, device, args.batch_size,
            captions=captions if modality == "text" else None,
            image_tokens=None if modality == "text" else image_tokens,
            token_readouts=args.token_readouts,
        )
        metrics = semantic_metrics(features, matrix, pairs, args, device)
        destination.write_text(json.dumps({
            "protocol": {"model": str(path), "modality": modality,
                         "train_mode": model_args.train_mode, "scenes": len(worlds)},
            "metrics": metrics,
        }, indent=2) + "\n")
        store[(modality, label)] = features
        best = max(metrics, key=lambda k: metrics[k]["rsa_scene"])
        print(f"{label} [{modality}]: L7 probe {metrics['L7']['probe_accuracy']*100:.1f}% "
              f"rsa {metrics['L7']['rsa_scene']:+.3f} | best rsa {best} "
              f"{metrics[best]['rsa_scene']:+.3f} (probe {metrics[best]['probe_accuracy']*100:.1f}%)",
              flush=True)
        del model
        torch.cuda.empty_cache()

    cross = {}
    for (modality_a, label_a), features_a in store.items():
        for (modality_b, label_b), features_b in store.items():
            if modality_a != "text" or modality_b != "image":
                continue
            per_layer = {}
            for name in features_a:
                if name not in features_b:
                    continue
                a, b = features_a[name], features_b[name]
                cosine_a = (a[pairs[:, 0]] * a[pairs[:, 1]]).sum(1)
                cosine_b = (b[pairs[:, 0]] * b[pairs[:, 1]]).sum(1)
                per_layer[name] = {
                    "cka": linear_cka(a, b),
                    "rsa_cross": spearman(1.0 - cosine_a, 1.0 - cosine_b),
                    "effective_rank_text": effective_rank(a),
                    "effective_rank_image": effective_rank(b),
                }
            cross[f"{label_a}|{label_b}"] = per_layer
            best = max(per_layer, key=lambda k: per_layer[k]["cka"])
            print(f"cross {label_a} vs {label_b}: L7 cka {per_layer['L7']['cka']:.3f} "
                  f"rsa {per_layer['L7']['rsa_cross']:+.3f} | best cka {best} "
                  f"{per_layer[best]['cka']:.3f}", flush=True)
    if cross:
        # Merge into any existing table: a run that evaluates a subset of the
        # models must not erase pairings measured by earlier runs.
        destination = out / "cross_modal.json"
        if destination.exists():
            merged = json.loads(destination.read_text())
            merged.update(cross)
            cross = merged
        destination.write_text(json.dumps(cross, indent=2) + "\n")


if __name__ == "__main__":
    main()
