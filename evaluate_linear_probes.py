#!/usr/bin/env python3
"""Frozen-encoder linear probes for image models, following the I-JEPA protocol.

I-JEPA (Assran et al., 2023) evaluates low-level scene understanding with the
VTAB CLEVR/Count and CLEVR/Dist tasks: the pretrained encoder is frozen, the
patch features are average-pooled (it has no [CLS]), and only a linear
classifier is trained.  Both the last layer and the concatenation of the last
four layers are tried and the better one is reported.  This script applies that
protocol to every checkpoint given, with identical data, splits, probe and
hyper-parameter search, and adds CLEVR factors the VTAB tasks do not cover.

Scene-level tasks (mean-pooled image tokens, one vector per image):

* ``count``  -- number of objects, one class per count (VTAB CLEVR/Count).
* ``dist``   -- camera depth of the closest object, binned at
  [0, 8, 8.5, 9, 9.5, 10, inf) exactly as VTAB CLEVR/Dist.
* ``color`` / ``shape`` / ``material`` / ``size`` -- how many objects carry
  each value (e.g. #red, #cube), ridge regression, mean R^2 over values.
* ``position`` -- 3-D (x, y) of the closest object, ridge regression, R^2.

Object-level tasks (the token under each object's projected centre, one vector
per object): ``obj_color`` (8-way), ``obj_shape`` (3), ``obj_material`` (2),
``obj_size`` (2).  These say whether an attribute is linearly readable where
the object is, which pooled features cannot show for multi-object scenes.

Fairness: every representation is probed on the same 20k held-out scenes, split
once into probe-train / probe-val / test.  Probe regularization and, where a
choice exists, the layer are picked on probe-val only; test is touched once.
Baselines are run through the same code path:

* ``chance``  -- majority class of probe-train (classification), train mean
  (regression, R^2 = 0 by construction).
* ``easy``    -- features with no learning: bag of VQ codes, mean VQ codebook
  vector, 2x-downsampled raw pixels; for objects, the VQ code one-hot and the
  12x12 pixel patch at the object centre.
* ``untrained`` -- the same architecture at random initialization, read
  exactly like the trained checkpoints (``--random-init LABEL=CKPT`` takes its
  architecture from CKPT and ignores its weights).
"""

from __future__ import annotations

from clevr_paths import CLEVR_DATA_ROOT, CLEVR_GEN_ROOT
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import ClevrTextTokenizer, MultimodalCollator
from evaluate_shared_private_retrieval import checkpoint_args, load_model
from train_multimodal import build_model

DATA_ROOT = CLEVR_DATA_ROOT + "/platonic_clevr_v1_5M_train_gpu_visible"
VALUES = {
    "color": ("gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"),
    "shape": ("cube", "sphere", "cylinder"),
    "material": ("rubber", "metal"),
    "size": ("large", "small"),
}
DIST_BINS = (8.0, 8.5, 9.0, 9.5, 10.0)  # VTAB clevr closest_object_distance
GRID = (16, 24)
PATCH = 4  # 96x64 image, 24x16 token grid
PREFIX = 2  # <image>, <bos> before the first image token
CLASSIFICATION_WD = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
RIDGE_ALPHA = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH")
    p.add_argument("--random-init", action="append", default=[], metavar="LABEL=PATH",
                   help="Untrained baseline: architecture of PATH, fresh random weights.")
    p.add_argument("--easy-baselines", action="store_true")
    p.add_argument("--manifest", default=f"{DATA_ROOT}/val_image_only.jsonl")
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--image-cache", default="outputs/image_only_1_2m_token_cache/val_tokens.pt")
    p.add_argument("--vqvae", default="outputs/vqvae_training_bs128/best.pt")
    p.add_argument("--split", type=int, nargs=3, default=(12000, 3000, 5000),
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


# --------------------------------------------------------------------- labels

def load_scenes(manifest, data_root, limit):
    rows = []
    with open(manifest) as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            rows.append(json.loads(line))
    root = Path(data_root)

    def read(row):
        path = root / row["image_path"].replace("images/", "scenes/", 1).replace(".png", ".json")
        return json.loads(path.read_text())

    with ThreadPoolExecutor(32) as pool:
        scenes = list(pool.map(read, rows))
    for row, scene in zip(rows, scenes):
        if scene["image_index"] != row["image_index"]:
            raise RuntimeError(f"scene/manifest mismatch at {row['image_path']}")
        world = [(o["color"], o["shape"], o["material"], o["size"]) for o in row["world"]["objects"]]
        render = [(o["color"], o["shape"], o["material"], o["size"]) for o in scene["objects"]]
        if sorted(world) != sorted(render):
            raise RuntimeError(f"scene objects disagree with manifest world at {row['image_path']}")
    return rows, scenes


def scene_labels(scenes):
    counts = np.asarray([len(s["objects"]) for s in scenes])
    depth = np.asarray([min(o["pixel_coords"][2] for o in s["objects"]) for s in scenes])
    closest = [min(s["objects"], key=lambda o: o["pixel_coords"][2]) for s in scenes]
    labels = {
        "count": ("class", counts),
        "dist": ("class", np.digitize(depth, DIST_BINS)),
        "position": ("regress", np.asarray([o["3d_coords"][:2] for o in closest], dtype=np.float64)),
    }
    for factor, values in VALUES.items():
        labels[factor] = ("regress", np.asarray(
            [[sum(o[factor] == v for o in s["objects"]) for v in values] for s in scenes], dtype=np.float64))
    return labels


def object_table(scenes):
    """One row per object: scene index, flat token index, attribute classes."""
    scene_of, token_of, pixel_xy, attributes = [], [], [], {f: [] for f in VALUES}
    for index, scene in enumerate(scenes):
        for o in scene["objects"]:
            x, y = o["pixel_coords"][:2]
            row = int(np.clip(y // PATCH, 0, GRID[0] - 1))
            col = int(np.clip(x // PATCH, 0, GRID[1] - 1))
            scene_of.append(index)
            token_of.append(row * GRID[1] + col)
            pixel_xy.append((int(np.clip(x, 0, 95)), int(np.clip(y, 0, 63))))
            for factor, values in VALUES.items():
                attributes[factor].append(values.index(o[factor]))
    return (np.asarray(scene_of), np.asarray(token_of), np.asarray(pixel_xy),
            {f"obj_{f}": ("class", np.asarray(v)) for f, v in attributes.items()})


# ------------------------------------------------------------------- features

@torch.no_grad()
def model_features(model, tokenizer, model_args, tokens, object_scene, object_token, batch_size, device):
    """Mean-pooled residual stream after the embedding and every block, plus the
    residual stream at each object's centre token.  Clean (unmasked) input."""
    collator = MultimodalCollator(tokenizer, model_args.num_image_codes, model_args.max_text_length)
    captured, hooks = {}, []
    hooks.append(model.blocks[0].register_forward_pre_hook(
        lambda _m, inputs: captured.__setitem__("emb", inputs[0])))
    for index, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(
            lambda _m, _i, output, index=index: captured.__setitem__(f"L{index}", output)))
    order = ["emb"] + [f"L{i}" for i in range(len(model.blocks))]
    pooled = {name: [] for name in order}
    local = {name: np.zeros((len(object_scene), model_args.d_model), np.float32) for name in order}
    starts = np.searchsorted(object_scene, np.arange(tokens.shape[0] + 1))
    try:
        for start in range(0, tokens.shape[0], batch_size):
            chunk = tokens[start:start + batch_size]
            batch = collator([{"kind": "image", "image_tokens": grid, "pair_index": start + k}
                              for k, grid in enumerate(chunk)])
            batch = {name: value.to(device) for name, value in batch.items()}
            captured.clear()
            model(batch["input_ids"], batch["attention_mask"], batch["position_ids"],
                  batch["modality_ids"], batch["route_ids"])
            weights = batch["eligible_mask"].float().unsqueeze(-1)
            lo, hi = starts[start], starts[min(start + batch_size, tokens.shape[0])]
            rows = torch.as_tensor(object_scene[lo:hi] - start, device=device)
            cols = torch.as_tensor(object_token[lo:hi] + PREFIX, device=device)
            for name in order:
                hidden = captured[name].float()
                pooled[name].append(((hidden * weights).sum(1) / weights.sum(1).clamp_min(1)).cpu())
                local[name][lo:hi] = hidden[rows, cols].cpu().numpy()
    finally:
        for hook in hooks:
            hook.remove()
    scene = {name: torch.cat(parts).numpy() for name, parts in pooled.items()}
    last = len(model.blocks) - 1
    scene["last4"] = np.concatenate([scene[f"L{i}"] for i in range(last - 3, last + 1)], axis=1)
    local["last4"] = np.concatenate([local[f"L{i}"] for i in range(last - 3, last + 1)], axis=1)
    return scene, local, f"L{last}"


def easy_features(tokens, rows, data_root, vqvae_path, object_scene, object_token, object_xy):
    flat = tokens.reshape(tokens.shape[0], -1).numpy()
    histogram = np.zeros((flat.shape[0], 512), np.float32)
    np.add.at(histogram, (np.repeat(np.arange(flat.shape[0]), flat.shape[1]), flat.ravel()), 1.0)
    vq = torch.load(vqvae_path, map_location="cpu", weights_only=False)
    state = vq.get("model", vq)
    codebook_key = next(k for k, v in state.items() if v.ndim == 2 and v.shape[0] == 512
                        and ("embed" in k or "codebook" in k))
    codebook = state[codebook_key].float().numpy()
    if codebook.shape[1] != 64:
        codebook = codebook.T
    root = Path(data_root)

    def read(row):
        return np.asarray(Image.open(root / row["image_path"]).convert("RGB"), dtype=np.float32) / 255.0

    with ThreadPoolExecutor(32) as pool:
        pixels = np.stack(list(pool.map(read, rows)))  # N, 64, 96, 3
    small = pixels.reshape(pixels.shape[0], 32, 2, 48, 2, 3).mean((2, 4)).reshape(pixels.shape[0], -1)
    padded = np.pad(pixels, ((0, 0), (6, 6), (6, 6), (0, 0)))
    patches = np.stack([padded[s, y:y + 12, x:x + 12].ravel()
                        for s, (x, y) in zip(object_scene, object_xy)])
    centre_code = flat[object_scene, object_token]
    one_hot = np.zeros((len(centre_code), 512), np.float32)
    one_hot[np.arange(len(centre_code)), centre_code] = 1.0
    scene = {"vq_code_histogram": histogram,
             "vq_codebook_mean": codebook[flat].mean(1),
             "pixels_32x48": small}
    local = {"vq_code_onehot": one_hot, "vq_codebook_vector": codebook[centre_code],
             "pixel_patch_12x12": patches}
    return scene, local, f"codebook tensor '{codebook_key}'"


# --------------------------------------------------------------------- probes

def standardize(train, *others):
    mean, scale = train.mean(0), train.std(0).clamp_min(1e-6)
    return [(x - mean) / scale for x in (train, *others)]


def fit_logistic(x, y, classes, weight_decay, steps=100):
    weight = torch.zeros(x.shape[1], classes, device=x.device, dtype=x.dtype, requires_grad=True)
    bias = torch.zeros(classes, device=x.device, dtype=x.dtype, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], lr=1, max_iter=steps, history_size=20,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(x @ weight + bias, y) + weight_decay * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return weight.detach(), bias.detach()


def fit_ridge(x, y, alpha):
    n, d = x.shape
    ym = y.mean(0)
    if d <= n:
        w = torch.linalg.solve(x.T @ x + alpha * n * torch.eye(d, device=x.device, dtype=x.dtype), x.T @ (y - ym))
    else:
        w = x.T @ torch.linalg.solve(x @ x.T + alpha * n * torch.eye(n, device=x.device, dtype=x.dtype), y - ym)
    return w, ym


def r2(pred, target, mean):
    residual = ((target - pred) ** 2).sum(0)
    total = ((target - mean) ** 2).sum(0).clamp_min(1e-12)
    return 1 - residual / total  # per output, relative to the probe-train mean


def bootstrap(values, rng, draws):
    values = np.asarray(values)
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(1)
    return [float(x) for x in np.quantile(means, (0.025, 0.975))]


def probe(kind, features, target, index, device, rng, draws):
    """Fit on probe-train, pick the regularizer on probe-val, score test once."""
    train, val, test = index
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    xt, xv, xs = standardize(x[train], x[val], x[test])
    if kind == "class":
        _, y = np.unique(target, return_inverse=True)
        y = torch.as_tensor(y, device=device)
        classes = int(y.max()) + 1
        best = None
        for wd in CLASSIFICATION_WD:
            w, b = fit_logistic(xt, y[train], classes, wd)
            score = ((xv @ w + b).argmax(1) == y[val]).float().mean().item()
            if best is None or score > best[0]:
                best = (score, wd, w, b)
        score, wd, w, b = best
        correct = ((xs @ w + b).argmax(1) == y[test]).float().cpu().numpy()
        return {"metric": "accuracy", "val": score, "test": float(correct.mean()),
                "ci95": bootstrap(correct, rng, draws), "weight_decay": wd, "_items": correct}
    y = torch.as_tensor(target, dtype=torch.float32, device=device)
    y = y[:, None] if y.ndim == 1 else y
    best = None
    for alpha in RIDGE_ALPHA:
        w, ym = fit_ridge(xt, y[train], alpha)
        score = r2(xv @ w + ym, y[val], ym).mean().item()
        if best is None or score > best[0]:
            best = (score, alpha, w, ym)
    score, alpha, w, ym = best
    pred = xs @ w + ym
    per_output = r2(pred, y[test], ym).cpu().numpy()
    # Per-scene contribution for the bootstrap: resample scenes, recompute R^2.
    err = ((y[test] - pred) ** 2).cpu().numpy()
    dev = ((y[test] - ym) ** 2).cpu().numpy()
    picks = rng.integers(0, len(err), (draws, len(err)))
    boot = (1 - err[picks].sum(1) / np.maximum(dev[picks].sum(1), 1e-12)).mean(1)
    rounded = (pred.round() == y[test]).float().mean().item()
    return {"metric": "r2", "val": score, "test": float(per_output.mean()),
            "ci95": [float(x) for x in np.quantile(boot, (0.025, 0.975))],
            "per_output_r2": [float(v) for v in per_output], "rounded_exact": rounded, "alpha": alpha,
            "_items": err}


def chance(kind, target, index):
    train, _, test = index
    if kind == "class":
        values, counts = np.unique(target[train], return_counts=True)
        majority = values[counts.argmax()]
        return {"metric": "accuracy", "test": float((target[test] == majority).mean()),
                "uniform": 1.0 / len(np.unique(target)), "classes": int(len(np.unique(target)))}
    return {"metric": "r2", "test": 0.0}


def probe_all(features, labels, index, primary, device, rng, draws):
    """I-JEPA headline = better of last layer and last-4 concat (chosen on val);
    ``best_layer`` additionally scans every single layer (also chosen on val)."""
    out = {}
    if not features:
        return out
    for task, (kind, target) in labels.items():
        per = {name: probe(kind, f, target, index, device, rng, draws) for name, f in features.items()}
        entry = {"layers": per}
        if primary is not None:
            ijepa = max((primary, "last4"), key=lambda n: per[n]["val"])
            layer = max((n for n in per if n != "last4"), key=lambda n: per[n]["val"])
            entry.update(ijepa_feature=ijepa, ijepa=per[ijepa], best_layer_feature=layer, best_layer=per[layer])
            # Per-test-item scores (correct 0/1, or squared error per output) for
            # the two reported features only, so models can be compared paired.
            entry["items"] = {"ijepa": per[ijepa]["_items"].tolist(),
                              "best_layer": per[layer]["_items"].tolist()}
        for result in per.values():
            result.pop("_items")
        out[task] = entry
    return out


# ----------------------------------------------------------------------- main

def split_indices(count, sizes, rng, groups=None):
    order = rng.permutation(count)
    a, b, c = sizes
    if a + b + c > count:
        raise ValueError(f"split {sizes} needs {a + b + c} scenes, have {count}")
    parts = (order[:a], order[a:a + b], order[a + b:a + b + c])
    return tuple(np.sort(p) for p in parts)


def summary_line(label, result):
    cells = []
    for task, entry in result.items():
        head = entry.get("ijepa") or next(iter(entry["layers"].values()))
        cells.append(f"{task}={100 * head['test']:.1f}" if head["metric"] == "accuracy"
                     else f"{task}=R2 {head['test']:.3f}")
    return f"{label}: " + " ".join(cells)


def main():
    args = arguments()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cache = torch.load(args.image_cache, map_location="cpu", weights_only=False)
    tokens = (cache["tokens"] if isinstance(cache, dict) else cache).long()
    need = sum(args.split)
    rows, scenes = load_scenes(args.manifest, args.data_root, need)
    tokens = tokens[:len(rows)]
    scene_y = scene_labels(scenes)
    object_scene, object_token, object_xy, object_y = object_table(scenes)

    rng = np.random.default_rng(args.seed)
    scene_index = split_indices(len(rows), args.split, rng)
    split_of = np.empty(len(rows), int)
    for k, part in enumerate(scene_index):
        split_of[part] = k
    # Objects inherit their scene's split, so no scene contributes to two splits.
    object_index = tuple(np.flatnonzero(split_of[object_scene] == k) for k in range(3))

    construction = {
        "scenes": len(rows), "split_scenes": list(args.split), "objects_per_split": [len(p) for p in object_index],
        "seed": args.seed, "manifest": args.manifest, "image_cache": args.image_cache,
        "dist_bins": list(DIST_BINS), "classification_weight_decay_grid": list(CLASSIFICATION_WD),
        "ridge_alpha_grid": list(RIDGE_ALPHA),
        "chance": {**{t: chance(k, y, scene_index) for t, (k, y) in scene_y.items()},
                   **{t: chance(k, y, object_index) for t, (k, y) in object_y.items()}},
        "label_histograms": {t: {str(v): int(c) for v, c in zip(*np.unique(y, return_counts=True))}
                             for t, (k, y) in {**scene_y, **object_y}.items() if k == "class"},
    }
    (out / "construction.json").write_text(json.dumps(construction, indent=2) + "\n")
    print(f"scenes {len(rows)} objects {len(object_scene)}; chance "
          + " ".join(f"{t}={100 * c['test']:.1f}" for t, c in construction["chance"].items()
                     if c["metric"] == "accuracy"), flush=True)

    def run(label, scene_features, local_features, primary, protocol):
        destination = out / f"{label}.json"
        if destination.exists():
            print(f"{label}: exists, skipping", flush=True)
            return
        began = time.time()
        task_rng = np.random.default_rng(args.seed + 1)
        result = {"scene": probe_all(scene_features, scene_y, scene_index, primary, device, task_rng, args.bootstrap),
                  "object": probe_all(local_features, object_y, object_index, primary, device, task_rng, args.bootstrap)}
        destination.write_text(json.dumps({"protocol": protocol, **result}, indent=2) + "\n")
        print(summary_line(label, {**result["scene"], **result["object"]})
              + f"  ({time.time() - began:.0f}s)", flush=True)

    if args.easy_baselines:
        scene_f, local_f, note = easy_features(tokens, rows, args.data_root, args.vqvae,
                                               object_scene, object_token, object_xy)
        for name in scene_f:
            run(f"easy_{name}", {name: scene_f[name]}, {}, None, {"baseline": name, "vqvae": note})
        for name in local_f:
            run(f"easy_{name}", {}, {name: local_f[name]}, None, {"baseline": name, "vqvae": note})

    for spec, random in [(s, False) for s in args.checkpoint] + [(s, True) for s in args.random_init]:
        label, path = spec.split("=", 1)
        if (out / f"{label}.json").exists():
            print(f"{label}: exists, skipping", flush=True)
            continue
        if random:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model_args = checkpoint_args(payload)
            tokenizer = ClevrTextTokenizer(payload["text_vocabulary"])
            torch.manual_seed(args.seed)
            model, _ = build_model(model_args, len(tokenizer))
            model.to(device).eval()
            del payload
        else:
            model, tokenizer, model_args = load_model(path, device)
        scene_f, local_f, primary = model_features(model, tokenizer, model_args, tokens,
                                                   object_scene, object_token, args.batch_size, device)
        run(label, scene_f, local_f, primary,
            {"model": path, "random_init": random, "train_mode": model_args.train_mode,
             "headline": "better of last block and last-4 concat on probe-val (I-JEPA)"})
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
