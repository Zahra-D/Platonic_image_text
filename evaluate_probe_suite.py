"""Layer-wise frozen linear probes on CLEVR scene structure, with a noise sweep.

Freezes the encoder, extracts pooled features at every layer and sublayer, and
fits a linear read-out for each ground-truth property taken from the scene JSON:

    count        number of objects                     (regression, R^2 / MAE)
    color        which of 8 colours are present        (multi-label, mean AP)
    shape        which of 3 shapes are present         (multi-label, mean AP)
    size/material                                      (multi-label, mean AP)
    conj         which (colour, shape) pairs present   (multi-label, mean AP)

`conj` is the binding-sensitive one: attribute-level probes can be solved by
counting features, a conjunction probe cannot.

With --noise-levels the same probes are refit on inputs corrupted to each
masking fraction, which is this model family's analogue of diffusion t.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from data import MultimodalCollator
from evaluate_shared_private_retrieval import load_model

def load_model_or_random(path, device, seed=0):
    """Load a checkpoint, or build the SAME architecture with fresh random
    weights when the path is prefixed with RANDOM:.

    A genuinely untrained encoder is the control that says how much of a probe
    score is learned structure and how much a random projection already
    affords.  Note epoch_000.pt is NOT this -- it has already seen a full
    epoch.
    """
    if not str(path).startswith("RANDOM:"):
        return load_model(path, device)
    import torch as _t
    from train_multimodal import build_model
    from analyze_paired_representations import checkpoint_args
    from data import ClevrTextTokenizer
    real = str(path)[len("RANDOM:"):]
    payload = _t.load(real, map_location="cpu", weights_only=False)
    margs = checkpoint_args(payload)
    tokenizer = ClevrTextTokenizer(payload["text_vocabulary"])
    _t.manual_seed(seed)
    model, _ = build_model(margs, len(tokenizer))
    model.to(device).eval()
    return model, tokenizer, margs


COLORS = ["red", "cyan", "green", "blue", "yellow", "brown", "gray", "purple"]
SHAPES = ["cylinder", "cube", "sphere"]
SIZES = ["small", "large"]
MATS = ["rubber", "metal"]


def labels_from(worlds):
    """Counts for attributes, existence for conjunctions.

    With ~5.4 objects per scene almost every scene contains every shape, size
    and material, so an existence probe for those is ~0.95 at chance and says
    nothing.  Counting how many objects carry each value is informative, and
    conjunctions stay as existence because their base rate is only 0.18.
    """
    out = {}
    out["count"] = np.array([[len(w["objects"])] for w in worlds], dtype=np.float64)
    def counts(vals, key):
        return np.array([[float(sum(o[key] == v for o in w["objects"])) for v in vals]
                         for w in worlds])
    out["color_count"] = counts(COLORS, "color")
    out["shape_count"] = counts(SHAPES, "shape")
    out["size_count"] = counts(SIZES, "size")
    out["material_count"] = counts(MATS, "material")
    pairs = [(c, s) for c in COLORS for s in SHAPES]
    out["conj"] = np.array([[float(any(o["color"] == c and o["shape"] == s for o in w["objects"]))
                             for c, s in pairs] for w in worlds])
    out["conj_count"] = np.array([[float(sum(o["color"] == c and o["shape"] == s
                                             for o in w["objects"])) for c, s in pairs]
                                  for w in worlds])
    return out


@torch.no_grad()
def encode(model, tokenizer, margs, items, batch_size, device, mask_fraction, seed,
           modality="image"):
    collator = MultimodalCollator(tokenizer, margs.num_image_codes, margs.max_text_length)
    captured, hooks = {}, []
    hooks.append(model.blocks[0].register_forward_pre_hook(
        lambda _m, i: captured.__setitem__("embedding", i[0].detach())))
    for i, b in enumerate(model.blocks):
        hooks.append(b.register_forward_hook(
            lambda _m, _i, o, i=i: captured.__setitem__(f"L{i}", o.detach())))
        hooks.append(b.mlp[3].register_forward_hook(
            lambda _m, _i, o, i=i: captured.__setitem__(f"L{i}.mlp_out", o.detach())))
    gen = torch.Generator().manual_seed(seed)
    vals = {}
    try:
        total = len(items) if modality == "text" else items.shape[0]
        for s in range(0, total, batch_size):
            chunk = items[s:s + batch_size]
            if modality == "text":
                ex = [{"kind": "text", "text": t, "pair_index": s + k}
                      for k, t in enumerate(chunk)]
            else:
                ex = [{"kind": "image", "image_tokens": t, "pair_index": s + k}
                      for k, t in enumerate(chunk)]
            batch = {k: v.to(device) for k, v in collator(ex).items()}
            if mask_fraction > 0:
                el = batch["eligible_mask"]
                r = torch.rand(el.shape, generator=gen).to(device)
                hit = el & (r < mask_fraction)
                batch["input_ids"] = batch["input_ids"].masked_fill(hit, tokenizer.mask_id)
            captured.clear()
            model(batch["input_ids"], batch["attention_mask"], batch["position_ids"],
                  batch["modality_ids"], batch["route_ids"])
            w = batch["eligible_mask"].double().unsqueeze(-1)
            for n, t in captured.items():
                pooled = (t.double() * w).sum(1) / w.sum(1).clamp_min(1)
                vals.setdefault(n, []).append(F.normalize(pooled, dim=1).float().cpu())
    finally:
        for h in hooks: h.remove()
    return {k: torch.cat(v).numpy().astype(np.float64) for k, v in vals.items()}


ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def tuned_probe(task, X, Y, fit, val, test):
    """Ridge readout with the penalty chosen on a validation split.

    A fixed penalty is not scale-free: pooled residual features of dense
    models are ~5x less spread than JEPA's, so one alpha shrinks them far
    harder and under-reports them.  The penalty is relative to the mean
    eigenvalue of the centred design, chosen per readout and task on `val`,
    then the readout is refitted on fit+val and scored on `test`.
    """
    def path(rows):
        mx, my = X[rows].mean(0), Y[rows].mean(0)
        xc = X[rows] - mx
        s, v = np.linalg.eigh(xc.T @ xc)
        s = np.clip(s, 0.0, None)
        vtb = v.T @ (xc.T @ (Y[rows] - my))
        scale = max(float(s.mean()), 1e-12)
        return lambda a, q: (X[q] - mx) @ (v @ (vtb / (s + a * scale)[:, None])) + my
    key = (lambda r: r["mAP"]) if task == "conj" else (lambda r: r["r2"])
    pf = path(fit)
    a_best = max(ALPHAS, key=lambda a: key(score(task, Y[val], pf(a, val))))
    full = np.concatenate([fit, val])
    out = score(task, Y[test], path(full)(a_best, test))
    out["alpha"] = a_best
    return out


def ridge(Xtr, Ytr, Xte, alpha=1.0):
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    A = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    return Xte @ W


def average_precision(y, s):
    o = np.argsort(-s); y = y[o]
    if y.sum() == 0: return np.nan
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def score(task, Yte, P):
    if task == "conj":
        pass
    elif task.endswith("count") or task == "count":
        # per-column R^2, averaged
        ss = ((Yte - P) ** 2).sum(0); st = ((Yte - Yte.mean(0)) ** 2).sum(0)
        r2 = 1 - ss / np.clip(st, 1e-12, None)
        return {"r2": float(np.mean(r2)), "mae": float(np.abs(Yte - P).mean()),
                "per_label": [float(x) for x in r2]}
    if False:
        ss = ((Yte - P) ** 2).sum(); st = ((Yte - Yte.mean()) ** 2).sum()
        return {"r2": float(1 - ss / st), "mae": float(np.abs(Yte - P).mean())}
    aps = [average_precision(Yte[:, j], P[:, j]) for j in range(Yte.shape[1])]
    aps = [x for x in aps if not np.isnan(x)]
    return {"mAP": float(np.mean(aps)), "per_label": [float(x) for x in aps]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--modality", choices=["image", "text"], default="image")
    p.add_argument("--caption-field", default="caption_human")
    p.add_argument("--manifest", default="/home/zd25e122/clevr-dataset-gen_clone/output/platonic_clevr_v1_5M_train_gpu_visible/val_image_only.jsonl")
    p.add_argument("--image-cache", default="outputs/image_only_2_5m_token_cache/val_tokens.pt")
    p.add_argument("--num-scenes", type=int, default=8000)
    p.add_argument("--train-frac", type=float, default=0.75)
    p.add_argument("--noise-levels", type=float, nargs="+", default=[0.0])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=20261005)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

    worlds = []
    if a.modality == "text":
        captions = []
        with open(a.manifest) as fh:
            for line in fh:
                if len(worlds) == a.num_scenes: break
                row = json.loads(line)
                worlds.append(row["world"]); captions.append(row[a.caption_field])
        items = captions
    else:
        cache = torch.load(a.image_cache, map_location="cpu", weights_only=False)["tokens"]
        rows = []
        with open(a.manifest) as fh:
            for r, line in enumerate(fh):
                if r >= cache.shape[0] or len(worlds) == a.num_scenes: break
                worlds.append(json.loads(line)["world"]); rows.append(r)
        items = torch.from_numpy(cache.numpy()[rows].astype("int64"))
    Y = labels_from(worlds)
    n = len(worlds); ntr = int(n * a.train_frac)
    rng = np.random.default_rng(a.seed); order = rng.permutation(n)
    tr, te = order[:ntr], order[ntr:]
    nval = len(tr) // 6
    fit, val = tr[nval:], tr[:nval]
    print(f"{n} scenes: {len(fit)} fit / {len(val)} val (penalty selection) / {n-ntr} test", flush=True)
    fdir = out / "features"; fdir.mkdir(exist_ok=True)

    for spec in a.checkpoint:
        label, path = spec.split("=", 1)
        dest = out / f"{label}.json"
        if dest.exists():
            print(f"{label}: exists, skipping", flush=True); continue
        model, tokenizer, margs = load_model_or_random(path, a.device)
        rec = {}
        for t in a.noise_levels:
            cached = fdir / f"{label}_t{t}.npz"
            if cached.exists():
                feats = dict(np.load(cached))
            else:
                feats = encode(model, tokenizer, margs, items, a.batch_size, a.device, t, a.seed,
                               modality=a.modality)
                np.savez(cached, **{k: v.astype(np.float32) for k, v in feats.items()})
            per_layer = {}
            for name, X in feats.items():
                X = X.astype(np.float64)
                per_layer[name] = {task: tuned_probe(task, X, Y[task], fit, val, te) for task in Y}
            rec[f"t={t}"] = per_layer
            best = max(per_layer, key=lambda k: per_layer[k]["conj"]["mAP"])
            print(f"{label} t={t}: count R2 {max(v['count']['r2'] for v in per_layer.values()):+.3f} | "
                  f"colour-count R2 {max(v['color_count']['r2'] for v in per_layer.values()):+.3f} | "
                  f"conj mAP {per_layer[best]['conj']['mAP']:.3f} @{best}", flush=True)
        dest.write_text(json.dumps({"protocol": {"model": str(path), "scenes": n,
                                                 "noise_levels": a.noise_levels},
                                    "metrics": rec}, indent=2) + "\n")
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
