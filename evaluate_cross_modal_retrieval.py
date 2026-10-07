"""Cross-modal retrieval through a linear map fitted between frozen encoders.

CKA says whether the text and image representations have similar geometry,
but it never places them in one coordinate system, so it cannot say whether
one space can be read in the other's terms.  This fits that map explicitly.
Both encoders stay frozen; only W is learned, from paired training scenes:

    W* = argmin_W  sum_i ||x_i W - y_i||^2 + lam ||W||^2        (ridge)
    s_ij = cos(x_i W, y_j)

Each held-out image retrieves its own caption among all test captions
(image -> text), and each caption its own image (text -> image, the same
similarity matrix read the other way).

Also reported, because a full linear map has 384^2 free parameters:

  * orthogonal Procrustes, W restricted to a rotation: the strict version,
    and the one closest to what CKA itself is invariant to;
  * a sample-efficiency curve, retrieval against the number of training
    pairs, which is what "how easily can the spaces be aligned" means;
  * a map fitted on shuffled pairs, which must land at chance or something
    leaks between the splits.

The readout pair (text layer, image layer) and the ridge strength are chosen
on a validation split carved from the training scenes.  Every reported number
comes from test scenes the map never saw.  Scenes and captions are drawn by
evaluate_cross_modal_structure.load_scenes with the same seed, so the first
4000 are exactly the pairs the CKA run used.
"""
from __future__ import annotations

from clevr_paths import CLEVR_DATA_ROOT, CLEVR_GEN_ROOT
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_cross_modal_structure as X
from evaluate_probe_suite import load_model_or_random

READOUTS = (["embedding"] + [f"L{i}" for i in range(8)]
            + [f"L{i}.mlp_out" for i in range(8)])
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
CURVE = (100, 300, 1000, 3000)
KS = (1, 5, 10)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text-checkpoint", action="append", default=[], metavar="LABEL=PATH")
    p.add_argument("--image-checkpoint", action="append", default=[], metavar="LABEL=PATH")
    p.add_argument("--pair", action="append", default=[], metavar="TEXT_LABEL|IMAGE_LABEL",
                   help="pairings to score; default is every text x image combination")
    p.add_argument("--image-manifest", default=CLEVR_DATA_ROOT + "/platonic_clevr_v1_5M_train_gpu_visible/val_image_only.jsonl")
    p.add_argument("--image-cache", default="outputs/image_only_2_5m_token_cache/val_tokens.pt")
    p.add_argument("--caption-generator", default=CLEVR_GEN_ROOT + "/generate_human_captions.py")
    p.add_argument("--num-scenes", type=int, default=8000)
    p.add_argument("--test", type=int, default=1000)
    p.add_argument("--val", type=int, default=1000)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


# --------------------------------------------------------------------------
# retrieval

def unit(m):
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


def both_ways(queries, candidates):
    """Rank of the true partner, both directions, from one similarity matrix.

    Row i of `queries` (mapped images) belongs with row i of `candidates`
    (captions).  Ranks are 1-based; ties count against the true partner.
    """
    s = unit(queries) @ unit(candidates).T
    d = np.diag(s)
    image_to_text = 1 + (s > d[:, None]).sum(1)
    text_to_image = 1 + (s > d[None, :]).sum(0)
    return image_to_text, text_to_image


def summary(ranks):
    r = np.asarray(ranks, dtype=np.float64)
    out = {f"R@{k}": float((r <= k).mean()) for k in KS}
    out["MRR"] = float((1.0 / r).mean())
    out["median_rank"] = float(np.median(r))
    return out


def with_ci(ranks, reps, seed):
    """Summary plus a 95% interval from resampling the query set."""
    r = np.asarray(ranks, dtype=np.float64)
    point = summary(r)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        s = r[rng.integers(0, len(r), len(r))]      # one resample for every metric
        draws.append([(s <= k).mean() for k in KS] + [(1.0 / s).mean()])
    lo, hi = np.quantile(np.asarray(draws), (0.025, 0.975), axis=0)
    for j, key in enumerate([f"R@{k}" for k in KS] + ["MRR"]):
        point[key + "_ci95"] = [float(lo[j]), float(hi[j])]
    return point


def chance(n):
    out = {f"R@{k}": k / n for k in KS}
    out["MRR"] = float(np.mean(1.0 / np.arange(1, n + 1)))
    return out


# --------------------------------------------------------------------------
# maps

class Ridge:
    """Closed-form ridge for every penalty from one eigendecomposition."""

    def __init__(self, x):
        self.x = x
        s, v = np.linalg.eigh(x.T @ x)
        self.s, self.v = np.clip(s, 0.0, None), v
        self.scale = max(float(self.s.mean()), 1e-12)

    def maps(self, y, alphas):
        vtb = self.v.T @ (self.x.T @ y)
        return {a: self.v @ (vtb / (self.s + a * self.scale)[:, None]) for a in alphas}


def procrustes(x, y):
    u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    return u @ vt


def centred(feature, fit_rows, *apply_rows):
    mu = feature[fit_rows].mean(0, keepdims=True)
    return [feature[r] - mu for r in (fit_rows,) + apply_rows]


def mean_mrr(fwd, rev):
    return 0.5 * float((1.0 / fwd).mean() + (1.0 / rev).mean())


# --------------------------------------------------------------------------

def encode_all(a, out, tokens, captions):
    fdir = out / "features"
    fdir.mkdir(parents=True, exist_ok=True)
    jobs = ([(*s.split("=", 1), "text") for s in a.text_checkpoint]
            + [(*s.split("=", 1), "image") for s in a.image_checkpoint])
    for label, path, modality in jobs:
        dest = fdir / f"{label}.npz"
        if dest.exists():
            print(f"{label}: features cached", flush=True)
            continue
        model, tokenizer, margs = load_model_or_random(path, a.device)
        kw = {"captions": captions} if modality == "text" else {"image_tokens": tokens}
        feats = X.encode_features(model, tokenizer, margs, a.device, a.batch_size, **kw)
        np.savez(dest, **{k: feats[k].astype(np.float32) for k in READOUTS if k in feats})
        print(f"{label}: encoded {len(feats)} readouts ({modality})", flush=True)
        del model
        torch.cuda.empty_cache()


def score_pair(ft, fi, tr, va, te, a):
    trva = np.concatenate([tr, va])
    names_i = [k for k in READOUTS if k in fi]
    names_t = [k for k in READOUTS if k in ft]

    # 1. choose readout pair + penalty on validation, for ridge and for Procrustes
    grid, best_r, best_p = {}, (-1.0, None), (-1.0, None)
    for ri in names_i:
        xtr, xva = centred(fi[ri], tr, va)
        ridge = Ridge(xtr)
        for rt in names_t:
            ytr, yva = centred(ft[rt], tr, va)
            scores = {al: mean_mrr(*both_ways(xva @ w, yva))
                      for al, w in ridge.maps(ytr, ALPHAS).items()}
            al = max(scores, key=scores.get)
            pv = mean_mrr(*both_ways(xva @ procrustes(xtr, ytr), yva))
            grid[f"{rt}|{ri}"] = {"alpha": al, "val_mrr_ridge": scores[al], "val_mrr_procrustes": pv}
            if scores[al] > best_r[0]:
                best_r = (scores[al], (rt, ri, al))
            if pv > best_p[0]:
                best_p = (pv, (rt, ri))

    def test_ridge(rt, ri, al, fit=trva, shuffle=False):
        x_fit, x_te = centred(fi[ri], fit, te)
        y_fit, y_te = centred(ft[rt], fit, te)
        if shuffle:
            y_fit = y_fit[np.random.default_rng(a.seed + 7).permutation(len(y_fit))]
        w = Ridge(x_fit).maps(y_fit, (al,))[al]
        return both_ways(x_te @ w, y_te), X.linear_cka(x_te, y_te)

    out = {"grid": grid}

    # 2. the selected cell on test, with intervals
    rt, ri, al = best_r[1]
    (fwd, rev), cka = test_ridge(rt, ri, al)
    out["ridge"] = {"text_readout": rt, "image_readout": ri, "alpha": al, "cka_test": cka,
                    "image_to_text": with_ci(fwd, a.bootstrap, a.seed),
                    "text_to_image": with_ci(rev, a.bootstrap, a.seed + 1)}

    # 3. Procrustes at its own selected cell
    prt, pri = best_p[1]
    x_fit, x_te = centred(fi[pri], trva, te)
    y_fit, y_te = centred(ft[prt], trva, te)
    fwd_p, rev_p = both_ways(x_te @ procrustes(x_fit, y_fit), y_te)
    out["procrustes"] = {"text_readout": prt, "image_readout": pri,
                         "image_to_text": with_ci(fwd_p, a.bootstrap, a.seed + 2),
                         "text_to_image": with_ci(rev_p, a.bootstrap, a.seed + 3)}

    # 4. the layer diagonal, as CKA was tabulated (no cross-layer selection)
    diag = {}
    for k in [f"L{i}" for i in range(8)]:
        if k in fi and k in ft:
            (f2, r2), c2 = test_ridge(k, k, grid[f"{k}|{k}"]["alpha"])
            diag[k] = {"cka_test": c2, "image_to_text": summary(f2), "text_to_image": summary(r2)}
    out["diagonal"] = diag

    # 5. sample efficiency at the selected cell; penalty re-chosen per size on val
    rng = np.random.default_rng(a.seed + 11)
    curve = {}
    for n in [c for c in CURVE if c < len(tr)] + [len(tr)]:
        sub = np.sort(rng.choice(tr, n, replace=False))
        xs, xv = centred(fi[ri], sub, va)
        ys, yv = centred(ft[rt], sub, va)
        maps = Ridge(xs).maps(ys, ALPHAS)
        al_n = max(maps, key=lambda q: mean_mrr(*both_ways(xv @ maps[q], yv)))
        (f3, r3), _ = test_ridge(rt, ri, al_n, fit=sub)
        curve[str(n)] = {"alpha": al_n, "image_to_text": summary(f3), "text_to_image": summary(r3)}
    out["sample_efficiency"] = curve

    # 6. leakage control
    (f4, r4), _ = test_ridge(rt, ri, al, shuffle=True)
    out["shuffled_control"] = {"image_to_text": summary(f4), "text_to_image": summary(r4)}

    # 7. no map at all: raw cosine between a caption and an image at the same
    # readout.  Only meaningful when one trunk encodes both modalities; for two
    # separate encoders it is a control that should sit at chance.
    ident = {}
    for k in [f"L{i}" for i in range(8)] + [f"L{i}.mlp_out" for i in range(8)]:
        if k in fi and k in ft and fi[k].shape[1] == ft[k].shape[1]:
            f5, r5 = both_ways(fi[k][te], ft[k][te])
            ci = centred(fi[k], trva, te)[1]       # per-modality mean removed,
            ct = centred(ft[k], trva, te)[1]       # estimated on training scenes
            f6, r6 = both_ways(ci, ct)
            ident[k] = {"raw": {"image_to_text": summary(f5), "text_to_image": summary(r5)},
                        "modality_centred": {"image_to_text": summary(f6), "text_to_image": summary(r6)}}
    out["identity_map"] = ident
    return out


def main():
    a = arguments()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ns = argparse.Namespace(num_scenes=a.num_scenes, seed=a.seed, caption_generator=a.caption_generator,
                            image_manifest=a.image_manifest, image_cache=a.image_cache)
    worlds, tokens, captions = X.load_scenes(ns)
    n = len(worlds)
    print(f"{n} paired scenes", flush=True)
    encode_all(a, out, tokens, captions)

    perm = np.random.default_rng(a.seed).permutation(n)
    te = np.sort(perm[:a.test])
    va = np.sort(perm[a.test:a.test + a.val])
    tr = np.sort(perm[a.test + a.val:])
    print(f"split: {len(tr)} train / {len(va)} val / {len(te)} test   "
          f"chance R@1 {1 / len(te):.4f}  MRR {chance(len(te))['MRR']:.4f}", flush=True)

    texts = [s.split("=", 1)[0] for s in a.text_checkpoint]
    images = [s.split("=", 1)[0] for s in a.image_checkpoint]
    pairs = [p.split("|", 1) for p in a.pair] or [(t, i) for t in texts for i in images]
    load = lambda lbl: dict(np.load(out / "features" / f"{lbl}.npz"))
    rdir = out / "results"
    rdir.mkdir(exist_ok=True)
    for t, i in pairs:
        dest = rdir / f"{t}__{i}.json"
        if dest.exists():
            print(f"{t} x {i}: exists, skipping", flush=True)
            continue
        ft = {k: v.astype(np.float64) for k, v in load(t).items()}
        fi = {k: v.astype(np.float64) for k, v in load(i).items()}
        res = score_pair(ft, fi, tr, va, te, a)
        res["protocol"] = {"text": t, "image": i, "scenes": n, "train": len(tr), "val": len(va),
                           "test": len(te), "chance": chance(len(te)), "seed": a.seed}
        dest.write_text(json.dumps(res, indent=2) + "\n")
        r = res["ridge"]
        print(f"{t} x {i}: ridge {r['text_readout']}|{r['image_readout']} "
              f"i->t R@1 {r['image_to_text']['R@1']:.3f} MRR {r['image_to_text']['MRR']:.3f} | "
              f"t->i R@1 {r['text_to_image']['R@1']:.3f} MRR {r['text_to_image']['MRR']:.3f} | "
              f"procrustes i->t MRR {res['procrustes']['image_to_text']['MRR']:.3f} | "
              f"shuffled MRR {res['shuffled_control']['image_to_text']['MRR']:.4f}", flush=True)


if __name__ == "__main__":
    main()
