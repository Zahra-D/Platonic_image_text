"""Unpaired cross-modal alignment, layer by layer (Schnaus et al., arXiv:2610.09411).

"Shared Geometry As A Rosetta Stone": an orthogonal map between two embedding
sets is estimated with no paired data at all,
  1. coarse init: S times, k-means (C centres) on a random batch of each side,
     match the centres by the permutation maximising linear CKA (a quadratic
     assignment problem, solved here by FAQ with random restarts), and average
     the low-rank plans  M = 1/(SC) sum_s A_s^T P_s B_s ;
  2. read-out: one linear assignment on X M Y^T gives pseudo-pairs, W = polar(X^T T Y);
  3. Wasserstein Procrustes: R rounds of (Hungarian on a batch of X W Y^T,
     W = polar(X_B^T T Y_B)).
Inputs are centred and L2-normalised.

Fitting here is strictly unpaired: images of one half of the fit+val scenes,
captions of the *other* half (disjoint scenes; the pairing is never used).
Evaluation on the 1000 held-out test pairs of evaluate_cross_modal_retrieval:
  FOSCTTM (mean fraction of candidates closer than the true partner, both
  directions; 0 perfect, 0.5 random), image->text / text->image R@1 and MRR,
  and hard R@1 against 7 same-word binding swaps (where candidate features
  exist), all with the unpaired W.  References at the same readout:
  identity (no map), paired orthogonal Procrustes on all 7000 fit pairs (the
  ceiling of this map family), and linear CKA of the test pairs.

  python3 evaluate_unpaired_rosetta.py --pair TEXT_LABEL|IMAGE_LABEL [...] --output-dir DIR
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment, quadratic_assignment
from sklearn.cluster import KMeans

from evaluate_cross_modal_hard_retrieval import hard_scores

READOUTS = ["embedding"] + [f"L{i}" for i in range(8)] + [f"L{i}.mlp_out" for i in range(8)]


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", action="append", required=True, metavar="TEXT_LABEL|IMAGE_LABEL")
    p.add_argument("--xret-dir", default="outputs/eval_all/xret")
    p.add_argument("--candidates-dir", default="outputs/eval_hard_xret_variants")
    p.add_argument("--hard-tasks", default="outputs/eval_hard_xret/tasks.json")
    p.add_argument("--readouts", nargs="+", default=READOUTS)
    p.add_argument("--num-scenes", type=int, default=8000)
    p.add_argument("--test", type=int, default=1000)
    p.add_argument("--val", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=30, help="S")
    p.add_argument("--centres", type=int, default=30, help="C")
    p.add_argument("--qap-restarts", type=int, default=1000)
    p.add_argument("--rounds", type=int, default=100, help="R")
    p.add_argument("--batch", type=int, default=1500)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def prep(z, mu):
    z = z - mu
    return z / np.linalg.norm(z, axis=1, keepdims=True).clip(1e-12)


def polar(n):
    u, _, vt = np.linalg.svd(n, full_matrices=False)
    return u @ vt


def lin_cka(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return float(np.linalg.norm(a.T @ b) ** 2 / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)))


def two_opt(ka, kb, p):
    """Pairwise swaps while tr(ka P kb P^T) improves."""
    p = p.copy()
    cur = np.sum(ka * kb[np.ix_(p, p)])
    improved = True
    while improved:
        improved = False
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                q = p.copy(); q[i], q[j] = q[j], q[i]
                v = np.sum(ka * kb[np.ix_(q, q)])
                if v > cur + 1e-12:
                    p, cur, improved = q, v, True
    return p


def match_centres(a, b, rng, restarts, polish=10):
    """Permutation of b's rows maximising linear CKA with a: FAQ from many random
    starts, the best `polish` solutions refined by 2-opt (stands in for MPOpt)."""
    ka = (a - a.mean(0)) @ (a - a.mean(0)).T
    kb = (b - b.mean(0)) @ (b - b.mean(0)).T
    sols = []
    for _ in range(restarts):
        res = quadratic_assignment(ka, kb, method="faq",
                                   options={"maximize": True, "P0": "randomized", "rng": int(rng.integers(2**31))})
        sols.append((res.fun, res.col_ind))
    sols.sort(key=lambda t: -t[0])
    best = (-np.inf, None)
    for _, p in sols[:polish]:
        q = two_opt(ka, kb, p)
        v = np.sum(ka * kb[np.ix_(q, q)])
        if v > best[0]:
            best = (v, q)
    return best[1]


def fit_unpaired(x, y, a, rng, xp=None, yp=None):
    """Orthogonal W with x W ~ y, from unpaired x (images) and y (captions).

    Optional known pairs (xp[i] <-> yp[i]) enter as a linear term in every
    Procrustes step, weighted as in the paper so that their total contribution
    is comparable to the pseudo-pairs: min(n, m) / p at initialisation and
    b / p in refinement (the coarse plan gets them with weight 1 / p).
    """
    known = None if xp is None or len(xp) == 0 else xp.T @ yp
    d = x.shape[1]
    m = np.zeros((d, y.shape[1]))
    for s in range(a.repeats):
        bx = x[rng.choice(len(x), min(len(x), 10_000), replace=False)]
        by = y[rng.choice(len(y), min(len(y), 10_000), replace=False)]
        ca = KMeans(a.centres, n_init=1, random_state=int(rng.integers(2**31))).fit(bx).cluster_centers_
        cb = KMeans(a.centres, n_init=1, random_state=int(rng.integers(2**31))).fit(by).cluster_centers_
        perm = match_centres(ca, cb, rng, a.qap_restarts)
        m += ca.T @ cb[perm]
    m /= a.repeats * a.centres
    if known is not None:
        m = m + known / len(xp)
    n = min(len(x), len(y))
    r, c = linear_sum_assignment(-(x[:n] @ m @ y[:n].T))
    w = polar(x[:n][r].T @ y[:n][c] + (0 if known is None else known * n / len(xp)))
    for _ in range(a.rounds):
        ix = rng.choice(len(x), a.batch, replace=False)
        iy = rng.choice(len(y), a.batch, replace=False)
        r, c = linear_sum_assignment(-(x[ix] @ w @ y[iy].T))
        w = polar(x[ix][r].T @ y[iy][c] + (0 if known is None else known * a.batch / len(xp)))
    return w


def retrieval(q, c):
    """q[i] should match c[i]: FOSCTTM, R@1, MRR in both directions (cosine)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-12)
    c = c / np.linalg.norm(c, axis=1, keepdims=True).clip(1e-12)
    s = q @ c.T
    d = np.diag(s)
    fwd = (s > d[:, None]).sum(1)          # captions closer than the true one, per image
    rev = (s > d[None, :]).sum(0)          # images closer than the true one, per caption
    n = len(d)
    out = {"FOSCTTM": float(0.5 * (fwd.mean() + rev.mean()) / (n - 1))}
    for name, rk in (("image_to_text", fwd + 1), ("text_to_image", rev + 1)):
        out[name] = {"R@1": float((rk == 1).mean()), "MRR": float((1.0 / rk).mean())}
    return out


def main():
    a = arguments()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    perm = np.random.default_rng(a.seed).permutation(a.num_scenes)
    te = np.sort(perm[:a.test])
    fit = perm[a.test:]                                   # fit + val scenes, never paired below
    half = len(fit) // 2
    img_rows, txt_rows = np.sort(fit[:half]), np.sort(fit[half:])   # disjoint scenes
    trva = np.sort(fit)
    built = json.load(open(a.hard_tasks))["test"]
    tasks = built["tasks"]
    task_scenes = [t["scene"] for t in tasks]
    for spec in a.pair:
        tl, il = spec.split("|", 1)
        dest = out / f"{tl}__{il}.json"
        if dest.exists():
            print(f"{dest.name}: exists, skipping", flush=True); continue
        ft = dict(np.load(Path(a.xret_dir) / "features" / f"{tl}.npz"))
        fi = dict(np.load(Path(a.xret_dir) / "features" / f"{il}.npz"))
        cpath = Path(a.candidates_dir) / f"candidates_{tl}.npz"
        cand = dict(np.load(cpath)) if cpath.exists() else {}
        res = {}
        for r in a.readouts:
            if r not in ft or r not in fi:
                continue
            rng = np.random.default_rng(a.seed + 1)
            X, Y = fi[r].astype(np.float64), ft[r].astype(np.float64)
            mx, my = X[img_rows].mean(0), Y[txt_rows].mean(0)
            x_fit, y_fit = prep(X[img_rows], mx), prep(Y[txt_rows], my)
            x_te, y_te = prep(X[te], mx), prep(Y[te], my)
            w = fit_unpaired(x_fit, y_fit, a, rng)
            # references at the same readout
            mxp, myp = X[trva].mean(0), Y[trva].mean(0)
            w_pair = polar(prep(X[trva], mxp).T @ prep(Y[trva], myp))
            row = {"unpaired": retrieval(x_te @ w, y_te),
                   "paired_procrustes": retrieval(prep(X[te], mxp) @ w_pair, prep(Y[te], myp)),
                   "identity": retrieval(x_te, y_te) if X.shape[1] == Y.shape[1] else None,
                   "cka_test": lin_cka(X[te], Y[te]),
                   "w_agreement": float(np.trace(w.T @ w_pair) / w.shape[1])}   # 1 = same map as paired
            key = f"test/{r}"
            if key in cand:
                c = prep(cand[key].astype(np.float64), my)
                q = prep(X[task_scenes], mx) @ w
                r1, rr = hard_scores(q, c, tasks)
                row["unpaired_hard_R@1"] = float(r1.mean())
                cp = prep(cand[key].astype(np.float64), myp)
                r1p, _ = hard_scores(prep(X[task_scenes], mxp) @ w_pair, cp, tasks)
                row["paired_procrustes_hard_R@1"] = float(r1p.mean())
            res[r] = row
            u, p, idm = row["unpaired"], row["paired_procrustes"], row["identity"]
            ident = "" if idm is None else f", identity {idm['FOSCTTM']:.3f}"
            print(f"{tl} x {il} [{r}] FOSCTTM unpaired {u['FOSCTTM']:.3f} (paired {p['FOSCTTM']:.3f}{ident}) "
                  f"i2t R@1 {u['image_to_text']['R@1']:.3f} t2i R@1 {u['text_to_image']['R@1']:.3f} "
                  f"hard {row.get('unpaired_hard_R@1', float('nan')):.3f} cka {row['cka_test']:.3f} "
                  f"agree {row['w_agreement']:.2f}", flush=True)
        res["protocol"] = {"text": tl, "image": il, "fit_images": len(img_rows), "fit_captions": len(txt_rows),
                           "disjoint_scenes": True, "test_pairs": len(te), "S": a.repeats, "C": a.centres,
                           "R": a.rounds, "batch": a.batch, "seed": a.seed, "paper": "arXiv:2610.09411"}
        dest.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
