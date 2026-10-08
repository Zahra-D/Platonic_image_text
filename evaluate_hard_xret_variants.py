"""Why does hard cross-modal retrieval fail when each model encodes binding?

Same scenes, split and hard tasks as evaluate_cross_modal_hard_retrieval.py
(true caption vs 7 same-word binding swaps; chance R@1 0.125, image -> text).
Five scorers, each with its readout pair (and hyper-parameter) chosen on the
val hard tasks and refitted on fit + val before the test tasks:

  ridge        the original map: ridge image -> text, cosine in text space
  whitened     both modalities PCA-whitened on the fit rows, then ridge and
               cosine in whitened text space (weak directions count equally)
  cca          regularised CCA, cosine in the top-k shared components
  map_probe    ridge image -> text, then a text fact probe applied to the
               mapped vector and to each candidate; cosine of centred logits
  to_facts     ridge from image features straight to the text probe's logits
               (the map only has to carry the fact directions)
  fact_match   image fact probe on the image vs text fact probe on each
               candidate (uses fact labels on both sides: an upper bound)

Facts are the binding-dependent conjunctions "some object has A=a and B=b"
(binding_swap_captions.conjunction_labels); probes are class-balanced
logistic regressions trained on the fit scenes only.
"""
from __future__ import annotations

from clevr_paths import CLEVR_DATA_ROOT, CLEVR_GEN_ROOT
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import binding_swap_captions as C
import evaluate_cross_modal_structure as X
from evaluate_binding_swap import fit_probe, label_matrix
from evaluate_cross_modal_hard_retrieval import ci, hard_scores
from evaluate_cross_modal_retrieval import ALPHAS, Ridge
from evaluate_probe_suite import load_model_or_random

TEXT_READOUTS = ("L5", "L6", "L7", "L5.mlp_out", "L6.mlp_out")
IMAGE_READOUTS = ("L5", "L6", "L7", "L6.mlp_out")
WHITEN_EPS = (1e-3, 1e-2, 1e-1)
CCA_K = (8, 16, 32, 64, 128)
CCA_REG = (1e-2, 1e-1)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", action="append", required=True, metavar="TEXT_LABEL=TEXT_PATH|IMAGE_LABEL")
    p.add_argument("--xret-dir", default="outputs/eval_all/xret")
    p.add_argument("--hard-dir", default="outputs/eval_hard_xret", help="holds tasks.json")
    p.add_argument("--image-manifest", default=CLEVR_DATA_ROOT + "/platonic_clevr_v1_5M_train_gpu_visible/val_image_only.jsonl")
    p.add_argument("--image-cache", default="outputs/image_only_2_5m_token_cache/val_tokens.pt")
    p.add_argument("--caption-generator", default=CLEVR_GEN_ROOT + "/generate_human_captions.py")
    p.add_argument("--num-scenes", type=int, default=8000)
    p.add_argument("--test", type=int, default=1000)
    p.add_argument("--val", type=int, default=1000)
    p.add_argument("--min-label-count", type=int, default=50)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def whitener(x, eps):
    mu = x.mean(0, keepdims=True)
    s, v = np.linalg.eigh(np.cov(x - mu, rowvar=False))
    w = v / np.sqrt(np.clip(s, 0, None) + eps * s.mean())
    return lambda z: (z - mu) @ w


def cca(x, y, reg):
    """Regularised CCA; returns projectors for image and text (all components, sorted)."""
    mx, my = x.mean(0, keepdims=True), y.mean(0, keepdims=True)
    xc, yc = x - mx, y - my
    n = len(x)
    def inv_sqrt(c):
        s, v = np.linalg.eigh(c + reg * np.trace(c) / len(c) * np.eye(len(c)))
        return v / np.sqrt(np.clip(s, 1e-12, None))
    wx, wy = inv_sqrt(xc.T @ xc / n), inv_sqrt(yc.T @ yc / n)
    u, _, vt = np.linalg.svd(wx.T @ (xc.T @ yc / n) @ wy)
    ax, ay = wx @ u, wy @ vt.T
    return (lambda z, k: (z - mx) @ ax[:, :k]), (lambda z, k: (z - my) @ ay[:, :k])


def main():
    a = arguments()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    ns = argparse.Namespace(num_scenes=a.num_scenes, seed=a.seed, caption_generator=a.caption_generator,
                            image_manifest=a.image_manifest, image_cache=a.image_cache)
    worlds, _, _ = X.load_scenes(ns)
    perm = np.random.default_rng(a.seed).permutation(len(worlds))
    te, va = np.sort(perm[:a.test]), np.sort(perm[a.test:a.test + a.val])
    tr = np.sort(perm[a.test + a.val:])
    trva = np.sort(np.concatenate([tr, va]))
    labels = C.all_conjunction_labels()
    facts = label_matrix(worlds, labels)
    keep = (facts[tr].sum(0) >= a.min_label_count) & ((1 - facts[tr]).sum(0) >= a.min_label_count)
    facts = facts[:, keep]
    built = json.load(open(Path(a.hard_dir) / "tasks.json"))
    tasks = {s: built[s]["tasks"] for s in ("val", "test")}
    scenes = {s: [t["scene"] for t in tasks[s]] for s in tasks}
    print(f"{int(keep.sum())} fact labels | {len(tasks['val'])} val / {len(tasks['test'])} test hard tasks", flush=True)
    xret = Path(a.xret_dir)
    dev = torch.device(a.device)

    for spec in a.pair:
        tpart, ilabel = spec.split("|", 1)
        tlabel, tpath = tpart.split("=", 1)
        dest = out / f"{tlabel}__{ilabel}.json"
        if dest.exists():
            print(f"{dest.name}: exists, skipping", flush=True); continue
        cache = out / f"candidates_{tlabel}.npz"
        if not cache.exists():
            model, tok, margs = load_model_or_random(tpath, a.device)
            enc = {s: X.encode_features(model, tok, margs, a.device, a.batch_size, captions=built[s]["texts"])
                   for s in ("val", "test")}
            np.savez(cache, **{f"{s}/{k}": enc[s][k].astype(np.float32) for s in enc for k in TEXT_READOUTS})
            del model; torch.cuda.empty_cache()
        cz = np.load(cache)
        cand = {s: {k: cz[f"{s}/{k}"].astype(np.float64) for k in TEXT_READOUTS} for s in ("val", "test")}
        ft = {k: v.astype(np.float64) for k, v in np.load(xret / "features" / f"{tlabel}.npz").items()}
        fi = {k: v.astype(np.float64) for k, v in np.load(xret / "features" / f"{ilabel}.npz").items()}

        def scorers(fit, split):
            """Yield (method, readout pair, hyper-parameter, R@1 array, RR array) on `split` tasks."""
            sc = scenes[split]
            probes_t = {rt: fit_probe(ft[rt][fit], facts[fit], dev) for rt in TEXT_READOUTS}
            probes_i = {ri: fit_probe(fi[ri][fit], facts[fit], dev) for ri in IMAGE_READOUTS}
            for ri in IMAGE_READOUTS:
                x_fit, x_q = fi[ri][fit], fi[ri][sc]
                mu_x = x_fit.mean(0, keepdims=True)
                ridge = Ridge(x_fit - mu_x)
                li = probes_i[ri](x_q); li_mu = probes_i[ri](x_fit).mean(0, keepdims=True)
                for rt in TEXT_READOUTS:
                    y_fit, c = ft[rt][fit], cand[split][rt]
                    mu_y = y_fit.mean(0, keepdims=True)
                    lt_fit = probes_t[rt](y_fit); lt_mu = lt_fit.mean(0, keepdims=True)
                    lc = probes_t[rt](c) - lt_mu
                    for al, w in ridge.maps(y_fit - mu_y, ALPHAS).items():
                        q = (x_q - mu_x) @ w
                        yield ("ridge", ri, rt, al, *hard_scores(q, c - mu_y, tasks[split]))
                        yield ("map_probe", ri, rt, al, *hard_scores(probes_t[rt](q + mu_y) - lt_mu, lc, tasks[split]))
                    for al, w in ridge.maps(lt_fit - lt_mu, ALPHAS).items():
                        yield ("to_facts", ri, rt, al, *hard_scores((x_q - mu_x) @ w, lc, tasks[split]))
                    yield ("fact_match", ri, rt, None, *hard_scores(li - li_mu, lc, tasks[split]))
                    for eps in WHITEN_EPS:
                        wx, wy = whitener(x_fit, eps), whitener(y_fit, eps)
                        xw = wx(x_fit)
                        wmap = Ridge(xw).maps(wy(y_fit), ALPHAS)
                        for al, w in wmap.items():
                            yield ("whitened", ri, rt, (eps, al), *hard_scores(wx(x_q) @ w, wy(c), tasks[split]))
                    for reg in CCA_REG:
                        px, py = cca(x_fit, y_fit, reg)
                        for k in CCA_K:
                            yield ("cca", ri, rt, (reg, k), *hard_scores(px(x_q, k), py(c, k), tasks[split]))

        best = {}
        for m, ri, rt, hp, r1, _ in scorers(tr, "val"):
            if r1.mean() > best.get(m, (-1,))[0]:
                best[m] = (float(r1.mean()), ri, rt, hp)
        res = {}
        test = {(m, ri, rt, str(hp)): (r1, rr) for m, ri, rt, hp, r1, rr in scorers(trva, "test")}
        for m, (v, ri, rt, hp) in best.items():
            r1, rr = test[(m, ri, rt, str(hp))]
            res[m] = {"image_readout": ri, "text_readout": rt, "hyper": hp, "val_R@1": v,
                      "R@1": ci(r1, a.bootstrap, a.seed), "MRR": ci(rr, a.bootstrap, a.seed + 1)}
        res["protocol"] = {"text": tlabel, "image": ilabel, "facts": int(keep.sum()), "chance_R@1": 0.125,
                           "test_tasks": len(tasks["test"]), "seed": a.seed}
        dest.write_text(json.dumps(res, indent=2, default=str) + "\n")
        print(f"{tlabel} x {ilabel}: " + " | ".join(
            f"{m} {res[m]['R@1']['value']:.3f}" for m in ("ridge", "whitened", "cca", "map_probe", "to_facts", "fact_match")), flush=True)


if __name__ == "__main__":
    main()
