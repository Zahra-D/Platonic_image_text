"""What drives each modality's pooled features, and what CCA keeps of it.

For the 8000 retrieval scenes (same rows and split as evaluate_cross_modal_retrieval)
nested descriptor sets are built from the full CLEVR scene files:
  content   object count; counts per colour / shape / material / size; colour x shape counts   (40)
  layout    size-weighted occupancy on a 4x6 image grid, mean / std of pixel x, y and depth,
            min and mean nearest-neighbour pixel distance, mean sin/cos of 4*rotation         (34)
  bound     where each attribute value sits: mean pixel x, y per colour / shape / material / size
            (0 when absent, plus a presence flag folded into content)                       (30)
  camera    sin / cos of the camera azimuth (from the scene's 'directions')                   (2)
  vq        the image's VQ-code histogram (512 codes; ridge handles the dimension)           (512)
  words     caption bag of words and token length (text side only)                          (vocab+1)
Held-out R^2 (ridge fit on the 6000 fit scenes, scored on the 2000 others) of the
whole feature matrix, variance-weighted.  Then CCA (reg 0.01, fitted on the 7000
fit + val pairs) is applied, and the same R^2 is reported for the image and text
projections onto the top-k canonical directions, together with the share of the
feature variance those k directions carry.

  python3 scripts/feature_variance_sources.py --pair TEXT_LABEL|IMAGE_LABEL [...] --readout L6.mlp_out --output OUT.json
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clevr_paths import CLEVR_DATA_ROOT, CLEVR_GEN_ROOT
import evaluate_cross_modal_structure as X
from evaluate_hard_xret_variants import cca

VALS = {"color": ["red", "yellow", "blue", "green", "gray", "brown", "purple", "cyan"],
        "shape": ["cube", "sphere", "cylinder"], "material": ["rubber", "metal"], "size": ["large", "small"]}
SHAPE_ALIAS = {"cylinder": "cylinder", "cube": "cube", "sphere": "sphere"}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", action="append", required=True)
    p.add_argument("--readout", default="L6.mlp_out")
    p.add_argument("--xret-dir", default="outputs/eval_all/xret")
    p.add_argument("--image-manifest", default=CLEVR_DATA_ROOT + "/platonic_clevr_v1_5M_train_gpu_visible/val_image_only.jsonl")
    p.add_argument("--image-cache", default="outputs/image_only_2_5m_token_cache/val_tokens.pt")
    p.add_argument("--caption-generator", default=CLEVR_GEN_ROOT + "/generate_human_captions.py")
    p.add_argument("--num-scenes", type=int, default=8000)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64, 128, 384])
    p.add_argument("--output", required=True)
    return p.parse_args()


def scene_file(manifest, image_path):
    root = Path(manifest).parent
    return root / image_path.replace("images/", "scenes/", 1).replace(".png", ".json")


def descriptors(a):
    rows = [json.loads(l) for _, l in zip(range(a.num_scenes), open(a.image_manifest))]
    content, layout, bound, camera = [], [], [], []
    for r in rows:
        sc = json.load(open(scene_file(a.image_manifest, r["image_path"])))
        objs = sc["objects"]
        c = [len(objs)] + [sum(o[k] == v for o in objs) for k in VALS for v in VALS[k]]
        c += [sum(o["color"] == col and o["shape"] == sh for o in objs) for col in VALS["color"] for sh in VALS["shape"]]
        content.append(c)
        px = np.array([o["pixel_coords"] for o in objs], float)          # x (0..96), y (0..64), depth
        occ = np.zeros((4, 6))
        for o, (x, y, _) in zip(objs, px):
            occ[min(3, int(y / 16)), min(5, int(x / 16))] += 2.0 if o["size"] == "large" else 1.0   # 96x64 render
        d = np.sqrt(((px[:, None, :2] - px[None, :, :2]) ** 2).sum(-1)) + np.eye(len(objs)) * 1e9
        nn = d.min(1)
        rot = np.array([o["rotation"] for o in objs]) * math.pi / 180
        layout.append(list(occ.ravel()) + list(px.mean(0)) + list(px.std(0)) + [nn.min(), nn.mean(),
                      np.sin(4 * rot).mean(), np.cos(4 * rot).mean()])
        b = []
        for k in VALS:
            for v in VALS[k]:
                sel = [i for i, o in enumerate(objs) if o[k] == v]
                b += list(px[sel, :2].mean(0)) if sel else [0.0, 0.0]
        bound.append(b)
        bh = sc["directions"]["behind"]
        az = math.atan2(bh[1], bh[0])
        camera.append([math.sin(az), math.cos(az)])
    cache = torch.load(a.image_cache, map_location="cpu", weights_only=False)["tokens"]
    toks = cache.numpy()[:len(rows)].astype(np.int64)
    vq = np.stack([np.bincount(t.ravel(), minlength=512) for t in toks]).astype(float)
    out = {k: np.asarray(v, float) for k, v in (("content", content), ("layout", layout), ("bound", bound), ("camera", camera))}
    out["vq"] = vq
    return out


def caption_words(a):
    ns = argparse.Namespace(num_scenes=a.num_scenes, seed=a.seed, caption_generator=a.caption_generator,
                            image_manifest=a.image_manifest, image_cache=a.image_cache)
    _, _, caps = X.load_scenes(ns)
    toks = [c.lower().replace(",", " ").replace(";", " ").replace(".", " ").split() for c in caps]
    vocab = sorted({w for t in toks for w in t})
    idx = {w: i for i, w in enumerate(vocab)}
    m = np.zeros((len(toks), len(vocab) + 1))
    for r, t in enumerate(toks):
        for w, n in Counter(t).items():
            m[r, idx[w]] = n
        m[r, -1] = len(t)
    return m


def zs(m, rows):
    mu, sd = m[rows].mean(0), m[rows].std(0)
    return (m - mu) / np.where(sd > 1e-9, sd, 1.0)


def heldout_r2(F, P, fit, test, alphas=(1e-2, 1e-1, 1, 10, 100)):
    """Variance-weighted held-out R^2 of ridge F ~ P (penalty chosen on an inner split of fit)."""
    F = F - F[fit].mean(0)
    P = zs(P, fit)
    inner = np.random.default_rng(0).permutation(fit)
    tr_i, va_i = inner[: int(0.8 * len(inner))], inner[int(0.8 * len(inner)):]
    def solve(rows, al):
        Pr = np.c_[P[rows], np.ones(len(rows))]
        g = Pr.T @ Pr; g[:-1, :-1] += al * np.trace(g[:-1, :-1]) / max(1, g.shape[0] - 1) * np.eye(g.shape[0] - 1)
        return np.linalg.solve(g, Pr.T @ F[rows])
    score = lambda rows, B: 1 - ((F[rows] - np.c_[P[rows], np.ones(len(rows))] @ B) ** 2).sum() / ((F[rows] - F[rows].mean(0)) ** 2).sum()
    al = max(alphas, key=lambda q: score(va_i, solve(tr_i, q)))
    return float(score(test, solve(fit, al)))


def main():
    a = arguments()
    perm = np.random.default_rng(a.seed).permutation(a.num_scenes)
    te_ret = perm[:1000]
    fitva = np.sort(perm[1000:])                     # CCA fit rows (7000), as everywhere
    rng = np.random.default_rng(1)
    split = rng.permutation(a.num_scenes)
    fit, test = np.sort(split[:6000]), np.sort(split[6000:])     # descriptor regressions
    D = descriptors(a)
    words = caption_words(a)
    sets_img = [("content", ["content"]), ("+layout", ["content", "layout"]), ("+bound", ["content", "layout", "bound"]),
                ("+camera", ["content", "layout", "bound", "camera"]), ("+vq", ["content", "layout", "bound", "camera", "vq"]),
                ("layout_only", ["layout", "bound", "camera"]), ("vq_only", ["vq"])]
    stack = lambda names, extra=None: np.concatenate([D[n] for n in names] + ([extra] if extra is not None else []), 1)
    res = {}
    for spec in a.pair:
        tl, il = spec.split("|", 1)
        ft = np.load(Path(a.xret_dir) / "features" / f"{tl}.npz")[a.readout].astype(float)
        fi = np.load(Path(a.xret_dir) / "features" / f"{il}.npz")[a.readout].astype(float)
        out = {"image": {n: heldout_r2(fi, stack(s), fit, test) for n, s in sets_img},
               "text": {"content": heldout_r2(ft, D["content"], fit, test),
                        "+layout": heldout_r2(ft, stack(["content", "layout", "bound", "camera"]), fit, test),
                        "+words": heldout_r2(ft, stack(["content"], words), fit, test),
                        "words_only": heldout_r2(ft, words, fit, test),
                        "vq_only": heldout_r2(ft, D["vq"], fit, test)}}
        # CCA subspaces
        px, py = cca(fi[fitva], ft[fitva], 0.01)
        cc = {}
        for k in a.ks:
            zi, zt = px(fi, k), py(ft, k)
            # share of feature variance carried by the k canonical coordinates (held-out linear reconstruction)
            share_i = heldout_r2(fi, zi, fit, test, alphas=(1e-6,))
            share_t = heldout_r2(ft, zt, fit, test, alphas=(1e-6,))
            cc[str(k)] = {"image_var_share": share_i, "text_var_share": share_t,
                          "image": {n: heldout_r2(zi, stack(s), fit, test) for n, s in sets_img if n in ("content", "+layout", "+vq", "layout_only", "vq_only")},
                          "text": {"content": heldout_r2(zt, D["content"], fit, test),
                                   "words_only": heldout_r2(zt, words, fit, test),
                                   "+words": heldout_r2(zt, stack(["content"], words), fit, test)}}
        out["cca"] = cc
        res[f"{tl}|{il}"] = out
        print(tl, il, a.readout, json.dumps({"image": {k: round(v, 3) for k, v in out["image"].items()},
                                              "text": {k: round(v, 3) for k, v in out["text"].items()}}), flush=True)
        for k, v in cc.items():
            print(f"  CCA k={k}: var share img {v['image_var_share']:.3f} txt {v['text_var_share']:.3f} | img " +
                  " ".join(f"{n} {x:.3f}" for n, x in v["image"].items()) + " | txt " +
                  " ".join(f"{n} {x:.3f}" for n, x in v["text"].items()), flush=True)
        Path(a.output).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
