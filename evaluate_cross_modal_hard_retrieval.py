"""Hard cross-modal retrieval: an image picks its caption among binding swaps.

The linear map is fitted exactly as in evaluate_cross_modal_retrieval (same
8000 scenes, same seeded fit / val / test split, per-modality centring on the
fit rows, ridge from image to text features).  The candidates change: for
every val / test scene, its own caption is scored against K captions of
*other* scenes made by exchanging one attribute between two of its objects
(a red cube and a blue sphere become a blue cube and a red sphere).  All
K + 1 candidates are rendered with one shared phrasing plan, each with its
own random object order, so they contain exactly the same word multiset
(verified, not assumed): an inventory of colours / shapes / materials / sizes
cannot separate them, only which attribute goes with which object can.
Chance R@1 = 1 / (K + 1); ties count at random.

Only image -> text: swapped images would need rendering.  Captions carry no
relations (the manifests store world["relationships"], which the caption
generator does not read), as in the easy retrieval.

Cells: the easy retrieval's chosen (text readout | image readout, alpha), read
from its result json, and a cell chosen on the val hard tasks.

  python3 evaluate_cross_modal_hard_retrieval.py --xret-dir outputs/eval_all/xret \
     --pair denseT_e040_TEXT=PATH|denseI_e040_IMAGE --output-dir outputs/eval_hard_xret
"""
from __future__ import annotations

from clevr_paths import CLEVR_DATA_ROOT, CLEVR_GEN_ROOT
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import binding_swap_captions as C
import evaluate_cross_modal_structure as X
from evaluate_cross_modal_retrieval import ALPHAS, READOUTS, Ridge, centred
from evaluate_hard_retrieval import attribute_swaps
from evaluate_probe_suite import load_model_or_random


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", action="append", required=True, metavar="TEXT_LABEL=TEXT_PATH|IMAGE_LABEL",
                   help="text model to encode the candidates; image features are read from the xret cache")
    p.add_argument("--xret-dir", default="outputs/eval_all/xret")
    p.add_argument("--image-manifest", default=CLEVR_DATA_ROOT + "/platonic_clevr_v1_5M_train_gpu_visible/val_image_only.jsonl")
    p.add_argument("--image-cache", default="outputs/image_only_2_5m_token_cache/val_tokens.pt")
    p.add_argument("--caption-generator", default=CLEVR_GEN_ROOT + "/generate_human_captions.py")
    p.add_argument("--num-scenes", type=int, default=8000)
    p.add_argument("--test", type=int, default=1000)
    p.add_argument("--val", type=int, default=1000)
    p.add_argument("--negatives", type=int, default=7)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def build_tasks(a, worlds, rows, tokenizer):
    """Per scene: K+1 lexically identical captions, the true one first."""
    gen = C.load_generator(a.caption_generator)
    texts, tasks, dropped = [], [], Counter()
    for r in rows:
        world = worlds[r]
        if C.has_twins(world) or len(world["objects"]) < 3:
            dropped["twins_or_<3_objects"] += 1
            continue
        rng = random.Random((a.seed + 31) * 1_000_003 + int(r))
        pool, seen, negs = attribute_swaps(world), {C.scene_signature(world)}, []
        rng.shuffle(pool)
        for w in pool:
            s = C.scene_signature(w)
            if s not in seen:
                seen.add(s); negs.append(w)
        if len(negs) < a.negatives:
            dropped["too_few_swaps"] += 1
            continue
        n = len(world["objects"])
        plan = C.make_plan(gen, rng.randrange(2**31), n, 0)
        cands = [C.render(gen, w, plan, rng.sample(range(n), n), [], rng.randrange(2))
                 for w in [world] + negs[:a.negatives]]
        bags = [Counter(tokenizer.tokenize(t)) for t in cands]
        if any(b != bags[0] for b in bags) or len(set(cands)) != len(cands):
            dropped["not_lexically_matched"] += 1
            continue
        tasks.append({"scene": int(r), "candidates": list(range(len(texts), len(texts) + len(cands)))})
        texts += cands
    return texts, tasks, dropped


def hard_scores(query, cand_feats, tasks):
    """query [S,D] mapped image features (one per task), cand_feats [N,D] centred captions."""
    q = query / np.linalg.norm(query, axis=1, keepdims=True).clip(1e-12)
    c = cand_feats / np.linalg.norm(cand_feats, axis=1, keepdims=True).clip(1e-12)
    r1, rr = np.empty(len(tasks)), np.empty(len(tasks))
    for k, t in enumerate(tasks):
        s = c[t["candidates"]] @ q[k]
        tie = np.abs(s - s[0]) <= 1e-6
        ranks = ((s > s[0]) & ~tie).sum() + np.arange(1, tie.sum() + 1)
        r1[k], rr[k] = (ranks == 1).mean(), (1.0 / ranks).mean()
    return r1, rr


def ci(values, reps, seed):
    rng = np.random.default_rng(seed)
    draws = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(reps)]
    return {"value": float(values.mean()), "ci95": [float(x) for x in np.quantile(draws, (0.025, 0.975))]}


def main():
    a = arguments()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    ns = argparse.Namespace(num_scenes=a.num_scenes, seed=a.seed, caption_generator=a.caption_generator,
                            image_manifest=a.image_manifest, image_cache=a.image_cache)
    worlds, _, _ = X.load_scenes(ns)
    n = len(worlds)
    perm = np.random.default_rng(a.seed).permutation(n)          # the easy retrieval's split
    te, va = np.sort(perm[:a.test]), np.sort(perm[a.test:a.test + a.val])
    trva = np.sort(np.concatenate([perm[a.test + a.val:], va]))
    tr = np.sort(perm[a.test + a.val:])
    xret = Path(a.xret_dir)
    tokenizer = None
    for spec in a.pair:
        tpart, ilabel = spec.split("|", 1)
        tlabel, tpath = tpart.split("=", 1)
        dest = out / f"{tlabel}__{ilabel}.json"
        if dest.exists():
            print(f"{dest.name}: exists, skipping", flush=True); continue
        model, tokenizer, margs = load_model_or_random(tpath, a.device)
        if not (out / "tasks.json").exists():
            built = {split: build_tasks(a, worlds, rows, tokenizer) for split, rows in (("val", va), ("test", te))}
            (out / "tasks.json").write_text(json.dumps({s: {"texts": t, "tasks": k, "dropped": d}
                                                         for s, (t, k, d) in built.items()}) + "\n")
        built = json.load(open(out / "tasks.json"))
        cand = {}
        for split in ("val", "test"):
            cand[split] = X.encode_features(model, tokenizer, margs, a.device, a.batch_size, captions=built[split]["texts"])
        del model; torch.cuda.empty_cache()
        ft = {k: v.astype(np.float64) for k, v in np.load(xret / "features" / f"{tlabel}.npz").items()}
        fi = {k: v.astype(np.float64) for k, v in np.load(xret / "features" / f"{ilabel}.npz").items()}
        tv, tt = built["val"]["tasks"], built["test"]["tasks"]
        print(f"{tlabel} x {ilabel}: {len(tv)} val / {len(tt)} test hard tasks, "
              f"{a.negatives + 1} candidates (chance R@1 {1 / (a.negatives + 1):.3f})", flush=True)

        def run(rt, ri, al, fit, tasks, split):
            x_fit = fi[ri][fit]; mu_x = x_fit.mean(0, keepdims=True)
            y_fit = ft[rt][fit]; mu_y = y_fit.mean(0, keepdims=True)
            w = Ridge(x_fit - mu_x).maps(y_fit - mu_y, (al,))[al]
            q = (fi[ri][[t["scene"] for t in tasks]] - mu_x) @ w
            return hard_scores(q, cand[split][rt].astype(np.float64) - mu_y, tasks)

        # 1. cell chosen on val hard tasks (map fitted on the 6000 fit scenes)
        grid, best = {}, (-1.0, None)
        names_t = [k for k in READOUTS if k in ft and k in cand["val"]]
        for ri in [k for k in READOUTS if k in fi]:
            x_fit = fi[ri][tr]
            mu_x = x_fit.mean(0, keepdims=True); ridge = Ridge(x_fit - mu_x)
            q_scenes = fi[ri][[t["scene"] for t in tv]] - mu_x
            for rt in names_t:
                y_fit = ft[rt][tr]; mu_y = y_fit.mean(0, keepdims=True)
                for al, w in ridge.maps(y_fit - mu_y, ALPHAS).items():
                    r1, _ = hard_scores(q_scenes @ w, cand["val"][rt].astype(np.float64) - mu_y, tv)
                    v = float(r1.mean())
                    if v > grid.get(f"{rt}|{ri}", {"val_R@1": -1})["val_R@1"]:
                        grid[f"{rt}|{ri}"] = {"alpha": al, "val_R@1": v}
                    if v > best[0]:
                        best = (v, (rt, ri, al))
        res = {"grid": grid}
        rt, ri, al = best[1]
        r1, rr = run(rt, ri, al, trva, tt, "test")
        res["hard_selected"] = {"text_readout": rt, "image_readout": ri, "alpha": al,
                                "R@1": ci(r1, a.bootstrap, a.seed), "MRR": ci(rr, a.bootstrap, a.seed + 1)}
        # 2. the easy retrieval's chosen cell
        easy = json.load(open(xret / "results" / f"{tlabel}__{ilabel}.json"))["ridge"]
        r1e, rre = run(easy["text_readout"], easy["image_readout"], easy["alpha"], trva, tt, "test")
        res["easy_cell"] = {"text_readout": easy["text_readout"], "image_readout": easy["image_readout"],
                            "alpha": easy["alpha"], "R@1": ci(r1e, a.bootstrap, a.seed + 2),
                            "MRR": ci(rre, a.bootstrap, a.seed + 3),
                            "easy_i2t_R@1": easy["image_to_text"]["R@1"]}
        # 3. shuffled-pairs control at the selected cell
        x_fit = fi[ri][trva]; mu_x = x_fit.mean(0, keepdims=True)
        y_fit = ft[rt][trva]; mu_y = y_fit.mean(0, keepdims=True)
        y_sh = (y_fit - mu_y)[np.random.default_rng(a.seed + 7).permutation(len(y_fit))]
        w = Ridge(x_fit - mu_x).maps(y_sh, (al,))[al]
        r1s, _ = hard_scores((fi[ri][[t["scene"] for t in tt]] - mu_x) @ w, cand["test"][rt].astype(np.float64) - mu_y, tt)
        res["shuffled_control_R@1"] = float(r1s.mean())
        res["protocol"] = {"text": tlabel, "text_path": tpath, "image": ilabel, "test_tasks": len(tt), "val_tasks": len(tv),
                           "candidates": a.negatives + 1, "chance_R@1": 1 / (a.negatives + 1), "seed": a.seed,
                           "dropped_test": built["test"]["dropped"]}
        dest.write_text(json.dumps(res, indent=2) + "\n")
        h, e = res["hard_selected"], res["easy_cell"]
        print(f"  hard-selected {rt}|{ri} a={al}: R@1 {h['R@1']['value']:.3f} {h['R@1']['ci95']} MRR {h['MRR']['value']:.3f} | "
              f"easy cell {e['text_readout']}|{e['image_readout']}: R@1 {e['R@1']['value']:.3f} | shuffled {r1s.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
