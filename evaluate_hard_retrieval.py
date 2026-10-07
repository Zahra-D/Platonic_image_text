#!/usr/bin/env python3
"""Hard retrieval: find the true scene among lexically identical binding swaps.

For every held-out scene the query is one caption, and the candidate set is
the same scene plus K different scenes made by exchanging one binding (two
objects swap a color, shape, material, or size; or a relation's two objects
swap roles).  All candidates share one phrasing plan that is independent of
the query's (different synonyms, phrase styles, and relation wording), so
they contain exactly the same tokens.  Each candidate, and the query, gets its
own random object order, relation order, and sentence pattern, so no
candidate is systematically closer to the query in word position either.
Word counts and token-embedding averages therefore score exactly chance, and
an untrained model is checked to score near chance.

Two tasks: ``attribute`` (K attribute-swap negatives, chance R@1 = 1/(K+1))
and ``relation`` (every distinct relation-role swap of the scene).  Ties are
scored by their expectation under random tie-breaking, so any feature that
cannot tell the candidates apart scores exactly chance.  The token-multiset
equality of every candidate set is verified, not assumed.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import binding_swap_captions as C
from evaluate_decomposed_representations import encode
from evaluate_shared_private_retrieval import load_model
from train_multimodal import build_model
from analyze_paired_representations import checkpoint_args


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH")
    p.add_argument("--untrained-from", default=None, metavar="PATH",
                   help="Also score a randomly initialized model with this checkpoint's architecture.")
    p.add_argument("--manifest", default="/home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_1m/val_text_only_human.jsonl")
    p.add_argument("--caption-generator", default="/home/zd25e122/clevr-dataset-gen_clone/image_generation/generate_human_captions.py")
    p.add_argument("--num-worlds", type=int, default=1000)
    p.add_argument("--negatives", type=int, default=7)
    p.add_argument("--max-tokens", type=int, default=190)
    p.add_argument("--sublayer-layers", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--delete-checkpoint-when-done", action="store_true",
        help="Remove each scored checkpoint afterwards; for the trainer's periodic probe, "
             "whose slim checkpoints are disposable.",
    )
    return p.parse_args()


def attribute_swaps(world):
    """Every single-attribute exchange that keeps the tokens and changes the scene."""
    objects = world["objects"]
    mentions = Counter()
    for r in world.get("relations", []):
        mentions[r["subject"]] += 1
        mentions[r["anchor"]] += 1
    out = []
    for kind in C.ATTRIBUTES:
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                a, b = objects[i], objects[j]
                if a[kind] == b[kind] or mentions[i] != mentions[j]:
                    continue
                if not any(a[o] != b[o] for o in C.ATTRIBUTES if o != kind):
                    continue
                w = copy.deepcopy(world)
                w["objects"][i][kind], w["objects"][j][kind] = b[kind], a[kind]
                if not C.has_twins(w):
                    out.append(w)
    return out


def relation_swaps(world):
    out = []
    for k in range(len(world.get("relations", []))):
        w = copy.deepcopy(world)
        r = w["relations"][k]
        r["subject"], r["anchor"] = r["anchor"], r["subject"]
        out.append(w)
    return out


def build(args, tokenizer):
    gen = C.load_generator(args.caption_generator)
    rows = []
    with open(args.manifest) as h:
        for line in h:
            w = json.loads(line)["world"]
            if not C.has_twins(w) and len(w["objects"]) >= 3:
                rows.append(w)
            if len(rows) == args.num_worlds:
                break
    texts, index = [], {}

    def add(t):
        if t not in index:
            index[t] = len(texts)
            texts.append(t)
        return index[t]

    bag = lambda t: Counter(tokenizer.tokenize(t))
    tasks, stats = [], Counter()
    for wi, world in enumerate(rows):
        rng = random.Random(args.seed * 1_000_003 + wi)
        n, m = len(world["objects"]), len(world.get("relations", []))
        plan = C.make_plan(gen, rng.randrange(2**31), n, m)
        other = C.make_plan(gen, rng.randrange(2**31), n, m)
        perm = lambda k: rng.sample(range(k), k)
        query = C.render(gen, world, plan, perm(n), perm(m), rng.randrange(2))
        sig = C.scene_signature(world)
        for task, pool in (("attribute", attribute_swaps(world)), ("relation", relation_swaps(world))):
            seen, negs = {sig}, []
            rng.shuffle(pool)
            for w in pool:
                s = C.scene_signature(w)
                if s not in seen:
                    seen.add(s)
                    negs.append(w)
            if task == "attribute":
                if len(negs) < args.negatives:
                    stats["attribute_too_few_swaps"] += 1
                    continue
                negs = negs[:args.negatives]
            elif not negs:
                stats["relation_no_relations"] += 1
                continue
            for cond in ("reworded",):
                cands = [C.render(gen, w, other, perm(n), perm(m), rng.randrange(2)) for w in [world] + negs]
                bags = [bag(t) for t in cands]
                if any(b != bags[0] for b in bags) or len(set(cands)) != len(cands):
                    stats[f"{task}_{cond}_not_lexically_matched"] += 1
                    continue
                if max(len(tokenizer.tokenize(t)) for t in [query] + cands) > args.max_tokens:
                    stats[f"{task}_{cond}_too_long"] += 1
                    continue
                tasks.append({"world": wi, "task": task, "condition": cond, "query": add(query),
                              "candidates": [add(t) for t in cands]})
    return texts, tasks, stats


def rank_metrics(features, tasks):
    """Expected R@1 and reciprocal rank of the true candidate (index 0), ties at random."""
    r1 = np.empty(len(tasks)); rr = np.empty(len(tasks)); chance = np.empty(len(tasks))
    for k, t in enumerate(tasks):
        s = features[t["candidates"]].astype(np.float64) @ features[t["query"]].astype(np.float64)
        tie = np.abs(s - s[0]) <= 1e-6  # float noise is a tie; includes the positive itself
        better, equal = ((s > s[0]) & ~tie).sum(), tie.sum()
        ranks = better + np.arange(1, equal + 1)
        r1[k] = (ranks == 1).mean()
        rr[k] = (1.0 / ranks).mean()
        chance[k] = 1.0 / len(s)
    return r1, rr, chance


def summarize(values, worlds, reps, seed):
    by = {w: np.flatnonzero(worlds == w) for w in np.unique(worlds)}
    keys = list(by)
    rng = np.random.default_rng(seed)
    draws = [values[np.concatenate([by[w] for w in rng.choice(keys, len(keys))])].mean() for _ in range(reps)]
    return {"value": float(values.mean()), "ci95": [float(x) for x in np.quantile(draws, (0.025, 0.975))]}


def score(features, tasks, reps, seed):
    out = {}
    for task in ("attribute", "relation"):
        for cond in ("reworded",):
            sel = [t for t in tasks if t["task"] == task and t["condition"] == cond]
            if not sel:
                continue
            r1, rr, ch = rank_metrics(features, sel)
            w = np.asarray([t["world"] for t in sel])
            out[f"{task}|{cond}"] = {"n": len(sel), "R@1": summarize(r1, w, reps, seed), "MRR": summarize(rr, w, reps, seed),
                                     "chance_R@1": float(ch.mean()), "mean_candidates": float(np.mean([len(t["candidates"]) for t in sel]))}
    return out


def main():
    args = arguments()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    specs = [s.split("=", 1) for s in args.checkpoint]
    first = specs[0][1] if specs else args.untrained_from
    _, tokenizer, _ = load_model(first, torch.device("cpu"))
    texts, tasks, stats = build(args, tokenizer)
    (out / "texts.jsonl").write_text("".join(json.dumps({"i": i, "caption": t}) + "\n" for i, t in enumerate(texts)))
    (out / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    (out / "construction.json").write_text(json.dumps({"texts": len(texts), "tasks": Counter(f"{t['task']}|{t['condition']}" for t in tasks),
                                                       "dropped": stats, "negatives": args.negatives, "seed": args.seed}, indent=2) + "\n")
    print("tasks:", dict(Counter(f"{t['task']}|{t['condition']}" for t in tasks)), "dropped:", dict(stats), flush=True)

    # model-free controls
    ctrl = out / "controls.json"
    if not ctrl.exists():
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.preprocessing import normalize
        wc = normalize(CountVectorizer(token_pattern=r"[a-z]+").fit_transform(texts)).toarray()
        bg = normalize(CountVectorizer(token_pattern=r"[a-z]+", ngram_range=(2, 2)).fit_transform(texts)).toarray()
        rnd = np.random.default_rng(args.seed).normal(size=(len(texts), 384)); rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
        ctrl.write_text(json.dumps({"word_counts": score(wc, tasks, args.bootstrap, args.seed),
                                    "word_bigrams": score(bg, tasks, args.bootstrap, args.seed),
                                    "random_features": score(rnd, tasks, args.bootstrap, args.seed)}, indent=2) + "\n")
    runs = list(specs)
    if args.untrained_from:
        runs.append(("untrained", None))
    for label, path in runs:
        dest = out / f"{label}.json"
        if dest.exists():
            print("skip", label); continue
        if path is None:
            payload = torch.load(args.untrained_from, map_location="cpu", weights_only=False)
            margs = checkpoint_args(payload)
            torch.manual_seed(args.seed)
            model, _ = build_model(margs, len(tokenizer))
            model.to(device).eval()
            source = f"random initialization with the architecture of {args.untrained_from}"
        else:
            model, tokenizer, margs = load_model(path, device)
            source = str(Path(path).resolve())
        feats, worst = encode(model, tokenizer, margs, texts, args.batch_size, device, set(args.sublayer_layers), check=True)
        if worst > 1e-3:
            raise RuntimeError(f"{label}: decomposition error {worst:.2e}")
        metrics = {name: score(v, tasks, args.bootstrap, args.seed) for name, v in feats.items()}
        dest.write_text(json.dumps({"protocol": {"model": source, "train_mode": margs.train_mode}, "metrics": metrics}, indent=2) + "\n")
        g = lambda f, k: f"{100*metrics[f][k]['R@1']['value']:.1f}" if f in metrics else "-"
        print(f"{label}: R@1 attribute / relation | residual L4 {g('L4.residual','attribute|reworded')} / {g('L4.residual','relation|reworded')}"
              f" | residual L7 {g('L7.residual','attribute|reworded')} / {g('L7.residual','relation|reworded')}", flush=True)
        del model; torch.cuda.empty_cache()
        if args.delete_checkpoint_when_done and args.untrained_from is None:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
