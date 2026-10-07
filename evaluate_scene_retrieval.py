"""Does representation geometry mirror scene-graph structure?

Two ground-truth scene descriptions are compared against the representation:

  bag  : counts of each colour / shape / size / material   (binding-blind --
         two scenes with swapped bindings are IDENTICAL under this)
  conj : counts of each (colour, shape) pair               (binding-aware)

Reporting both separates "the geometry tracks which ingredients are present"
from "the geometry tracks which attribute belongs to which object".  A model
that only does the former scores well on bag and poorly on conj.

Metrics: Spearman between representation distance and scene distance over
sampled pairs, and precision@k of representation neighbours against scene
neighbours.
"""
from __future__ import annotations
from clevr_paths import CLEVR_DATA_ROOT, CLEVR_GEN_ROOT
import argparse, json
from pathlib import Path
import numpy as np, torch
from evaluate_probe_suite import encode, COLORS, SHAPES, SIZES, MATS
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



def scene_vectors(worlds):
    bag, conj = [], []
    pairs = [(c, s) for c in COLORS for s in SHAPES]
    for w in worlds:
        o = w["objects"]
        bag.append([sum(x["color"] == v for x in o) for v in COLORS] +
                   [sum(x["shape"] == v for x in o) for v in SHAPES] +
                   [sum(x["size"] == v for x in o) for v in SIZES] +
                   [sum(x["material"] == v for x in o) for v in MATS])
        conj.append([sum(x["color"] == c and x["shape"] == s for x in o) for c, s in pairs])
    unit = lambda M: (lambda A: A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None))(np.asarray(M, float))
    return unit(bag), unit(conj)


def spearman(a, b):
    r = lambda v: np.argsort(np.argsort(v)).astype(np.float64)
    ra, rb = r(a), r(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def precision_at_k(R, S, k):
    """Fraction of each scene's k representation-neighbours that are also
    among its k scene-graph neighbours."""
    nn_r = np.argsort(-R, axis=1)[:, 1:k + 1]
    nn_s = np.argsort(-S, axis=1)[:, 1:k + 1]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(nn_r, nn_s)]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--modality", choices=["image", "text"], default="image")
    p.add_argument("--manifest", default=CLEVR_DATA_ROOT + "/platonic_clevr_v1_5M_train_gpu_visible/val_image_only.jsonl")
    p.add_argument("--image-cache", default="outputs/image_only_2_5m_token_cache/val_tokens.pt")
    p.add_argument("--caption-field", default="caption_human")
    p.add_argument("--num-scenes", type=int, default=2500)
    p.add_argument("--pairs", type=int, default=200000)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=20261005)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

    worlds = []
    if a.modality == "text":
        caps = []
        for line in open(a.manifest):
            if len(worlds) == a.num_scenes: break
            r = json.loads(line); worlds.append(r["world"]); caps.append(r[a.caption_field])
        items = caps
    else:
        cache = torch.load(a.image_cache, map_location="cpu", weights_only=False)["tokens"]
        rows = []
        for r, line in enumerate(open(a.manifest)):
            if r >= cache.shape[0] or len(worlds) == a.num_scenes: break
            worlds.append(json.loads(line)["world"]); rows.append(r)
        items = torch.from_numpy(cache.numpy()[rows].astype("int64"))
    n = len(worlds)
    BAG, CONJ = scene_vectors(worlds)
    Sbag, Sconj = BAG @ BAG.T, CONJ @ CONJ.T
    rng = np.random.default_rng(a.seed)
    i = rng.integers(0, n, a.pairs); j = rng.integers(0, n, a.pairs)
    keep = i != j; i, j = i[keep], j[keep]
    print(f"{n} scenes, {len(i)} sampled pairs", flush=True)

    for spec in a.checkpoint:
        label, path = spec.split("=", 1)
        dest = out / f"{label}.json"
        if dest.exists():
            print(f"{label}: exists, skipping", flush=True); continue
        model, tok, margs = load_model_or_random(path, a.device)
        feats = encode(model, tok, margs, items, a.batch_size, a.device, 0.0, a.seed,
                       modality=a.modality)
        rec = {}
        for name, X in feats.items():
            R = X @ X.T
            rec[name] = {
                "spearman_bag": spearman(R[i, j], Sbag[i, j]),
                "spearman_conj": spearman(R[i, j], Sconj[i, j]),
                f"p@{a.topk}_bag": precision_at_k(R, Sbag, a.topk),
                f"p@{a.topk}_conj": precision_at_k(R, Sconj, a.topk),
            }
        best = max(rec, key=lambda k: rec[k]["spearman_conj"])
        dest.write_text(json.dumps({"protocol": {"model": str(path), "scenes": n,
                                                 "modality": a.modality},
                                    "metrics": rec}, indent=2) + "\n")
        print(f"{label}: best conj rho {rec[best]['spearman_conj']:+.3f} @{best} "
              f"(bag {rec[best]['spearman_bag']:+.3f}, p@{a.topk} conj "
              f"{rec[best][f'p@{a.topk}_conj']:.3f})", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
