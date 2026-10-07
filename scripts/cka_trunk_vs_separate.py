"""Text<->image CKA: shared trunk (same model, private LoRA ignored) vs two separately
trained dense models, from the pooled features cached by evaluate_cross_modal_retrieval.

Per readout: linear CKA (centred, 8000 scenes), debiased CKA (unbiased HSIC, 3000
scenes) and a permutation null (scene correspondence shuffled, 50 draws).
"""
import glob, os, sys
import numpy as np

FEATS = {}
for f in glob.glob("outputs/eval_*/xret/features/*.npz"):
    FEATS.setdefault(os.path.basename(f)[:-4], f)
READOUTS = ["embedding"] + [f"L{i}" for i in range(8)]
rng = np.random.default_rng(0)
SUB = rng.permutation(8000)[:3000]

def center(x): return x - x.mean(0, keepdims=True)
def lin_cka(x, y):
    x, y = center(x), center(y)
    return float(np.linalg.norm(y.T @ x) ** 2 / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))
def unbiased_hsic(k, l):
    n = k.shape[0]; k = k.copy(); l = l.copy(); np.fill_diagonal(k, 0); np.fill_diagonal(l, 0)
    one = np.ones(n)
    return (np.sum(k * l) + (one @ k @ one) * (one @ l @ one) / ((n - 1) * (n - 2))
            - 2 * (one @ k @ l @ one) / (n - 2)) / (n * (n - 3))
def debiased_cka(x, y):
    k, l = x @ x.T, y @ y.T
    return float(unbiased_hsic(k, l) / np.sqrt(unbiased_hsic(k, k) * unbiased_hsic(l, l)))

def pair(text, image, perms=50):
    a, b = np.load(FEATS[f"{text}_TEXT"]), np.load(FEATS[f"{image}_IMAGE"])
    out = {}
    for r in READOUTS:
        x, y = a[r].astype(np.float64), b[r].astype(np.float64)
        null = [lin_cka(x, y[rng.permutation(len(y))]) for _ in range(perms)]
        out[r] = dict(cka=lin_cka(x, y), dcka=debiased_cka(x[SUB], y[SUB]),
                      null_mean=float(np.mean(null)), null_p95=float(np.percentile(null, 95)))
    return out

PAIRS = [  # (name, text model, image model, kind)
    ("random × random (separate)", "random", "random", "separate"),
    ("dense 2 ep (separate)", "denseT2", "denseI2", "separate"),
    ("dense 4 ep (separate)", "denseT4", "denseI4", "separate"),
    ("dense 7 ep (separate)", "denseT7", "denseI7", "separate"),
    ("trunk 1 ep (diffusion)", "mmT1", "mmT1", "trunk"),
    ("trunk 2 ep (diffusion)", "mmT2", "mmT2", "trunk"),
    ("trunk 4 ep (diffusion)", "mmT4", "mmT4", "trunk"),
    ("trunk 6 ep (diffusion, JEPA start)", "start6", "start6", "trunk"),
    ("A: +1 ep trunk I-JEPA", "A1", "A1", "trunk"),
    ("A: +2 ep trunk I-JEPA", "A2", "A2", "trunk"),
    ("B: +1 ep I-JEPA + private diff", "B1", "B1", "trunk"),
    ("C: +1 ep I-JEPA + SIGReg", "C1", "C1", "trunk"),
    ("C: +2 ep I-JEPA + SIGReg", "C2", "C2", "trunk"),
]
extra = sys.argv[1:]  # more trunk labels, e.g. A3 A4 B2 B3 C3 C4 trunk10 denseT10:denseI10
for e in extra:
    t, i = e.split(":") if ":" in e else (e, e)
    PAIRS.append((f"{e}", t, i, "separate" if ":" in e else "trunk"))
print("| pair | kind | L7 CKA | L7 debiased | L7 null mean / p95 | best CKA (layer) | mean CKA L0–L7 | mean debiased L0–L7 |")
print("|---|---|---|---|---|---|---|---|")
for name, t, i, kind in PAIRS:
    if f"{t}_TEXT" not in FEATS or f"{i}_IMAGE" not in FEATS: continue
    r = pair(t, i)
    deep = [r[f"L{k}"] for k in range(8)]
    b = max(READOUTS[1:], key=lambda k: r[k]["cka"])
    print(f"| {name} | {kind} | {r['L7']['cka']:.3f} | {r['L7']['dcka']:.3f} | {r['L7']['null_mean']:.3f} / {r['L7']['null_p95']:.3f} | "
          f"{r[b]['cka']:.3f} ({b}) | {np.mean([d['cka'] for d in deep]):.3f} | {np.mean([d['dcka'] for d in deep]):.3f} |", flush=True)
