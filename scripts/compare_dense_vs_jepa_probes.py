#!/usr/bin/env python3
"""Did the JEPA/data2vec objective help? Paired, budget-matched probe differences.

Each JEPA-style checkpoint is compared with the dense-diffusion checkpoint that
has the same initialization and has seen the same number of images:

* from-dense data2vec (init = dense epoch 3, 4.8M) at 6.0M / 7.2M images
  vs dense diffusion continued from the same epoch 3 to 6.0M / 7.2M;
* scratch data2vec at epoch k vs dense diffusion from scratch at epoch k.

Both models are probed on the same test items, so the difference is bootstrapped
over items (paired), which removes the between-item variance that makes the two
separate intervals overlap.  Accuracy tasks: difference in accuracy (points).
R^2 tasks: difference in mean R^2 with each output's total variance held fixed.
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/linear_probes_image")
FEATURE = sys.argv[2] if len(sys.argv) > 2 else "ijepa"  # or best_layer
TASKS = ["count", "dist", "color", "shape", "material", "size", "position",
         "obj_color", "obj_shape", "obj_material", "obj_size"]
DRAWS = 2000
D2V_FROM_DENSE = ["avg_all_randt", "avg_l4to7_randt", "layerwise_all_block2d",
                  "layerwise_all_randt", "layerwise_l4to7_randt"]
D2V_SCRATCH = ["block2d_30pct_avg_l4to7", "block2d_30pct_layerwise_l4to7",
               "block2d_65pct_avg_l4to7", "block2d_65pct_layerwise_l4to7"]


def pairs():
    for ep, dense in ((0, "dense_diffusion_1_2m_6e_continued__epoch_004"),
                      (1, "dense_diffusion_1_2m_6e_continued__epoch_005")):
        for v in D2V_FROM_DENSE:
            yield "from-dense", f"{4.8 + 1.2 * (ep + 1):.1f}M", dense, f"data2vec_from_dense_{v}_1_2m_2e__epoch_00{ep}"
    for ep in range(4):
        for v in D2V_SCRATCH:
            yield "scratch", f"{1.2 * (ep + 1):.1f}M", f"dense_diffusion_1_2m_4e__epoch_00{ep}", \
                f"data2vec_scratch_{v}_1_2m_4e__epoch_00{ep}"


def load(label):
    path = OUT / f"{label}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {**data["scene"], **data["object"]}


def entry_items(entry):
    head = entry[FEATURE]
    items = np.asarray(entry["items"][FEATURE], dtype=np.float64)
    if head["metric"] == "accuracy":
        return "accuracy", items, None, head["test"]
    items = items.reshape(len(items), -1)
    r2 = np.asarray(head["per_output_r2"])
    total = items.sum(0) / np.maximum(1 - r2, 1e-9)  # sum of squared deviations per output
    return "r2", items, total, head["test"]


def paired(a, b, rng):
    kind, ia, total, sa = entry_items(a)
    _, ib, _, sb = entry_items(b)
    n = len(ia)
    picks = rng.integers(0, n, (DRAWS, n))
    if kind == "accuracy":
        diff = ib - ia
        boot = diff[picks].mean(1)
    else:
        diff = (ia - ib) / (total / n)  # per-item R^2 gain per output
        boot = diff[picks].mean(1).mean(1)
    low, high = np.quantile(boot, (0.025, 0.975))
    return sb - sa, low, high, kind


def fmt(delta, low, high, kind):
    scale, unit = (100, "") if kind == "accuracy" else (1, "")
    star = "*" if low > 0 or high < 0 else " "
    return f"{scale * delta:+6.2f}{star}" if kind == "accuracy" else f"{delta:+.3f}{star}"


def main():
    rng = np.random.default_rng(0)
    rows, missing = [], []
    for family, budget, dense, jepa in pairs():
        a, b = load(dense), load(jepa)
        if a is None or b is None:
            missing.append(jepa if b is None else dense)
            continue
        if "items" not in a["count"] or "items" not in b["count"]:
            missing.append(f"{dense if 'items' not in a['count'] else jepa} (no per-item scores)")
            continue
        row = {"family": family, "budget": budget, "dense": dense, "jepa": jepa}
        for t in TASKS:
            delta, low, high, kind = paired(a[t], b[t], rng)
            row[t] = {"delta": delta, "ci95": [low, high], "kind": kind,
                      "dense": a[t][FEATURE]["test"], "jepa": b[t][FEATURE]["test"]}
        rows.append(row)
    (OUT / f"dense_vs_jepa_{FEATURE}.json").write_text(json.dumps(rows, indent=2) + "\n")

    print(f"JEPA minus dense ({FEATURE} feature); accuracy in points, R^2 in units; "
          f"* = paired 95% CI excludes 0")
    print(f"{'family':10} {'budget':6} {'jepa variant':34} " + " ".join(f"{t[:9]:>9}" for t in TASKS))
    for r in rows:
        name = r["jepa"].replace("data2vec_from_dense_", "").replace("data2vec_scratch_", "").split("_1_2m")[0]
        print(f"{r['family']:10} {r['budget']:6} {name[:34]:34} "
              + " ".join(f"{fmt(r[t]['delta'], *r[t]['ci95'], r[t]['kind']):>9}" for t in TASKS))
    dense_seen = {}
    for r in rows:
        dense_seen[r["dense"]] = r
    print("\ndense reference scores")
    for label, r in dense_seen.items():
        print(f"{label[:45]:45} " + " ".join(
            f"{(100 * r[t]['dense'] if r[t]['kind'] == 'accuracy' else r[t]['dense']):9.3f}" for t in TASKS))
    if missing:
        print("\nnot yet available:", ", ".join(sorted(set(missing))))


if __name__ == "__main__":
    main()
