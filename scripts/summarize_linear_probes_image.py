#!/usr/bin/env python3
"""Table of frozen linear-probe results with training budget and baselines."""
import csv
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/linear_probes_image")
BUDGET = [Path("outputs/image_training_budget.json"), Path("outputs/image_training_budget_epochs.json")]
SCENE = ["count", "dist", "color", "shape", "material", "size", "position"]
OBJECT = ["obj_color", "obj_shape", "obj_material", "obj_size"]

budget = {}
for path in BUDGET:
    if path.exists():
        for row in json.loads(path.read_text()):
            budget[row["checkpoint"]] = row


def head(entry, which):
    if which in entry:
        return entry[which]
    return next(iter(entry["layers"].values()))


rows = []
construction = json.loads((OUT / "construction.json").read_text())
rows.append({"label": "chance (majority class / train mean)", "group": "baseline",
             **{t: construction["chance"][t]["test"] for t in SCENE + OBJECT}})
for path in sorted(OUT.glob("*.json")):
    if path.name == "construction.json" or path.name.startswith("dense_vs_jepa"):
        continue
    result = json.loads(path.read_text())
    protocol = result["protocol"]
    b = budget.get(protocol.get("model", ""), {})
    group = ("easy" if "baseline" in protocol else "untrained" if protocol.get("random_init")
             else "trained_100k" if "100k" in b.get("train_manifest", "100k" if "__last" in path.stem else "")
             else "trained_1.2M")
    row = {"label": path.stem, "group": group,
           "images_seen_M": round(b["cumulative_images"] / 1e6, 2) if b else "",
           "image_tokens_B": round(b["cumulative_image_tokens"] / 1e9, 3) if b else ""}
    for level, tasks in (("scene", SCENE), ("object", OBJECT)):
        for t in tasks:
            if t in result.get(level, {}):
                e = result[level][t]
                row[t] = head(e, "ijepa")["test"]
                row[f"{t}_feat"] = e.get("ijepa_feature", "")
                row[f"{t}_ci"] = head(e, "ijepa")["ci95"]
                if "best_layer" in e:
                    row[f"{t}_bestlayer"] = e["best_layer"]["test"]
                    row[f"{t}_bestlayer_feat"] = e["best_layer_feature"]
    rows.append(row)

fields = ["group", "label", "images_seen_M", "image_tokens_B"] + SCENE + OBJECT
extra = sorted({k for r in rows for k in r} - set(fields))
with open(OUT / "summary.csv", "w", newline="") as handle:
    writer = csv.DictWriter(handle, fields + extra)
    writer.writeheader()
    writer.writerows(rows)
order = {"baseline": 0, "easy": 1, "untrained": 2, "trained_1.2M": 3, "trained_100k": 4}
rows.sort(key=lambda r: (order[r["group"]], r["label"]))


def cell(r, t):
    v = r.get(t, "")
    if v == "":
        return "  -  "
    return f"{100 * v:5.1f}" if t in ("count", "dist") or t.startswith("obj_") else f"{v:5.3f}"


print(f"{'group':13} {'label':62} {'img(M)':>6} " + " ".join(f"{t[:8]:>8}" for t in SCENE + OBJECT))
for r in rows:
    print(f"{r['group']:13} {r['label'][:62]:62} {str(r.get('images_seen_M', '')):>6} "
          + " ".join(f"{cell(r, t):>8}" for t in SCENE + OBJECT))
