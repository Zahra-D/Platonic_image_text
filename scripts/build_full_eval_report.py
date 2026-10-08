"""Full evaluation report from outputs/eval_all (run_full_eval.py).

Writes outputs/eval_all/all_metrics_long.csv (every model, eval, layer, metric) and
docs/FULL_EVAL.md: retrieval (R@1 / MRR) and binding first, then per-modality
summaries at layer 7 and the best layer, then the JEPA prediction check.
"""
from __future__ import annotations
import csv, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from collect_existing_evals import rows_of  # noqa: E402

Q = ROOT / "outputs" / "eval_all"
inv = {r["run"]: r for r in json.load(open(ROOT / "outputs" / "model_inventory.json")) if r.get("include")}
DIRS = {"structure": "structure", "probe_suite/text": "probe_suite", "probe_suite/image": "probe_suite",
        "scene_retrieval/text": "scene_retrieval", "scene_retrieval/image": "scene_retrieval", "binding": "binding",
        "sens_text": "sens_text", "sens_image": "sens_image", "triples": "triples", "jepa_prediction": "jepa_prediction"}


def model_of(stem):
    for suffix in ("_TEXT", "_IMAGE"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


long = []
for sub, kind in DIRS.items():
    for f in sorted((Q / sub).glob("*.json")):
        if f.name == "construction.json":
            continue
        d = json.load(open(f))
        for mod, readout, metric, value in rows_of(kind, d, f):
            if isinstance(value, (int, float)):
                long.append([model_of(f.stem), mod, kind, readout, metric, value])
# cross-modal retrieval
xret = {}
for f in sorted((Q / "xret" / "results").glob("*.json")):
    d = json.load(open(f)); t, i = f.stem.split("__", 1)[0], f.stem
    pair = f.stem.replace("_TEXT__", "|").removesuffix("_IMAGE")
    r = d["ridge"]
    xret[pair] = {"cell": f"{r['text_readout']}|{r['image_readout']}",
                  "i2t_R1": r["image_to_text"]["R@1"], "i2t_MRR": r["image_to_text"]["MRR"],
                  "t2i_R1": r["text_to_image"]["R@1"], "t2i_MRR": r["text_to_image"]["MRR"],
                  "procrustes_i2t_MRR": d.get("procrustes", {}).get("image_to_text", {}).get("MRR"),
                  "cka_test": r.get("cka_test")}
    idm = d.get("identity_map") or {}
    if idm:
        best = max(idm, key=lambda L: idm[L]["modality_centred"]["image_to_text"]["MRR"])
        xret[pair]["identity_i2t_MRR"] = idm[best]["modality_centred"]["image_to_text"]["MRR"]
        xret[pair]["identity_layer"] = best
    for layer, v in (d.get("diagonal") or {}).items():
        for direction in ("image_to_text", "text_to_image"):
            if direction in v:
                for k in ("R@1", "MRR"):
                    long.append([pair, "cross", "xret_diagonal", layer, f"{direction}_{k}", v[direction][k]])
        if "cka_test" in v:
            long.append([pair, "cross", "xret_diagonal", layer, "cka_test", v["cka_test"]])
    for k, v in xret[pair].items():
        if isinstance(v, (int, float)):
            long.append([pair, "cross", "xret", "best_cell", k, v])
with open(Q / "all_metrics_long.csv", "w", newline="") as h:
    w = csv.writer(h); w.writerow(["model", "modality", "eval", "readout", "metric", "value"]); w.writerows(long)

T = defaultdict(dict)
for m, mod, kind, r, metric, v in long:
    T[(m, mod)][(kind, r, metric)] = v
L7 = lambda t, k, m: t.get((k, "L7", m))
def best(t, k, m, skip_at=True):
    vals = [(v, r) for (kk, r, mm), v in t.items() if kk == k and mm == m and not (skip_at and "@" in r)]
    return max(vals) if vals else None
f3 = lambda x: "—" if x is None else f"{x:.3f}"
fp = lambda x: "—" if x is None else f"{100 * x:.1f}"
fb = lambda b, pct=False: "—" if b is None else (f"{100 * b[0]:.1f} ({b[1]})" if pct else f"{b[0]:.3f} ({b[1]})")
family = lambda m: inv.get(m.split("__e")[0], {}).get("objective", "") if "__e" in m else "baseline"


def epoch(m):
    return int(m.split("__e")[1]) + 1 if "__e" in m else None


lines = ["# Full evaluation (last checkpoint of every eligible run, all layers)", "",
         "Built by `scripts/build_full_eval_report.py` from `outputs/eval_all` (`scripts/run_full_eval.py`, Oct 7–8 2026). "
         "Shared-trunk (`dense_private`) models are scored **trunk only**. Every value at every layer is in "
         "`outputs/eval_all/all_metrics_long.csv`; metric definitions in `EVALUATIONS.md`. "
         "`ep` = epochs of the run's own last checkpoint (`epoch_NNN` + 1).", ""]
# 1. retrieval
lines += ["## 1. Cross-modal retrieval (text × image of the same model; dense = two separately trained models)", "",
          "Ridge map chosen on val, scored on 1000 test scenes. Chance R@1 0.001, MRR 0.0075. Identity = no map (trunk models).", "",
          "| pair | best cell | i→t R@1 | i→t MRR | t→i R@1 | t→i MRR | Procrustes i→t MRR | identity i→t MRR |",
          "|---|---|---|---|---|---|---|---|"]
for pair, v in sorted(xret.items(), key=lambda kv: -kv[1]["t2i_MRR"]):
    lines.append(f"| `{pair}` | {v['cell']} | {v['i2t_R1']:.3f} | {v['i2t_MRR']:.3f} | {v['t2i_R1']:.3f} | {v['t2i_MRR']:.3f} | "
                 f"{f3(v.get('procrustes_i2t_MRR'))} | {f3(v.get('identity_i2t_MRR'))} |")
# 2. binding
img = sorted(k for k in T if k[1] == "image")
lines += ["", "## 2. Image binding (397 content-matched pairs; pairwise chance 50%, strict = both scenes classified)", "",
          "| model | ep | objective | binding L7 | binding best (layer) | strict L7 | strict best | clean probe L7 |", "|---|---|---|---|---|---|---|---|"]
rows = []
for k in img:
    t = T[k]
    b = best(t, "binding", "binding_accuracy")
    if b is None:
        continue
    rows.append((b[0], f"| `{k[0].split('__e')[0]}` | {epoch(k[0]) or ''} | {family(k[0])} | {fp(L7(t, 'binding', 'binding_accuracy'))} | "
                       f"{fb(b, True)} | {fp(L7(t, 'binding', 'binding_strict'))} | {fb(best(t, 'binding', 'binding_strict'), True)} | "
                       f"{fp(L7(t, 'binding', 'binding_clean_probe'))} |"))
lines += [r for _, r in sorted(rows, reverse=True)]
# 3/4. per-modality summaries
for mod in ("image", "text"):
    keys = sorted(k for k in T if k[1] == mod)
    lines += ["", f"## {3 if mod == 'image' else 4}. {mod.title()} summary (L7 and best layer)", ""]
    head = ["model", "ep", "probe L7", "probe best", "RSA L7", "cos / rank L7", "conj mAP best", "conj mAP t=.75",
            "scene ρ_conj best"] + (["triples d′ centred L7", "S L7 raw / centred / z (d_s / d_n centred)"] if mod == "image"
                                     else ["S_bind L7 raw / centred / z", "S_cont L7 raw / centred / z"])
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for k in keys:
        t = T[k]; m = k[0]
        cos, rank = L7(t, "structure", "mean_pairwise_cosine"), L7(t, "structure", "effective_rank")
        row = [f"`{m.split('__e')[0]}`", str(epoch(m) or ""), fp(L7(t, "structure", "probe_accuracy")),
               fb(best(t, "structure", "probe_accuracy"), True), f3(L7(t, "structure", "rsa_scene")),
               "—" if cos is None else f"{cos:.2f} / {rank:.0f}", fb(best(t, "probe_suite", "conj_mAP@t=0.0")),
               f3((best(t, "probe_suite", "conj_mAP@t=0.75") or (None,))[0]), fb(best(t, "scene_retrieval", "spearman_conj"))]
        if mod == "image":
            g = lambda geo: L7(t, "sens_image", f"S_ratio_of_means_{geo}")
            row += [f3(L7(t, "triples", "d_binding_centred")),
                    f"{f3(g('raw'))} / {f3(g('centred'))} / {f3(g('zscore'))} "
                    f"({f3(L7(t, 'sens_image', 'semantic_mean_centred'))} / {f3(L7(t, 'sens_image', 'nuisance_mean_centred'))})"]
        else:
            key = "L7.residual" if ("sens_text", "L7.residual", "S_binding_raw") in t else "L7"
            s = lambda m_, geo: t.get(("sens_text", key, f"{m_}_{geo}"))
            row += [" / ".join(f3(s("S_binding", g)) for g in ("raw", "centred", "zscore")),
                    " / ".join(f3(s("S_content", g)) for g in ("raw", "centred", "zscore"))]
        lines.append("| " + " | ".join(row) + " |")
# 5. JEPA prediction
lines += ["", "## 5. JEPA prediction check (mean over supervised layers)", "",
          "| model | modality | raw C₊ / C₋ / Δ | centred C₊ / C₋ / Δ |", "|---|---|---|---|"]
for k in sorted(T):
    t = T[k]
    vals = {mm: [v for (kk, r, m2), v in t.items() if kk == "jepa_prediction" and m2 == mm]
            for mm in ("C_plus_raw", "C_minus_raw", "delta_raw", "C_plus_centred", "C_minus_centred", "delta_centred")}
    if not vals["delta_raw"]:
        continue
    a = {mm: sum(v) / len(v) for mm, v in vals.items()}
    lines.append(f"| `{k[0]}` | {k[1]} | {a['C_plus_raw']:.3f} / {a['C_minus_raw']:.3f} / {a['delta_raw']:.3f} | "
                 f"{a['C_plus_centred']:.3f} / {a['C_minus_centred']:.3f} / **{a['delta_centred']:.3f}** |")
(ROOT / "docs" / "FULL_EVAL.md").write_text("\n".join(lines) + "\n")
print(f"{len(long)} values | {len(T)} model views | {len(xret)} retrieval pairs -> docs/FULL_EVAL.md")
