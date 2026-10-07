"""Collect every existing eval result for the inventory's eligible checkpoints.

Writes outputs/existing_evals_long.csv (run, epoch, modality, view, eval, readout,
metric, value, source) with every layer, and docs/EXISTING_EVALS.md with one row
per checkpoint (layer 7 and best layer). When one checkpoint was evaluated several
times by the same eval, the newest file wins. view = "trunk" for TRUNK: (shared
trunk only) evaluations, "full" otherwise.
"""
from __future__ import annotations
import csv, json, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
KINDS = [("structure", "structure"), ("cka", "structure"), ("cross_modal_structure", "structure"),
         ("probe_suite", "probe_suite"), ("scene_retrieval", "scene_retrieval"), ("binding", "binding"),
         ("semantic_sensitivity_text", "sens_text"), ("sens_text", "sens_text"),
         ("sens_image", "sens_image"), ("semantic_sensitivity", "sens_image"), ("triples", "triples"),
         ("jepa_pred", "jepa_prediction")]


def kind_of(path: Path):
    s = "/".join(path.relative_to(OUT).parts[:-1])
    return next((name for key, name in KINDS if key in s), None)


def rows_of(kind, data, path):
    """(modality, readout, metric, value) tuples for one result file."""
    m = data.get("metrics", {})
    if kind == "structure":
        mod = (data.get("protocol") or {}).get("modality") or ("text" if path.stem.endswith("_TEXT") else "image")
        for r, v in m.items():
            for k in ("probe_accuracy", "rsa_scene", "mean_pairwise_cosine", "effective_rank"):
                if k in v: yield mod, r, k, v[k]
    elif kind == "probe_suite":
        mod = "image" if "/image" in str(path) else "text"
        for t, readouts in m.items():
            for r, tasks in readouts.items():
                for task, v in tasks.items():
                    key = "mAP" if "mAP" in v else "r2"
                    yield mod, r, f"{task}_{key}@{t}", v[key]
    elif kind == "scene_retrieval":
        mod = "image" if "/image" in str(path) else "text"
        for r, v in m.items():
            for k, x in v.items(): yield mod, r, k, x
    elif kind == "binding":
        for r, v in (data.get("layers") or {"L7": data}).items():
            yield "image", r, "binding_accuracy", v["binding_accuracy"]["value"]
            if "binding_strict_accuracy" in v: yield "image", r, "binding_strict", v["binding_strict_accuracy"]["value"]
            yield "image", r, "binding_clean_probe", v["clean_probe_balanced_accuracy"]
    elif kind == "triples":
        for r, v in m.items():
            for g in ("d_binding", "d_binding_centred"):
                if g in v: yield "image", r, g, v[g]["d"]["value"]
    elif kind == "sens_image":
        for r, v in m.items():
            for g, vv in (("raw", v), ("centred", v.get("centred")), ("zscore", v.get("zscore"))):
                if vv:
                    for k in ("S_ratio_of_means", "semantic_mean", "nuisance_mean", "S_median_paired"):
                        yield "image", r, f"{k}_{g}", vv[k]
    elif kind == "sens_text":
        for r, v in m.items():
            for g, vv in (("raw", v), ("centred", v.get("centred")), ("zscore", v.get("zscore"))):
                if vv:
                    for k in ("S_binding", "S_content", "binding_mean", "content_mean", "nuisance_mean"):
                        yield "text", r, f"{k}_{g}", vv[k]
    elif kind == "jepa_prediction":
        for mod, layers in m.items():
            for r, v in layers.items():
                for g in ("raw", "centred"):
                    for k in ("C_plus", "C_minus", "delta"): yield mod, r, f"{k}_{g}", v[g][k]


def main():
    inv = json.load(open(OUT / "model_inventory.json"))
    meta = {r["run"]: r for r in inv if r.get("include")}
    newest = {}  # (run, epoch, kind, view, modality-hint, file-stem-kind) -> (mtime, path, data)
    for f in OUT.rglob("*.json"):
        kind = kind_of(f)
        if not kind or f.stat().st_size > 60_000_000:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        p = (d.get("protocol") or {}).get("model") if isinstance(d, dict) else None
        if not isinstance(p, str) or p.startswith("RANDOM"):
            continue
        view = "trunk" if p.startswith("TRUNK:") else "full"
        mm = re.search(r"outputs/([^/]+)/epoch_(\d+)\.pt", p)
        if not mm or mm.group(1) not in meta:
            continue
        sub = "image" if "/image" in str(f) else "text" if "/text" in str(f) else ""
        if kind == "structure":
            sub = "text" if f.stem.endswith("_TEXT") else "image" if f.stem.endswith("_IMAGE") else sub
        key = (mm.group(1), int(mm.group(2)), kind, view, sub)
        t = f.stat().st_mtime
        if key not in newest or t > newest[key][0]:
            newest[key] = (t, f, d)
    long_rows = []
    for (run, ep, kind, view, _), (_, f, d) in newest.items():
        for mod, readout, metric, value in rows_of(kind, d, f):
            if isinstance(value, (int, float)):
                long_rows.append([run, ep, mod, view, kind, readout, metric, value, str(f.relative_to(ROOT))])
    with open(OUT / "existing_evals_long.csv", "w", newline="") as h:
        w = csv.writer(h); w.writerow(["run", "epoch", "modality", "view", "eval", "readout", "metric", "value", "source"])
        w.writerows(sorted(long_rows))
    # per-checkpoint summary
    table = defaultdict(dict)
    for run, ep, mod, view, kind, r, metric, v, _ in long_rows:
        table[(run, ep, mod, view)][(kind, r, metric)] = v
    def L7(t, kind, metric): return t.get((kind, "L7", metric))
    def best(t, kind, metric):
        vals = [(v, r) for (k, r, m), v in t.items() if k == kind and m == metric and "@" not in r]
        return max(vals) if vals else None
    f3 = lambda x: "—" if x is None else f"{x:.3f}"
    fb = lambda b: "—" if b is None else f"{b[0]:.3f} ({b[1]})"
    lines = ["# Existing evaluation results (collected, not re-run)", "",
             "Collected by `scripts/collect_existing_evals.py` from earlier eval folders; newest result per checkpoint and eval. "
             "**Mixed script versions**: strict binding, centred/z-scored sensitivity and the JEPA prediction check exist only "
             "for recent runs. `view`: trunk = shared trunk only (private LoRA dropped), full = whole model. All layers are in "
             "`outputs/existing_evals_long.csv`.", ""]
    for mod in ("text", "image"):
        keys = sorted(k for k in table if k[2] == mod)
        lines += [f"## {mod} ({len(keys)} checkpoint views)", "",
                  "| run | epoch | view | probe L7 | probe best | RSA L7 | cos / rank L7 | conj mAP best | scene ρ_conj best | "
                  + ("binding L7 / best | triples d′ centred L7 | S L7 raw / centred |" if mod == "image"
                     else "S_bind L7 | S_cont L7 |"),
                  "|" + "---|" * (13 if mod == "image" else 11)]
        for k in keys:
            t = table[k]; run, ep, _, view = k
            cos, rank = L7(t, "structure", "mean_pairwise_cosine"), L7(t, "structure", "effective_rank")
            conj = best(t, "probe_suite", "conj_mAP@t=0.0")
            row = [f"`{run}`", str(ep), view, f3(L7(t, "structure", "probe_accuracy")),
                   fb(best(t, "structure", "probe_accuracy")), f3(L7(t, "structure", "rsa_scene")),
                   "—" if cos is None else f"{cos:.2f} / {rank:.0f}", fb(conj),
                   fb(best(t, "scene_retrieval", "spearman_conj"))]
            if mod == "image":
                b7, bb = L7(t, "binding", "binding_accuracy"), best(t, "binding", "binding_accuracy")
                row += ["—" if b7 is None else f"{b7:.3f} / {bb[0]:.3f}", f3(L7(t, "triples", "d_binding_centred")),
                        "—" if L7(t, "sens_image", "S_ratio_of_means_raw") is None else
                        f"{L7(t, 'sens_image', 'S_ratio_of_means_raw'):.3f} / {f3(L7(t, 'sens_image', 'S_ratio_of_means_centred'))}"]
            else:
                key = "L7.residual" if ("sens_text", "L7.residual", "S_binding_raw") in t else "L7"
                row += [f3(t.get(("sens_text", key, "S_binding_raw"))), f3(t.get(("sens_text", key, "S_content_raw")))]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    (ROOT / "docs" / "EXISTING_EVALS.md").write_text("\n".join(lines) + "\n")
    print(f"{len(long_rows)} metric values from {len(newest)} result files; {len(table)} checkpoint views")


if __name__ == "__main__":
    main()
