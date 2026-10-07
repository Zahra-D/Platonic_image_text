"""One report for the whole representation-evaluation suite.

Reads whatever has finished under outputs/ and prints markdown tables for:
  1. layer-wise linear probes          (outputs/probe_suite/{image,text})
  2. semantic-vs-nuisance sensitivity  (outputs/semantic_sensitivity, ..._text)
  3. scene-graph retrieval             (outputs/scene_retrieval/{image,text})
  4. probe accuracy vs corruption      (same files as 1, every noise level)
  5. cross-modal linear-map retrieval  (outputs/cross_modal_retrieval/results)

"best" columns take the best readout over all layers and sublayers on the
evaluation split itself, so they are mildly optimistic; the readout is always
printed next to the number so it can be checked.
"""
from __future__ import annotations
import json
from pathlib import Path

O = Path("outputs")
CHANCE = {"count": 0.0, "color_count": 0.0, "shape_count": 0.0, "conj": 0.184, "conj_count": 0.0}
IMAGE_ORDER = ["random_init", "dense_ep1", "dense_ep12", "dense_ep16", "dense_ep25",
               "jepa_s15_t45", "jepa_s30_t60", "jepa_scratch"]
TEXT_ORDER = ["random_init", "dense_ep4", "dense_ep20", "dense_ep41",
              "jepa_from_ep4", "jepa_from_ep20", "jepa_scratch"]


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def ordered(folder, order):
    have = {p.stem: p for p in Path(folder).glob("*.json")}
    return [(k, have[k]) for k in order if k in have] + \
           [(k, v) for k, v in sorted(have.items()) if k not in order]


def short(name):
    return name.replace(".residual", "").replace(".mlp_out", ".mlp").replace(".attn_out", ".attn")


def best(layers, key, sub=None):
    vals = [(v[key] if sub is None else v[key][sub], k) for k, v in layers.items()
            if key in v and (sub is None or sub in v[key])]
    return max(vals) if vals else (None, None)


def probes(modality, order):
    rows = ordered(O / "probe_suite" / modality, order)
    if not rows:
        return f"_no {modality} probe results yet_\n"
    out = [f"| model | count R² | colour-count R² | shape-count R² | **conj mAP** (chance 0.184) | conj-count R² |",
           "|---|---|---|---|---|---|"]
    for label, path in rows:
        m = load(path)
        if not m or "t=0.0" not in m["metrics"]:
            continue
        L = m["metrics"]["t=0.0"]
        cells = []
        for task, sub in (("count", "r2"), ("color_count", "r2"), ("shape_count", "r2"),
                          ("conj", "mAP"), ("conj_count", "r2")):
            v, k = best(L, task, sub)
            cells.append(f"{v:.3f} @{short(k)}" if v is not None else "-")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def noise(modality, order):
    rows = ordered(O / "probe_suite" / modality, order)
    if not rows:
        return ""
    out = ["| model | conj mAP t=0 | t=0.25 | t=0.5 | t=0.75 | count R² t=0 | t=0.25 | t=0.5 | t=0.75 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for label, path in rows:
        m = load(path)
        if not m:
            continue
        mets = m["metrics"]
        conj = [best(mets[t], "conj", "mAP")[0] if t in mets else None
                for t in ("t=0.0", "t=0.25", "t=0.5", "t=0.75")]
        cnt = [best(mets[t], "count", "r2")[0] if t in mets else None
               for t in ("t=0.0", "t=0.25", "t=0.5", "t=0.75")]
        f = lambda x: f"{x:.3f}" if x is not None else "-"
        out.append(f"| {label} | " + " | ".join(map(f, conj)) + " | " + " | ".join(map(f, cnt)) + " |")
    return "\n".join(out) + "\n"


def sensitivity():
    parts = []
    rows = ordered(O / "semantic_sensitivity", IMAGE_ORDER)
    if rows:
        parts.append("**Image** — nuisance = camera/light change, semantic = colour swap at fixed camera\n")
        parts.append("| model | S at L7 | best S | semantic Δ | nuisance Δ | frac semantic > nuisance |")
        parts.append("|---|---|---|---|---|---|")
        for label, path in rows:
            m = load(path)
            if not m:
                continue
            M = m["metrics"]
            v, k = best(M, "S_ratio_of_means")
            b = M[k]
            l7 = M.get("L7", {}).get("S_ratio_of_means")
            parts.append(f"| {label} | {l7:.3f} | {v:.3f} @{short(k)} | {b['semantic_mean']:.5f} | "
                         f"{b['nuisance_mean']:.5f} | {b['fraction_semantic_larger']:.3f} |")
    rows = ordered(O / "semantic_sensitivity_text", TEXT_ORDER)
    if rows:
        parts.append("\n**Text** — nuisance = paraphrase, binding = word-identical swap, content = template twin\n")
        parts.append("| model | S_binding at L7 | best S_binding | S_content at L7 | binding Δ | nuisance Δ |")
        parts.append("|---|---|---|---|---|---|")
        for label, path in rows:
            m = load(path)
            if not m:
                continue
            M = m["metrics"]
            v, k = best(M, "S_binding")
            l7 = M.get("L7.residual", {})
            parts.append(f"| {label} | {l7.get('S_binding', float('nan')):.3f} | {v:.3f} @{short(k)} | "
                         f"{l7.get('S_content', float('nan')):.3f} | {M[k]['binding_mean']:.5f} | "
                         f"{M[k]['nuisance_mean']:.5f} |")
    return "\n".join(parts) + "\n" if parts else "_no sensitivity results yet_\n"


def retrieval(modality, order):
    rows = ordered(O / "scene_retrieval" / modality, order)
    if not rows:
        return f"_no {modality} retrieval results yet_\n"
    out = ["| model | ρ bag (binding-blind) | **ρ conj** (binding-aware) | p@10 bag | p@10 conj |",
           "|---|---|---|---|---|"]
    for label, path in rows:
        m = load(path)
        if not m:
            continue
        M = m["metrics"]
        vb, kb = best(M, "spearman_bag")
        vc, kc = best(M, "spearman_conj")
        pb, _ = best(M, "p@10_bag")
        pc, _ = best(M, "p@10_conj")
        out.append(f"| {label} | {vb:+.3f} @{short(kb)} | {vc:+.3f} @{short(kc)} | {pb:.3f} | {pc:.3f} |")
    return "\n".join(out) + "\n"


def cross_modal():
    d = O / "cross_modal_retrieval" / "results"
    files = sorted(d.glob("*.json")) if d.exists() else []
    if not files:
        return "_no cross-modal retrieval results yet_\n"
    cka = load(O / "cka_matched" / "cross_modal.json") or {}
    out = []
    first = load(files[0])
    ch = first["protocol"]["chance"]
    out.append(f"{first['protocol']['test']} test pairs, {first['protocol']['train']} train, "
               f"{first['protocol']['val']} val. Chance: R@1 {ch['R@1']:.3f}, MRR {ch['MRR']:.4f}.\n")
    out.append("| text × image | readouts (val-selected) | img→txt R@1 | R@10 | MRR [95% CI] | "
               "txt→img R@1 | MRR | Procrustes MRR | shuffled MRR | CKA (test, same cell) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for f in files:
        r = load(f)
        p, rg = r["protocol"], r["ridge"]
        it, ti = rg["image_to_text"], rg["text_to_image"]
        ci = it.get("MRR_ci95", [float("nan")] * 2)
        out.append(f"| {p['text']} × {p['image']} | {short(rg['text_readout'])} / {short(rg['image_readout'])} | "
                   f"{it['R@1']:.3f} | {it['R@10']:.3f} | {it['MRR']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
                   f"{ti['R@1']:.3f} | {ti['MRR']:.3f} | {r['procrustes']['image_to_text']['MRR']:.3f} | "
                   f"{r['shuffled_control']['image_to_text']['MRR']:.4f} | {rg['cka_test']:.3f} |")
    out.append("\n**Sample efficiency** (img→txt MRR at the selected cell, by number of training pairs)\n")
    sizes = sorted({int(n) for f in files for n in load(f)["sample_efficiency"]})
    out.append("| text × image | " + " | ".join(str(s) for s in sizes) + " |")
    out.append("|---|" + "---|" * len(sizes))
    for f in files:
        r = load(f)
        se = r["sample_efficiency"]
        cells = [f"{se[str(s)]['image_to_text']['MRR']:.3f}" if str(s) in se else "-" for s in sizes]
        out.append(f"| {r['protocol']['text']} × {r['protocol']['image']} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main():
    sections = [
        ("1. Layer-wise linear probes — image (t = 0)", probes("image", IMAGE_ORDER)),
        ("1. Layer-wise linear probes — text (t = 0)", probes("text", TEXT_ORDER)),
        ("2. Semantic vs nuisance sensitivity", sensitivity()),
        ("3. Scene-graph retrieval — image", retrieval("image", IMAGE_ORDER)),
        ("3. Scene-graph retrieval — text", retrieval("text", TEXT_ORDER)),
        ("4. Probe accuracy vs input corruption — image", noise("image", IMAGE_ORDER)),
        ("4. Probe accuracy vs input corruption — text", noise("text", TEXT_ORDER)),
        ("5. Cross-modal retrieval through a fitted linear map", cross_modal()),
    ]
    for title, body in sections:
        print(f"## {title}\n\n{body}")


if __name__ == "__main__":
    main()
