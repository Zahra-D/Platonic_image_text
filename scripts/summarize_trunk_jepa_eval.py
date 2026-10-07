"""Tables for outputs/eval_trunk_jepa: trunk-only JEPA variants vs start trunk and dense."""
import json, os, re, sys
# usage: summarize_trunk_jepa_eval.py [--dense N] LABEL ...  (reads every outputs/eval_trunk_jepa* dir)
import glob
DIRS = sorted(glob.glob("outputs/eval_trunk_jepa*"))
args = sys.argv[1:]
DE = "7"
if args[:1] == ["--dense"]: DE, args = args[1], args[2:]
LABS = args or ["denseT7", "start6", "A1", "A2", "B1", "C1", "C2"]
def find(rel):
    for d in DIRS:
        if os.path.exists(f"{d}/{rel}"): return f"{d}/{rel}"
    return f"{DIRS[0]}/{rel}"
Q = None
def J(p):
    p = find(p)
    if not os.path.exists(p): return None
    d = json.load(open(p)); return d.get("metrics", d)
def best(m, f): k = max(m, key=lambda r: f(m[r])); return f(m[k])
def cell(v, fmt="{:.3f}"): return "—" if v is None else fmt.format(v)
def table(title, rows, labs):
    print(f"\n### {title}\n\n| metric | " + " | ".join(labs) + " |\n|" + "---|" * (len(labs) + 1))
    for name, fn in rows:
        vals = []
        for l in labs:
            try: vals.append(fn(l))
            except Exception: vals.append("—")
        print(f"| {name} | " + " | ".join(vals) + " |")
for mod in ("TEXT", "IMAGE"):
    md = mod.lower(); dl = f"denseT{DE}" if mod == "TEXT" else f"denseI{DE}"
    labs = [dl if l.startswith("denseT") else l for l in LABS]
    st = lambda l: J(f"structure/{l}_{mod}.json")
    ps = lambda l: json.load(open(find(f"probe_suite/{md}/{l}.json")))["metrics"]
    pb = lambda l, t, task, key: max(ps(l)[f"t={t}"][r][task][key] for r in ps(l)[f"t={t}"])
    rows = [
        ("structure probe L7 (best)", lambda l: f"{100*st(l)['L7']['probe_accuracy']:.1f} ({100*best(st(l), lambda v: v['probe_accuracy']):.1f})"),
        ("rsa_scene L7", lambda l: cell(st(l)['L7']['rsa_scene'])),
        ("mean cos / eff. rank L7", lambda l: f"{st(l)['L7']['mean_pairwise_cosine']:.2f} / {st(l)['L7']['effective_rank']:.0f}"),
        ("conj mAP best (t=0)", lambda l: cell(pb(l, 0.0, 'conj', 'mAP'))),
        ("conj mAP best (t=0.75)", lambda l: cell(pb(l, 0.75, 'conj', 'mAP'))),
        ("conj-count R² best", lambda l: cell(pb(l, 0.0, 'conj_count', 'r2'))),
        ("count R² best", lambda l: cell(pb(l, 0.0, 'count', 'r2'))),
        ("scene ρ_conj best", lambda l: cell(best(json.load(open(find(f"scene_retrieval/{md}/{l}.json")))["metrics"], lambda v: v["spearman_conj"]))),
    ]
    if mod == "TEXT":
        se = lambda l: J(f"sens_text/{l}.json")
        k = lambda m: "L7.residual" if "L7.residual" in m else "L7"
        rows += [("S_cont L7", lambda l: cell(se(l)[k(se(l))]["S_content"])),
                 ("S_bind L7 (best)", lambda l: f"{se(l)[k(se(l))]['S_binding']:.3f} ({best(se(l), lambda v: v['S_binding']):.3f})")]
    else:
        bd = lambda l: json.load(open(find(f"binding/{l}.json")))["layers"]
        tr = lambda l: J(f"triples/{l}.json")
        si = lambda l: J(f"sens_image/{l}.json")
        rows += [("binding % L7 (best)", lambda l: f"{100*bd(l)['L7']['binding_accuracy']['value']:.1f} ({100*best(bd(l), lambda v: v['binding_accuracy']['value']):.1f})"),
                 ("triples centred d′ L7 (best)", lambda l: f"{tr(l)['L7']['d_binding_centred']['d']['value']:.3f} ({best(tr(l), lambda v: v['d_binding_centred']['d']['value']):.3f})"),
                 ("image S L7 / median", lambda l: f"{si(l)['L7']['S_ratio_of_means']:.3f} / {si(l)['L7']['S_median_paired']:.3f}")]
    table(mod, rows, labs)
c, xr = {}, {}
for d in DIRS:
    if os.path.exists(f"{d}/structure/cross_modal.json"): c.update(json.load(open(f"{d}/structure/cross_modal.json")))
for f in sorted(glob.glob("outputs/eval_trunk_jepa*/xret/results/*.json")):
    r = json.load(open(f)); rd = r["ridge"]; f = os.path.basename(f)
    t, i = f[:-5].split("__")
    xr[f"{t}|{i}"] = (f"{rd['image_to_text']['R@1']:.3f}", f"{rd['image_to_text']['MRR']:.3f}",
                      f"{rd['text_to_image']['R@1']:.3f}", f"{rd['text_to_image']['MRR']:.3f}")
print("\n### CROSS-MODAL (text × image of the same model; dense = two separate models)\n")
print("| pair | CKA L7 | best CKA | rsa_cross L7 | ridge R@1 i→t / t→i | ridge MRR i→t / t→i |\n|---|---|---|---|---|---|")
for l in LABS:
    k = f"{l}_TEXT|denseI{l[6:]}_IMAGE" if l.startswith("denseT") else f"{l}_TEXT|{l}_IMAGE"
    v = c.get(k); x = xr.get(k)
    if not v and not x: continue
    cka = f"{v['L7']['cka']:.3f} | {max(v[L]['cka'] for L in v):.3f} | {v['L7']['rsa_cross']:+.3f}" if v else "— | — | —"
    print(f"| {('dense' + l[6:]) if l.startswith('denseT') else l} | {cka} | " + (f"{x[0]} / {x[2]} | {x[1]} / {x[3]}" if x else "— | —") + " |")
