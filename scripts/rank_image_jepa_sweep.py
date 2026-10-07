"""Rank the image I-JEPA masking sweep by image-binding d'."""
import json, re
from pathlib import Path

D = Path("outputs/image_binding_sweep")
CFG = Path("configs")
rows = []
for f in sorted(D.glob("image_ijepa_sweep_*.json")):
    name = f.stem
    try: m = json.load(open(f))
    except Exception: continue
    pl = m.get("per_layer") or m.get("metrics") or {}
    def best(sub):
        v = [(d.get("d_binding", {}).get("d", {}).get("value"), k)
             for k, d in pl.items() if k.endswith(sub) or (sub == "" and "." not in k)]
        v = [(a, b) for a, b in v if a is not None]
        return max(v) if v else (None, None)
    head = m.get("d_binding", {}).get("d", {}).get("value")
    br, bl = best("")
    mr, ml = best(".mlp_out")
    ar, al = best(".attn_out")
    cfgp = CFG / f"{name}.yaml"
    t = sc = asp = blk = "?"
    if cfgp.exists():
        s = cfgp.read_text()
        t = (re.search(r"(?m)^  train_fixed_t: (.*)$", s) or [None, "?"])[1]
        blk = (re.search(r"(?m)^  mask_block_2d: (.*)$", s) or [None, "?"])[1]
        sm = re.search(r"(?m)^  mask_block_scale:\n  - (.*)\n  - (.*)$", s)
        am = re.search(r"(?m)^  mask_block_aspect:\n  - (.*)\n  - (.*)$", s)
        sc = f"{sm.group(1)}-{sm.group(2)}" if sm else "-"
        asp = f"{am.group(1)}-{am.group(2)}" if am else "-"
    rows.append(dict(name=name.replace("image_ijepa_sweep_", ""), head=head,
                     res=br, res_at=bl, mlp=mr, mlp_at=ml, attn=ar, attn_at=al,
                     t=t, blocks=blk, scale=sc, aspect=asp))

key = lambda r: (r["mlp"] if r["mlp"] is not None else -9)
rows.sort(key=key, reverse=True)
print("# Image I-JEPA masking sweep\n")
print("Ranked by best MLP-write image-binding d'. Reference: the two settings used")
print("so far scored 0.819 (blk_s15_t45) and 0.822 (blk_s10_t30) from a 4-epoch trunk.\n")
print("| setting | blocks | mask ratio | block area | aspect | best residual | best attn | best MLP |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    f = lambda v, at: f"{v:.3f} @{at}" if v is not None else "-"
    print(f"| `{r['name']}` | {r['blocks']} | {r['t']} | {r['scale']} | {r['aspect']} | "
          f"{f(r['res'], r['res_at'])} | {f(r['attn'], r['attn_at'])} | **{f(r['mlp'], r['mlp_at'])}** |")
if rows and rows[0]["mlp"] is not None:
    print(f"\n**Best setting: `{rows[0]['name']}`** — MLP d' {rows[0]['mlp']:.3f}")
