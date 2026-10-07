"""Turn freshly rendered scenes into anchor and binding-swap replay specs."""
import json, glob, random, argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--scene-dir", required=True)
p.add_argument("--out", required=True)
p.add_argument("--num", type=int, default=2000)
p.add_argument("--seed", type=int, default=20261005)
a = p.parse_args()
rng = random.Random(a.seed)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

anchors, swaps, meta, skipped = [], [], [], 0
for f in sorted(glob.glob(f"{a.scene_dir}/*.json")):
    if len(anchors) >= a.num: break
    try: s = json.load(open(f))
    except Exception: continue
    objs = s.get("objects") or []
    if len(objs) < 2: skipped += 1; continue
    # Two objects that differ in BOTH colour and shape: swapping their colours
    # keeps every attribute multiset identical while changing the binding, and
    # the change is visible because the shapes differ.
    cand = [(i, j) for i in range(len(objs)) for j in range(i + 1, len(objs))
            if objs[i]["color"] != objs[j]["color"] and objs[i]["shape"] != objs[j]["shape"]]
    if not cand: skipped += 1; continue
    i, j = rng.choice(cand)
    sw = [dict(o) for o in objs]
    sw[i]["color"], sw[j]["color"] = sw[j]["color"], sw[i]["color"]
    anchors.append({"objects": [dict(o) for o in objs]})
    swaps.append({"objects": sw})
    meta.append({"scene_file": Path(f).name, "swapped": [i, j],
                 "colors": [objs[i]["color"], objs[j]["color"]],
                 "shapes": [objs[i]["shape"], objs[j]["shape"]],
                 "num_objects": len(objs)})

(out / "anchors.json").write_text(json.dumps({"scenes": anchors}))
(out / "swaps.json").write_text(json.dumps({"scenes": swaps}))
(out / "meta.json").write_text(json.dumps(meta, indent=1))
print(f"{len(anchors)} triples, {skipped} scenes skipped (no valid swap pair)")
