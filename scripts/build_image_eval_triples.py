"""Build anchor / paraphrase / binding-swap scene specs for image evaluation.

anchor      : a source scene, replayed exactly
paraphrase  : the same objects in the same places, re-rendered with different
              camera and light jitter  (same content, different view)
swap        : the same objects in the same places with the colours of two
              objects exchanged  (identical attribute multiset, different
              binding), rendered with the SAME camera as the anchor

The swap is the image analogue of the text binding swap: every attribute that
appears in the anchor also appears in the swap, exactly once, so a model that
only counts attributes cannot tell them apart.
"""
import json, glob, random, argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--source", default="/home/zd25e122/clevr-dataset-gen_clone/output/platonic_clevr_v1_5M_train_gpu_visible/batches")
p.add_argument("--out", default="outputs/image_eval_triples")
p.add_argument("--num", type=int, default=2000)
p.add_argument("--seed", type=int, default=20261005)
a = p.parse_args()

rng = random.Random(a.seed)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

anchors, swaps, meta = [], [], []
# Draw from batches far past the 1.2M mark so these scenes were never trained on.
files = sorted(glob.glob(f"{a.source}/train_*_scenes.json"))
files = [f for f in files if int(f.split("_")[-2]) >= 1200000]
rng.shuffle(files)

for path in files:
    if len(anchors) >= a.num: break
    try: d = json.load(open(path))
    except Exception: continue
    scenes = d["scenes"] if isinstance(d, dict) and "scenes" in d else d
    for s in scenes:
        if len(anchors) >= a.num: break
        objs = s["objects"]
        if len(objs) < 2: continue
        # pick two objects with different colour AND different shape, so the
        # swap changes the binding and is visible at 96x64
        cand = [(i, j) for i in range(len(objs)) for j in range(i + 1, len(objs))
                if objs[i]["color"] != objs[j]["color"] and objs[i]["shape"] != objs[j]["shape"]]
        if not cand: continue
        i, j = rng.choice(cand)
        sw = [dict(o) for o in objs]
        sw[i]["color"], sw[j]["color"] = sw[j]["color"], sw[i]["color"]
        anchors.append({"objects": [dict(o) for o in objs]})
        swaps.append({"objects": sw})
        meta.append({"source_index": s["image_index"], "swapped": [i, j],
                     "colors": [objs[i]["color"], objs[j]["color"]],
                     "shapes": [objs[i]["shape"], objs[j]["shape"]],
                     "num_objects": len(objs)})

(out / "anchors.json").write_text(json.dumps({"scenes": anchors}))
(out / "swaps.json").write_text(json.dumps({"scenes": swaps}))
(out / "meta.json").write_text(json.dumps(meta, indent=1))
print(f"built {len(anchors)} triples -> {out}")
print(f"  mean objects/scene: {sum(m['num_objects'] for m in meta)/max(1,len(meta)):.2f}")
