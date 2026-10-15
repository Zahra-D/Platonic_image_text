"""One paraphrase per caption of a text manifest: the same scene record rendered again by
the caption generator with an independent phrasing plan, object order, relation order and
sentence pattern (relations included).  Line i of the output = paraphrase of manifest row i.

  python3 scripts/make_paraphrase_cache.py --manifest M --output outputs/text_paraphrase_cache/train_para.txt
"""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import binding_swap_captions as C
from clevr_paths import CLEVR_DATA_ROOT, CLEVR_GEN_ROOT

p = argparse.ArgumentParser()
p.add_argument("--manifest", default=CLEVR_DATA_ROOT + "/platonic_text_only_v1_2m/train_text_only_human.jsonl")
p.add_argument("--output", default="outputs/text_paraphrase_cache/train_para.txt")
p.add_argument("--seed", type=int, default=20261014)
a = p.parse_args()
gen = C.load_generator(CLEVR_GEN_ROOT + "/generate_human_captions.py")
with open(a.manifest) as src, open(a.output, "w") as out:
    for i, line in enumerate(src):
        w = json.loads(line)["world"]; rng = random.Random(a.seed * 1_000_003 + i)
        n, m = len(w["objects"]), len(w.get("relations", []))
        plan = C.make_plan(gen, rng.randrange(2**31), n, m)
        cap = C.render(gen, w, plan, rng.sample(range(n), n), rng.sample(range(m), m), rng.randrange(2))
        out.write(cap.replace("\n", " ") + "\n")
        if i % 500000 == 0:
            print(i, flush=True)
print("done", i + 1)
