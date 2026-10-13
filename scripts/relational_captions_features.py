"""Re-encode text models on retrieval-scene captions that include spatial relations.

load_scenes renders each caption from the manifest world, which carries CLEVR
'relationships' but no 'relations', so the 8000 retrieval captions had no relation
sentences.  Here each world gets the scene file's 'latent_relations' (the
generator's format, 0-2 relations, minimum projection 0.7, as in the text corpus)
and the caption is rendered with the same per-scene seed and plan recipe.
Features are saved as <label>_REL.npz (xret format).

  python3 scripts/relational_captions_features.py --checkpoint LABEL=PATH [...] [--mpnet]
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clevr_paths import CLEVR_DATA_ROOT, CLEVR_GEN_ROOT
import binding_swap_captions as C
import evaluate_cross_modal_structure as X
from evaluate_cross_modal_retrieval import READOUTS
from evaluate_probe_suite import load_model_or_random


def relational_captions(manifest, n, seed, generator_path):
    gen = C.load_generator(generator_path)
    root = Path(manifest).parent
    caps, worlds = [], []
    for pos, line in zip(range(n), open(manifest)):
        row = json.loads(line)
        sc = json.load(open(root / row["image_path"].replace("images/", "scenes/", 1).replace(".png", ".json")))
        world = dict(row["world"]); world["relations"] = sc.get("latent_relations", [])
        rng = random.Random(seed * 1_000_003 + pos)
        no, nr = len(world["objects"]), len(world["relations"])
        plan = C.make_plan(gen, rng.randrange(2**31), no, nr)
        caps.append(C.render(gen, world, plan, rng.sample(range(no), no), rng.sample(range(nr), nr), rng.randrange(2)))
        worlds.append(world)
    return caps, worlds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", default=[])
    p.add_argument("--mpnet", action="store_true")
    p.add_argument("--manifest", default=CLEVR_DATA_ROOT + "/platonic_clevr_v1_5M_train_gpu_visible/val_image_only.jsonl")
    p.add_argument("--caption-generator", default=CLEVR_GEN_ROOT + "/generate_human_captions.py")
    p.add_argument("--num-scenes", type=int, default=8000)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--output-dir", default="outputs/eval_all/xret/features")
    a = p.parse_args()
    caps, worlds = relational_captions(a.manifest, a.num_scenes, a.seed, a.caption_generator)
    out = Path(a.output_dir)
    (out.parent / "relational_captions.jsonl").write_text("".join(json.dumps({"i": i, "caption": c}) + "\n" for i, c in enumerate(caps)))
    print("relations per caption:", np.bincount([len(w["relations"]) for w in worlds]).tolist(), "| example:", caps[0][:300], flush=True)
    for spec in a.checkpoint:
        label, path = spec.split("=", 1)
        model, tok, margs = load_model_or_random(path, "cuda")
        feats = X.encode_features(model, tok, margs, "cuda", 64, captions=caps)
        np.savez(out / f"{label}_REL.npz", **{k: feats[k].astype(np.float32) for k in READOUTS if k in feats})
        print(label, "saved", flush=True)
        del model; torch.cuda.empty_cache()
    if a.mpnet:
        from transformers import AutoModel, AutoTokenizer
        tk = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        m = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").cuda().eval()
        embs = []
        with torch.no_grad():
            for s in range(0, len(caps), 128):
                b = tk(caps[s:s + 128], padding=True, truncation=True, max_length=384, return_tensors="pt").to("cuda")
                h = m(**b).last_hidden_state; mk = b["attention_mask"].unsqueeze(-1).float()
                embs.append(F.normalize((h * mk).sum(1) / mk.sum(1), dim=1).cpu())
        np.savez(out / "mpnet_TEXT_REL.npz", mean=torch.cat(embs).numpy()); print("mpnet saved", flush=True)


if __name__ == "__main__":
    main()
