"""VQ-encode the anchor / paraphrase / swap renders into one aligned cache."""
import torch, argparse, re
from pathlib import Path
from PIL import Image
import numpy as np
import sys
sys.path.insert(0, "/home/zd25e122/clevr_discrete_diffusion")
from models.vqvae import VQVAE

p = argparse.ArgumentParser()
p.add_argument("--root", default="outputs/image_eval_triples")
p.add_argument("--tokenizer", default="outputs/vqvae_training_bs128/best.pt")
p.add_argument("--splits", nargs="+", default=["anchor", "para", "swap"])
p.add_argument("--batch-size", type=int, default=256)
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(a.tokenizer, map_location=dev, weights_only=False)
ka = ck.get("args", {})
vq = VQVAE(embed_dim=ka.get("embed_dim", 64), num_codes=ka.get("num_codes", 512)).to(dev).eval()
vq.load_state_dict(ck["model"])

root = Path(a.root)
idx_of = lambda p: int(re.search(r"_(\d+)\.png$", p.name).group(1))
out = {}
for split in a.splits:
    files = sorted((root / split).glob("*.png"), key=idx_of)
    order = [idx_of(f) for f in files]
    toks = []
    with torch.no_grad():
        for s in range(0, len(files), a.batch_size):
            batch = np.stack([np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
                              for f in files[s:s + a.batch_size]])
            x = torch.from_numpy(batch).permute(0, 3, 1, 2).to(dev)
            toks.append(vq.encode_to_indices(x).cpu().to(torch.uint16))
    out[split] = {"tokens": torch.cat(toks), "index": torch.tensor(order)}
    print(f"{split}: {tuple(out[split]['tokens'].shape)} indices {min(order)}..{max(order)}")

# Keep only scene indices present in every split, in one shared order.
common = sorted(set.intersection(*[set(v["index"].tolist()) for v in out.values()]))
pos = {s: {int(i): r for r, i in enumerate(v["index"].tolist())} for s, v in out.items()}
aligned = {s: out[s]["tokens"][[pos[s][i] for i in common]] for s in out}
torch.save({"tokens": aligned, "scene_index": torch.tensor(common),
            "splits": list(out)}, root / "triples_tokens.pt")
print(f"aligned {len(common)} triples -> {root/'triples_tokens.pt'}")
