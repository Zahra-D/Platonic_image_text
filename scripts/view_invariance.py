"""How invariant are pooled image features to the photometric views of make_augmented_views?

For the first --n retrieval scenes, two independent augmented views are tokenised
(same augmentation as training), encoded, mean-pooled per readout; view-A features
retrieve view-B features among all n (cosine).  Reported: R@1, mean cos(A_i, B_i)
centred, and the same for clean vs one view.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from clevr_paths import CLEVR_DATA_ROOT
from data.clevr_dataset import ClevrImageFolder
from models.vqvae import VQVAE
from make_augmented_views import augment
import evaluate_cross_modal_structure as X
from evaluate_probe_suite import load_model_or_random

p = argparse.ArgumentParser(); p.add_argument("--checkpoint", action="append", required=True); p.add_argument("--n", type=int, default=2000)
p.add_argument("--readouts", nargs="+", default=["L5.mlp_out", "L6.mlp_out", "L6", "L7"])
a = p.parse_args()
m = CLEVR_DATA_ROOT + "/platonic_clevr_v1_5M_train_gpu_visible/val_image_only.jsonl"
ds = ClevrImageFolder(str(Path(m).parent), (64, 96), m)
x = torch.stack([ds[i] for i in range(a.n)]).cuda()
ck = torch.load("outputs/vqvae_training_bs128/best.pt", map_location="cpu", weights_only=False)
vq = VQVAE(64, 512).cuda().eval(); vq.load_state_dict(ck["model"])
g = torch.Generator(device="cuda"); g.manual_seed(7)
with torch.no_grad():
    toks = {"clean": vq.encode_to_indices(x), "A": vq.encode_to_indices(augment((x + 1) / 2, g) * 2 - 1),
            "B": vq.encode_to_indices(augment((x + 1) / 2, g) * 2 - 1)}
toks = {k: v.cpu().long() for k, v in toks.items()}
def r1(q, c):
    q = q - q.mean(0); c = c - c.mean(0)
    q = q / np.linalg.norm(q, axis=1, keepdims=True); c = c / np.linalg.norm(c, axis=1, keepdims=True)
    s = q @ c.T; d = np.diag(s)
    return float((s.max(1) <= d + 1e-9).mean()), float(d.mean())
for spec in a.checkpoint:
    lab, path = spec.split("=", 1)
    model, tok, margs = load_model_or_random(path, "cuda")
    f = {k: X.encode_features(model, tok, margs, "cuda", 64, image_tokens=v) for k, v in toks.items()}
    print(lab, " | ".join("%s A->B R@1 %.3f cos %.2f, clean->A R@1 %.3f" % (r, *r1(f["A"][r], f["B"][r]), r1(f["clean"][r], f["A"][r])[0]) for r in a.readouts), flush=True)
    del model; torch.cuda.empty_cache()
