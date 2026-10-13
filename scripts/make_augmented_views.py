"""Two photometrically augmented VQ token views per training image.

For cross-view data2vec: the same scene under different lighting / exposure must
map to the same representation.  Each image is loaded once and augmented twice on
the GPU, independently, then tokenised with the training VQ-VAE:
  brightness x U(0.6, 1.4), contrast x U(0.6, 1.4) about the image mean,
  gamma U(0.7, 1.4), Gaussian blur (sigma 0.6) with p = 0.3, pixel noise sigma U(0, 0.04).
No hue / saturation change (colour is content) and no geometric change (token
positions must stay aligned between the views).  Rows are the first --num-images
of the manifest (the 1.2M block, disjoint from the retrieval eval scenes), in
manifest order, so row i matches row i of the clean token cache.

  python3 scripts/make_augmented_views.py --shard K --num-shards S --gpu G
  python3 scripts/make_augmented_views.py --merge --num-shards S
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clevr_paths import CLEVR_DATA_ROOT
from data.clevr_dataset import ClevrImageFolder
from models.vqvae import VQVAE

OUT = Path("outputs/image_views_cache")


def augment(x, g):
    """x in [0, 1], [B, 3, H, W]; independent photometric jitter per image."""
    b = x.size(0)
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(b, 1, 1, 1, device=x.device, generator=g)
    x = x * u(0.6, 1.4)
    mean = x.mean((1, 2, 3), keepdim=True)
    x = (x - mean) * u(0.6, 1.4) + mean
    x = x.clamp(1e-4, 1).pow(u(0.7, 1.4))
    k = torch.tensor([0.2, 0.6, 0.2], device=x.device)
    blurred = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate").reshape(-1, 1, x.size(2) + 2, x.size(3) + 2),
                       (k[:, None] * k[None, :])[None, None]).reshape_as(x)
    pick = (torch.rand(b, 1, 1, 1, device=x.device, generator=g) < 0.3).float()
    x = pick * blurred + (1 - pick) * x
    x = x + u(0.0, 0.04) * torch.randn(x.shape, device=x.device, generator=g)
    return x.clamp(0, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=CLEVR_DATA_ROOT + "/platonic_clevr_v1_5M_train_gpu_visible/train_image_only_2_5m.jsonl")
    p.add_argument("--num-images", type=int, default=600_000)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--tokenizer-checkpoint", default="outputs/vqvae_training_bs128/best.pt")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    size = (a.num_images + a.num_shards - 1) // a.num_shards
    if a.merge:
        parts = [torch.load(OUT / f"shard_{k:02d}.pt", weights_only=False) for k in range(a.num_shards)]
        merged = {v: torch.cat([q[v] for q in parts]) for v in ("view_a", "view_b")}
        merged["num_images"] = len(merged["view_a"]); merged["manifest"] = a.manifest
        torch.save(merged, OUT / "train_views.pt")
        print("merged", merged["view_a"].shape)
        return
    dev = f"cuda:{a.gpu}"
    ck = torch.load(a.tokenizer_checkpoint, map_location="cpu", weights_only=False)
    vq = VQVAE(embed_dim=64, num_codes=512).to(dev).eval()
    vq.load_state_dict(ck["model"])
    ds = ClevrImageFolder(str(Path(a.manifest).parent), image_size=(64, 96), manifest_path=a.manifest)
    rows = range(a.shard * size, min(a.num_images, (a.shard + 1) * size))
    loader = DataLoader(Subset(ds, list(rows)), batch_size=512, num_workers=16, pin_memory=True)
    g = torch.Generator(device=dev); g.manual_seed(20261012 + a.shard)
    va, vb, t0 = [], [], time.time()
    with torch.no_grad():
        for i, x in enumerate(loader):
            x = (x.to(dev, non_blocking=True) + 1) / 2
            for store in (va, vb):
                store.append(vq.encode_to_indices(augment(x, g) * 2 - 1).cpu().to(torch.uint16))
            if i % 50 == 0:
                print(f"shard {a.shard}: {(i + 1) * 512} / {len(rows)} images, {time.time() - t0:.0f}s", flush=True)
    torch.save({"view_a": torch.cat(va), "view_b": torch.cat(vb), "rows": [rows.start, rows.stop]}, OUT / f"shard_{a.shard:02d}.pt")
    print(f"shard {a.shard} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
