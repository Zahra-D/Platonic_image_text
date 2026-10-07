"""Encode CLEVR once with a frozen VQ-VAE and save compact token caches."""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.clevr_dataset import ClevrImageFolder
from data.multimodal_dataset import image_manifest_fingerprint, read_jsonl
from models.vqvae import VQVAE


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer-checkpoint", default="outputs/vqvae_training_bs128/best.pt")
    p.add_argument("--train-dir", default=str(Path.home() / "Omni/data/clevr/train"))
    p.add_argument("--val-dir", default=str(Path.home() / "Omni/data/clevr/val"))
    p.add_argument("--train-manifest", default=None,
                   help="Optional paired JSONL whose image_path values are relative to --train-dir.")
    p.add_argument("--val-manifest", default=None,
                   help="Optional paired JSONL whose image_path values are relative to --val-dir.")
    p.add_argument("--cache-dir", default="outputs/token_cache")
    p.add_argument("--image-size", type=int, nargs=2, default=[64, 96], metavar=("H", "W"))
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def checkpoint_metadata(path, ckpt):
    stat = Path(path).expanduser().stat()
    args = ckpt.get("args", {})
    return {
        "tokenizer_checkpoint": str(Path(path).expanduser().resolve()),
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "num_codes": args.get("num_codes", 512),
        "embed_dim": args.get("embed_dim", 64),
    }


@torch.no_grad()
def encode_split(name, split_dir, manifest_path, out_path, vqvae, args, metadata, device):
    if out_path.exists() and not args.overwrite:
        print(f"[{name}] using existing cache: {out_path}")
        return

    ds = ClevrImageFolder(split_dir, image_size=tuple(args.image_size), manifest_path=manifest_path)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )
    chunks = []
    start = time.perf_counter()
    for batch_idx, images in enumerate(loader, start=1):
        indices = vqvae.encode_to_indices(images.to(device, non_blocking=True))
        chunks.append(indices.cpu().to(torch.uint16))
        if batch_idx % 25 == 0 or batch_idx == len(loader):
            n = min(batch_idx * args.batch_size, len(ds))
            print(f"[{name}] {n}/{len(ds)} images ({n / (time.perf_counter() - start):.1f} img/s)")

    tokens = torch.cat(chunks)
    manifest = Path(manifest_path).expanduser() if manifest_path else Path(split_dir).expanduser() / "images.jsonl"
    payload = {
        "tokens": tokens,
        "metadata": {
            **metadata,
            "split": name,
            "source_dir": str(Path(split_dir).expanduser().resolve()),
            "image_size": list(args.image_size),
            "grid_size": list(tokens.shape[1:]),
            "num_images": len(ds),
            "dtype": str(tokens.dtype),
            **(
                {
                    "image_manifest_sha256": image_manifest_fingerprint(
                        read_jsonl(manifest)
                    )
                }
                if manifest.exists()
                else {}
            ),
        },
    }
    tmp_path = out_path.with_suffix(".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(out_path)
    size_mb = out_path.stat().st_size / 2**20
    print(f"[{name}] wrote {out_path} ({size_mb:.1f} MiB)")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.tokenizer_checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    vqvae = VQVAE(
        embed_dim=ckpt_args.get("embed_dim", 64),
        num_codes=ckpt_args.get("num_codes", 512),
    ).to(device)
    vqvae.load_state_dict(ckpt["model"])
    vqvae.eval()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = checkpoint_metadata(args.tokenizer_checkpoint, ckpt)
    encode_split("train", args.train_dir, args.train_manifest, cache_dir / "train_tokens.pt", vqvae, args, metadata, device)
    encode_split("val", args.val_dir, args.val_manifest, cache_dir / "val_tokens.pt", vqvae, args, metadata, device)


if __name__ == "__main__":
    main()
