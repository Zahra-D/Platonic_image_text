#!/usr/bin/env python3
"""Tokenize a large image-only manifest in parallel shards, then merge.

Reading a million small PNGs over NFS is latency-bound, so one process with
more workers does not help (64 workers measured slower than 32). Splitting the
manifest across several processes does. Each shard is tokenized by
``pretokenize_clevr.py`` into its own cache, then the shards are concatenated
**in manifest order** and saved as one cache whose fingerprint covers the whole
manifest, exactly as the single-process path would have produced.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.multimodal_dataset import image_manifest_fingerprint, read_jsonl


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split-name", required=True, choices=["train", "val"])
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--num-workers", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--tokenizer-checkpoint", default="outputs/vqvae_training_bs128/best.pt")
    p.add_argument("--python", default="/home/zd25e122/miniconda3/envs/unix/bin/python3.10")
    return p.parse_args()


def main():
    args = arguments()
    cache = Path(args.cache_dir); cache.mkdir(parents=True, exist_ok=True)
    final = cache / f"{args.split_name}_tokens.pt"
    if final.exists():
        print(f"{final} already exists")
        return
    records = read_jsonl(args.manifest)
    work = cache / f"{args.split_name}_shards"; work.mkdir(exist_ok=True)
    size = (len(records) + args.shards - 1) // args.shards
    shard_paths, processes = [], []
    for index in range(args.shards):
        rows = records[index * size:(index + 1) * size]
        if not rows:
            continue
        shard_manifest = work / f"shard_{index:02d}.jsonl"
        shard_manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        shard_cache = work / f"cache_{index:02d}"
        shard_paths.append(shard_cache / "train_tokens.pt")
        if (shard_cache / "train_tokens.pt").exists():
            print(f"shard {index}: already tokenized"); continue
        gpu = args.gpus[index % len(args.gpus)]
        log = open(work / f"shard_{index:02d}.log", "w")
        processes.append(subprocess.Popen(
            [args.python, "pretokenize_clevr.py", "--tokenizer-checkpoint", args.tokenizer_checkpoint,
             "--train-dir", args.dataset, "--val-dir", args.dataset,
             "--train-manifest", str(shard_manifest), "--val-manifest", str(shard_manifest),
             "--cache-dir", str(shard_cache), "--image-size", "64", "96",
             "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers)],
            env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
            stdout=log, stderr=subprocess.STDOUT))
        print(f"shard {index}: {len(rows)} images on GPU {gpu}", flush=True)
    for process in processes:
        if process.wait() != 0:
            raise RuntimeError("a shard failed; see the shard logs")
    tokens = torch.cat([torch.load(p, map_location="cpu", weights_only=False)["tokens"] for p in shard_paths])
    if len(tokens) != len(records):
        raise RuntimeError(f"merged {len(tokens)} tokens for {len(records)} manifest rows")
    reference = torch.load(shard_paths[0], map_location="cpu", weights_only=False)["metadata"]
    metadata = {**reference, "split": args.split_name, "num_images": len(tokens),
                "grid_size": list(tokens.shape[1:]),
                "image_manifest_sha256": image_manifest_fingerprint(records),
                "sharded_tokenization": {"shards": len(shard_paths), "manifest": str(Path(args.manifest).resolve())}}
    temporary = final.with_suffix(".tmp")
    torch.save({"tokens": tokens, "metadata": metadata}, temporary)
    temporary.replace(final)
    print(f"wrote {final}: {tuple(tokens.shape)} {tokens.dtype}, {final.stat().st_size / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
