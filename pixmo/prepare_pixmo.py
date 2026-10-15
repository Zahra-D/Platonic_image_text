"""Build the PixMo-Cap data of the three-stage pipeline (strictly unpaired), on the machine with the downloaded images.

Sources (this machine only): the image chunks and metadata downloaded under ~/platonic_align/data/pixmo_cap and
MaskGIT's ImageNet VQGAN (taming f16, 1024 codes) from ~/platonic_align/external/maskgit.

Splits (stable when more chunks are downloaded later):
  test   the first 5,000 downloaded images (the PixMo test set of all oracle-W results in ~/platonic_align)
  A / B  every other image goes to A (images) or B (captions) by a hash of its PixMo id, so no image is ever
         trained together with its own caption
  probe  the first 10,000 A images in download order, paired with their own (never trained) captions; used only
         to fit the oracle W of the evaluation

Outputs in $PIXMO_DATA (default ~/platonic_pixmo_data):
  images_train.jsonl / images_val.jsonl   image_path "pixmo/<id>", pixmo_id    (A images / test images)
  text_train.jsonl / text_val.jsonl       caption                              (B captions / test captions)
  token_cache/{train,val}_tokens.pt       {"tokens": uint16 [N,16,16], "metadata": {...}}, rows = manifest rows
  views/train_views.pt                    {"view_a", "view_b": uint16 [N,16,16]}: two photometric views per A image
  paraphrase/train_para.txt               one transcript per text_train row (the spoken description the caption
                                          was written from), newlines removed
  eval_pairs.pt                           {"test" | "probe": {"ids", "image_tokens", "captions"}}
  split.json

Images are squashed to 256 x 256 (no crop), the same view the DINOv2 teachers saw.

    # GPU part, one process per GPU, then merge (re-runnable; finished shards are skipped)
    python pixmo/prepare_pixmo.py --tokenize --shard 0 --num-shards 2 --gpu 0 &
    python pixmo/prepare_pixmo.py --tokenize --shard 1 --num-shards 2 --gpu 1 & wait
    python pixmo/prepare_pixmo.py --merge
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
ALIGN = Path(os.environ.get("PLATONIC_ALIGN", Path.home() / "platonic_align"))
OUT = Path(os.environ.get("PIXMO_DATA", Path.home() / "platonic_pixmo_data"))
CHUNKS = ALIGN / "data" / "pixmo_cap" / "images"
METADATA = ALIGN / "data" / "pixmo_cap" / "data"
NUM_TEST, NUM_PROBE, SIZE = 5000, 10000, 256
sys.path.insert(0, str(REPO))
from data.multimodal_dataset import image_manifest_fingerprint  # noqa: E402


def augment(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Photometric jitter of scripts/make_augmented_views.py (CLEVR stage 2), unchanged: x in [0, 1]."""
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


def chunk_paths() -> list[str]:
    paths = sorted(glob.glob(str(CHUNKS / "chunk_*.parquet")))
    limit = os.environ.get("PIXMO_MAX_CHUNKS")  # smoke tests only
    return paths[: int(limit)] if limit else paths


def splits() -> dict:
    downloaded = [i for path in chunk_paths() for i in pq.read_table(path, columns=["id"]).column(0).to_pylist()]
    test, rest = downloaded[:NUM_TEST], downloaded[NUM_TEST:]
    half = lambda i: int(hashlib.sha256(f"pixmo-{i}".encode()).hexdigest(), 16) % 2
    images = [i for i in rest if half(i) == 0]
    captions = [i for i in rest if half(i) == 1]
    return {"test": test, "images": images, "captions": captions, "probe": images[:NUM_PROBE],
            "num_chunks": len(chunk_paths())}


@torch.inference_mode()
def tokenize(shard: int, num_shards: int, gpu: int) -> None:
    """Codes of every test and A image (clean + two photometric views), one cached file per download chunk, so the
    work is incremental: rerun after more chunks are downloaded and only the new chunks are tokenized."""
    sys.path.insert(0, str(ALIGN / "distill"))
    sys.path.insert(0, str(ALIGN / "scripts"))
    from prepare import load_vqgan  # noqa: E402  (platonic_align/distill/prepare.py: MaskGIT VQGAN only)
    from PIL import Image
    import io

    split = splits()
    wanted = set(split["test"]) | set(split["images"])
    (OUT / "parts").mkdir(parents=True, exist_ok=True)
    todo = [p for k, p in enumerate(chunk_paths()) if k % num_shards == shard and not part_path(p).exists()]
    if not todo:
        print(f"shard {shard}: nothing to do", flush=True)
        return
    device = torch.device(f"cuda:{gpu}")
    vqgan = load_vqgan(device)

    def encode(x: torch.Tensor) -> torch.Tensor:
        _, _, (_, _, index) = vqgan.encode(2 * x - 1)
        return index.view(-1, 16, 16).to(torch.int16).cpu()

    start = time.time()
    for n, path in enumerate(todo):
        g = torch.Generator(device=device)
        g.manual_seed(20261015 + int(Path(path).stem.split("_")[1]))  # per chunk: the same views whatever the sharding
        table = pq.read_table(path, columns=["id", "image"])
        rows = [(i, im["bytes"]) for i, im in zip(table.column(0).to_pylist(), table.column(1).to_pylist()) if i in wanted]
        ids, clean, view_a, view_b = [], [], [], []
        for b in range(0, len(rows), 64):
            batch = rows[b : b + 64]
            images = [torch.from_numpy(np.asarray(Image.open(io.BytesIO(data)).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)).copy()).permute(2, 0, 1)
                      for _, data in batch]
            x = torch.stack(images).to(device).float() / 255
            ids.extend(i for i, _ in batch)
            clean.append(encode(x))
            view_a.append(encode(augment(x, g)))
            view_b.append(encode(augment(x, g)))
        empty = torch.zeros(0, 16, 16, dtype=torch.int16)
        torch.save({"ids": torch.tensor(ids, dtype=torch.long), "clean": torch.cat(clean) if clean else empty,
                    "view_a": torch.cat(view_a) if view_a else empty, "view_b": torch.cat(view_b) if view_b else empty},
                   part_path(path))
        print(f"shard {shard}: {n + 1}/{len(todo)} chunks ({Path(path).name}, {len(ids)} images), {time.time() - start:.0f}s", flush=True)


def part_path(chunk: str) -> Path:
    return OUT / "parts" / f"{Path(chunk).stem}_codes.pt"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def merge(num_shards: int) -> None:
    split = splits()
    missing = [p for p in chunk_paths() if not part_path(p).exists()]
    assert not missing, f"{len(missing)} chunks not tokenized yet, e.g. {missing[0]}"
    parts = [torch.load(part_path(p), weights_only=False) for p in chunk_paths()]
    ids = torch.cat([p["ids"] for p in parts])
    row = {int(i): r for r, i in enumerate(ids.tolist())}
    codes = {k: torch.cat([p[k] for p in parts]) for k in ("clean", "view_a", "view_b")}
    gather = lambda key, which: codes[key][torch.tensor([row[i] for i in which])].to(torch.uint16)

    meta = pq.read_table(sorted(glob.glob(str(METADATA / "*.parquet"))), columns=["caption", "transcripts"])
    captions, transcripts = meta.column(0).to_pylist(), meta.column(1).to_pylist()
    caption = lambda i: captions[i]
    paraphrase = lambda i: " ".join((transcripts[i] or [captions[i]])[0].split())

    for d in ("token_cache", "views", "paraphrase"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    image_rows = {"train": split["images"], "val": split["test"]}
    for name, which in image_rows.items():
        records = [{"image_path": f"pixmo/{i}", "pixmo_id": i} for i in which]
        write_jsonl(OUT / f"images_{name}.jsonl", records)
        metadata = {"tokenizer": "maskgit-vqgan-imagenet-f16-1024", "num_codes": 1024, "grid_size": [16, 16],
                    "image_size": [SIZE, SIZE], "preprocess": "squash", "split": name,
                    "image_manifest_sha256": image_manifest_fingerprint(records)}
        torch.save({"tokens": gather("clean", which), "metadata": metadata}, OUT / "token_cache" / f"{name}_tokens.pt")
    torch.save({"view_a": gather("view_a", split["images"]), "view_b": gather("view_b", split["images"]),
                "num_images": len(split["images"]), "manifest": "images_train.jsonl"}, OUT / "views" / "train_views.pt")
    write_jsonl(OUT / "text_train.jsonl", [{"caption": caption(i), "pixmo_id": i} for i in split["captions"]])
    write_jsonl(OUT / "text_val.jsonl", [{"caption": caption(i), "pixmo_id": i} for i in split["test"]])
    with open(OUT / "paraphrase" / "train_para.txt", "w") as handle:
        for i in split["captions"]:
            handle.write(paraphrase(i) + "\n")
    torch.save({name: {"ids": torch.tensor(which), "image_tokens": gather("clean", which),
                       "captions": [caption(i) for i in which]}
                for name, which in (("test", split["test"]), ("probe", split["probe"]))}, OUT / "eval_pairs.pt")
    (OUT / "split.json").write_text(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in split.items()}, indent=2))
    print(f"wrote {OUT}: {json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in split.items()})}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenize", action="store_true")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--gpu", type=int, default=0)
    a = p.parse_args()
    if a.tokenize:
        tokenize(a.shard, a.num_shards, a.gpu)
    if a.merge:
        merge(a.num_shards)


if __name__ == "__main__":
    main()
