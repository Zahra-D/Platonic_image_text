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
OMNI = Path(os.environ.get("PIXMO_OMNI", Path.home() / "Omni" / "data" / "pixmo_cap"))  # full PixMo download (Omni project)
OMNI_BLOCK = 5000
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


def omni_paths() -> dict[int, Path]:
    """PixMo id (= row of the metadata parquet) -> image file of the Omni project's full PixMo download."""
    paths = {}
    if (OMNI / "paired.jsonl").exists() and not os.environ.get("PIXMO_MAX_CHUNKS"):
        with open(OMNI / "paired.jsonl") as handle:
            for line in handle:
                record = json.loads(line)
                paths[int(record["id"])] = OMNI / record["image_path"]
    return paths


def splits() -> dict:
    """test: the first 5,000 images of our own download (the PixMo test set of every oracle-W result); the training pool
    is every other image of the Omni full download (or of our chunks when it is absent / for smoke tests)."""
    test = []
    for path in chunk_paths():
        test += pq.read_table(path, columns=["id"]).column(0).to_pylist()
        if len(test) >= NUM_TEST:
            break
    test = test[:NUM_TEST]
    omni = omni_paths()
    if omni:
        pool = sorted(set(omni) - set(test))
    else:
        downloaded = [i for path in chunk_paths() for i in pq.read_table(path, columns=["id"]).column(0).to_pylist()]
        pool = [i for i in downloaded if i not in set(test)]
    half = lambda i: int(hashlib.sha256(f"pixmo-{i}".encode()).hexdigest(), 16) % 2
    images = [i for i in pool if half(i) == 0]
    captions = [i for i in pool if half(i) == 1]
    return {"test": test, "images": images, "captions": captions, "probe": images[:NUM_PROBE],
            "num_chunks": len(chunk_paths()), "source": "omni" if omni else "chunks"}


class OmniImages(torch.utils.data.Dataset):
    def __init__(self, items: list[tuple[int, Path]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, k: int):
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        i, path = self.items[k]
        try:
            image = Image.open(path)
            image.draft("RGB", (2 * SIZE, 2 * SIZE))  # JPEG: decode at reduced scale, much faster for large photos
            image = image.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
            return i, torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1), True
        except Exception:  # a few downloaded files are truncated / not images: skipped, dropped from the split at merge
            return i, torch.zeros(3, SIZE, SIZE, dtype=torch.uint8), False


@torch.inference_mode()
def tokenize(shard: int, num_shards: int, gpu: int, batch_size: int = 64, workers: int = 6) -> None:
    """Codes (clean + two photometric views) of every test and A image. Images already tokenized from our own
    download chunks (parts/chunk_*_codes.pt) are reused; every other A image is read from the Omni download, in fixed
    blocks of 5,000 ids (parts/omni_XXXX_codes.pt) shared out over the shards. Rerunning skips finished blocks."""
    sys.path.insert(0, str(ALIGN / "distill"))
    sys.path.insert(0, str(ALIGN / "scripts"))
    from prepare import load_vqgan  # noqa: E402  (platonic_align/distill/prepare.py: MaskGIT VQGAN only)

    split = splits()
    (OUT / "parts").mkdir(parents=True, exist_ok=True)
    covered = set()
    for p in sorted((OUT / "parts").glob("chunk_*_codes.pt")):
        covered.update(torch.load(p, weights_only=False)["ids"].tolist())
    missing_test = [i for i in split["test"] if i not in covered]
    if missing_test:
        raise RuntimeError(f"{len(missing_test)} test images are not tokenized; tokenize chunks 0-1 first")
    omni = omni_paths()
    images = split["images"]
    blocks = [images[b : b + OMNI_BLOCK] for b in range(0, len(images), OMNI_BLOCK)]
    todo = [k for k in range(len(blocks)) if k % num_shards == shard and not omni_part(k).exists()]
    print(f"shard {shard}: {len(covered)} images reused from chunk parts; {len(todo)} Omni blocks to do", flush=True)
    if not todo:
        return
    device = torch.device(f"cuda:{gpu}")
    vqgan = load_vqgan(device)

    def encode(x: torch.Tensor) -> torch.Tensor:
        _, _, (_, _, index) = vqgan.encode(2 * x - 1)
        return index.view(-1, 16, 16).to(torch.int16).cpu()

    start = time.time()
    for n, k in enumerate(todo):
        items = [(i, omni[i]) for i in blocks[k] if i not in covered]
        g = torch.Generator(device=device)
        g.manual_seed(20261015 + 100_000 + k)
        ids, clean, view_a, view_b, failed = [], [], [], [], []
        loader = torch.utils.data.DataLoader(OmniImages(items), batch_size=batch_size, num_workers=workers)
        for batch_ids, pixels, ok in loader:
            failed.extend(batch_ids[~ok].tolist())
            batch_ids, pixels = batch_ids[ok], pixels[ok]
            if len(batch_ids) == 0:
                continue
            x = pixels.to(device).float() / 255
            ids.extend(batch_ids.tolist())
            clean.append(encode(x))
            view_a.append(encode(augment(x, g)))
            view_b.append(encode(augment(x, g)))
        empty = torch.zeros(0, 16, 16, dtype=torch.int16)
        part = {"ids": torch.tensor(ids, dtype=torch.long), "clean": torch.cat(clean) if clean else empty,
                "view_a": torch.cat(view_a) if view_a else empty, "view_b": torch.cat(view_b) if view_b else empty,
                "failed": failed}
        torch.save(part, omni_part(k).with_suffix(".partial"))
        omni_part(k).with_suffix(".partial").rename(omni_part(k))
        print(f"shard {shard}: block {n + 1}/{len(todo)} (omni_{k:04d}, {len(ids)} images, {len(failed)} unreadable), "
              f"{time.time() - start:.0f}s", flush=True)


def omni_part(k: int) -> Path:
    return OUT / "parts" / f"omni_{k:04d}_codes.pt"


def part_path(chunk: str) -> Path:
    return OUT / "parts" / f"{Path(chunk).stem}_codes.pt"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def merge(num_shards: int) -> None:
    split = splits()
    parts = [torch.load(p, weights_only=False) for p in sorted((OUT / "parts").glob("*_codes.pt"))]
    ids = torch.cat([p["ids"] for p in parts])
    row = {int(i): r for r, i in enumerate(ids.tolist())}  # duplicates (an id in a chunk part and an Omni block) keep the last
    unreadable = {i for p in parts for i in p.get("failed", [])}
    absent = [i for i in split["test"] + split["images"] if i not in row and i not in unreadable]
    assert not absent, f"{len(absent)} test / A images have no codes yet, e.g. {absent[:3]}; run --tokenize"
    assert not [i for i in split["test"] if i not in row], "a test image has no codes"
    split["images"] = [i for i in split["images"] if i in row]  # unreadable files dropped
    split["probe"] = split["images"][:NUM_PROBE]
    split["unreadable_dropped"] = len(unreadable)
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
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    if a.tokenize:
        tokenize(a.shard, a.num_shards, a.gpu, a.batch_size, a.workers)
    if a.merge:
        merge(a.num_shards)


if __name__ == "__main__":
    main()
