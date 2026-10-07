#!/usr/bin/env python3
"""Build leakage-audited one-caption-per-image COCO JEPA manifests.

Exactly one caption from each image is written to the primary manifest.  The
other captions are written only to the retrieval manifest and are never read
by the training dataset or used to build its tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-annotations", required=True)
    parser.add_argument("--val-annotations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260923)
    return parser.parse_args()


def stable_choice(seed: int, split: str, image_id: int, count: int) -> int:
    digest = hashlib.sha256(f"{seed}:{split}:{image_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % count


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_split(source: Path, split: str, destination: Path, seed: int) -> dict:
    payload = json.loads(source.read_text())
    image_ids = {int(row["id"]) for row in payload["images"]}
    grouped: dict[int, list[dict]] = defaultdict(list)
    for annotation in payload["annotations"]:
        grouped[int(annotation["image_id"])].append(annotation)
    if image_ids != set(grouped):
        missing = sorted(image_ids.difference(grouped))[:10]
        extra = sorted(set(grouped).difference(image_ids))[:10]
        raise ValueError(f"image/caption mismatch for {split}: missing={missing}, extra={extra}")

    primary_path = destination / f"{split}_primary.jsonl"
    retrieval_path = destination / f"{split}_retrieval.jsonl"
    primary_ids: set[int] = set()
    heldout_ids: set[int] = set()
    caption_counts: dict[int, int] = defaultdict(int)
    with primary_path.open("w", encoding="utf-8") as primary_file, retrieval_path.open(
        "w", encoding="utf-8"
    ) as retrieval_file:
        for image_id in sorted(image_ids):
            captions = sorted(grouped[image_id], key=lambda row: int(row["id"]))
            if len(captions) < 2:
                raise ValueError(f"image {image_id} in {split} has fewer than two captions")
            selected = stable_choice(seed, split, image_id, len(captions))
            query = captions[selected]
            heldout = captions[:selected] + captions[selected + 1 :]
            query_id = int(query["id"])
            positive_ids = [int(row["id"]) for row in heldout]
            primary_ids.add(query_id)
            heldout_ids.update(positive_ids)
            caption_counts[len(captions)] += 1
            primary_file.write(
                json.dumps(
                    {
                        "image_id": image_id,
                        "split": split,
                        "caption_id": query_id,
                        "caption": query["caption"],
                        "heldout_caption_count": len(heldout),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            retrieval_file.write(
                json.dumps(
                    {
                        "image_id": image_id,
                        "split": split,
                        "query_caption_id": query_id,
                        "query": query["caption"],
                        "positives": [
                            {"caption_id": int(row["id"]), "caption": row["caption"]}
                            for row in heldout
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    overlap = primary_ids.intersection(heldout_ids)
    if overlap:
        raise RuntimeError(f"caption-ID leakage in {split}: {sorted(overlap)[:10]}")
    return {
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "images": len(image_ids),
        "primary_captions": len(primary_ids),
        "heldout_captions": len(heldout_ids),
        "captions_per_image_histogram": {str(k): v for k, v in sorted(caption_counts.items())},
        "primary_manifest": str(primary_path.resolve()),
        "primary_manifest_sha256": sha256(primary_path),
        "retrieval_manifest": str(retrieval_path.resolve()),
        "retrieval_manifest_sha256": sha256(retrieval_path),
        "caption_id_overlap": 0,
    }


def main() -> None:
    args = arguments()
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    train = prepare_split(Path(args.train_annotations), "train2017", destination, args.seed)
    val = prepare_split(Path(args.val_annotations), "val2017", destination, args.seed)
    report = {
        "schema_version": 1,
        "seed": args.seed,
        "selection": "sha256(seed:split:image_id) modulo caption count",
        "invariant": "only *_primary.jsonl may be passed to train_multimodal.py",
        "train": train,
        "validation": val,
    }
    (destination / "split_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
