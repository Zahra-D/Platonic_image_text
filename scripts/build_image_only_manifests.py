#!/usr/bin/env python3
"""Build image-only train/val manifests from a rendered CLEVR dataset.

Reads the aggregated ``batches/*_scenes.json`` files (one per rendered batch,
each with a ``.done`` marker) rather than the per-image scene JSONs, so a
million-image dataset can be indexed in minutes.  Writes one JSONL record per
image with the fields the training pipeline needs plus the scene labels later
evaluations need:

* ``image_path``: path relative to the dataset root, as
  ``ClevrImageFolder``/``pretokenize_clevr.py`` expect;
* ``image_index``: integer index from the renderer;
* ``world``: objects with their four attributes, and the CLEVR relationship
  adjacency lists, kept verbatim.

The split is by image index: the highest indices become validation, so
validation never shares a rendered batch with training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ATTRIBUTES = ("size", "color", "material", "shape")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="Dataset root containing batches/ and images/")
    p.add_argument("--split", default="train", help="Renderer split name inside batches/ and images/")
    p.add_argument("--train-size", type=int, default=1_200_000)
    p.add_argument("--val-size", type=int, default=20_000)
    p.add_argument("--train-out", required=True)
    p.add_argument("--val-out", required=True)
    return p.parse_args()


def main():
    args = arguments()
    root = Path(args.dataset)
    scene_files = sorted(root.glob(f"batches/{args.split}_*_scenes.json"))
    records, skipped = [], 0
    for path in scene_files:
        if not path.with_name(path.name.replace("_scenes.json", ".done")).exists():
            skipped += 1
            continue
        for scene in json.loads(path.read_text())["scenes"]:
            records.append({
                "image_index": int(scene["image_index"]),
                "image_path": f"images/{args.split}/{scene['image_filename']}",
                "world": {
                    "objects": [
                        {"id": index + 1, **{key: obj[key] for key in ATTRIBUTES}}
                        for index, obj in enumerate(scene["objects"])
                    ],
                    "relationships": scene.get("relationships", {}),
                    "rule_set": scene.get("world_rule_set"),
                },
            })
    records.sort(key=lambda r: r["image_index"])
    indices = [r["image_index"] for r in records]
    if len(set(indices)) != len(indices):
        raise RuntimeError("duplicate image_index values in the scene files")
    if args.train_size + args.val_size > len(records):
        raise RuntimeError(
            f"requested {args.train_size} + {args.val_size} images but only {len(records)} are rendered"
        )
    # Validation takes the highest indices, training the lowest, with the
    # remaining middle indices left unused so the two never touch.
    val = records[-args.val_size:]
    train = records[:args.train_size]
    for rows, out in ((train, args.train_out), (val, args.val_out)):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    print(json.dumps({
        "scene_files": len(scene_files), "skipped_unfinished_batches": skipped,
        "records_indexed": len(records), "train": len(train), "val": len(val),
        "train_index_range": [train[0]["image_index"], train[-1]["image_index"]],
        "val_index_range": [val[0]["image_index"], val[-1]["image_index"]],
        "unused_middle_images": len(records) - len(train) - len(val),
    }, indent=2))


if __name__ == "__main__":
    main()
