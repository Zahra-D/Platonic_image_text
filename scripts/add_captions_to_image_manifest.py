#!/usr/bin/env python3
"""Render a caption for every row of an image manifest, in place of a new file.

Unpaired training needs one manifest carrying both a caption and an image per
row -- the dataset then de-pairs them with a derangement, so the caption is
never seen alongside its own image. The 1.2M image corpus ships without
captions, so this renders one per scene from its `world` with the same
generator the text corpus used.

Row order and `image_path` are preserved exactly, and the token cache's
fingerprint hashes only those, so the existing cache stays valid.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import binding_swap_captions as C


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--caption-generator", default="/home/zd25e122/clevr-dataset-gen_clone/image_generation/generate_human_captions.py")
    p.add_argument("--caption-field", default="caption_human")
    p.add_argument("--seed", type=int, default=20260923)
    args = p.parse_args()

    generator = C.load_generator(args.caption_generator)
    written = 0
    with open(args.manifest) as source, open(args.output, "w") as destination:
        for index, line in enumerate(source):
            record = json.loads(line)
            world = record["world"]
            rng = random.Random(args.seed * 1_000_003 + index)
            objects, relations = len(world["objects"]), len(world.get("relations", []))
            plan = C.make_plan(generator, rng.randrange(2**31), objects, relations)
            record[args.caption_field] = C.render(
                generator, world, plan, rng.sample(range(objects), objects),
                rng.sample(range(relations), relations), rng.randrange(2),
            )
            destination.write(json.dumps(record) + "\n")
            written += 1
            if written % 200_000 == 0:
                print(f"{written} rows", flush=True)
    print(f"wrote {written} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
