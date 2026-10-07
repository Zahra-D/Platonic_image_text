#!/usr/bin/env python3
"""Images and image tokens seen by each image checkpoint, own run + inherited init."""
import json, sys, torch
from pathlib import Path

TOKENS_PER_IMAGE = 16 * 24


def info(path, cache={}):
    path = str(path)
    if path not in cache:
        p = torch.load(path, map_location="cpu", weights_only=False)
        a = p["args"]
        per_step = a["batch_size"] * a.get("gradient_accumulation_steps", 1)
        if a.get("objective") == "both":  # half the carriers are text
            per_step //= 2
        images = p["step"] * per_step  # resume keeps counting steps, so this is cumulative
        parent = a.get("init_checkpoint")  # init restarts the step counter
        inherited = info(parent)["cumulative_images"] if parent else 0
        cache[path] = {"checkpoint": path, "step": p["step"], "epochs_done": p["epoch"] + 1,
                       "images_per_step": per_step, "train_manifest": a["train_manifest"],
                       "train_mode": a["train_mode"], "init_from": parent, "resumed_from": a.get("resume"),
                       "run_images": images, "cumulative_images": images + inherited,
                       "cumulative_image_tokens": (images + inherited) * TOKENS_PER_IMAGE}
    return cache[path]


print(json.dumps([info(p) for p in sys.argv[1:]], indent=1))
