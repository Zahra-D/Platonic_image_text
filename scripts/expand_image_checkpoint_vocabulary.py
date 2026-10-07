#!/usr/bin/env python3
"""Give an image-only checkpoint the text vocabulary it never had.

An `image_only` run builds its tokenizer from an empty caption list, so it ends
up with the 7 special tokens and nothing else: its embedding and output head
have 7 + 512 = 519 rows. A multimodal model over the same images plus the 2M
caption corpus needs 170 + 512 = 682. Initializing the second from the first
therefore needs the rows moved into their new positions:

    specials    0..6      ->  0..6        (unchanged)
    image codes 7..518    ->  170..681    (shifted by the word-vocabulary size)
    words       -         ->  7..169      (newly initialized, as in a fresh model)

Everything outside the embedding and head is copied unchanged. The result is a
checkpoint that loads into a multimodal model and starts from the image model's
learned weights everywhere they exist.
"""

from __future__ import annotations

import argparse

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-checkpoint", required=True)
    p.add_argument("--vocabulary-from", required=True,
                   help="Checkpoint whose text_vocabulary the result should adopt.")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20260923)
    args = p.parse_args()

    source = torch.load(args.image_checkpoint, map_location="cpu", weights_only=False)
    target_vocabulary = torch.load(args.vocabulary_from, map_location="cpu", weights_only=False)["text_vocabulary"]
    old_vocabulary = source["text_vocabulary"]
    if list(old_vocabulary) != list(target_vocabulary[:len(old_vocabulary)]):
        raise ValueError("The image checkpoint's tokens are not a prefix of the target vocabulary")
    old_words, new_words = len(old_vocabulary), len(target_vocabulary)
    state = dict(source["model"])
    generator = torch.Generator().manual_seed(args.seed)
    for key in ("token_embed.weight", "head.weight"):
        old = state[key]
        codes = old.shape[0] - old_words
        expanded = torch.empty(new_words + codes, old.shape[1], dtype=old.dtype)
        # Fresh word rows use the same initialization scale as the rest.
        expanded.normal_(0.0, float(old.std()), generator=generator)
        expanded[:old_words] = old[:old_words]
        expanded[new_words:] = old[old_words:]
        state[key] = expanded
        print(f"{key}: {tuple(old.shape)} -> {tuple(expanded.shape)} "
              f"({old_words} specials kept, {codes} image codes moved to {new_words}..{new_words + codes - 1})")
    if "head.bias" in state:
        old = state["head.bias"]
        codes = old.shape[0] - old_words
        expanded = torch.zeros(new_words + codes, dtype=old.dtype)
        expanded[:old_words] = old[:old_words]
        expanded[new_words:] = old[old_words:]
        state["head.bias"] = expanded
    payload = {"model": state, "args": source["args"], "text_vocabulary": target_vocabulary,
               "epoch": source.get("epoch"), "step": source.get("step"),
               "lora_modules": source.get("lora_modules", [])}
    torch.save(payload, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
