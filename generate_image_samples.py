#!/usr/bin/env python3
"""Unconditional image samples from an image checkpoint, both reveal orders.

Every image-code position starts as a mask token; nothing is conditioned on.
The model fills the 16x24 grid over ``--num-steps`` rounds, and the two reveal
orders differ only in which masked positions are committed each round:

``confidence``  commit the positions the model is most sure about first, which
                is the usual inference rule for a masked-diffusion model;
``random``      commit a random subset, which removes the model's ability to
                sequence its own generation and therefore shows how much the
                sample quality depends on that ordering.

The same seed is used for both, so the grids are directly comparable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import ClevrTextTokenizer, MultimodalCollator
from image_utils import save_image_grid
from multimodal_diffusion import generate_conditioned_images
from train_multimodal import build_model, load_vqvae
from analyze_paired_representations import checkpoint_args


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--reveal-order", nargs="+", default=["confidence", "random"])
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    for spec in args.checkpoint:
        label, path = spec.split("=", 1)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model_args = checkpoint_args(payload)
        tokenizer = ClevrTextTokenizer(payload["text_vocabulary"])
        model, _ = build_model(model_args, len(tokenizer))
        model.load_state_dict(payload["model"], strict=False)
        model.to(device).eval()
        vqvae = load_vqvae(model_args, device)
        collator = MultimodalCollator(tokenizer, model_args.num_image_codes, model_args.max_text_length)
        height, width = model_args.grid_size
        blank = torch.zeros(height, width, dtype=torch.long)
        batch = collator([{"kind": "image", "image_tokens": blank, "pair_index": index}
                          for index in range(args.num_samples)])
        batch = {name: value.to(device) for name, value in batch.items()}
        image_positions = batch["eligible_mask"] & batch["modality_ids"].eq(2)
        batch["input_ids"] = batch["input_ids"].masked_fill(image_positions, tokenizer.mask_id)

        for order in args.reveal_order:
            torch.manual_seed(args.seed)
            with torch.no_grad():
                codes = generate_conditioned_images(
                    model, batch, len(tokenizer), model_args.num_image_codes,
                    tokenizer.mask_id, args.num_steps, args.temperature, order,
                )
                images = vqvae.decode_from_indices(codes.view(-1, height, width))
            destination = out / f"{label}_{order}.png"
            save_image_grid(images, destination, nrow=min(8, args.num_samples), value_range=(-1, 1))
            print(f"{label} [{order}]: wrote {destination} "
                  f"({args.num_samples} samples, {args.num_steps} steps, T={args.temperature})", flush=True)
        del model, vqvae
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
