"""Generate CLEVR images conditioned on text using a multimodal checkpoint."""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

import torch

from data import ClevrTextTokenizer, MultimodalCollator
from multimodal_diffusion import generate_conditioned_images
from train_multimodal import build_model, load_vqvae, move_batch
from image_utils import save_image_grid


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/multimodal_generated.png")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--reveal-order", choices=["confidence", "random"], default="confidence")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


@torch.no_grad()
def main():
    cli = parse_args()
    torch.manual_seed(cli.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    saved = dict(checkpoint["args"])
    args = Namespace(**saved)
    tokenizer = ClevrTextTokenizer(checkpoint["text_vocabulary"])
    collator = MultimodalCollator(tokenizer, args.num_image_codes, args.max_text_length)
    model, _ = build_model(args, len(tokenizer))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    examples = [
        {
            "kind": "paired",
            "text": prompt,
            "image_tokens": torch.zeros(args.grid_size, dtype=torch.long),
            "pair_index": index,
        }
        for index, prompt in enumerate(cli.prompts)
    ]
    batch = move_batch(collator(examples), device)
    tokens = generate_conditioned_images(
        model, batch, collator.image_offset, args.num_image_codes, tokenizer.mask_id,
        cli.num_steps, cli.temperature, cli.reveal_order,
    ).view(-1, *args.grid_size)
    vqvae = load_vqvae(args, device)
    images = vqvae.decode_from_indices(tokens)
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image_grid(images, output, nrow=len(images), value_range=(-1, 1))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
