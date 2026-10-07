"""Render unconditional all-mask image samples for pretrained CLEVR checkpoints.

No caption, reference image, or image token is passed to a model.  Each sample
starts from a 16x24 image-token segment whose every eligible content position
is replaced with the mask token before the first forward pass.
"""

from __future__ import annotations

import argparse
import html
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from data import ClevrTextTokenizer, MultimodalCollator
from multimodal_diffusion import generate_conditioned_images
from train_multimodal import build_model, load_vqvae, move_batch


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower()


def save_image(image: torch.Tensor, path: Path) -> None:
    pixels = ((image.detach().float().cpu().clamp(-1, 1) + 1) / 2)
    array = (pixels.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def render_report(output_dir: Path, results: dict, checkpoint_paths: dict, cli) -> None:
    names = list(checkpoint_paths)
    headers = "".join(f"<th>{html.escape(name)}</th>" for name in names)
    sections = []
    for order in ("confidence", "random"):
        rows = []
        for sample_index in range(cli.num_samples):
            cells = "".join(
                f'<td><img src="{html.escape(results[order][name][sample_index])}"><div>{html.escape(name)}</div></td>'
                for name in names
            )
            rows.append(f"<tr><td>{sample_index + 1}</td>{cells}</tr>")
        sections.append(
            f"<h2>{order.title()} reveal order</h2>"
            f"<table><thead><tr><th>Independent sample</th>{headers}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
    checkpoint_list = "".join(
        f"<li><strong>{html.escape(name)}</strong>: <code>{html.escape(path)}</code></li>"
        for name, path in checkpoint_paths.items()
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Pretraining marginal samples</title>
<style>
body {{ font-family: sans-serif; margin: 28px; background: #101318; color: #e8ecf1; min-width: 1800px; }}
table {{ border-collapse: collapse; margin: 18px 0 42px; }} th,td {{ border: 1px solid #343b45; padding: 8px; vertical-align: top; background: #161b22; }} th {{ background: #1d232c; }}
img {{ width: 160px; display: block; }} code {{ color: #9ecbff; }}
.note {{ border: 1px solid #6a5815; background: #28220f; padding: 12px; max-width: 1350px; line-height: 1.45; }}
</style></head><body>
<h1>Pretrained checkpoints: unconditional image marginal samples</h1>
<p class="note"><strong>Input for every cell:</strong> an image-only sequence with all 384 image-code positions masked. No text segment exists, no ground-truth image code is supplied, and no reference image is displayed. Temperature={cli.temperature:g}; {cli.num_steps} denoising steps; each sample is decoded only after generation. “Confidence” and “random” change the reveal-position criterion, not the initial masked input.</p>
<p class="note">This is qualitative evidence that a checkpoint learned the image marginal distribution. It is not evidence of text–image correspondence; that requires the matched-versus-shuffled tests.</p>
<h2>Checkpoints</h2><ul>{checkpoint_list}</ul>
{''.join(sections)}
</body></html>"""
    (output_dir / "index.html").write_text(document)


@torch.no_grad()
def main():
    cli = arguments()
    if cli.num_samples < 1 or cli.num_steps < 1:
        raise ValueError("num-samples and num-steps must be positive")
    specifications = [value.split("=", 1) for value in cli.checkpoint]
    if len({name for name, _ in specifications}) != len(specifications):
        raise ValueError("Checkpoint names must be unique")
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = {name: path for name, path in specifications}
    results = {order: {} for order in ("confidence", "random")}

    for model_index, (name, checkpoint_path) in enumerate(specifications):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        args = Namespace(**dict(checkpoint["args"]))
        tokenizer = ClevrTextTokenizer(checkpoint["text_vocabulary"])
        collator = MultimodalCollator(tokenizer, args.num_image_codes, args.max_text_length)
        model, _ = build_model(args, len(tokenizer))
        model.load_state_dict(checkpoint["model"])
        model.to(cli.device).eval()
        vqvae = load_vqvae(args, cli.device)
        # Zeros only establish the fixed token-grid shape.  The generation helper
        # masks every eligible image position before the first model forward.
        examples = [
            {"kind": "image", "image_tokens": torch.zeros(args.grid_size, dtype=torch.long), "pair_index": index}
            for index in range(cli.num_samples)
        ]
        batch = move_batch(collator(examples), cli.device)
        for order_index, order in enumerate(("confidence", "random")):
            # Same per-order RNG stream for every checkpoint; output differences
            # come from checkpoint distributions and reveal-order policy.
            torch.manual_seed(cli.seed + order_index)
            tokens = generate_conditioned_images(
                model, batch, collator.image_offset, args.num_image_codes,
                tokenizer.mask_id, cli.num_steps, cli.temperature, order,
            ).view(-1, *args.grid_size)
            images = vqvae.decode_from_indices(tokens)
            paths = []
            for sample_index, image in enumerate(images):
                relative = f"{safe_name(name)}_{order}_{sample_index}.png"
                save_image(image, output_dir / relative)
                paths.append(relative)
            results[order][name] = paths
            print(f"{name} {order}: generated {len(paths)} all-mask image samples", flush=True)
        del model, vqvae, checkpoint
        if cli.device.startswith("cuda"):
            torch.cuda.empty_cache()

    metadata = {
        "input": "image-only; all 16x24 eligible image-code positions masked; no text or reference tokens",
        "num_steps": cli.num_steps,
        "temperature": cli.temperature,
        "seed": cli.seed,
        "orders": ["confidence", "random"],
        "checkpoints": checkpoint_paths,
        "results": results,
    }
    (output_dir / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    render_report(output_dir, results, checkpoint_paths, cli)
    print(f"report written to {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
