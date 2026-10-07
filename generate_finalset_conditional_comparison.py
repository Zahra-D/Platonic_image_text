"""Generate matched text-to-image comparisons for final-set checkpoints.

Every row uses a held-out caption and its reference image for display only.
All 16x24 eligible image-code positions are masked before sampling, so no
ground-truth image token is supplied to any model.  Each requested reveal
order is generated with one code committed per denoising step.
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
from data.multimodal_dataset import read_jsonl
from multimodal_diffusion import generate_conditioned_images
from train_multimodal import build_model, load_vqvae, move_batch


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[3, 4, 6, 9])
    parser.add_argument("--num-steps", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def save_tensor_image(image, path):
    pixels = ((image.detach().float().cpu().clamp(-1, 1) + 1) / 2)
    array = (pixels.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def safe_name(value):
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower()


def render_report(output_dir, records, results, checkpoints, num_steps, temperature):
    model_names = list(checkpoints)
    headers = "".join(f"<th>{html.escape(name)}</th>" for name in model_names)
    sections = []
    for order in ("confidence", "random"):
        rows = []
        for row_index, record in enumerate(records):
            cells = "".join(
                f'<td><img src="{html.escape(results[order][name][row_index])}"><div>{html.escape(name)}</div></td>'
                for name in model_names
            )
            rows.append(
                f'<tr><td><img src="source_{row_index}.png"><div>reference only</div></td>'
                f'<td class="prompt"><strong>Held-out pair {record["_index"]}</strong><br>'
                f'{html.escape(record["caption_human"])}</td>{cells}</tr>'
            )
        sections.append(
            f'<h2>{order.title()} reveal order</h2>'
            f'<table><thead><tr><th>Reference image</th><th>Input caption</th>{headers}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>'
        )
    checkpoint_list = "".join(
        f'<li><strong>{html.escape(name)}</strong>: <code>{html.escape(path)}</code></li>'
        for name, path in checkpoints.items()
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Conditional final-set comparison</title>
<style>
body {{ font-family: sans-serif; margin: 28px; background: #101318; color: #e8ecf1; min-width: 1800px; }}
table {{ border-collapse: collapse; margin: 18px 0 42px; }} th,td {{ border: 1px solid #343b45; padding: 8px; vertical-align: top; background: #161b22; }} th {{ background: #1d232c; }}
img {{ width: 160px; display: block; }} .prompt {{ width: 310px; line-height: 1.35; }} code {{ color: #9ecbff; }}
.note {{ border: 1px solid #6a5815; background: #28220f; padding: 12px; max-width: 1300px; }}
</style></head><body>
<h1>Held-out caption → image conditional comparison</h1>
<p class="note">Each generation starts with every eligible image code masked. The reference image is shown only for human comparison; it is never supplied to the model. Temperature={temperature:g}; {num_steps} denoising steps commit one of the 384 image codes per step.</p>
<ul>{checkpoint_list}</ul>{"".join(sections)}</body></html>"""
    (output_dir / "index.html").write_text(document)


@torch.no_grad()
def main():
    cli = arguments()
    if cli.num_steps < 1:
        raise ValueError("num_steps must be positive")
    specs = [value.split("=", 1) for value in cli.checkpoint]
    if len({name for name, _ in specs}) != len(specs):
        raise ValueError("Checkpoint names must be unique")
    records = read_jsonl(cli.manifest)
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for output_index, source_index in enumerate(cli.indices):
        record = dict(records[source_index])
        record["_index"] = source_index
        selected.append(record)
        with Image.open(Path(cli.dataset_root) / record["image_path"]) as image:
            image.convert("RGB").save(output_dir / f"source_{output_index}.png")

    results = {order: {} for order in ("confidence", "random")}
    checkpoints = {name: path for name, path in specs}
    for model_index, (name, checkpoint_path) in enumerate(specs):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        args = Namespace(**dict(checkpoint["args"]))
        tokenizer = ClevrTextTokenizer(checkpoint["text_vocabulary"])
        collator = MultimodalCollator(tokenizer, args.num_image_codes, args.max_text_length)
        model, _ = build_model(args, len(tokenizer))
        model.load_state_dict(checkpoint["model"])
        model.to(cli.device).eval()
        vqvae = load_vqvae(args, cli.device)
        examples = [
            {
                "kind": "paired",
                "text": record["caption_human"],
                # Shape only: generation masks every eligible image position before its first forward.
                "image_tokens": torch.zeros(args.grid_size, dtype=torch.long),
                "pair_index": record["_index"],
            }
            for record in selected
        ]
        batch = move_batch(collator(examples), cli.device)
        for order_index, order in enumerate(("confidence", "random")):
            # Resetting the seed makes the two orders reproducible and gives every model
            # the same RNG stream for a given order.
            torch.manual_seed(cli.seed + order_index)
            tokens = generate_conditioned_images(
                model, batch, collator.image_offset, args.num_image_codes,
                tokenizer.mask_id, cli.num_steps, cli.temperature, order,
            ).view(-1, *args.grid_size)
            images = vqvae.decode_from_indices(tokens)
            paths = []
            for sample_index, image in enumerate(images):
                relative = f"{safe_name(name)}_{order}_{sample_index}.png"
                save_tensor_image(image, output_dir / relative)
                paths.append(relative)
            results[order][name] = paths
            print(f"{name} {order}: generated {len(paths)} images", flush=True)
        del model, vqvae, checkpoint
        if cli.device.startswith("cuda"):
            torch.cuda.empty_cache()

    metadata = {
        "manifest": cli.manifest,
        "indices": cli.indices,
        "checkpoints": checkpoints,
        "num_steps": cli.num_steps,
        "temperature": cli.temperature,
        "orders": ["confidence", "random"],
        "samples": [
            {"index": record["_index"], "caption": record["caption_human"], "reference": record["image_path"]}
            for record in selected
        ],
    }
    (output_dir / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    render_report(output_dir, selected, results, checkpoints, cli.num_steps, cli.temperature)
    print(f"wrote {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
