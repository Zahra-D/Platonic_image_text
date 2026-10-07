"""Generate text-to-image and image-to-text examples from multimodal checkpoints."""

from __future__ import annotations

import argparse
import html
import json
import math
from argparse import Namespace
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from data import ClevrTextTokenizer, MultimodalCollator
from data.multimodal_dataset import read_jsonl
from multimodal_diffusion import generate_conditioned_images
from train_multimodal import build_model, load_vqvae, move_batch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="NAME=CHECKPOINT entries")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[3, 4, 6, 9])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument(
        "--reveal-orders", nargs="+", choices=["confidence", "random"],
        default=["confidence", "random"],
        help="Reveal-position policies to render independently in both directions.",
    )
    parser.add_argument("--image-temperature", type=float, default=1.0)
    parser.add_argument("--text-temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default=None, help="cuda, cpu, or a CUDA device such as cuda:0")
    return parser.parse_args()


@torch.no_grad()
def generate_conditioned_text(
    model,
    batch,
    tokenizer,
    num_steps,
    temperature=0.0,
    reveal_order="confidence",
):
    """Fill caption-content positions while keeping paired image tokens fixed."""
    ids = batch["input_ids"].clone()
    text_positions = batch["eligible_mask"] & batch["modality_ids"].eq(1)
    ids[text_positions] = tokenizer.mask_id
    remaining = text_positions.clone()
    lexical_start = len(tokenizer.SPECIAL_TOKENS)
    model.eval()

    for step in range(max(1, num_steps)):
        logits = model(
            ids,
            batch["attention_mask"],
            batch["position_ids"],
            batch["modality_ids"],
            batch["route_ids"],
        )[..., : len(tokenizer)]
        logits[..., :lexical_start] = -torch.inf
        if temperature <= 0:
            sampled = logits.argmax(dim=-1)
            confidence = logits.softmax(dim=-1).amax(dim=-1)
        else:
            probabilities = (logits / temperature).softmax(dim=-1)
            sampled = torch.multinomial(
                probabilities.reshape(-1, len(tokenizer)), 1
            ).view(ids.shape)
            confidence = probabilities.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        for row in range(ids.size(0)):
            candidates = remaining[row].nonzero(as_tuple=False).flatten()
            if not len(candidates):
                continue
            steps_left = max(1, num_steps - step)
            reveal_count = min(len(candidates), math.ceil(len(candidates) / steps_left))
            if reveal_order == "confidence":
                chosen = candidates[confidence[row, candidates].topk(reveal_count).indices]
            else:
                chosen = candidates[
                    torch.randperm(len(candidates), device=ids.device)[:reveal_count]
                ]
            ids[row, chosen] = sampled[row, chosen]
            remaining[row, chosen] = False
        if not remaining.any():
            break

    return [tokenizer.decode(ids[row, text_positions[row]].tolist()) for row in range(ids.size(0))]


def save_tensor_image(tensor, path):
    tensor = ((tensor.detach().float().cpu().clamp(-1, 1) + 1) / 2)
    array = (tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def render_report(output_dir, records, results, checkpoint_paths, orders):
    model_names = list(results)
    rows_text_to_image = []
    rows_image_to_text = []
    for sample_index, record in enumerate(records):
        prompt = html.escape(record["caption_human"])
        generated_cells = "".join(
            f'<td><img src="{html.escape(results[name]["text_to_image"][order][sample_index])}"><div>{html.escape(name)} · {order}</div></td>'
            for name in model_names for order in orders
        )
        rows_text_to_image.append(
            f'<tr><td><img src="source_{sample_index}.png"><div>Reference image</div></td>'
            f'<td class="prompt"><strong>Pair index {record["_index"]}</strong><br>{prompt}</td>{generated_cells}</tr>'
        )
        caption_cells = "".join(
            f'<td><strong>{html.escape(name)} · {order}</strong><br>{html.escape(results[name]["image_to_text"][order][sample_index])}</td>'
            for name in model_names for order in orders
        )
        rows_image_to_text.append(
            f'<tr><td><img src="source_{sample_index}.png"><div>Pair index {record["_index"]}</div></td>'
            f'<td class="prompt">{prompt}</td>{caption_cells}</tr>'
        )

    headers = "".join(f"<th>{html.escape(name)} · {order}</th>" for name in model_names for order in orders)
    checkpoint_list = "".join(
        f"<li><strong>{html.escape(name)}</strong>: <code>{html.escape(checkpoint_paths[name])}</code></li>"
        for name in model_names
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Instruction-tuned bidirectional samples</title>
<style>
body {{ font-family: sans-serif; max-width: 1500px; margin: 30px auto; padding: 0 20px; background: #101318; color: #e8ecf1; }}
table {{ width: 100%; border-collapse: collapse; margin-bottom: 42px; }}
th, td {{ border: 1px solid #343b45; padding: 10px; vertical-align: top; }}
th {{ background: #1d232c; }} td {{ background: #161b22; }}
img {{ width: 192px; image-rendering: auto; display: block; margin: auto; }}
.prompt {{ min-width: 360px; line-height: 1.4; }} code {{ color: #9ecbff; }}
.note {{ background: #28220f; border: 1px solid #6a5815; padding: 12px; border-radius: 6px; }}
</style></head><body>
<h1>Instruction-tuned models: bidirectional samples</h1>
<ul>{checkpoint_list}</ul>
<h2>Text → image</h2>
<table><thead><tr><th>Reference image</th><th>Input text</th>{headers}</tr></thead><tbody>{''.join(rows_text_to_image)}</tbody></table>
<h2>Image → text</h2>
<p class="note">The image is the only semantic input. The ground-truth caption is fully masked, but its number of token slots is retained because EOS was not trained as a denoising target in these checkpoints. Both reveal policies are rendered.</p>
<table><thead><tr><th>Input image</th><th>Reference caption</th>{headers}</tr></thead><tbody>{''.join(rows_image_to_text)}</tbody></table>
</body></html>"""
    (output_dir / "index.html").write_text(document)


@torch.no_grad()
def main():
    cli = parse_args()
    torch.manual_seed(cli.seed)
    device = cli.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(cli.dataset_root)
    all_records = read_jsonl(cli.manifest)
    token_payload = torch.load(cli.token_cache, map_location="cpu", weights_only=False)
    image_tokens = token_payload["tokens"] if isinstance(token_payload, dict) else token_payload
    records = []
    for output_index, source_index in enumerate(cli.indices):
        record = dict(all_records[source_index])
        record["_index"] = source_index
        records.append(record)
        with Image.open(dataset_root / record["image_path"]) as source:
            source.convert("RGB").save(output_dir / f"source_{output_index}.png")

    checkpoint_paths = {}
    results = {}
    for checkpoint_spec in cli.checkpoints:
        name, checkpoint_path = checkpoint_spec.split("=", 1)
        checkpoint_paths[name] = checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        args = Namespace(**dict(checkpoint["args"]))
        tokenizer = ClevrTextTokenizer(checkpoint["text_vocabulary"])
        collator = MultimodalCollator(tokenizer, args.num_image_codes, args.max_text_length)
        model, _ = build_model(args, len(tokenizer))
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        vqvae = load_vqvae(args, device)

        examples = [
            {
                "kind": "paired",
                "text": record["caption_human"],
                "image_tokens": image_tokens[record["_index"]].long(),
                "pair_index": record["_index"],
            }
            for record in records
        ]
        image_batch = move_batch(collator(examples), device)
        safe_name = name.lower().replace(" ", "_")
        image_paths = {}
        captions = {}
        for order_index, order in enumerate(cli.reveal_orders):
            torch.manual_seed(cli.seed + order_index)
            image_batch = move_batch(collator(examples), device)
            generated_tokens = generate_conditioned_images(
                model,
                image_batch,
                collator.image_offset,
                args.num_image_codes,
                tokenizer.mask_id,
                cli.num_steps,
                cli.image_temperature,
                order,
            ).view(-1, *args.grid_size)
            generated_images = vqvae.decode_from_indices(generated_tokens)
            image_paths[order] = []
            for sample_index, generated_image in enumerate(generated_images):
                relative_path = f"{safe_name}_text_to_image_{order}_{sample_index}.png"
                save_tensor_image(generated_image, output_dir / relative_path)
                image_paths[order].append(relative_path)

            text_batch = move_batch(collator(examples), device)
            captions[order] = generate_conditioned_text(
                model, text_batch, tokenizer, cli.num_steps, cli.text_temperature, order,
            )
        results[name] = {"text_to_image": image_paths, "image_to_text": captions}
        print(f"{name}: generated {len(cli.reveal_orders)} reveal policies in both directions")
        del model, vqvae, checkpoint
        torch.cuda.empty_cache()

    payload = {
        "dataset_root": str(dataset_root),
        "manifest": cli.manifest,
        "indices": cli.indices,
        "checkpoints": checkpoint_paths,
        "samples": [
            {
                "index": record["_index"],
                "image_path": record["image_path"],
                "reference_caption": record["caption_human"],
                "generated": {
                    name: {
                        "text_to_image": {
                            order: results[name]["text_to_image"][order][sample_index]
                            for order in cli.reveal_orders
                        },
                        "image_to_text": {
                            order: results[name]["image_to_text"][order][sample_index]
                            for order in cli.reveal_orders
                        },
                    }
                    for name in results
                },
            }
            for sample_index, record in enumerate(records)
        ],
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    render_report(output_dir, records, results, checkpoint_paths, cli.reveal_orders)
    print(f"report written to {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
