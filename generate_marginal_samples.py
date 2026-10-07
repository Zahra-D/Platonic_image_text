"""Create a visual/textual report of unconditional (marginal) checkpoint samples.

The reference captions are shown only to make the fixed generated caption length
auditable.  They are not supplied to the model.  Image samples are completely
unconditional: every image-code position starts as a mask token.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image
import torch

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
    parser.add_argument("--indices", type=int, nargs="+", default=[7, 81, 421, 1337])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--image-temperature", type=float, default=1.0)
    parser.add_argument("--text-temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


@torch.no_grad()
def generate_text(model, batch, tokenizer, num_steps, temperature):
    ids = batch["input_ids"].clone()
    target = batch["eligible_mask"] & batch["modality_ids"].eq(1)
    ids[target] = tokenizer.mask_id
    remaining = target.clone()
    lexical_start = len(tokenizer.SPECIAL_TOKENS)
    for step in range(max(1, num_steps)):
        logits = model(ids, batch["attention_mask"], batch["position_ids"], batch["modality_ids"], batch["route_ids"])[..., :len(tokenizer)]
        logits[..., :lexical_start] = -torch.inf
        if temperature <= 0:
            sampled = logits.argmax(dim=-1)
            confidence = logits.softmax(dim=-1).amax(dim=-1)
        else:
            probabilities = (logits / temperature).softmax(dim=-1)
            sampled = torch.multinomial(probabilities.flatten(0, 1), 1).view(ids.shape)
            confidence = probabilities.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        for row in range(ids.size(0)):
            candidates = remaining[row].nonzero().flatten()
            if not len(candidates):
                continue
            reveal = min(len(candidates), math.ceil(len(candidates) / max(1, num_steps - step)))
            chosen = candidates[confidence[row, candidates].topk(reveal).indices]
            ids[row, chosen] = sampled[row, chosen]
            remaining[row, chosen] = False
        if not remaining.any():
            break
    return [tokenizer.decode(ids[row, target[row]].tolist()) for row in range(ids.size(0))]


def save_image(tensor, path):
    array = ((tensor.detach().float().cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
    Image.fromarray((array * 255).round().astype(np.uint8)).save(path)


def write_report(output_dir, records, results, paths):
    names = list(results)
    headers = "".join(f"<th>{html.escape(name)}</th>" for name in names)
    text_rows = []
    image_rows = []
    for index, record in enumerate(records):
        text_cells = "".join(f"<td>{html.escape(results[name]['text'][index])}</td>" for name in names)
        image_cells = "".join(
            f'<td><img src="{html.escape(results[name]["images"][index])}"><div>{html.escape(name)}</div></td>'
            for name in names
        )
        text_rows.append(f"<tr><td>{record['_index']}</td><td>{html.escape(record['caption_human'])}</td>{text_cells}</tr>")
        image_rows.append(f"<tr><td>{index + 1}</td>{image_cells}</tr>")
    checkpoints = "".join(f"<li><b>{html.escape(k)}</b>: <code>{html.escape(v)}</code></li>" for k, v in paths.items())
    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>Marginal samples</title>
<style>body{{font-family:sans-serif;max-width:1400px;margin:30px auto;padding:0 20px;background:#101318;color:#e8ecf1}}table{{border-collapse:collapse;width:100%;margin:16px 0 42px}}th,td{{border:1px solid #343b45;padding:10px;vertical-align:top}}th{{background:#1d232c}}td{{background:#161b22}}img{{width:192px;display:block;margin:auto}}.note{{background:#28220f;border:1px solid #6a5815;padding:12px;border-radius:6px;line-height:1.4}}code{{color:#9ecbff}}</style></head><body>
<h1>Pretraining checkpoints: marginal (unconditional) samples</h1><ul>{checkpoints}</ul>
<p class="note">These are qualitative samples, not reconstructions. Every content token of the generated modality is masked. Text samples retain only their source caption's number of content-token slots; the reference text is not given to the model. Images always use the fixed 16×24 token grid.</p>
<h2>Text-only generation</h2><table><thead><tr><th>Slot source index</th><th>Reference caption (not input)</th>{headers}</tr></thead><tbody>{''.join(text_rows)}</tbody></table>
<h2>Image-only generation</h2><table><thead><tr><th>Sample</th>{headers}</tr></thead><tbody>{''.join(image_rows)}</tbody></table></body></html>'''
    (output_dir / "index.html").write_text(page)


@torch.no_grad()
def main():
    cli = parse_args()
    torch.manual_seed(cli.seed)
    device = cli.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_all = read_jsonl(cli.manifest)
    cache = torch.load(cli.token_cache, map_location="cpu", weights_only=False)
    tokens = cache["tokens"] if isinstance(cache, dict) else cache
    records = [dict(records_all[i], _index=i) for i in cli.indices]
    checkpoint_paths, results = {}, {}
    for specification in cli.checkpoints:
        name, checkpoint_path = specification.split("=", 1)
        checkpoint_paths[name] = checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        args = Namespace(**dict(checkpoint["args"]))
        tokenizer = ClevrTextTokenizer(checkpoint["text_vocabulary"])
        collator = MultimodalCollator(tokenizer, args.num_image_codes, args.max_text_length)
        model, _ = build_model(args, len(tokenizer))
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        text_examples = [
            {"kind": "text", "text": record["caption_human"], "pair_index": record["_index"]}
            for record in records
        ]
        captions = generate_text(model, move_batch(collator(text_examples), device), tokenizer, cli.num_steps, cli.text_temperature)
        image_examples = [
            {"kind": "image", "image_tokens": tokens[record["_index"]].long(), "pair_index": record["_index"]}
            for record in records
        ]
        image_batch = move_batch(collator(image_examples), device)
        generated = generate_conditioned_images(model, image_batch, collator.image_offset, args.num_image_codes, tokenizer.mask_id, cli.num_steps, cli.image_temperature, "confidence").view(-1, *args.grid_size)
        vqvae = load_vqvae(args, device)
        images = vqvae.decode_from_indices(generated)
        paths = []
        for index, image in enumerate(images):
            path = f"{name.lower().replace(' ', '_')}_image_only_{index}.png"
            save_image(image, output_dir / path)
            paths.append(path)
        results[name] = {"text": captions, "images": paths}
        print(f"{name}: generated {len(captions)} text-only and image-only samples")
        del model, vqvae, checkpoint
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    (output_dir / "results.json").write_text(json.dumps({"indices": cli.indices, "checkpoints": checkpoint_paths, "results": results}, indent=2) + "\n")
    write_report(output_dir, records, results, checkpoint_paths)
    print(f"report written to {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
