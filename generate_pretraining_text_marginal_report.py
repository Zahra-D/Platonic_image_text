"""Render all-mask, text-only marginal samples for pretrained CLEVR checkpoints.

The model uses a fixed text-token scaffold: <text>, <bos>, N fully masked
lexical positions, <eos>.  No caption or image is supplied.  Because EOS is
not predicted by this architecture, N is an externally chosen slot count rather
than a generated caption length.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from argparse import Namespace
from pathlib import Path

import torch

from data import ClevrTextTokenizer, MultimodalCollator
from train_multimodal import build_model, move_batch


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--slot-counts", type=int, nargs="+", default=[8, 12, 16, 20])
    parser.add_argument("--num-steps", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def generate_text(model, batch, tokenizer, num_steps, temperature, reveal_order):
    ids = batch["input_ids"].clone()
    target = batch["eligible_mask"] & batch["modality_ids"].eq(1)
    ids[target] = tokenizer.mask_id
    remaining = target.clone()
    lexical_start = len(tokenizer.SPECIAL_TOKENS)
    for step in range(max(1, num_steps)):
        logits = model(
            ids, batch["attention_mask"], batch["position_ids"],
            batch["modality_ids"], batch["route_ids"],
        )[..., :len(tokenizer)]
        logits[..., :lexical_start] = -torch.inf
        if temperature <= 0:
            sampled = logits.argmax(dim=-1)
            confidence = logits.softmax(dim=-1).amax(dim=-1)
        else:
            probabilities = (logits / temperature).softmax(dim=-1)
            sampled = torch.multinomial(probabilities.flatten(0, 1), 1).view(ids.shape)
            confidence = probabilities.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        for row in range(ids.size(0)):
            candidates = remaining[row].nonzero(as_tuple=False).flatten()
            if not len(candidates):
                continue
            reveal = min(len(candidates), math.ceil(len(candidates) / max(1, num_steps - step)))
            if reveal_order == "confidence":
                chosen = candidates[confidence[row, candidates].topk(reveal).indices]
            else:
                chosen = candidates[torch.randperm(len(candidates), device=ids.device)[:reveal]]
            ids[row, chosen] = sampled[row, chosen]
            remaining[row, chosen] = False
        if not remaining.any():
            break
    return [tokenizer.decode(ids[row, target[row]].tolist()) for row in range(ids.size(0))]


def render_report(output_dir: Path, results: dict, checkpoint_paths: dict, cli) -> None:
    names = list(checkpoint_paths)
    headers = "".join(f"<th>{html.escape(name)}</th>" for name in names)
    sections = []
    for order in ("confidence", "random"):
        rows = []
        for index, slots in enumerate(cli.slot_counts):
            cells = "".join(
                f"<td>{html.escape(results[order][name][index])}</td>" for name in names
            )
            rows.append(f"<tr><td>{slots}</td>{cells}</tr>")
        sections.append(
            f"<h2>{order.title()} reveal order</h2>"
            f"<table><thead><tr><th>Masked lexical slots</th>{headers}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
    checkpoint_list = "".join(
        f"<li><strong>{html.escape(name)}</strong>: <code>{html.escape(path)}</code></li>"
        for name, path in checkpoint_paths.items()
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Pretraining text marginal samples</title>
<style>
body {{ font-family: sans-serif; margin: 28px; background: #101318; color: #e8ecf1; min-width: 1800px; }}
table {{ border-collapse: collapse; margin: 18px 0 42px; width: 100%; }} th,td {{ border: 1px solid #343b45; padding: 10px; vertical-align: top; background: #161b22; line-height: 1.4; }} th {{ background: #1d232c; }} code {{ color: #9ecbff; }}
.note {{ border: 1px solid #6a5815; background: #28220f; padding: 12px; max-width: 1350px; line-height: 1.45; }}
</style></head><body>
<h1>Pretrained checkpoints: unconditional text marginal samples</h1>
<p class="note"><strong>Input for every cell:</strong> <code>&lt;text&gt; &lt;bos&gt; [MASK] × N &lt;eos&gt;</code>. No reference caption, image, or image token is supplied. Temperature={cli.temperature:g}; {cli.num_steps} denoising steps. “Confidence” and “random” alter only the reveal-position criterion.</p>
<p class="note"><strong>Length limitation:</strong> N is externally fixed because this masked-token architecture does not generate EOS position. This evaluates lexical/syntactic text marginal quality conditional on a chosen number of content slots—not the caption-length distribution.</p>
<p class="note">This is qualitative marginal evidence only. It cannot demonstrate text–image correspondence; matched-versus-shuffled evaluation is required for that.</p>
<h2>Checkpoints</h2><ul>{checkpoint_list}</ul>
{''.join(sections)}
</body></html>"""
    (output_dir / "index.html").write_text(document)


@torch.no_grad()
def main():
    cli = arguments()
    if not cli.slot_counts or min(cli.slot_counts) < 1 or cli.num_steps < 1:
        raise ValueError("slot-counts and num-steps must be positive")
    specifications = [value.split("=", 1) for value in cli.checkpoint]
    if len({name for name, _ in specifications}) != len(specifications):
        raise ValueError("Checkpoint names must be unique")
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = {name: path for name, path in specifications}
    results = {order: {} for order in ("confidence", "random")}

    for name, checkpoint_path in specifications:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        args = Namespace(**dict(checkpoint["args"]))
        tokenizer = ClevrTextTokenizer(checkpoint["text_vocabulary"])
        collator = MultimodalCollator(tokenizer, args.num_image_codes, args.max_text_length)
        model, _ = build_model(args, len(tokenizer))
        model.load_state_dict(checkpoint["model"])
        model.to(cli.device).eval()
        filler = next(token for token in tokenizer.vocabulary if token not in tokenizer.SPECIAL_TOKENS)
        examples = [
            {"kind": "text", "text": " ".join([filler] * slots), "pair_index": index}
            for index, slots in enumerate(cli.slot_counts)
        ]
        batch = move_batch(collator(examples), cli.device)
        for order_index, order in enumerate(("confidence", "random")):
            torch.manual_seed(cli.seed + order_index)
            results[order][name] = generate_text(
                model, batch, tokenizer, cli.num_steps, cli.temperature, order
            )
            print(f"{name} {order}: generated {len(cli.slot_counts)} all-mask text samples", flush=True)
        del model, checkpoint
        if cli.device.startswith("cuda"):
            torch.cuda.empty_cache()

    metadata = {
        "input": "text-only; <text> <bos> N masked lexical slots <eos>; no caption or image supplied",
        "slot_counts": cli.slot_counts,
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
