"""Layerwise paired retrieval from shared-only and private-only LoRA readouts.

Tri-LoRA models always execute a normal full forward, so shared and private
branches both contribute to every hidden state. Hooks only observe each
branch's delta. For each block, token-pooled adapter deltas are adaptively
reduced to d_model and averaged across qkv, attention output, and both MLP
adapters. Retrieval uses only that branch-specific pooled vector. Dense models
use the pooled residual output of each block as a branch-free reference.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from analyze_paired_representations import checkpoint_args
from data import ClevrMultimodalDataset, ClevrTextTokenizer, MultimodalCollator
from models import TriLoRALinear
from train_multimodal import build_model


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--caption-field", default="caption_human")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def single_modality_batch(paired_batch, modality_id, route_id, device):
    keep = paired_batch["attention_mask"] & paired_batch["modality_ids"].eq(modality_id)
    batch_size = keep.size(0)
    lengths = keep.sum(dim=1)
    max_length = int(lengths.max())
    single = {
        "input_ids": torch.zeros((batch_size, max_length), dtype=torch.long, device=device),
        "attention_mask": torch.zeros((batch_size, max_length), dtype=torch.bool, device=device),
        "position_ids": torch.zeros((batch_size, max_length), dtype=torch.long, device=device),
        "modality_ids": torch.zeros((batch_size, max_length), dtype=torch.long, device=device),
        "eligible_mask": torch.zeros((batch_size, max_length), dtype=torch.bool, device=device),
        "route_ids": torch.full((batch_size,), route_id, dtype=torch.long, device=device),
    }
    for row in range(batch_size):
        length = int(lengths[row])
        for field in ("input_ids", "position_ids", "modality_ids", "eligible_mask"):
            single[field][row, :length] = paired_batch[field][row, keep[row]]
        single["attention_mask"][row, :length] = True
    return single


def reduce_and_pool(activation, eligible_mask, width):
    weights = eligible_mask.to(activation.dtype).unsqueeze(-1)
    pooled = (activation * weights).sum(1) / weights.sum(1).clamp_min(1)
    if pooled.shape[-1] != width:
        pooled = F.adaptive_avg_pool1d(pooled.unsqueeze(1), width).squeeze(1)
    return pooled.float()


@torch.no_grad()
def encode_model(model, loader, device):
    width = model.token_embed.embedding_dim
    is_tri = any(isinstance(module, TriLoRALinear) for module in model.modules())
    vectors = defaultdict(lambda: defaultdict(list))
    state = {"shared": {}, "private": {}, "blocks": {}}
    current_private_branch = {"name": "text"}
    hooks = []

    if is_tri:
        for module_name, module in model.named_modules():
            match = re.fullmatch(r"blocks\.(\d+)\..+", module_name)
            if not isinstance(module, TriLoRALinear) or match is None:
                continue
            layer = int(match.group(1))
            def pre_hook(mod, inputs, name=module_name, layer_index=layer):
                x = inputs[0].detach()
                state["shared"][(layer_index, name)] = mod._delta(x, "shared").detach()
                branch = current_private_branch["name"]
                state["private"][(layer_index, name)] = mod._delta(x, branch).detach()
            hooks.append(module.register_forward_pre_hook(pre_hook))
    else:
        for layer, block in enumerate(model.blocks):
            def block_hook(_module, _inputs, output, layer_index=layer):
                state["blocks"][layer_index] = output.detach()
            hooks.append(block.register_forward_hook(block_hook))

    model.eval()
    try:
        for cpu_batch in loader:
            paired_batch = {key: value.to(device) for key, value in cpu_batch.items()}
            for modality, modality_id, route_id, private_branch in (
                ("text", 1, 0, "text"), ("image", 2, 1, "image")
            ):
                single = single_modality_batch(paired_batch, modality_id, route_id, device)
                current_private_branch["name"] = private_branch
                state["shared"].clear(); state["private"].clear(); state["blocks"].clear()
                model(
                    single["input_ids"], single["attention_mask"], single["position_ids"],
                    single["modality_ids"], single["route_ids"],
                )
                if is_tri:
                    for branch in ("shared", "private"):
                        per_layer = defaultdict(list)
                        for (layer, _name), activation in state[branch].items():
                            per_layer[layer].append(
                                reduce_and_pool(activation, single["eligible_mask"], width)
                            )
                        if sorted(per_layer) != list(range(len(model.blocks))):
                            raise RuntimeError(f"Missing {branch} activations: {sorted(per_layer)}")
                        for layer, activations in per_layer.items():
                            if len(activations) != 4:
                                raise RuntimeError(
                                    f"Expected four {branch} adapters in layer {layer}, got {len(activations)}"
                                )
                            vectors[branch][(layer, modality)].append(
                                torch.stack(activations).mean(0).cpu()
                            )
                else:
                    weights = single["eligible_mask"].float().unsqueeze(-1)
                    for layer, activation in state["blocks"].items():
                        pooled = (activation.float() * weights).sum(1) / weights.sum(1).clamp_min(1)
                        vectors["dense_reference"][(layer, modality)].append(pooled.cpu())
    finally:
        for hook in hooks:
            hook.remove()
    return {
        branch: {
            key: torch.cat(parts)
            for key, parts in values.items()
        }
        for branch, values in vectors.items()
    }


def recall_metrics(text, image):
    text = F.normalize(text - text.mean(0, keepdim=True), dim=1)
    image = F.normalize(image - image.mean(0, keepdim=True), dim=1)
    target = torch.arange(len(text))
    def direction(left, right):
        ranks = (left @ right.T).argsort(dim=1, descending=True).eq(target[:, None]).nonzero()[:, 1]
        return {f"r@{k}": float(ranks.lt(k).float().mean()) for k in (1, 5, 10)}
    text_to_image = direction(text, image)
    image_to_text = direction(image, text)
    return {
        "text_to_image": text_to_image,
        "image_to_text": image_to_text,
        "bidirectional_mean": {
            key: (text_to_image[key] + image_to_text[key]) / 2
            for key in text_to_image
        },
    }


def main():
    args = arguments()
    specs = [value.split("=", 1) for value in args.checkpoint]
    first = torch.load(specs[0][1], map_location="cpu", weights_only=False)
    tokenizer = ClevrTextTokenizer(first["text_vocabulary"])
    first_args = checkpoint_args(first)
    dataset = ClevrMultimodalDataset(
        args.split_dir, args.token_cache, "paired",
        pair_manifest=args.pair_manifest, caption_field=args.caption_field,
    )
    dataset = Subset(dataset, range(min(args.num_samples, len(dataset))))
    collator = MultimodalCollator(tokenizer, first_args.num_image_codes, first_args.max_text_length)
    loader = DataLoader(
        dataset, args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collator, pin_memory=args.device.startswith("cuda"),
    )
    output = {
        "dataset": args.pair_manifest,
        "num_samples": len(dataset),
        "retrieval": "center each modality, L2-normalize, exact paired index target",
        "tri_lora_forward": "full shared+active-private forward; hooks only observe branch deltas",
        "layer_pooling": "eligible-token pool each adapter, adaptive reduce to d_model, average four adapters per block",
        "models": {},
    }
    for index, (name, path) in enumerate(specs):
        payload = first if index == 0 else torch.load(path, map_location="cpu", weights_only=False)
        if payload["text_vocabulary"] != first["text_vocabulary"]:
            raise ValueError(f"Tokenizer vocabulary differs for {name}")
        model_args = checkpoint_args(payload)
        model, _ = build_model(model_args, len(tokenizer))
        model.load_state_dict(payload["model"])
        model.to(args.device)
        is_tri = any(isinstance(module, TriLoRALinear) for module in model.modules())
        if is_tri and not model_args.delete_base_weights:
            raise ValueError(f"{name} is not a no-base checkpoint")
        encoded = encode_model(model, loader, args.device)
        model_output = {
            "checkpoint": {"path": path, "epoch": payload.get("epoch"), "step": payload.get("step")},
            "train_mode": model_args.train_mode,
            "delete_base_weights": bool(model_args.delete_base_weights),
            "representations": {},
        }
        for branch, values in encoded.items():
            layers = {}
            for layer in range(len(model.blocks)):
                layers[f"layer_{layer:02d}"] = recall_metrics(
                    values[(layer, "text")], values[(layer, "image")]
                )
            model_output["representations"][branch] = layers
        output["models"][name] = model_output
        compact = {
            branch: {
                layer: metrics["bidirectional_mean"]
                for layer, metrics in layers.items()
            }
            for branch, layers in model_output["representations"].items()
        }
        print(name, json.dumps(compact, sort_keys=True), flush=True)
        del model, payload
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
