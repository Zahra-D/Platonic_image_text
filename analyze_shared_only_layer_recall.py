"""Layerwise cross-modal recall by pooling only shared Tri-LoRA deltas.

Dense has no branch decomposition, so its reference is the pooled block output.
For Tri-LoRA models, the forward pass is unchanged: base, shared, and the active
private branch all run normally. The reported retrieval vector nevertheless
contains only that layer's shared LoRA deltas (qkv, out_proj, mlp.0, mlp.3),
token-pooled, adaptively reduced to d_model, and averaged within the layer.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from analyze_paired_representations import checkpoint_args, representation_metrics
from data import ClevrMultimodalDataset, ClevrTextTokenizer, MultimodalCollator
from models import TriLoRALinear
from train_multimodal import build_model


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--token-cache", required=True)
    p.add_argument("--caption-field", default="caption_human")
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--permutations", type=int, default=30)
    p.add_argument("--seed", type=int, default=20260825)
    p.add_argument("--output", required=True)
    return p.parse_args()


def single_modality_batch(paired, modality_id, route_id, device):
    paired = {key: value.to(device) for key, value in paired.items()}
    keep = paired["attention_mask"] & paired["modality_ids"].eq(modality_id)
    batch_size = keep.size(0)
    lengths = keep.sum(dim=1)
    max_length = int(lengths.max())
    result = {
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
            result[field][row, :length] = paired[field][row, keep[row]]
        result["attention_mask"][row, :length] = True
    return result


@torch.no_grad()
def encode(model, loader, device):
    is_tri = any(isinstance(module, TriLoRALinear) for module in model.modules())
    vectors = defaultdict(lambda: {"text": [], "image": []})
    state = {"blocks": {}, "shared": defaultdict(list)}
    current_eligible = {"value": None}
    hooks = []

    if is_tri:
        for module_name, module in model.named_modules():
            if not isinstance(module, TriLoRALinear):
                continue
            layer = int(module_name.split(".")[1])

            def shared_hook(layer_index):
                def hook(active_module, inputs):
                    delta = active_module._delta(inputs[0].detach(), "shared").float()
                    weights = current_eligible["value"].float().unsqueeze(-1)
                    pooled = (delta * weights).sum(1) / weights.sum(1).clamp_min(1)
                    if pooled.shape[-1] != model.token_embed.embedding_dim:
                        pooled = F.adaptive_avg_pool1d(
                            pooled.unsqueeze(1), model.token_embed.embedding_dim
                        ).squeeze(1)
                    state["shared"][layer_index].append(pooled.cpu())
                return hook
            hooks.append(module.register_forward_pre_hook(shared_hook(layer)))
    else:
        for layer, block in enumerate(model.blocks):
            def block_hook(layer_index):
                def hook(_module, _inputs, output):
                    state["blocks"][layer_index] = output.detach().float()
                return hook
            hooks.append(block.register_forward_hook(block_hook(layer)))

    model.eval()
    # Keep the real trained forward path. Private branches may shape the input
    # reaching later shared adapters, but private outputs are never pooled into
    # the reported layer representation.
    context = nullcontext()
    try:
        with context:
            for paired in loader:
                for name, modality_id, route_id in (("text", 1, 0), ("image", 2, 1)):
                    batch = single_modality_batch(paired, modality_id, route_id, device)
                    state["blocks"] = {}; state["shared"] = defaultdict(list)
                    current_eligible["value"] = batch["eligible_mask"]
                    model(
                        batch["input_ids"], batch["attention_mask"], batch["position_ids"],
                        batch["modality_ids"], batch["route_ids"],
                    )
                    if is_tri:
                        for layer in range(len(model.blocks)):
                            vectors[f"layer_{layer:02d}_shared_only"][name].append(
                                torch.stack(state["shared"][layer], dim=0).mean(0)
                            )
                    else:
                        weights = batch["eligible_mask"].float().unsqueeze(-1)
                        for layer in range(len(model.blocks)):
                            pooled = (state["blocks"][layer] * weights).sum(1) / weights.sum(1).clamp_min(1)
                            vectors[f"layer_{layer:02d}_dense_block"][name].append(pooled.cpu())
    finally:
        for hook in hooks:
            hook.remove()
    return {
        level: {modality: torch.cat(items) for modality, items in modalities.items()}
        for level, modalities in vectors.items()
    }


def main():
    cli = arguments()
    specs = [item.split("=", 1) for item in cli.checkpoint]
    first = torch.load(specs[0][1], map_location="cpu", weights_only=False)
    tokenizer = ClevrTextTokenizer(first["text_vocabulary"])
    first_args = checkpoint_args(first)
    dataset = ClevrMultimodalDataset(
        cli.dataset_root, cli.token_cache, "paired",
        pair_manifest=cli.manifest, caption_field=cli.caption_field,
    )
    dataset = Subset(dataset, range(min(cli.num_samples, len(dataset))))
    collator = MultimodalCollator(tokenizer, first_args.num_image_codes, first_args.max_text_length)
    loader = DataLoader(dataset, cli.batch_size, shuffle=False, num_workers=0, collate_fn=collator)
    results = {
        "dataset": cli.manifest,
        "num_samples": len(dataset),
        "comparison": {
            "dense": "pooled residual block output",
            "tri_lora": "normal full forward; retrieval vector is the average shared LoRA delta within each layer only",
        },
        "models": {},
    }
    for index, (name, path) in enumerate(specs):
        payload = first if index == 0 else torch.load(path, map_location="cpu", weights_only=False)
        model_args = checkpoint_args(payload)
        model, _ = build_model(model_args, len(tokenizer))
        model.load_state_dict(payload["model"]); model.to(cli.device).eval()
        representations = encode(model, loader, cli.device)
        results["models"][name] = {
            "checkpoint": {"path": path, "epoch": payload.get("epoch"), "step": payload.get("step")},
            "representation": "dense_block" if not any(isinstance(m, TriLoRALinear) for m in model.modules()) else "shared_lora_delta_only_from_full_forward",
            "layers": {
                level: representation_metrics(values["text"], values["image"], cli.permutations, cli.seed)
                for level, values in representations.items()
            },
        }
        print(name, "complete", flush=True)
        del model, payload
    target = Path(cli.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
