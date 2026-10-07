"""Compare text/image representations on held-out paired CLEVR data.

This is a read-only diagnostic: checkpoints and token caches are never changed.
It separately measures the final residual representation (available for every
model) and the actual final d_model-wide shared Tri-LoRA update.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data import ClevrMultimodalDataset, ClevrTextTokenizer, MultimodalCollator
from models import TriLoRALinear
from train_multimodal import build_model


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--caption-field", default="caption_human")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", default="outputs/paired_representation_analysis/results.json")
    return parser.parse_args()


def checkpoint_specification(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Checkpoint must be NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name, Path(path).expanduser()


def checkpoint_args(payload) -> Namespace:
    values = dict(payload["args"])
    # Defaults make older checkpoints compatible with the current constructor.
    values.setdefault("lora_architecture", "tri")
    # Current ``build_model`` constructs every LoRA checkpoint as strict
    # no-base Tri-LoRA, even though older saved argument dictionaries did not
    # serialize this derived flag.  Match that construction here so analysis
    # tools do not incorrectly reject valid no-base checkpoints.
    values.setdefault("delete_base_weights", values.get("train_mode") == "lora")
    values.setdefault("modality_adversarial", False)
    values.setdefault("modality_discriminator_hidden", None)
    return Namespace(**values)


def last_shared_module(model):
    candidates = [module for module in model.modules() if isinstance(module, TriLoRALinear) and module.base.out_features == model.token_embed.embedding_dim]
    return candidates[-1] if candidates else None


@torch.no_grad()
def encode_modalities(model, loader, device):
    vectors = defaultdict(lambda: {"text": [], "image": []})
    state = {"final": None, "blocks": {}, "shared": {}}

    def final_hook(_module, _inputs, output):
        state["final"] = output.detach()

    def block_hook(layer_index):
        def hook(_module, _inputs, output):
            state["blocks"][layer_index] = output.detach()
        return hook

    def shared_pre_hook(module_name):
        def hook(module, inputs):
            state["shared"][module_name] = module._delta(inputs[0].detach(), "shared").detach()
        return hook

    hooks = [model.norm_out.register_forward_hook(final_hook)]
    for layer_index, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(block_hook(layer_index)))
    shared_module_names = []
    for module_name, module in model.named_modules():
        if isinstance(module, TriLoRALinear):
            shared_module_names.append(module_name)
            hooks.append(module.register_forward_pre_hook(shared_pre_hook(module_name)))
    model.eval()
    try:
        for paired_batch in loader:
            paired_batch = {key: value.to(device) for key, value in paired_batch.items()}
            for modality, modality_id, route_id in (("text", 1, 0), ("image", 2, 1)):
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
                state["final"] = None
                state["blocks"] = {}
                state["shared"] = {}
                model_output = model(
                    single["input_ids"], single["attention_mask"], single["position_ids"],
                    single["modality_ids"], single["route_ids"],
                    return_shared=bool(shared_module_names),
                )
                weights = single["eligible_mask"].float().unsqueeze(-1)
                final = (state["final"].float() * weights).sum(1) / weights.sum(1).clamp_min(1)
                vectors["final"][modality].append(final.cpu())
                for layer_index, activation in state["blocks"].items():
                    pooled = (activation.float() * weights).sum(1) / weights.sum(1).clamp_min(1)
                    vectors[f"block_{layer_index:02d}_final"][modality].append(pooled.cpu())
                for module_name, activation in state["shared"].items():
                    pooled = (activation.float() * weights).sum(1) / weights.sum(1).clamp_min(1)
                    vectors[f"shared_lora/{module_name}"][modality].append(pooled.cpu())
                if shared_module_names:
                    # This is the exact all-adapter representation consumed by
                    # the corrected DANN discriminator: token-pool each shared
                    # delta, reduce it to d_model, then average all adapters.
                    _, shared_aggregate = model_output
                    vectors["shared_lora_aggregate"][modality].append(
                        shared_aggregate.float().cpu()
                    )
    finally:
        for hook in hooks:
            hook.remove()
    result = {
        level: {modality: torch.cat(values) for modality, values in modalities.items()}
        for level, modalities in vectors.items()
    }
    # Backward-compatible alias: this was the only shared activation captured
    # by the original analysis and corresponds to blocks.7.mlp.3.
    if shared_module_names:
        last_name = shared_module_names[-1]
        result["shared_lora"] = result[f"shared_lora/{last_name}"]
    return result


def linear_cka(x, y):
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    cross = x.T @ y
    denominator = (x.T @ x).norm() * (y.T @ y).norm()
    return float(cross.square().sum() / denominator.clamp_min(1e-12))


def representation_metrics(text, image, permutations, seed):
    if text.shape != image.shape:
        raise ValueError(f"Representation shapes differ: {text.shape} and {image.shape}")
    generator = torch.Generator().manual_seed(seed)
    raw_text, raw_image = F.normalize(text, dim=1), F.normalize(image, dim=1)
    centered_text = F.normalize(text - text.mean(0, keepdim=True), dim=1)
    centered_image = F.normalize(image - image.mean(0, keepdim=True), dim=1)

    def retrieval(left, right):
        similarities = left @ right.T
        target = torch.arange(len(left))
        ranks = similarities.argsort(dim=1, descending=True).eq(target[:, None]).nonzero()[:, 1]
        return {f"r@{k}": float(ranks.lt(k).float().mean()) for k in (1, 5, 10)}

    similarity = centered_text @ centered_image.T
    diagonal = float(similarity.diag().mean())
    shuffled = float(similarity[torch.arange(len(text)), torch.arange(len(text)).roll(1)].mean())
    cka = linear_cka(text, image)
    permuted_cka = []
    for _ in range(permutations):
        permuted_cka.append(linear_cka(text, image[torch.randperm(len(image), generator=generator)]))

    text_geometry = centered_text @ centered_text.T
    image_geometry = centered_image @ centered_image.T
    upper = torch.triu_indices(len(text), len(text), offset=1)
    geometry_correlation = float(torch.corrcoef(torch.stack([
        text_geometry[upper[0], upper[1]], image_geometry[upper[0], upper[1]]
    ]))[0, 1])
    combined = torch.cat((text, image), dim=0).float()
    centered_combined = combined - combined.mean(0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered_combined)
    energy = singular_values.square()
    effective_rank = float(energy.sum().square() / energy.square().sum().clamp_min(1e-12))
    numerical_rank = int((singular_values > singular_values.max().clamp_min(1e-12) * 1e-3).sum())
    feature_std = centered_combined.std(dim=0, unbiased=False)
    normalized_combined = F.normalize(combined, dim=1)
    pairwise_cosine = normalized_combined @ normalized_combined.T
    upper_all = torch.triu_indices(len(combined), len(combined), offset=1)
    return {
        "num_pairs": len(text),
        "mean_norm_text": float(text.norm(dim=1).mean()),
        "mean_norm_image": float(image.norm(dim=1).mean()),
        "raw_paired_cosine": float((raw_text * raw_image).sum(1).mean()),
        "centered_paired_cosine": diagonal,
        "centered_shuffled_cosine": shuffled,
        "paired_minus_shuffled_cosine": diagonal - shuffled,
        "text_to_image": retrieval(centered_text, centered_image),
        "image_to_text": retrieval(centered_image, centered_text),
        "linear_cka": cka,
        "permuted_linear_cka_mean": sum(permuted_cka) / len(permuted_cka),
        "linear_cka_gap": cka - sum(permuted_cka) / len(permuted_cka),
        "geometry_correlation": geometry_correlation,
        "collapse_diagnostics": {
            "centered_rms": float(centered_combined.square().mean().sqrt()),
            "mean_feature_std": float(feature_std.mean()),
            "near_zero_std_fraction": float(feature_std.lt(1e-4).float().mean()),
            "effective_rank_participation": effective_rank,
            "numerical_rank_1e-3": numerical_rank,
            "top_singular_energy_fraction": float(energy.max() / energy.sum().clamp_min(1e-12)),
            "mean_off_diagonal_cosine": float(
                pairwise_cosine[upper_all[0], upper_all[1]].mean()
            ),
        },
    }


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    specifications = [checkpoint_specification(value) for value in args.checkpoint]
    first_payload = torch.load(specifications[0][1], map_location="cpu", weights_only=False)
    tokenizer = ClevrTextTokenizer(first_payload["text_vocabulary"])
    first_args = checkpoint_args(first_payload)
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

    results = {
        "dataset": str(args.pair_manifest),
        "token_cache": str(args.token_cache),
        "num_samples": len(dataset),
        "models": {},
    }
    for model_index, (name, path) in enumerate(specifications):
        payload = first_payload if model_index == 0 else torch.load(path, map_location="cpu", weights_only=False)
        if payload["text_vocabulary"] != first_payload["text_vocabulary"]:
            raise ValueError(f"Tokenizer vocabulary differs for {name}")
        model_args = checkpoint_args(payload)
        model, _ = build_model(model_args, len(tokenizer))
        model.load_state_dict(payload["model"])
        model.to(args.device)
        representations = encode_modalities(model, loader, args.device)
        results["models"][name] = {
            level: representation_metrics(values["text"], values["image"], args.permutations, args.seed)
            for level, values in representations.items()
        }
        results["models"][name]["checkpoint"] = {
            "path": str(path), "epoch": payload.get("epoch"), "step": payload.get("step")
        }
        del model, payload

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
