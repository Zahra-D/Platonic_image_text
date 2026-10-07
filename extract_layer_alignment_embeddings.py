"""Extract per-layer text/image representations for modality-alignment visualization.

Read-only diagnostic. For each checkpoint and each layer level (every
Transformer block output, the final residual, and — for Tri-LoRA checkpoints —
the pooled shared branch and each modality's own pooled private branch), this
pulls paired held-out text and image representations, then reduces them with
PCA (3D, with the 2D view as its first two components so the projections stay
consistent) and t-SNE (2D) for plotting. Quantitative alignment metrics
(linear CKA, paired-vs-shuffled cosine gap, cross-modal retrieval) are
attached alongside each layer's points so a viewer isn't reading geometry
blind.

The shared and private branch representations use the identical pooling
recipe (attention-mask-weighted token pooling, adaptive-pooled to d_model
where an adapter's output width differs, averaged across every Tri-LoRA
module) so the two are directly comparable: the shared branch is expected to
show cross-modal alignment, the private branch is expected not to, since only
the shared branch feeds the modality-adversarial discriminator.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

from analyze_paired_representations import (
    checkpoint_args,
    checkpoint_specification,
    representation_metrics,
)
from data import ClevrMultimodalDataset, ClevrTextTokenizer, MultimodalCollator
from models import TriLoRALinear
from train_multimodal import build_model

LEVEL_ORDER = (
    [f"block_{i:02d}_final" for i in range(8)]
    + ["final", "shared_lora_aggregate", "private_lora_aggregate", "branch_combined"]
)
LEVEL_LABELS = {
    **{f"block_{i:02d}_final": f"Block {i}" for i in range(8)},
    "final": "Final (post-norm)",
    "shared_lora_aggregate": "Shared branch (pooled)",
    "private_lora_aggregate": "Private branch (pooled, own modality)",
    "branch_combined": "Shared vs private (combined)",
}
BRANCH_GROUP_NAMES = ["shared_text", "shared_image", "private_text", "private_image"]


def branch_pooled(delta: torch.Tensor, attention_mask: torch.Tensor, d_model: int) -> torch.Tensor:
    weights = attention_mask.to(delta.dtype).unsqueeze(-1)
    pooled = (delta * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
    if pooled.shape[-1] != d_model:
        pooled = F.adaptive_avg_pool1d(pooled.unsqueeze(1), d_model).squeeze(1)
    return pooled


@torch.no_grad()
def encode_modalities(model, loader, device):
    vectors = defaultdict(lambda: {"text": [], "image": []})
    state = {"final": None, "blocks": {}, "branch": None}

    def final_hook(_module, _inputs, output):
        state["final"] = output.detach()

    def block_hook(layer_index):
        def hook(_module, _inputs, output):
            state["blocks"][layer_index] = output.detach()
        return hook

    tri_lora_modules = [module for module in model.modules() if isinstance(module, TriLoRALinear)]
    d_model = model.token_embed.embedding_dim

    def branch_pre_hook(module):
        def hook(_module, inputs):
            branch = state["branch"]
            if branch is None:
                return
            x = inputs[0].detach()
            branch["shared"].append(branch_pooled(module._delta(x, "shared").detach(), branch["attention_mask"], d_model))
            branch["private"].append(branch_pooled(module._delta(x, branch["modality"]).detach(), branch["attention_mask"], d_model))
        return hook

    hooks = [model.norm_out.register_forward_hook(final_hook)]
    for layer_index, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(block_hook(layer_index)))
    for module in tri_lora_modules:
        hooks.append(module.register_forward_pre_hook(branch_pre_hook(module)))

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
                state["branch"] = (
                    {"attention_mask": single["attention_mask"], "modality": modality, "shared": [], "private": []}
                    if tri_lora_modules else None
                )
                model(
                    single["input_ids"], single["attention_mask"], single["position_ids"],
                    single["modality_ids"], single["route_ids"],
                )
                weights = single["eligible_mask"].float().unsqueeze(-1)
                final = (state["final"].float() * weights).sum(1) / weights.sum(1).clamp_min(1)
                vectors["final"][modality].append(final.cpu())
                for layer_index, activation in state["blocks"].items():
                    pooled = (activation.float() * weights).sum(1) / weights.sum(1).clamp_min(1)
                    vectors[f"block_{layer_index:02d}_final"][modality].append(pooled.cpu())
                if tri_lora_modules:
                    shared = torch.stack(state["branch"]["shared"], dim=0).mean(dim=0)
                    private = torch.stack(state["branch"]["private"], dim=0).mean(dim=0)
                    vectors["shared_lora_aggregate"][modality].append(shared.float().cpu())
                    vectors["private_lora_aggregate"][modality].append(private.float().cpu())
    finally:
        for hook in hooks:
            hook.remove()
    return {
        level: {modality: torch.cat(values) for modality, values in modalities.items()}
        for level, modalities in vectors.items()
    }


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--label", action="append", default=[], metavar="NAME=Display label")
    parser.add_argument("--state", action="append", default=[], metavar="NAME=Run state")
    parser.add_argument("--run-id", action="append", default=[], metavar="NAME=W&B run id")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--caption-field", default="caption_human")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-max-samples", type=int, default=300, help="Per-group cap for the t-SNE subsample; PCA always uses the full --num-samples.")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--decimals", type=int, default=4)
    parser.add_argument("--output", default="outputs/paired_representation_analysis/layer_alignment_embeddings.json")
    return parser.parse_args()


def name_value_map(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        name, _, payload = value.partition("=")
        result[name] = payload
    return result


def round_nested(values: list[list[float]], decimals: int) -> list[list[float]]:
    return np.asarray(values, dtype=np.float64).round(decimals).tolist()


def fit_pca(combined: np.ndarray, seed: int, decimals: int):
    centered = combined - combined.mean(axis=0, keepdims=True)
    n_components = min(3, centered.shape[0], centered.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    projected = pca.fit_transform(centered)
    if n_components < 3:
        pad = np.zeros((projected.shape[0], 3 - n_components))
        projected = np.concatenate([projected, pad], axis=1)
        explained = list(pca.explained_variance_ratio_) + [0.0] * (3 - n_components)
    else:
        explained = list(pca.explained_variance_ratio_)
    return projected, [round(float(v), decimals) for v in explained]


def fit_tsne_subsampled(combined: np.ndarray, group_sizes: list[int], seed: int, perplexity: float, tsne_max: int):
    """Run t-SNE on a random subsample so it stays fast at large N.

    Every group holds the same underlying examples in the same order (index i is
    the same CLEVR example in every group), so a single shared index draw is
    reused across all groups -- picking independently per group would silently
    break the text<->image pairing that the plot's pair-links depend on.
    """
    rng = np.random.default_rng(seed)
    per_group = min(tsne_max, min(group_sizes))
    offsets = np.cumsum([0] + group_sizes)
    chosen = rng.choice(min(group_sizes), size=per_group, replace=False)
    chosen.sort()
    picks = [chosen + offsets[group_index] for group_index in range(len(group_sizes))]
    subsample_idx = np.concatenate(picks)
    subsampled = combined[subsample_idx]
    perplexity = float(min(perplexity, max(5.0, (len(subsampled) - 1) / 3)))
    tsne = TSNE(
        n_components=2, perplexity=perplexity, random_state=seed,
        init="pca", learning_rate="auto",
    )
    projected = tsne.fit_transform(subsampled)
    return [projected[i * per_group:(i + 1) * per_group] for i in range(len(group_sizes))]


def reduce_level(text: torch.Tensor, image: torch.Tensor, seed: int, decimals: int, perplexity: float, tsne_max: int):
    text_np = text.numpy().astype(np.float64)
    image_np = image.numpy().astype(np.float64)
    combined = np.concatenate([text_np, image_np], axis=0)
    n_text = len(text_np)

    projected, explained = fit_pca(combined, seed, decimals)
    tsne_text, tsne_image = fit_tsne_subsampled(combined, [n_text, len(image_np)], seed, perplexity, tsne_max)

    return {
        "pca3d": {
            "text": round_nested(projected[:n_text, :3], decimals),
            "image": round_nested(projected[n_text:, :3], decimals),
        },
        "pca2d": {
            "text": round_nested(projected[:n_text, :2], decimals),
            "image": round_nested(projected[n_text:, :2], decimals),
        },
        "pca_explained_variance": explained,
        "tsne2d": {
            "text": round_nested(tsne_text, decimals),
            "image": round_nested(tsne_image, decimals),
        },
    }


def reduce_combined(shared_text, shared_image, private_text, private_image, seed, decimals, perplexity, tsne_max):
    arrays = [shared_text, shared_image, private_text, private_image]
    sizes = [len(a) for a in arrays]
    combined = np.concatenate([a.numpy().astype(np.float64) for a in arrays], axis=0)

    projected, explained = fit_pca(combined, seed, decimals)
    tsne_parts = fit_tsne_subsampled(combined, sizes, seed, perplexity, tsne_max)

    def split(arr, dims):
        out = {}
        idx = 0
        for name, size in zip(BRANCH_GROUP_NAMES, sizes):
            out[name] = round_nested(arr[idx:idx + size, :dims], decimals)
            idx += size
        return out

    return {
        "pca3d": split(projected, 3),
        "pca2d": split(projected, 2),
        "pca_explained_variance": explained,
        "tsne2d": {name: round_nested(part, decimals) for name, part in zip(BRANCH_GROUP_NAMES, tsne_parts)},
    }


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    labels = name_value_map(args.label)
    states = name_value_map(args.state)
    run_ids = name_value_map(args.run_id)

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

    output = {
        "dataset": str(args.pair_manifest),
        "token_cache": str(args.token_cache),
        "num_samples": len(dataset),
        "level_order": LEVEL_ORDER,
        "level_labels": LEVEL_LABELS,
        "models": {},
    }

    for model_index, (name, path) in enumerate(specifications):
        print(f"[{model_index + 1}/{len(specifications)}] {name} <- {path}")
        payload = first_payload if model_index == 0 else torch.load(path, map_location="cpu", weights_only=False)
        if payload["text_vocabulary"] != first_payload["text_vocabulary"]:
            raise ValueError(f"Tokenizer vocabulary differs for {name}")
        model_args = checkpoint_args(payload)
        model, _ = build_model(model_args, len(tokenizer))
        model.load_state_dict(payload["model"])
        model.to(args.device)

        representations = encode_modalities(model, loader, args.device)
        levels_out = {}
        for level in LEVEL_ORDER:
            if level == "branch_combined":
                continue
            if level not in representations:
                levels_out[level] = {"available": False}
                continue
            text_reps = representations[level]["text"]
            image_reps = representations[level]["image"]
            metrics = representation_metrics(text_reps, image_reps, args.permutations, args.seed)
            reduced = reduce_level(text_reps, image_reps, args.seed, args.decimals, args.tsne_perplexity, args.tsne_max_samples)
            levels_out[level] = {"available": True, "metrics": metrics, **reduced}

        if "shared_lora_aggregate" in representations:
            combined = reduce_combined(
                representations["shared_lora_aggregate"]["text"],
                representations["shared_lora_aggregate"]["image"],
                representations["private_lora_aggregate"]["text"],
                representations["private_lora_aggregate"]["image"],
                args.seed, args.decimals, args.tsne_perplexity, args.tsne_max_samples,
            )
            levels_out["branch_combined"] = {"available": True, **combined}
        else:
            levels_out["branch_combined"] = {"available": False}

        output["models"][name] = {
            "label": labels.get(name, name),
            "state": states.get(name, "unknown"),
            "run_id": run_ids.get(name, ""),
            "checkpoint": {
                "path": str(path), "epoch": payload.get("epoch"), "step": payload.get("step"),
            },
            "levels": levels_out,
        }
        del model, payload
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output) + "\n")
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
