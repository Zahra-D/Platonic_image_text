"""Uni-X-style layerwise text/image gradient-conflict analysis.

For dense models, the measured parameters are the weights of qkv, out_proj,
mlp.0, and mlp.3.  For strict no-base Tri-LoRA models, the measured parameters
are shared_A and shared_B in those same modules.  Modality-private LoRA is
allowed to participate in the forward pass but is never included in the
reported gradient vectors.

The diagnostic evaluates text-only and image-only masked-token objectives on
the same held-out carriers.  Multiple disjoint shards estimate within-modality
gradient reproducibility, which is used to distinguish genuine cross-modal
conflict from ordinary finite-minibatch noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from analyze_paired_representations import checkpoint_args
from data import ClevrMultimodalDataset, ClevrTextTokenizer, MultimodalCollator
from multimodal_diffusion import corrupt_batch, masked_loss
from train_multimodal import build_model


MODULE_GROUPS = {
    "qkv": "attn.qkv",
    "out_proj": "attn.out_proj",
    "mlp_in": "mlp.0",
    "mlp_out": "mlp.3",
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--caption-field", default="caption_human")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def checkpoint_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    return name, path


def one_modality(paired: dict[str, torch.Tensor], modality: str, device: str):
    modality_id, route_id = (1, 0) if modality == "text" else (2, 1)
    paired = {key: value.to(device, non_blocking=True) for key, value in paired.items()}
    keep = paired["attention_mask"] & paired["modality_ids"].eq(modality_id)
    lengths = keep.sum(1)
    width = int(lengths.max())
    result = {
        "input_ids": torch.zeros((len(lengths), width), dtype=torch.long, device=device),
        "attention_mask": torch.zeros((len(lengths), width), dtype=torch.bool, device=device),
        "position_ids": torch.zeros((len(lengths), width), dtype=torch.long, device=device),
        "modality_ids": torch.zeros((len(lengths), width), dtype=torch.long, device=device),
        "eligible_mask": torch.zeros((len(lengths), width), dtype=torch.bool, device=device),
        "route_ids": torch.full((len(lengths),), route_id, dtype=torch.long, device=device),
    }
    for row, length_tensor in enumerate(lengths):
        length = int(length_tensor)
        for field in ("input_ids", "position_ids", "modality_ids", "eligible_mask"):
            result[field][row, :length] = paired[field][row, keep[row]]
        result["attention_mask"][row, :length] = True
    return result


def selected_parameters(model, train_mode: str):
    """Return layer/group/name -> parameter for Uni-X-equivalent projections."""
    selected = defaultdict(lambda: defaultdict(dict))
    dense_pattern = re.compile(
        r"^blocks\.(\d+)\.(attn\.qkv|attn\.out_proj|mlp\.0|mlp\.3)\.weight$"
    )
    lora_pattern = re.compile(
        r"^blocks\.(\d+)\.(attn\.qkv|attn\.out_proj|mlp\.0|mlp\.3)\.(shared_A|shared_B)$"
    )
    inverse_groups = {module: group for group, module in MODULE_GROUPS.items()}
    pattern = dense_pattern if train_mode == "dense" else lora_pattern
    for name, parameter in model.named_parameters():
        match = pattern.match(name)
        if match:
            layer = int(match.group(1))
            group = inverse_groups[match.group(2)]
            selected[layer][group][name] = parameter
    expected_layers = set(range(len(model.blocks)))
    if set(selected) != expected_layers:
        raise RuntimeError(f"Missing target layers: found {sorted(selected)}")
    for layer in expected_layers:
        if set(selected[layer]) != set(MODULE_GROUPS):
            raise RuntimeError(
                f"Layer {layer} target groups are {sorted(selected[layer])}, "
                f"expected {sorted(MODULE_GROUPS)}"
            )
    return selected


def flatten_gradients(selected) -> dict[str, torch.Tensor]:
    result = {}
    for layer, groups in selected.items():
        layer_parts = []
        for group, parameters in groups.items():
            parts = []
            for name in sorted(parameters):
                parameter = parameters[name]
                gradient = parameter.grad
                parts.append(
                    torch.zeros_like(parameter, dtype=torch.float32).flatten().cpu()
                    if gradient is None
                    else gradient.detach().float().flatten().cpu()
                )
            vector = torch.cat(parts)
            result[f"layer_{layer:02d}/{group}"] = vector
            layer_parts.append(vector)
        result[f"layer_{layer:02d}/all"] = torch.cat(layer_parts)
    return result


def collect_gradients(
    model,
    selected,
    loader,
    modality: str,
    mask_token_id: int,
    mask_ratio: float,
    mask_seed: int,
    device: str,
):
    model.eval()
    model.zero_grad(set_to_none=True)
    batches = len(loader)
    if batches < 1:
        raise ValueError("A gradient shard must contain at least one batch")
    total_loss = 0.0
    total_accuracy = 0.0
    for batch_index, paired in enumerate(loader):
        batch = one_modality(paired, modality, device)
        # Reset before corruption so every checkpoint receives identical masks.
        torch.manual_seed(mask_seed + batch_index)
        corrupted, mask, t = corrupt_batch(
            batch["input_ids"], batch["eligible_mask"], batch["modality_ids"],
            mask_token_id, objective=modality, fixed_t=mask_ratio,
        )
        logits = model(
            corrupted, batch["attention_mask"], batch["position_ids"],
            batch["modality_ids"], batch["route_ids"],
        )
        # Fixed t makes 1/t a scalar and therefore irrelevant to cosine; use
        # ordinary CE to keep the reported loss directly interpretable.
        loss = masked_loss(logits, batch["input_ids"], mask, t, False)
        (loss / batches).backward()
        total_loss += loss.detach().item() / batches
        total_accuracy += (
            logits.detach().argmax(-1)[mask].eq(batch["input_ids"][mask]).float().mean().item()
            / batches
        )
    gradients = flatten_gradients(selected)
    model.zero_grad(set_to_none=True)
    return gradients, {"loss": total_loss, "masked_accuracy": total_accuracy}


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() != right.numel():
        raise ValueError("Gradient vectors differ in size")
    denominator = left.norm() * right.norm()
    if denominator <= 1e-20:
        return float("nan")
    return float(torch.dot(left, right) / denominator)


def finite_mean(values):
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def conflict_metrics(text_vectors, image_vectors, key: str):
    text = [gradient[key] for gradient in text_vectors]
    image = [gradient[key] for gradient in image_vectors]
    cross_values = [cosine(a, b) for a in text for b in image]
    text_within = [cosine(text[i], text[j]) for i in range(len(text)) for j in range(i + 1, len(text))]
    image_within = [cosine(image[i], image[j]) for i in range(len(image)) for j in range(i + 1, len(image))]
    cross = finite_mean(cross_values)
    within_text = finite_mean(text_within)
    within_image = finite_mean(image_within)
    within = finite_mean([within_text, within_image])
    adjusted_similarity = cross - within
    return {
        "cross_modal_cosine": cross,
        "within_text_cosine": within_text,
        "within_image_cosine": within_image,
        "within_modality_cosine": within,
        "adjusted_similarity_cross_minus_within": adjusted_similarity,
        "adjusted_conflict_within_minus_cross": -adjusted_similarity,
        "negative_cross_comparison_fraction": finite_mean([
            float(value < 0) for value in cross_values if math.isfinite(value)
        ]),
        "text_gradient_norm_mean": finite_mean([float(value.norm()) for value in text]),
        "image_gradient_norm_mean": finite_mean([float(value.norm()) for value in image]),
    }


def analyze_checkpoint(name, path, loaders, tokenizer, args, first_vocabulary):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["text_vocabulary"] != first_vocabulary:
        raise ValueError(f"Vocabulary mismatch for {name}")
    model_args = checkpoint_args(payload)
    model, _ = build_model(model_args, len(tokenizer))
    model.load_state_dict(payload["model"])
    model.to(args.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = selected_parameters(model, model_args.train_mode)
    for groups in selected.values():
        for parameters in groups.values():
            for parameter in parameters.values():
                parameter.requires_grad_(True)

    text_vectors, image_vectors = [], []
    shard_diagnostics = []
    for shard, loader in enumerate(loaders):
        text_gradient, text_diagnostic = collect_gradients(
            model, selected, loader, "text", tokenizer.mask_id, args.mask_ratio,
            args.seed + shard * 1000 + 100, args.device,
        )
        image_gradient, image_diagnostic = collect_gradients(
            model, selected, loader, "image", tokenizer.mask_id, args.mask_ratio,
            args.seed + shard * 1000 + 200, args.device,
        )
        text_vectors.append(text_gradient)
        image_vectors.append(image_gradient)
        shard_diagnostics.append({
            "shard": shard, "text": text_diagnostic, "image": image_diagnostic
        })
        print(f"{name}: completed gradient shard {shard + 1}/{len(loaders)}", flush=True)

    layers = {}
    for layer in range(len(model.blocks)):
        layers[f"layer_{layer:02d}"] = {
            group: conflict_metrics(
                text_vectors, image_vectors, f"layer_{layer:02d}/{group}"
            )
            for group in (*MODULE_GROUPS.keys(), "all")
        }
    result = {
        "checkpoint": str(path.resolve()),
        "epoch": payload.get("epoch"),
        "step": payload.get("step"),
        "train_mode": model_args.train_mode,
        "shared_only_epochs": getattr(model_args, "shared_only_epochs", 0),
        "sigreg": bool(getattr(model_args, "sigreg", False)),
        "sigreg_per_layer": bool(getattr(model_args, "sigreg_per_layer", False)),
        "measured_parameterization": (
            "dense projection weight" if model_args.train_mode == "dense"
            else "Tri-LoRA shared_A + shared_B (private branches excluded)"
        ),
        "shards": shard_diagnostics,
        "layers": layers,
    }
    del model, text_vectors, image_vectors
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def write_csv(report, output: Path):
    fields = [
        "model", "layer", "group", "cross_modal_cosine", "within_text_cosine",
        "within_image_cosine", "within_modality_cosine",
        "adjusted_similarity_cross_minus_within", "adjusted_conflict_within_minus_cross",
        "negative_cross_comparison_fraction", "text_gradient_norm_mean",
        "image_gradient_norm_mean",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_name, model_result in report["models"].items():
            for layer, layer_result in model_result["layers"].items():
                for group, metrics in layer_result.items():
                    writer.writerow({"model": model_name, "layer": layer, "group": group, **metrics})


def plot_curves(report, output_dir: Path, metric: str, ylabel: str, filename: str):
    plt.figure(figsize=(11, 6))
    for name, result in report["models"].items():
        values = [result["layers"][f"layer_{layer:02d}"]["all"][metric] for layer in range(8)]
        plt.plot(range(8), values, marker="o", linewidth=2, label=name)
    plt.axhline(0, color="black", linewidth=1, alpha=0.6)
    plt.xlabel("Transformer layer")
    plt.ylabel(ylabel)
    plt.xticks(range(8))
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=180)
    plt.close()


def write_markdown(report, output: Path):
    names = list(report["models"])
    lines = [
        "# Layerwise text–image gradient conflict",
        "",
        f"Held-out samples: {report['num_samples']} in {report['num_shards']} disjoint shards; "
        f"masking ratio: {report['mask_ratio']}.",
        "",
        "`cross cosine < 0` is direct gradient opposition. `adjusted conflict = "
        "within-modality cosine − cross-modal cosine`; larger positive values mean the two "
        "modalities disagree more than expected from finite-minibatch noise.",
        "",
        "## Aggregate projection conflict",
        "",
        "| Model | " + " | ".join(f"L{layer}" for layer in range(8)) + " | Least-conflict layers | Most-conflict layers |",
        "|---|" + "---:|" * 8 + "---|---|",
    ]
    all_values = defaultdict(list)
    for name in names:
        result = report["models"][name]
        values = [
            result["layers"][f"layer_{layer:02d}"]["all"]["adjusted_conflict_within_minus_cross"]
            for layer in range(8)
        ]
        for layer, value in enumerate(values):
            if math.isfinite(value):
                all_values[layer].append(value)
        ordered = sorted(range(8), key=lambda layer: values[layer])
        lines.append(
            f"| {name} | " + " | ".join(f"{value:+.3f}" for value in values)
            + " | " + ", ".join(f"L{layer}" for layer in ordered[:3])
            + " | " + ", ".join(f"L{layer}" for layer in ordered[-3:][::-1]) + " |"
        )
    medians = {
        layer: float(torch.tensor(values).median()) for layer, values in all_values.items()
    }
    ordered = sorted(medians, key=medians.get)
    lines.extend([
        "",
        "## Cross-model summary",
        "",
        "Median adjusted conflict by layer: "
        + ", ".join(f"L{layer}={medians[layer]:+.3f}" for layer in range(8)) + ".",
        "",
        "Lowest median conflict: " + ", ".join(f"L{layer}" for layer in ordered[:3]) + ".",
        "Highest median conflict: " + ", ".join(f"L{layer}" for layer in ordered[-3:][::-1]) + ".",
        "",
        "Do not compare the absolute dense and LoRA cosines as if their parameter coordinates "
        "were identical: dense uses W gradients, while LoRA uses shared A/B gradients. The "
        "within-model layer profile and agreement across models are the safer evidence for routing.",
        "",
        "See `layer_gradient_conflict.csv` for qkv, output-projection, MLP-input, and MLP-output breakdowns.",
    ])
    output.write_text("\n".join(lines) + "\n")


def main():
    args = arguments()
    if args.num_shards < 2:
        raise ValueError("num_shards must be at least 2 for within-modality controls")
    if args.num_samples < args.num_shards * args.batch_size:
        raise ValueError("num_samples must provide at least one full batch per shard")
    specs = [checkpoint_spec(value) for value in args.checkpoint]
    first_payload = torch.load(specs[0][1], map_location="cpu", weights_only=False)
    tokenizer = ClevrTextTokenizer(first_payload["text_vocabulary"])
    first_args = checkpoint_args(first_payload)
    dataset = ClevrMultimodalDataset(
        args.split_dir, args.token_cache, "paired", pair_manifest=args.pair_manifest,
        caption_field=args.caption_field,
    )
    generator = torch.Generator().manual_seed(args.seed)
    count = min(args.num_samples, len(dataset))
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    shard_indices = [indices[shard::args.num_shards] for shard in range(args.num_shards)]
    collator = MultimodalCollator(tokenizer, first_args.num_image_codes, first_args.max_text_length)
    loaders = [
        DataLoader(
            Subset(dataset, shard), batch_size=args.batch_size, shuffle=False,
            num_workers=0, collate_fn=collator, pin_memory=args.device.startswith("cuda"),
        )
        for shard in shard_indices
    ]
    report = {
        "method": "Uni-X-style layerwise gradient cosine with within-modality split control",
        "dataset": args.pair_manifest,
        "token_cache": args.token_cache,
        "num_samples": sum(len(shard) for shard in shard_indices),
        "num_shards": args.num_shards,
        "batch_size": args.batch_size,
        "mask_ratio": args.mask_ratio,
        "seed": args.seed,
        "loss": "ordinary masked-token cross entropy at fixed t; text and image processed independently",
        "models": {},
    }
    for name, path in specs:
        report["models"][name] = analyze_checkpoint(
            name, path, loaders, tokenizer, args, first_payload["text_vocabulary"]
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "layer_gradient_conflict.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(report, output_dir / "layer_gradient_conflict.csv")
    plot_curves(
        report, output_dir, "adjusted_conflict_within_minus_cross",
        "Adjusted conflict (within cosine − text/image cosine)",
        "adjusted_gradient_conflict.png",
    )
    plot_curves(
        report, output_dir, "cross_modal_cosine",
        "Raw text–image gradient cosine", "raw_gradient_cosine.png",
    )
    write_markdown(report, output_dir / "REPORT.md")
    print(f"Wrote report to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
