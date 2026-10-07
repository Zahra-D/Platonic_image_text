"""Test whether SIGReg-pretrained shared vectors have matching distributions.

This is deliberately separate from paired retrieval: distribution similarity
does not imply that the representation encodes which caption belongs to which
image.  For each checkpoint the extractor uses the exact ``return_shared``
readout consumed by SIGReg during training: attention-mask pool every shared
adapter activation, reduce it to d_model when needed, then average across all
shared adapters in all Transformer blocks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from analyze_paired_representations import checkpoint_args
from data import ClevrMultimodalDataset, ClevrTextTokenizer, MultimodalCollator
from sigreg import gaussianity_diagnostics, sigreg_loss
from train_multimodal import build_model


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--caption-field", default="caption_human")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-slices", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def one_modality(paired: dict[str, torch.Tensor], modality_id: int, route_id: int, device: str):
    paired = {key: value.to(device) for key, value in paired.items()}
    keep = paired["attention_mask"] & paired["modality_ids"].eq(modality_id)
    lengths = keep.sum(1)
    result = {
        "input_ids": torch.zeros((len(lengths), int(lengths.max())), dtype=torch.long, device=device),
        "attention_mask": torch.zeros((len(lengths), int(lengths.max())), dtype=torch.bool, device=device),
        "position_ids": torch.zeros((len(lengths), int(lengths.max())), dtype=torch.long, device=device),
        "modality_ids": torch.zeros((len(lengths), int(lengths.max())), dtype=torch.long, device=device),
        "route_ids": torch.full((len(lengths),), route_id, dtype=torch.long, device=device),
    }
    for row, length in enumerate(lengths.tolist()):
        for field in ("input_ids", "position_ids", "modality_ids"):
            result[field][row, :length] = paired[field][row, keep[row]]
        result["attention_mask"][row, :length] = True
    return result


@torch.no_grad()
def extract_shared(model, loader, device):
    """Return the aggregate readout and each block's own SIGReg readout."""
    output = {"aggregate": {"text": [], "image": []}, "per_layer": {}}
    model.eval()
    for paired in loader:
        for name, modality_id, route_id in (("text", 1, 0), ("image", 2, 1)):
            batch = one_modality(paired, modality_id, route_id, device)
            _, shared, shared_by_layer = model(
                batch["input_ids"], batch["attention_mask"], batch["position_ids"],
                batch["modality_ids"], batch["route_ids"], return_shared=True,
                return_shared_by_layer=True,
            )
            output["aggregate"][name].append(shared.float().cpu())
            for layer, representation in shared_by_layer.items():
                output["per_layer"].setdefault(layer, {"text": [], "image": []})[name].append(
                    representation.float().cpu()
                )
    return {
        "aggregate": {
            modality: torch.cat(parts) for modality, parts in output["aggregate"].items()
        },
        "per_layer": {
            layer: {modality: torch.cat(parts) for modality, parts in modalities.items()}
            for layer, modalities in output["per_layer"].items()
        },
    }


def covariance(x: torch.Tensor):
    centered = x - x.mean(0, keepdim=True)
    return centered.T @ centered / max(1, len(x) - 1)


def sliced_wasserstein(left: torch.Tensor, right: torch.Tensor, directions: torch.Tensor):
    if len(left) != len(right):
        count = min(len(left), len(right))
        left, right = left[:count], right[:count]
    a = (left @ directions).sort(dim=0).values
    b = (right @ directions).sort(dim=0).values
    return (a - b).abs().mean().item()


def metrics(text: torch.Tensor, image: torch.Tensor, args):
    generator = torch.Generator().manual_seed(args.seed)
    directions = F.normalize(torch.randn(text.size(1), args.num_slices, generator=generator), dim=0)
    text, image = text.float(), image.float()
    mean_gap = (text.mean(0) - image.mean(0)).norm().item()
    cov_text, cov_image = covariance(text), covariance(image)
    cov_gap = (cov_text - cov_image).norm().item()
    cov_scale = ((cov_text.norm() + cov_image.norm()) / 2).clamp_min(1e-12).item()
    cross_swd = sliced_wasserstein(text, image, directions)
    text_within = sliced_wasserstein(text[::2], text[1::2], directions)
    image_within = sliced_wasserstein(image[::2], image[1::2], directions)
    baseline = (text_within + image_within) / 2
    return {
        "text_gaussianity": {**gaussianity_diagnostics(text), "sigreg_statistic": sigreg_loss(text, num_slices=args.num_slices, seed=args.seed).item()},
        "image_gaussianity": {**gaussianity_diagnostics(image), "sigreg_statistic": sigreg_loss(image, num_slices=args.num_slices, seed=args.seed).item()},
        "text_image_mean_l2": mean_gap,
        "text_image_mean_l2_per_dimension": mean_gap / text.size(1) ** 0.5,
        "text_image_covariance_relative_frobenius": cov_gap / cov_scale,
        "text_image_sliced_wasserstein": cross_swd,
        "within_text_sliced_wasserstein": text_within,
        "within_image_sliced_wasserstein": image_within,
        "cross_to_within_swd_ratio": cross_swd / max(baseline, 1e-12),
    }


def main():
    args = arguments()
    specs = [item.split("=", 1) for item in args.checkpoint]
    first = torch.load(specs[0][1], map_location="cpu", weights_only=False)
    tokenizer = ClevrTextTokenizer(first["text_vocabulary"])
    model_args = checkpoint_args(first)
    dataset = ClevrMultimodalDataset(
        args.dataset_root, args.token_cache, "paired", pair_manifest=args.manifest,
        caption_field=args.caption_field,
    )
    dataset = Subset(dataset, range(min(args.num_samples, len(dataset))))
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=MultimodalCollator(tokenizer, model_args.num_image_codes, model_args.max_text_length))
    report = {
        "num_samples": len(dataset),
        "shared_readout": "exact training return_shared aggregate; valid-token pool each shared adapter, then average all adapters/layers",
        "interpretation": "cross-to-within SWD near 1 means text/image difference is no larger than sampling variation within modalities; it does not test semantic pairing",
        "models": {},
    }
    for name, path in specs:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        current_args = checkpoint_args(payload)
        model, _ = build_model(current_args, len(tokenizer))
        model.load_state_dict(payload["model"])
        model.to(args.device)
        vectors = extract_shared(model, loader, args.device)
        report["models"][name] = {
            "checkpoint": path,
            "epoch": payload.get("epoch"),
            "step": payload.get("step"),
            "aggregate_metrics": metrics(
                vectors["aggregate"]["text"], vectors["aggregate"]["image"], args
            ),
            "per_layer_metrics": {
                f"layer_{layer:02d}": metrics(values["text"], values["image"], args)
                for layer, values in sorted(vectors["per_layer"].items())
            },
        }
        print(name, json.dumps(report["models"][name]["aggregate_metrics"], sort_keys=True), flush=True)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
