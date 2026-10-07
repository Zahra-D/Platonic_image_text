"""Fixed-mask held-out evaluation of saved shared-JEPA checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from analyze_layer_gradient_conflict import one_modality
from analyze_paired_representations import checkpoint_args
from data import ClevrMultimodalDataset, ClevrTextTokenizer, MultimodalCollator
from multimodal_diffusion import corrupt_batch
from train_multimodal import build_model


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--caption-field", default="caption_human")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mask-ratios", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


class Moments:
    def __init__(self, width: int):
        self.count = 0
        self.total = torch.zeros(width, dtype=torch.float64)
        self.gram = torch.zeros(width, width, dtype=torch.float64)

    def update(self, values: torch.Tensor):
        values = values.detach().float().cpu().double()
        self.count += values.size(0)
        self.total += values.sum(0)
        self.gram += values.T @ values

    def metrics(self):
        if self.count < 2:
            return {}
        mean = self.total / self.count
        covariance = self.gram / self.count - mean[:, None] @ mean[None, :]
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        energy = eigenvalues.sum().clamp_min(1e-20)
        effective_rank = energy.square() / eigenvalues.square().sum().clamp_min(1e-20)
        return {
            "feature_variance_mean": float(eigenvalues.mean()),
            "effective_rank_participation": float(effective_rank),
            "top_covariance_energy_fraction": float(eigenvalues.max() / energy),
        }


def main():
    cli = arguments()
    specs = [item.split("=", 1) for item in cli.checkpoint]
    first = torch.load(specs[0][1], map_location="cpu", weights_only=False)
    tokenizer = ClevrTextTokenizer(first["text_vocabulary"])
    first_args = checkpoint_args(first)
    dataset = ClevrMultimodalDataset(
        cli.split_dir, cli.token_cache, "paired",
        pair_manifest=cli.pair_manifest, caption_field=cli.caption_field,
    )
    dataset = Subset(dataset, range(min(cli.num_samples, len(dataset))))
    loader = DataLoader(
        dataset, batch_size=cli.batch_size, shuffle=False, num_workers=0,
        collate_fn=MultimodalCollator(tokenizer, first_args.num_image_codes, first_args.max_text_length),
    )
    report = {
        "dataset": cli.pair_manifest,
        "num_samples": len(dataset),
        "mask_ratios": cli.mask_ratios,
        "fixed_mask_seed": cli.seed,
        "target": "same-checkpoint clean forward in eval mode with stop-gradient",
        "models": {},
    }
    for model_index, (name, path) in enumerate(specs):
        payload = first if model_index == 0 else torch.load(path, map_location="cpu", weights_only=False)
        args = checkpoint_args(payload)
        if not getattr(args, "shared_jepa", False):
            raise ValueError(f"{name} does not contain a shared JEPA predictor")
        model, _ = build_model(args, len(tokenizer))
        model.load_state_dict(payload["model"])
        model.to(cli.device).eval()
        layers = list(getattr(args, "shared_jepa_layers", None) or range(len(model.blocks)))
        accumulators = defaultdict(lambda: defaultdict(float))
        moments = {
            (modality, ratio, layer): Moments(model.token_embed.embedding_dim)
            for modality in ("text", "image") for ratio in cli.mask_ratios for layer in layers
        }
        for batch_index, paired in enumerate(loader):
            for modality_index, modality in enumerate(("text", "image")):
                batch = one_modality(paired, modality, cli.device)
                with torch.no_grad():
                    clean = model(
                        batch["input_ids"], batch["attention_mask"], batch["position_ids"],
                        batch["modality_ids"], batch["route_ids"], return_shared=True,
                        return_shared_tokens_by_layer=True,
                    )[3]
                for ratio_index, ratio in enumerate(cli.mask_ratios):
                    torch.manual_seed(cli.seed + batch_index * 101 + modality_index * 17 + ratio_index)
                    corrupted, mask, _ = corrupt_batch(
                        batch["input_ids"], batch["eligible_mask"], batch["modality_ids"],
                        tokenizer.mask_id, objective=modality, fixed_t=ratio,
                    )
                    with torch.no_grad():
                        masked = model(
                            corrupted, batch["attention_mask"], batch["position_ids"],
                            batch["modality_ids"], batch["route_ids"], return_shared=True,
                            return_shared_tokens_by_layer=True,
                        )[3]
                    for layer in layers:
                        prediction = (
                            model.predict_shared_jepa(layer, masked[layer])[mask]
                            .float().detach()
                        )
                        target = clean[layer][mask].float()
                        count = prediction.size(0)
                        key = (modality, ratio, layer)
                        raw = F.mse_loss(prediction, target)
                        pred_unit = F.normalize(prediction, dim=-1, eps=1e-6)
                        target_unit = F.normalize(target, dim=-1, eps=1e-6)
                        cosine = (pred_unit * target_unit).sum(-1).mean()
                        normalized_mse = (pred_unit - target_unit).square().sum(-1).mean()
                        values = accumulators[key]
                        values["count"] += count
                        values["raw_mse_sum"] += float(raw) * count
                        values["normalized_mse_sum"] += float(normalized_mse) * count
                        values["cosine_sum"] += float(cosine) * count
                        values["prediction_norm_sum"] += float(prediction.norm(dim=-1).mean()) * count
                        values["target_norm_sum"] += float(target.norm(dim=-1).mean()) * count
                        moments[key].update(target)
        metrics = {}
        for (modality, ratio, layer), values in accumulators.items():
            count = values.pop("count")
            metrics.setdefault(modality, {}).setdefault(f"t{ratio:g}", {})[f"layer_{layer:02d}"] = {
                key.removesuffix("_sum"): value / count for key, value in values.items()
            } | moments[(modality, ratio, layer)].metrics()
        report["models"][name] = {
            "checkpoint": {"path": path, "epoch": payload.get("epoch"), "step": payload.get("step")},
            "trained_loss": getattr(args, "shared_jepa_loss", "mse"),
            "layers": layers,
            "metrics": metrics,
        }
        print(f"{name} complete", flush=True)
        del model, payload
        if cli.device.startswith("cuda"):
            torch.cuda.empty_cache()
    target = Path(cli.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
