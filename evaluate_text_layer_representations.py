"""Compare frozen text-only post-block representations across checkpoints.

Unlike the shared/private probe, this records the ordinary output of each
Transformer block.  It is therefore directly comparable for dense and LoRA
models and answers whether LoRA changes layer-level semantic accessibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from analyze_layer_gradient_conflict import one_modality
from analyze_paired_representations import checkpoint_args
from data import ClevrMultimodalDataset, ClevrTextTokenizer, MultimodalCollator
from evaluate_jepa_modality import fit_semantic_probes, rows_and_labels, score_semantic_probes
from multimodal_diffusion import corrupt_batch
from train_multimodal import build_model


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--caption-field", default="caption_human")
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mask-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


class BlockRecorder:
    def __init__(self, model, layers):
        self.values = {}
        self.hooks = [
            model.blocks[layer].register_forward_hook(self._hook(layer))
            for layer in layers
        ]

    def _hook(self, layer):
        def hook(_module, _inputs, output):
            self.values[layer] = output.detach()
        return hook

    def clear(self):
        self.values.clear()

    def close(self):
        for hook in self.hooks:
            hook.remove()


def pooled(values, eligible):
    weights = eligible.float().unsqueeze(-1)
    return (values.float() * weights).sum(1) / weights.sum(1).clamp_min(1)


def loader(directory, manifest, cache, tokenizer, model_args, limit, batch_size):
    dataset = ClevrMultimodalDataset(
        directory, Path(cache) / "unused.pt", "text_only",
        pair_manifest=manifest, caption_field=model_args.caption_field,
    )
    dataset = Subset(dataset, range(min(limit, len(dataset))))
    return DataLoader(
        dataset, batch_size, shuffle=False, num_workers=0,
        collate_fn=MultimodalCollator(tokenizer, model_args.num_image_codes, model_args.max_text_length),
        pin_memory=torch.cuda.is_available(),
    ), len(dataset)


@torch.no_grad()
def vectors(model, data_loader, layers, device, *, masked, tokenizer, seed, ratio):
    recorder = BlockRecorder(model, layers)
    result = {layer: [] for layer in layers}
    model.eval()
    try:
        for batch_index, paired in enumerate(data_loader):
            batch = one_modality(paired, "text", device)
            if masked:
                torch.manual_seed(seed + batch_index * 1009 + 500_000)
                input_ids, _, _ = corrupt_batch(
                    batch["input_ids"], batch["eligible_mask"], batch["modality_ids"],
                    tokenizer.mask_id, objective="text", fixed_t=ratio,
                )
                batch = dict(batch); batch["input_ids"] = input_ids
            recorder.clear()
            model(
                batch["input_ids"], batch["attention_mask"], batch["position_ids"],
                batch["modality_ids"], batch["route_ids"],
            )
            for layer in layers:
                result[layer].append(pooled(recorder.values[layer], batch["eligible_mask"]).cpu())
    finally:
        recorder.close()
    return {layer: torch.cat(parts).numpy() for layer, parts in result.items()}


def main():
    cli = arguments()
    specs = [value.split("=", 1) for value in cli.checkpoint]
    first = torch.load(specs[0][1], map_location="cpu", weights_only=False)
    first_args = checkpoint_args(first)
    tokenizer = ClevrTextTokenizer(first["text_vocabulary"])
    train_loader, train_count = loader(
        cli.train_dir, cli.train_manifest, cli.token_cache, tokenizer, first_args,
        cli.train_samples, cli.batch_size,
    )
    val_loader, val_count = loader(
        cli.val_dir, cli.val_manifest, cli.token_cache, tokenizer, first_args,
        cli.val_samples, cli.batch_size,
    )
    labels = {
        "train": rows_and_labels(cli.train_manifest, train_count),
        "val": rows_and_labels(cli.val_manifest, val_count),
    }
    report = {"protocol": {
        "representation": "eligible-token mean of post-Transformer-block hidden state",
        "train_samples": train_count, "val_samples": val_count, "layers": cli.layers,
        "mask_ratio": cli.mask_ratio,
        "probe": "class-balanced logistic regression; clean-train fit and masked-held-out score",
    }, "models": {}}
    for name, path in specs:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model_args = checkpoint_args(payload)
        model, _ = build_model(model_args, len(tokenizer))
        model.load_state_dict(payload["model"])
        model.to(cli.device).eval()
        layers = [layer for layer in cli.layers if layer < len(model.blocks)]
        train = vectors(model, train_loader, layers, cli.device, masked=False, tokenizer=tokenizer, seed=cli.seed, ratio=cli.mask_ratio)
        val = vectors(model, val_loader, layers, cli.device, masked=True, tokenizer=tokenizer, seed=cli.seed, ratio=cli.mask_ratio)
        report["models"][name] = {
            "checkpoint": {"path": path, "epoch": payload.get("epoch"), "step": payload.get("step"), "train_mode": model_args.train_mode},
            "layers": {
                str(layer): {
                    "semantic_clean_train_fit": score_semantic_probes(
                        fit_semantic_probes(train[layer], labels["train"], cli.seed), train[layer], labels["train"],
                    ),
                    "semantic_clean_train_masked_test": score_semantic_probes(
                        fit_semantic_probes(train[layer], labels["train"], cli.seed), val[layer], labels["val"],
                    ),
                }
                for layer in layers
            },
        }
        print(f"{name} complete", flush=True)
    output = Path(cli.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
