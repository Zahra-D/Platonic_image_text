"""Evaluate whether frozen shared representations encode CLEVR semantics.

The diagnostic has two complementary parts:
1. label-free K-means followed by post-hoc cluster/attribute association;
2. frozen linear probes, including zero-refit cross-modal transfer.

For Tri-LoRA models, the representation is the per-layer average of the four
shared adapter deltas.  Dense models have no separate branch, so their block
residual is the corresponding common representation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from torch.utils.data import DataLoader, Subset

from analyze_layer_gradient_conflict import one_modality
from analyze_paired_representations import checkpoint_args
from data import ClevrMultimodalDataset, ClevrTextTokenizer, MultimodalCollator
from train_multimodal import build_model
from models.lora import TriLoRALinear


COLORS = ("gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow")
SHAPES = ("cube", "sphere", "cylinder")
MATERIALS = ("metal", "rubber")
SIZES = ("small", "large")
RELATIONS = ("left", "right", "front", "behind")
GROUPS = {
    "color": COLORS,
    "shape": SHAPES,
    "material": MATERIALS,
    "size": SIZES,
    "relation": RELATIONS,
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--caption-field", default="caption_human")
    parser.add_argument("--train-samples", type=int, default=4000)
    parser.add_argument("--val-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument(
        "--lora-representation", choices=("shared", "private"), default="shared",
        help=(
            "LoRA branch to probe. For 'private', text examples use the text-private "
            "branch and image examples use the image-private branch."
        ),
    )
    parser.add_argument("--clusters", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_rows(path, limit):
    rows = []
    with open(path) as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            rows.append(json.loads(line))
    return rows


def semantic_labels(rows):
    categorical = defaultdict(list)
    binary = {name: [] for values in GROUPS.values() for name in values}
    for row in rows:
        objects = row["world"]["objects"]
        relations = row["world"].get("relations", [])
        categorical["object_count"].append(str(len(objects)))
        for group, names in (("shape", SHAPES), ("material", MATERIALS), ("size", SIZES)):
            counts = Counter(obj[group] for obj in objects)
            categorical[f"{group}_profile"].append("|".join(str(counts[name]) for name in names))
        present = {group: {obj[group] for obj in objects} for group in ("color", "shape", "material", "size")}
        present["relation"] = {relation["relation"] for relation in relations}
        for group, names in GROUPS.items():
            for name in names:
                binary[name].append(int(name in present[group]))
    return {
        "categorical": {key: np.asarray(value) for key, value in categorical.items()},
        "binary": {key: np.asarray(value, dtype=np.int64) for key, value in binary.items()},
    }


@torch.no_grad()
def extract(model, loader, train_mode, layers, device, lora_representation="shared"):
    output = {modality: {layer: [] for layer in layers} for modality in ("text", "image")}
    dense_state = {}
    private_state = {layer: [] for layer in layers}
    private_branch = {"name": None}
    hooks = []
    if train_mode == "dense":
        for layer in layers:
            def hook(_module, _inputs, value, layer=layer):
                dense_state[layer] = value.detach()
            hooks.append(model.blocks[layer].register_forward_hook(hook))
    elif lora_representation == "private":
        representation_dim = model.token_embed.embedding_dim

        def private_hook(module, inputs):
            branch = private_branch["name"]
            if branch is None or module.layer_index not in private_state:
                return
            value = module._delta(inputs[0].detach(), branch)
            if value.shape[-1] != representation_dim:
                batch, length, width = value.shape
                value = F.adaptive_avg_pool1d(
                    value.reshape(batch * length, 1, width), representation_dim,
                ).reshape(batch, length, representation_dim)
            private_state[module.layer_index].append(value)

        for module in model.modules():
            if isinstance(module, TriLoRALinear) and module.layer_index in layers:
                hooks.append(module.register_forward_pre_hook(private_hook))
    model.eval()
    try:
        for paired in loader:
            for modality in ("text", "image"):
                batch = one_modality(paired, modality, device)
                weights = batch["eligible_mask"].float().unsqueeze(-1)
                dense_state.clear()
                for values in private_state.values():
                    values.clear()
                private_branch["name"] = modality
                if train_mode in {"lora", "dense_private"}:
                    if lora_representation == "shared":
                        activations = model(
                            batch["input_ids"], batch["attention_mask"], batch["position_ids"],
                            batch["modality_ids"], batch["route_ids"], return_shared=True,
                            return_shared_tokens_by_layer=True,
                        )[3]
                    else:
                        model(
                            batch["input_ids"], batch["attention_mask"], batch["position_ids"],
                            batch["modality_ids"], batch["route_ids"],
                        )
                        activations = {
                            layer: torch.stack(values, dim=0).mean(dim=0)
                            for layer, values in private_state.items()
                        }
                else:
                    model(
                        batch["input_ids"], batch["attention_mask"], batch["position_ids"],
                        batch["modality_ids"], batch["route_ids"],
                    )
                    activations = dense_state
                for layer in layers:
                    value = activations[layer].float()
                    pooled = (value * weights).sum(1) / weights.sum(1).clamp_min(1)
                    output[modality][layer].append(pooled.cpu())
    finally:
        for hook in hooks:
            hook.remove()
    return {
        modality: {layer: torch.cat(parts).numpy() for layer, parts in values.items()}
        for modality, values in output.items()
    }


def cluster_metrics(text, image, labels, cluster_counts, seed):
    result = {}
    for modality, raw in (("text", text), ("image", image)):
        values = Normalizer().fit_transform(raw)
        by_k = {}
        for k in cluster_counts:
            assignments = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(values)
            categorical = {
                name: float(normalized_mutual_info_score(target, assignments))
                for name, target in labels["categorical"].items()
            }
            binary = {
                name: float(normalized_mutual_info_score(target, assignments))
                for name, target in labels["binary"].items()
            }
            grouped = {
                group: float(np.mean([binary[name] for name in names]))
                for group, names in GROUPS.items()
            }
            by_k[str(k)] = {
                "silhouette_cosine": float(silhouette_score(
                    values, assignments, metric="cosine", sample_size=min(1000, len(values)),
                    random_state=seed,
                )),
                "categorical_nmi": categorical,
                "attribute_presence_nmi": binary,
                "attribute_group_nmi_mean": grouped,
            }
        result[modality] = by_k

    # Joint clustering asks whether paired modalities land in the same cluster.
    joint = Normalizer().fit_transform(np.concatenate((text, image), axis=0))
    paired = {}
    for k in cluster_counts:
        assignments = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(joint)
        left, right = assignments[:len(text)], assignments[len(text):]
        paired[str(k)] = {
            "paired_same_cluster_fraction": float(np.mean(left == right)),
            "paired_cluster_ari": float(adjusted_rand_score(left, right)),
        }
    result["joint_cross_modal"] = paired
    return result


def fitted_probe(x, y, seed):
    return make_pipeline(
        Normalizer(), StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    ).fit(x, y)


def score_binary(model, x, y):
    prediction = model.predict(x)
    probability = model.predict_proba(x)[:, 1]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "auroc": float(roc_auc_score(y, probability)),
    }


def probe_direction(train_x, test_x, train_labels, test_labels, seed):
    count_probe = fitted_probe(train_x, train_labels["categorical"]["object_count"], seed)
    count_prediction = count_probe.predict(test_x)
    result = {
        "object_count": {
            "balanced_accuracy": float(balanced_accuracy_score(
                test_labels["categorical"]["object_count"], count_prediction
            )),
            "macro_f1": float(f1_score(
                test_labels["categorical"]["object_count"], count_prediction, average="macro"
            )),
        },
        "attributes": {},
    }
    for name, y_train in train_labels["binary"].items():
        if len(np.unique(y_train)) < 2:
            continue
        model = fitted_probe(train_x, y_train, seed)
        result["attributes"][name] = score_binary(model, test_x, test_labels["binary"][name])
    result["attribute_groups"] = {
        group: {
            metric: float(np.mean([
                result["attributes"][name][metric] for name in names
                if name in result["attributes"]
            ]))
            for metric in ("balanced_accuracy", "f1", "auroc")
        }
        for group, names in GROUPS.items()
    }
    return result


def probe_metrics(train, val, train_labels, val_labels, seed):
    return {
        "text_to_text": probe_direction(train["text"], val["text"], train_labels, val_labels, seed),
        "image_to_image": probe_direction(train["image"], val["image"], train_labels, val_labels, seed),
        "text_to_image": probe_direction(train["text"], val["image"], train_labels, val_labels, seed),
        "image_to_text": probe_direction(train["image"], val["text"], train_labels, val_labels, seed),
    }


def main():
    cli = arguments()
    torch.manual_seed(cli.seed)
    specs = [item.split("=", 1) for item in cli.checkpoint]
    first = torch.load(specs[0][1], map_location="cpu", weights_only=False)
    tokenizer = ClevrTextTokenizer(first["text_vocabulary"])
    first_args = checkpoint_args(first)
    collator = MultimodalCollator(tokenizer, first_args.num_image_codes, first_args.max_text_length)

    def make_loader(directory, manifest, limit, split):
        token_cache = Path(cli.token_cache)
        if token_cache.is_dir():
            token_cache = token_cache / f"{split}_tokens.pt"
        dataset = ClevrMultimodalDataset(
            directory, token_cache, "paired", pair_manifest=manifest,
            caption_field=cli.caption_field,
        )
        dataset = Subset(dataset, range(min(limit, len(dataset))))
        return DataLoader(
            dataset, cli.batch_size, shuffle=False, num_workers=0, collate_fn=collator,
            pin_memory=cli.device.startswith("cuda"),
        ), len(dataset)

    train_loader, train_count = make_loader(
        cli.train_dir, cli.train_manifest, cli.train_samples, "train"
    )
    val_loader, val_count = make_loader(
        cli.val_dir, cli.val_manifest, cli.val_samples, "val"
    )
    train_labels = semantic_labels(load_rows(cli.train_manifest, train_count))
    val_labels = semantic_labels(load_rows(cli.val_manifest, val_count))
    report = {
        "protocol": {
            "train_samples": train_count, "val_samples": val_count,
            "layers": cli.layers, "clusters": cli.clusters,
            "lora_representation": (
                "eligible-token mean of per-layer average shared adapter delta"
                if cli.lora_representation == "shared" else
                "eligible-token mean of per-layer average matching private adapter delta "
                "(text-private for text; image-private for image)"
            ),
            "dense_representation": "eligible-token mean of block residual (dense has no separate shared branch)",
            "probe": "frozen backbone; L2 normalization + train-split standardization + balanced logistic regression",
        },
        "models": {},
    }
    for index, (name, path) in enumerate(specs):
        payload = first if index == 0 else torch.load(path, map_location="cpu", weights_only=False)
        args = checkpoint_args(payload)
        model, _ = build_model(args, len(tokenizer))
        model.load_state_dict(payload["model"])
        model.to(cli.device)
        train_vectors = extract(
            model, train_loader, args.train_mode, cli.layers, cli.device,
            cli.lora_representation,
        )
        val_vectors = extract(
            model, val_loader, args.train_mode, cli.layers, cli.device,
            cli.lora_representation,
        )
        model_report = {
            "checkpoint": {"path": path, "epoch": payload.get("epoch"), "step": payload.get("step")},
            "representation": (
                f"{cli.lora_representation}_lora"
                if args.train_mode in {"lora", "dense_private"} else "dense_block_residual"
            ),
            "layers": {},
        }
        for layer in cli.layers:
            model_report["layers"][str(layer)] = {
                "clustering": cluster_metrics(
                    val_vectors["text"][layer], val_vectors["image"][layer],
                    val_labels, cli.clusters, cli.seed,
                ),
                "probes": probe_metrics(
                    {m: train_vectors[m][layer] for m in ("text", "image")},
                    {m: val_vectors[m][layer] for m in ("text", "image")},
                    train_labels, val_labels, cli.seed,
                ),
            }
        report["models"][name] = model_report
        print(f"{name} complete", flush=True)
        del model, payload, train_vectors, val_vectors
        if cli.device.startswith("cuda"):
            torch.cuda.empty_cache()
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
