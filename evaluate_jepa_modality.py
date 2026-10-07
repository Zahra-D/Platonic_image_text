"""Modality-local evaluation for masked-to-clean shared-JEPA checkpoints.

This deliberately does *not* measure image--text retrieval.  It asks whether
the shared LoRA readout is stable under corruption in the modality on which
JEPA was trained, whether scene labels remain decodable, and whether the
branch is comparatively tied to exact discrete-token identity.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Subset

from analyze_layer_gradient_conflict import one_modality
from analyze_paired_representations import checkpoint_args
from data import ClevrMultimodalDataset, ClevrTextTokenizer, MultimodalCollator
from models.lora import TriLoRALinear
from multimodal_diffusion import corrupt_batch
from train_multimodal import build_model


GROUPS = {
    "color": ("gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"),
    "shape": ("cube", "sphere", "cylinder"),
    "material": ("metal", "rubber"),
    "size": ("small", "large"),
    "relation": ("left", "right", "front", "behind"),
}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--train-dir", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--token-cache", required=True)
    p.add_argument("--caption-field", default="caption_human")
    p.add_argument("--modality", choices=("text", "image"), required=True)
    p.add_argument(
        "--data-mode", choices=("paired", "text_only"), default="paired",
        help="Use text_only for a caption corpus without image tokens.",
    )
    p.add_argument("--layers", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--mask-ratios", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.6, 0.8])
    p.add_argument("--train-samples", type=int, default=1000)
    p.add_argument("--val-samples", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--token-probe-train", type=int, default=4000)
    p.add_argument("--token-probe-val", type=int, default=2000)
    p.add_argument("--probe-epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", required=True)
    return p.parse_args()


def rows_and_labels(path: str, count: int):
    rows = []
    with open(path) as h:
        for line in h:
            if len(rows) == count:
                break
            rows.append(json.loads(line))
    binary = {value: [] for names in GROUPS.values() for value in names}
    count_label = []
    for row in rows:
        world = row["world"]
        objects = world["objects"]
        count_label.append(str(len(objects)))
        present = {key: set() for key in GROUPS}
        for obj in objects:
            for key in ("color", "shape", "material", "size"):
                present[key].add(obj[key])
        for relation in world.get("relations", []):
            present["relation"].add(relation["relation"])
        for group, names in GROUPS.items():
            for name in names:
                binary[name].append(int(name in present[group]))
    return {
        "count": np.asarray(count_label),
        "binary": {key: np.asarray(value) for key, value in binary.items()},
    }


def make_loader(directory, manifest, token_cache, split, tokenizer, args, size):
    cache = Path(token_cache)
    if cache.is_dir():
        cache = cache / f"{split}_tokens.pt"
    dataset = ClevrMultimodalDataset(
        directory, cache, args.data_mode, pair_manifest=manifest, caption_field=args.caption_field,
    )
    dataset = Subset(dataset, range(min(size, len(dataset))))
    return DataLoader(
        dataset, args.batch_size, shuffle=False, num_workers=0,
        collate_fn=MultimodalCollator(tokenizer, args.num_image_codes, args.max_text_length),
        pin_memory=args.device.startswith("cuda"),
    ), len(dataset)


class PrivateRecorder:
    """Recreate the matching private delta at each selected block."""
    def __init__(self, model, layers, modality, width):
        self.values = {layer: [] for layer in layers}
        self.branch = modality
        self.width = width
        self.hooks = []
        for module in model.modules():
            if isinstance(module, TriLoRALinear) and module.layer_index in self.values:
                self.hooks.append(module.register_forward_pre_hook(self.hook))

    def hook(self, module, inputs):
        value = module._delta(inputs[0].detach(), self.branch)
        if value.shape[-1] != self.width:
            b, n, d = value.shape
            value = F.adaptive_avg_pool1d(value.reshape(b * n, 1, d), self.width).reshape(b, n, self.width)
        self.values[module.layer_index].append(value)

    def clear(self):
        for value in self.values.values():
            value.clear()

    def result(self):
        return {layer: torch.stack(value).mean(0) for layer, value in self.values.items()}

    def close(self):
        for hook in self.hooks:
            hook.remove()


@torch.no_grad()
def representations(model, batch, layers, private):
    if private is not None:
        private.clear()
    shared = model(
        batch["input_ids"], batch["attention_mask"], batch["position_ids"],
        batch["modality_ids"], batch["route_ids"], return_shared=True,
        return_shared_tokens_by_layer=True,
    )[3]
    return ({layer: shared[layer].float() for layer in layers},
            None if private is None else {layer: value.float() for layer, value in private.result().items()})


def cosine_sum(left, right):
    return F.cosine_similarity(left.float(), right.float(), dim=-1, eps=1e-8).sum().item(), left.size(0)


def pool(values, eligible):
    weights = eligible.float().unsqueeze(-1)
    return (values * weights).sum(1) / weights.sum(1).clamp_min(1)


def sample_tokens(features, labels, mask, limit, generator):
    positions = mask.nonzero(as_tuple=False)
    if not len(positions):
        return None, None
    if len(positions) > limit:
        # CPU generators cannot drive CUDA randperm.  The caller resets the
        # global seed before each corruption, which keeps this sampling
        # reproducible on the active device as well.
        positions = positions[torch.randperm(len(positions), device=positions.device)[:limit]]
    return features[positions[:, 0], positions[:, 1]].cpu(), labels[positions[:, 0], positions[:, 1]].cpu()


def fit_semantic_probes(train_x, train_y, seed):
    # ``lbfgs`` is needlessly slow here: the protocol fits 20 labels for each
    # branch/layer/checkpoint.  Liblinear remains an exact deterministic
    # class-balanced logistic-regression probe, supports the required binary
    # probabilities, and uses one-vs-rest for the small count task.
    logistic = dict(
        max_iter=100, tol=1e-3, solver="liblinear", class_weight="balanced",
        random_state=seed,
    )
    clf = make_pipeline(Normalizer(), StandardScaler(), LogisticRegression(
        **logistic,
    )).fit(train_x, train_y["count"])
    binary_models = {}
    for group, names in GROUPS.items():
        for name in names:
            if len(np.unique(train_y["binary"][name])) < 2:
                continue
            binary_models[name] = make_pipeline(Normalizer(), StandardScaler(), LogisticRegression(
                **logistic,
            )).fit(train_x, train_y["binary"][name])
    return clf, binary_models


def score_semantic_probes(probes, test_x, test_y):
    clf, binary_models = probes
    pred = clf.predict(test_x)
    result = {"object_count": {
        "balanced_accuracy": float(balanced_accuracy_score(test_y["count"], pred)),
        "macro_f1": float(f1_score(test_y["count"], pred, average="macro")),
    }}
    attributes = {}
    groups = {}
    for group, names in GROUPS.items():
        scores = []
        for name in names:
            if name not in binary_models:
                continue
            score = float(balanced_accuracy_score(
                test_y["binary"][name], binary_models[name].predict(test_x)
            ))
            attributes[name] = score
            scores.append(score)
        groups[group] = float(np.mean(scores)) if scores else float("nan")
    result["attribute_balanced_accuracy"] = attributes
    result["attribute_group_balanced_accuracy"] = groups
    result["attribute_group_mean"] = float(np.nanmean(list(groups.values())))
    return result


def fit_token_probe(train_x, train_y, vocab_size, epochs, device, seed):
    if len(train_x) < 2:
        return None
    torch.manual_seed(seed)
    linear = nn.Linear(train_x.shape[1], vocab_size).to(device)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=2e-3, weight_decay=1e-4)
    x = train_x.to(device); y = train_y.to(device)
    linear.train()
    for _ in range(epochs):
        order = torch.randperm(len(x), device=device)
        for rows in order.split(256):
            loss = F.cross_entropy(linear(x[rows]), y[rows])
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return linear.eval()


def score_token_probe(linear, test_x, test_y, device):
    if linear is None or len(test_x) < 1:
        return None
    with torch.no_grad():
        logits = linear(test_x.to(device))
        loss = F.cross_entropy(logits, test_y.to(device)).item()
        accuracy = logits.argmax(-1).eq(test_y.to(device)).float().mean().item()
    return {"cross_entropy": loss, "top1_accuracy": accuracy, "test_tokens": int(len(test_x))}


def load_ignoring_modulewise_predictors(model, state):
    """Load exactly, except for modulewise JEPA predictor tensors.

    Older modulewise checkpoints named these predictors ``att_lNN``/``mlp_lNN``
    before the scheme became ``out_proj_lNN``/``mlp_3_lNN``.  This evaluator
    never calls them, so their names may differ; every other tensor must match.
    """
    incompatible = model.load_state_dict(state, strict=False)
    prefix = "modulewise_jepa_predictors."
    missing = [key for key in incompatible.missing_keys if not key.startswith(prefix)]
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith(prefix)]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch outside unused modulewise predictors: "
            f"missing={missing}, unexpected={unexpected}"
        )


def create_teacher(model, payload, vocab_size, device):
    state = payload.get("shared_jepa_ema_teacher")
    if state is None:
        return model
    args = checkpoint_args(payload)
    teacher, _ = build_model(args, vocab_size)
    load_ignoring_modulewise_predictors(teacher, state)
    teacher.to(device).eval()
    return teacher


def evaluate_model(name, path, first_payload, tokenizer, cli, loaders, labels):
    payload = first_payload if path == first_payload.get("_path") else torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint_args(payload)
    if args.train_mode not in {"lora", "dense_private"}:
        raise ValueError(f"{name} is dense; this evaluator is for shared/private LoRA readouts")
    model, _ = build_model(args, len(tokenizer))
    load_ignoring_modulewise_predictors(model, payload["model"])
    model.to(cli.device).eval()
    layers = [layer for layer in cli.layers if layer < len(model.blocks)]
    private = PrivateRecorder(model, layers, cli.modality, model.token_embed.embedding_dim)
    teacher = create_teacher(model, payload, len(tokenizer), cli.device)
    teacher_private = None if teacher is model else PrivateRecorder(teacher, layers, cli.modality, teacher.token_embed.embedding_dim)
    metrics = {f"t{ratio:g}": {str(layer): defaultdict(float) for layer in layers} for ratio in cli.mask_ratios}
    train_clean = {"shared": {layer: [] for layer in layers}, "private": {layer: [] for layer in layers}}
    val_masked = {f"t{ratio:g}": {branch: {layer: [] for layer in layers} for branch in ("shared", "private")} for ratio in cli.mask_ratios}
    token_train = {branch: {layer: [] for layer in layers} for branch in ("shared", "private")}
    token_train_y = {branch: {layer: [] for layer in layers} for branch in ("shared", "private")}
    token_val = {f"t{ratio:g}": {branch: {layer: [] for layer in layers} for branch in ("shared", "private")} for ratio in cli.mask_ratios}
    token_val_y = {f"t{ratio:g}": {branch: {layer: [] for layer in layers} for branch in ("shared", "private")} for ratio in cli.mask_ratios}
    generator = torch.Generator().manual_seed(cli.seed)
    objective = cli.modality
    for split, loader in loaders.items():
        for bi, paired in enumerate(loader):
            batch = one_modality(paired, cli.modality, cli.device)
            clean_shared, clean_private = representations(model, batch, layers, private)
            if teacher is model:
                target_shared = clean_shared
            else:
                target_shared, _ = representations(teacher, batch, layers, teacher_private)
            if split == "train":
                for layer in layers:
                    train_clean["shared"][layer].append(pool(clean_shared[layer], batch["eligible_mask"]).cpu())
                    train_clean["private"][layer].append(pool(clean_private[layer], batch["eligible_mask"]).cpu())
                    # Train the local-detail probe on clean token positions.
                    available = batch["eligible_mask"]
                    remaining = max(0, cli.token_probe_train - sum(len(x) for x in token_train_y["shared"][layer]))
                    if remaining:
                        for branch, values in (("shared", clean_shared[layer]), ("private", clean_private[layer])):
                            x, y = sample_tokens(values, batch["input_ids"], available, remaining, generator)
                            if x is not None:
                                token_train[branch][layer].append(x); token_train_y[branch][layer].append(y)
                # Clean features are the only training input for both probes.
                # Re-running all corruption levels on the training split adds
                # substantial cost but cannot change either fitted probe.
                continue
            for ri, ratio in enumerate(cli.mask_ratios):
                torch.manual_seed(cli.seed + bi * 1009 + ri * 31 + (0 if split == "train" else 500_000))
                corrupted, mask, _ = corrupt_batch(
                    batch["input_ids"], batch["eligible_mask"], batch["modality_ids"], tokenizer.mask_id,
                    objective=objective, fixed_t=ratio,
                )
                masked_batch = dict(batch); masked_batch["input_ids"] = corrupted
                masked_shared, masked_private = representations(model, masked_batch, layers, private)
                key = f"t{ratio:g}"
                for layer in layers:
                    entry = metrics[key][str(layer)]
                    cs, count = cosine_sum(masked_shared[layer][mask], target_shared[layer][mask])
                    entry["shared_clean_cosine_sum"] += cs; entry["masked_tokens"] += count
                    cs, _ = cosine_sum(masked_private[layer][mask], clean_private[layer][mask])
                    entry["private_clean_cosine_sum"] += cs
                    if model.shared_jepa_predictors is not None:
                        predicted = model.predict_shared_jepa(layer, masked_shared[layer][mask]).detach()
                        cs, _ = cosine_sum(predicted, target_shared[layer][mask])
                        entry["predictor_teacher_cosine_sum"] += cs
                        entry["predictor_teacher_mse_sum"] += float(F.mse_loss(predicted.float(), target_shared[layer][mask].float())) * count
                    if split == "val":
                        val_masked[key]["shared"][layer].append(pool(masked_shared[layer], batch["eligible_mask"]).cpu())
                        val_masked[key]["private"][layer].append(pool(masked_private[layer], batch["eligible_mask"]).cpu())
                        remaining = max(0, cli.token_probe_val - sum(len(x) for x in token_val_y[key]["shared"][layer]))
                        if remaining:
                            for branch, values in (("shared", masked_shared[layer]), ("private", masked_private[layer])):
                                x, y = sample_tokens(values, batch["input_ids"], mask, remaining, generator)
                                if x is not None:
                                    token_val[key][branch][layer].append(x); token_val_y[key][branch][layer].append(y)
    result = {
        "checkpoint": {"path": path, "epoch": payload.get("epoch"), "global_step": payload.get("global_step")},
        "modality": cli.modality, "layers": layers,
        "teacher": "EMA clean teacher" if teacher is not model else "online clean stop-gradient target",
        "direct_masked_position_stability": {},
        # This is deliberately reported as an in-sample fit ceiling.  It is
        # useful to diagnose underfitting and to compare probe difficulty, but
        # it is not a generalization result and must not replace held-out scores.
        "semantic_clean_train_fit": {},
        "semantic_clean_train_masked_test": {},
        "exact_token_clean_train_masked_test": {},
    }
    for key, by_layer in metrics.items():
        result["direct_masked_position_stability"][key] = {}
        for layer, entry in by_layer.items():
            n = max(1, entry.pop("masked_tokens"))
            values = {
                "masked_tokens": n,
                "shared_clean_cosine": entry.pop("shared_clean_cosine_sum") / n,
                "private_clean_cosine": entry.pop("private_clean_cosine_sum") / n,
            }
            if "predictor_teacher_cosine_sum" in entry:
                values["predictor_teacher_cosine"] = entry.pop("predictor_teacher_cosine_sum") / n
                values["predictor_teacher_mse"] = entry.pop("predictor_teacher_mse_sum") / n
            result["direct_masked_position_stability"][key][layer] = values
    for layer in layers:
        for branch in ("shared", "private"):
            x_train = torch.cat(train_clean[branch][layer]).numpy()
            semantic_probes = fit_semantic_probes(x_train, labels["train"], cli.seed)
            result["semantic_clean_train_fit"].setdefault(str(layer), {})[branch] = score_semantic_probes(
                semantic_probes, x_train, labels["train"],
            )
            train_x = torch.cat(token_train[branch][layer])
            train_y = torch.cat(token_train_y[branch][layer])
            token_linear = fit_token_probe(
                train_x, train_y, model.vocab_size, cli.probe_epochs, cli.device, cli.seed + layer,
            )
            for key in val_masked:
                x_val = torch.cat(val_masked[key][branch][layer]).numpy()
                result["semantic_clean_train_masked_test"].setdefault(key, {}).setdefault(str(layer), {})[branch] = score_semantic_probes(
                    semantic_probes, x_val, labels["val"],
                )
                test_x = torch.cat(token_val[key][branch][layer])
                test_y = torch.cat(token_val_y[key][branch][layer])
                local = score_token_probe(token_linear, test_x, test_y, cli.device)
                if local is not None:
                    local["train_tokens"] = int(len(train_x))
                result["exact_token_clean_train_masked_test"].setdefault(key, {}).setdefault(str(layer), {})[branch] = local
            del token_linear
    private.close()
    if teacher_private is not None:
        teacher_private.close(); del teacher
    del model
    if cli.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def markdown(report):
    lines = [
        "# Modality-local JEPA evaluation", "",
        "This report intentionally excludes cross-modal retrieval and matched/shuffled metrics.",
        "It evaluates the modality on which each model's JEPA objective was trained.", "",
        "- **Direct stability:** cosine at exactly corrupted token positions between the student readout and clean target.",
        "- **Probe-fit ceiling:** in-sample semantic accuracy on the clean training features used to fit each probe; it is diagnostic only, not a generalization metric.",
        "- **Semantic robustness:** that frozen linear probe evaluated on masked held-out pooled features.",
        "- **Local detail:** a linear probe trained on clean per-token features to recover the original discrete token, evaluated at corrupted positions.",
        "",
    ]
    for name, model in report["models"].items():
        lines += [f"## {name} ({model['modality']})", "", f"Teacher: {model['teacher']}", ""]
        for ratio, layers in model["direct_masked_position_stability"].items():
            lines += [f"### Mask ratio {ratio[1:]}", "", "| Layer | Shared clean cosine | Private clean cosine | JEPA predictor cosine |", "|---:|---:|---:|---:|"]
            for layer, values in layers.items():
                pred = values.get("predictor_teacher_cosine")
                lines.append(f"| {layer} | {values['shared_clean_cosine']:.4f} | {values['private_clean_cosine']:.4f} | {'—' if pred is None else f'{pred:.4f}'} |")
            lines.append("")
        lines += ["Semantic and exact-token probe details are stored in `results.json`.", ""]
    return "\n".join(lines)


def main():
    cli = arguments()
    specs = [value.split("=", 1) for value in cli.checkpoint]
    first = torch.load(specs[0][1], map_location="cpu", weights_only=False)
    first["_path"] = specs[0][1]
    tokenizer = ClevrTextTokenizer(first["text_vocabulary"])
    model_args = checkpoint_args(first)
    # make_loader also needs model properties, but keeps the CLI specification compact.
    cli.num_image_codes = model_args.num_image_codes; cli.max_text_length = model_args.max_text_length
    train_loader, train_count = make_loader(cli.train_dir, cli.train_manifest, cli.token_cache, "train", tokenizer, cli, cli.train_samples)
    val_loader, val_count = make_loader(cli.val_dir, cli.val_manifest, cli.token_cache, "val", tokenizer, cli, cli.val_samples)
    report = {
        "protocol": {
            "modality": cli.modality, "layers": cli.layers, "mask_ratios": cli.mask_ratios,
            "data_mode": cli.data_mode,
            "train_samples": train_count, "val_samples": val_count,
            "token_probe": f"linear, clean train -> corrupted-position test; {cli.token_probe_train}/{cli.token_probe_val} max tokens",
            "semantic_probe": "L2-normalized + standardized balanced logistic regression, clean train -> masked test",
        }, "models": {},
    }
    labels = {"train": rows_and_labels(cli.train_manifest, train_count), "val": rows_and_labels(cli.val_manifest, val_count)}
    for name, path in specs:
        print(f"evaluating {name}", flush=True)
        report["models"][name] = evaluate_model(
            name, path, first, tokenizer, cli, {"train": train_loader, "val": val_loader}, labels,
        )
    output = Path(cli.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    output.with_name("REPORT.md").write_text(markdown(report))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
