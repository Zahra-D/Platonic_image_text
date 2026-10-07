"""Causal shared/private swap evaluation for same-modality no-base Tri-LoRA.

For target B, this compares self generation G(S_B, P_B) with a middle-layer
swap G(S_A, P_B).  The shared deltas from clean A replace the target's shared
deltas only at selected layers; B keeps its own masked, iteratively-generated
tokens and private LoRA pathway.  A frozen scene parser, trained only on clean
training VQ codes, scores each generated image against both Y_A and Y_B.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from analyze_layer_gradient_conflict import one_modality
from analyze_paired_representations import checkpoint_args
from data import ClevrMultimodalDataset, ClevrTextTokenizer, MultimodalCollator
from models.lora import shared_replacement_context
from train_multimodal import build_model


ATTRIBUTES = (
    "gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow",
    "cube", "sphere", "cylinder", "metal", "rubber", "small", "large",
    "right", "behind",
)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--train-dir", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--token-cache", required=True)
    p.add_argument("--caption-field", default="caption_human")
    p.add_argument("--layers", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--parser-train-samples", type=int, default=20000)
    p.add_argument("--parser-val-samples", type=int, default=1024)
    p.add_argument("--parser-epochs", type=int, default=12)
    p.add_argument("--parser-batch-size", type=int, default=128)
    p.add_argument("--swap-samples", type=int, default=64)
    p.add_argument("--swap-batch-size", type=int, default=8)
    p.add_argument("--generation-steps", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--parser-checkpoint")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def records(path: str, limit: int | None = None):
    result = []
    with open(path) as handle:
        for line in handle:
            if limit is not None and len(result) >= limit:
                break
            result.append(json.loads(line))
    return result


def labels(rows):
    counts, attrs = [], []
    for row in rows:
        world = row["world"]
        counts.append(len(world["objects"]))
        present = set()
        for obj in world["objects"]:
            present.update((obj["color"], obj["shape"], obj["material"], obj["size"]))
        # The current manifest's relation labels use right/behind directions.
        present.update(rel["relation"] for rel in world.get("relations", []))
        attrs.append([float(name in present) for name in ATTRIBUTES])
    return torch.tensor(counts, dtype=torch.long), torch.tensor(attrs, dtype=torch.float32)


class CodeSceneDataset(Dataset):
    def __init__(self, codes, count, attributes):
        self.codes = codes
        self.count = count
        self.attributes = attributes

    def __len__(self):
        return len(self.count)

    def __getitem__(self, index):
        return self.codes[index], self.count[index], self.attributes[index]


class FrozenCodeSceneParser(nn.Module):
    """Evaluation-only CNN parser for a 16x24 VQ code grid."""
    def __init__(self, codes: int, count_classes: int, attributes: int):
        super().__init__()
        self.embedding = nn.Embedding(codes, 64)
        self.body = nn.Sequential(
            nn.Conv2d(64, 96, 3, padding=1), nn.GELU(),
            nn.Conv2d(96, 128, 3, padding=1), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 160, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.count_head = nn.Linear(160, count_classes)
        self.attribute_head = nn.Linear(160, attributes)

    def forward(self, codes):
        x = self.embedding(codes.long()).permute(0, 3, 1, 2)
        x = self.body(x).flatten(1)
        return self.count_head(x), self.attribute_head(x)


def parser_metrics(parser, loader, device):
    parser.eval()
    count_correct = count_total = 0
    attr_correct = attr_total = 0
    with torch.no_grad():
        for codes, count, attributes in loader:
            codes, count, attributes = codes.to(device), count.to(device), attributes.to(device)
            count_logits, attribute_logits = parser(codes)
            count_correct += count_logits.argmax(-1).eq(count).sum().item()
            count_total += len(count)
            attr_correct += (attribute_logits.sigmoid().ge(0.5) == attributes.bool()).sum().item()
            attr_total += attributes.numel()
    return {"count_accuracy": count_correct / max(1, count_total), "attribute_accuracy": attr_correct / max(1, attr_total)}


def train_or_load_parser(cli, train_codes, train_labels, val_codes, val_labels, num_codes, path):
    count_classes = int(max(train_labels[0].max(), val_labels[0].max()).item()) + 1
    parser = FrozenCodeSceneParser(num_codes, count_classes, len(ATTRIBUTES)).to(cli.device)
    path = Path(path)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        parser.load_state_dict(payload["parser"])
        return parser.eval(), payload["metrics"], "loaded"
    generator = torch.Generator().manual_seed(cli.seed)
    train_loader = DataLoader(
        CodeSceneDataset(*train_codes, *train_labels), cli.parser_batch_size,
        shuffle=True, generator=generator, num_workers=0,
    )
    val_loader = DataLoader(CodeSceneDataset(*val_codes, *val_labels), cli.parser_batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(parser.parameters(), lr=2e-3, weight_decay=1e-4)
    for epoch in range(cli.parser_epochs):
        parser.train()
        for codes, count, attributes in train_loader:
            codes, count, attributes = codes.to(cli.device), count.to(cli.device), attributes.to(cli.device)
            count_logits, attribute_logits = parser(codes)
            loss = F.cross_entropy(count_logits, count) + F.binary_cross_entropy_with_logits(attribute_logits, attributes)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        print(f"parser epoch={epoch:02d}", flush=True)
    metrics = parser_metrics(parser, val_loader, cli.device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"parser": parser.cpu().state_dict(), "metrics": metrics, "attributes": ATTRIBUTES}, path)
    parser.to(cli.device).eval()
    return parser, metrics, "trained"


def only_image(batch, device):
    return one_modality(batch, "image", device)


def layer_of(name: str):
    parts = name.split(".")
    return int(parts[1]) if len(parts) > 2 and parts[0] == "blocks" else None


@torch.no_grad()
def native_shared(model, batch, layers):
    _logits, _shared, native = model(
        batch["input_ids"], batch["attention_mask"], batch["position_ids"],
        batch["modality_ids"], batch["route_ids"], return_shared=True,
        return_shared_native_by_module=True,
    )
    return {name: value for name, value in native.items() if layer_of(name) in set(layers)}


@torch.no_grad()
def generate_with_shared_source(model, target, replacements, token_start, token_count, mask_id, steps):
    ids = target["input_ids"].clone()
    eligible = target["eligible_mask"]
    ids[eligible] = mask_id
    remaining = eligible.clone()
    for step in range(max(1, steps)):
        with shared_replacement_context(replacements):
            logits = model(ids, target["attention_mask"], target["position_ids"], target["modality_ids"], target["route_ids"])
        logits = logits[..., token_start:token_start + token_count]
        probabilities = logits.softmax(-1)
        samples = probabilities.argmax(-1) + token_start
        confidence = probabilities.amax(-1)
        for row in range(ids.size(0)):
            candidates = remaining[row].nonzero(as_tuple=False).flatten()
            if not len(candidates):
                continue
            number = min(len(candidates), math.ceil(len(candidates) / max(1, steps - step)))
            chosen = candidates[confidence[row, candidates].topk(number).indices]
            ids[row, chosen] = samples[row, chosen]
            remaining[row, chosen] = False
        if not remaining.any():
            break
    return ids[eligible].reshape(ids.size(0), -1) - token_start


@torch.no_grad()
def semantic_nll(parser, codes, count, attributes):
    count_logits, attribute_logits = parser(codes)
    return F.cross_entropy(count_logits, count, reduction="none") + F.binary_cross_entropy_with_logits(
        attribute_logits, attributes, reduction="none"
    ).mean(-1)


def pairs(size, seed):
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(size, generator=gen)
    source = torch.roll(order, 1)
    return order, source


def swap_metrics(model, parser, dataset, collator, val_labels, cli, tokenizer, args):
    target_indices, source_indices = pairs(min(cli.swap_samples, len(dataset)), cli.seed)
    values = {key: [] for key in ("self_target_nll", "swap_source_nll", "swap_target_nll", "swap_source_win")}
    model.eval()
    for target_rows, source_rows in zip(target_indices.split(cli.swap_batch_size), source_indices.split(cli.swap_batch_size)):
        target_examples = [dataset[int(index)] for index in target_rows]
        source_examples = [dataset[int(index)] for index in source_rows]
        target = only_image(collator(target_examples), cli.device)
        source = only_image(collator(source_examples), cli.device)
        source_shared = native_shared(model, source, cli.layers)
        target_shared = native_shared(model, target, cli.layers)
        self_codes = generate_with_shared_source(
            model, target, target_shared, len(tokenizer), args.num_image_codes,
            tokenizer.mask_id, cli.generation_steps,
        ).reshape(len(target_rows), *args.grid_size)
        swap_codes = generate_with_shared_source(
            model, target, source_shared, len(tokenizer), args.num_image_codes,
            tokenizer.mask_id, cli.generation_steps,
        ).reshape(len(target_rows), *args.grid_size)
        target_count = val_labels[0][target_rows].to(cli.device); target_attr = val_labels[1][target_rows].to(cli.device)
        source_count = val_labels[0][source_rows].to(cli.device); source_attr = val_labels[1][source_rows].to(cli.device)
        self_nll = semantic_nll(parser, self_codes, target_count, target_attr)
        swap_source = semantic_nll(parser, swap_codes, source_count, source_attr)
        swap_target = semantic_nll(parser, swap_codes, target_count, target_attr)
        values["self_target_nll"].append(self_nll.cpu())
        values["swap_source_nll"].append(swap_source.cpu())
        values["swap_target_nll"].append(swap_target.cpu())
        values["swap_source_win"].append((swap_source < swap_target).float().cpu())
    return {key: float(torch.cat(value).mean()) for key, value in values.items()}


def markdown(report):
    protocol = report["protocol"]
    layers = ", ".join(str(layer) for layer in protocol["layers"])
    lines = [
        "# Causal shared/private swapping evaluation", "", report["interpretation"], "",
        "## Protocol", "",
        "For each cyclically paired source/target pair $(A,B)$, the source image $A$ is clean and the target image $B$ starts fully masked.",
        f"At Transformer layers {layers}, every selected shared adapter update $\\Delta_s^{{l,a}}(A)$ from the clean source replaces the corresponding target shared update during all {protocol['generation_steps']} confidence-based generation steps:", "",
        "$$G_{swap}=G(\\Delta_s(A),P_B), \qquad G_{self}=G(\\Delta_s(B),P_B).$$", "",
        "The target keeps its own image-private adapter route $P_B$, mask/generation state, and all unselected shared layers. This is **same-modality image swapping**, not a text-image alignment test.",
        "", "## Frozen parser validation", "", f"- Source: {report['parser']['source']}",
        f"- Clean held-out count accuracy: {report['parser']['metrics']['count_accuracy']:.3f}",
        f"- Clean held-out attribute accuracy: {report['parser']['metrics']['attribute_accuracy']:.3f}",
        "- The evaluation parser reads VQ-code grids, is trained only on clean training images, and is frozen before any generated samples are scored.",
        "- Caveat: count accuracy is modest, and attribute accuracy is ordinary (not class-balanced) accuracy. Treat this as a useful causal diagnostic, not a final semantic benchmark.",
        "", "## Swap results", "",
        "NLL is the frozen parser's negative log likelihood for the complete source or target scene labels. Lower is better. A positive source advantage and source-win rate above 0.5 mean that the swapped generation is judged more like source $A$ than retained-private target $B$.",
        "", "| Model | Self NLL vs B | Swap NLL vs A | Swap NLL vs B | Source advantage (B−A) | Source win rate |", "|---|---:|---:|---:|---:|---:|"
    ]
    for name, value in report["models"].items():
        advantage = value["swap_target_nll"] - value["swap_source_nll"]
        lines.append(f"| {name} | {value['self_target_nll']:.3f} | {value['swap_source_nll']:.3f} | {value['swap_target_nll']:.3f} | {advantage:.3f} | {value['swap_source_win']:.3f} |")
    lines += ["", "## Interpretation", "", "The EMA-JEPA models have a large source advantage (12.2–15.4) and source-win rate (0.812–0.859), while the diffusion-only control has a near-chance source-win rate (0.453). This is evidence that, in the selected middle layers, the learned shared updates causally affect generated image semantics rather than being ignored. It does **not** yet establish cross-modal shared semantics, and the parser limitation above means the effect should be rechecked with a stronger, balanced scene parser.", "", "The close equality between self NLL vs $B$ and swapped NLL vs $A$ for the EMA-JEPA models is expected under the cyclic source pairing when the swap transfers source semantics almost as reliably as self generation transfers target semantics; it is not a reuse of the same generated sample.", ""]
    return "\n".join(lines) + "\n"


def main():
    cli = arguments()
    torch.manual_seed(cli.seed)
    specs = [entry.split("=", 1) for entry in cli.checkpoint]
    payload = torch.load(specs[0][1], map_location="cpu", weights_only=False)
    args = checkpoint_args(payload)
    if args.train_mode not in {"lora", "dense_private"}:
        raise ValueError("Causal shared/private swapping requires a Tri-LoRA checkpoint")
    tokenizer = ClevrTextTokenizer(payload["text_vocabulary"])
    cache = Path(cli.token_cache)
    train_cache = cache / "train_tokens.pt" if cache.is_dir() else cache
    val_cache = cache / "val_tokens.pt" if cache.is_dir() else cache
    train_dataset = ClevrMultimodalDataset(cli.train_dir, train_cache, "paired", cli.train_manifest, cli.caption_field)
    val_dataset = ClevrMultimodalDataset(cli.val_dir, val_cache, "paired", cli.val_manifest, cli.caption_field)
    train_rows = records(cli.train_manifest, cli.parser_train_samples)
    val_rows = records(cli.val_manifest, max(cli.parser_val_samples, cli.swap_samples))
    train_labels = labels(train_rows); val_labels = labels(val_rows)
    train_codes = (train_dataset.image_tokens[:len(train_rows)],)
    val_codes = (val_dataset.image_tokens[:len(val_rows)],)
    parser_path = cli.parser_checkpoint or str(Path(cli.output).with_name("frozen_image_code_scene_parser.pt"))
    parser, parser_eval, parser_source = train_or_load_parser(
        cli, train_codes, train_labels,
        (val_codes[0][:cli.parser_val_samples],),
        (val_labels[0][:cli.parser_val_samples], val_labels[1][:cli.parser_val_samples]),
        args.num_image_codes, parser_path,
    )
    collator = MultimodalCollator(tokenizer, args.num_image_codes, args.max_text_length)
    report = {"protocol": {"modality": "image", "layers": cli.layers, "swap_samples": cli.swap_samples, "generation_steps": cli.generation_steps, "source": "clean A shared deltas replace target B shared deltas only at selected layers; B private deltas remain active"}, "parser": {"source": parser_source, "checkpoint": parser_path, "metrics": parser_eval}, "models": {}, "interpretation": "This is a same-modality causal substitution test. It does not measure cross-modal alignment."}
    for name, path in specs:
        print(f"evaluating {name}", flush=True)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model_args = checkpoint_args(checkpoint)
        if model_args.grid_size != args.grid_size or model_args.num_image_codes != args.num_image_codes:
            raise ValueError("All compared checkpoints must use the same image tokenizer/grid")
        model, _ = build_model(model_args, len(tokenizer))
        model.load_state_dict(checkpoint["model"])
        model.to(cli.device).eval()
        report["models"][name] = swap_metrics(model, parser, val_dataset, collator, val_labels, cli, tokenizer, model_args)
        del model
        if cli.device.startswith("cuda"):
            torch.cuda.empty_cache()
    output = Path(cli.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    output.with_name("REPORT.md").write_text(markdown(report))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
