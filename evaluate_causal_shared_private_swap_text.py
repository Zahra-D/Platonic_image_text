"""Text counterpart of the image causal shared/private swap evaluation.

For clean text source A and fully masked target text B, replace native shared
LoRA deltas at selected layers with clean-A deltas, retain B's text-private
LoRA route, generate B greedily, and score source versus target scene labels
with a frozen text-only scene parser.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from analyze_layer_gradient_conflict import one_modality
from analyze_paired_representations import checkpoint_args
from data import ClevrMultimodalDataset, ClevrTextTokenizer, MultimodalCollator
from evaluate_causal_shared_private_swap import ATTRIBUTES, labels, records, pairs
from models.lora import shared_replacement_context
from train_multimodal import build_model


def cli_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--train-dir", required=True); p.add_argument("--val-dir", required=True)
    p.add_argument("--train-manifest", required=True); p.add_argument("--val-manifest", required=True)
    p.add_argument("--token-cache", required=True); p.add_argument("--caption-field", default="caption_human")
    p.add_argument("--layers", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--parser-train-samples", type=int, default=20000); p.add_argument("--parser-val-samples", type=int, default=1024)
    p.add_argument("--parser-epochs", type=int, default=12); p.add_argument("--parser-batch-size", type=int, default=128)
    p.add_argument("--swap-samples", type=int, default=64); p.add_argument("--swap-batch-size", type=int, default=8)
    p.add_argument("--generation-steps", type=int, default=64); p.add_argument("--sample-count", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260915); p.add_argument("--parser-checkpoint"); p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class TextSceneDataset(Dataset):
    def __init__(self, ids, attention, count, attr): self.ids, self.attention, self.count, self.attr = ids, attention, count, attr
    def __len__(self): return len(self.ids)
    def __getitem__(self, i): return self.ids[i], self.attention[i], self.count[i], self.attr[i]


class FrozenTextSceneParser(nn.Module):
    def __init__(self, vocab, count_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab, 128, padding_idx=0)
        self.body = nn.Sequential(nn.Linear(128, 192), nn.GELU(), nn.Linear(192, 160), nn.GELU())
        self.count_head = nn.Linear(160, count_classes); self.attr_head = nn.Linear(160, len(ATTRIBUTES))
    def forward(self, ids, attention):
        x = self.embedding(ids)
        weights = attention.unsqueeze(-1).to(x.dtype)
        x = (x * weights).sum(1) / weights.sum(1).clamp_min(1)
        x = self.body(x)
        return self.count_head(x), self.attr_head(x)


def text_arrays(dataset, tokenizer, max_length, n):
    ids = torch.full((n, max_length), tokenizer.pad_id, dtype=torch.long)
    attention = torch.zeros((n, max_length), dtype=torch.bool)
    for i in range(n):
        values = tokenizer.encode(dataset.text_records[i]["text"], max_length)
        ids[i, :len(values)] = torch.tensor(values); attention[i, :len(values)] = True
    return ids, attention


def parser_metrics(parser, loader, device):
    parser.eval(); cc = ct = ac = at = 0
    with torch.no_grad():
        for ids, attention, count, attr in loader:
            out_c, out_a = parser(ids.to(device), attention.to(device)); count, attr = count.to(device), attr.to(device)
            cc += out_c.argmax(-1).eq(count).sum().item(); ct += len(count)
            ac += (out_a.sigmoid().ge(.5) == attr.bool()).sum().item(); at += attr.numel()
    return {"count_accuracy": cc / max(1, ct), "attribute_accuracy": ac / max(1, at)}


def train_or_load_parser(cli, train, val, vocab, count_classes, path):
    parser = FrozenTextSceneParser(vocab, count_classes).to(cli.device); path = Path(path)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False); parser.load_state_dict(payload["parser"])
        return parser.eval(), payload["metrics"], "loaded"
    loader = DataLoader(TextSceneDataset(*train), cli.parser_batch_size, shuffle=True, generator=torch.Generator().manual_seed(cli.seed))
    val_loader = DataLoader(TextSceneDataset(*val), cli.parser_batch_size)
    opt = torch.optim.AdamW(parser.parameters(), lr=2e-3, weight_decay=1e-4)
    for epoch in range(cli.parser_epochs):
        parser.train()
        for ids, attention, count, attr in loader:
            out_c, out_a = parser(ids.to(cli.device), attention.to(cli.device)); count, attr = count.to(cli.device), attr.to(cli.device)
            loss = F.cross_entropy(out_c, count) + F.binary_cross_entropy_with_logits(out_a, attr)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        print(f"text parser epoch={epoch:02d}", flush=True)
    metrics = parser_metrics(parser, val_loader, cli.device); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"parser": parser.cpu().state_dict(), "metrics": metrics, "attributes": ATTRIBUTES}, path)
    return parser.to(cli.device).eval(), metrics, "trained"


def only_text(batch, device): return one_modality(batch, "text", device)
def select(batch, indices): return {key: value[indices] if torch.is_tensor(value) else value for key, value in batch.items()}
def layer(name):
    parts = name.split("."); return int(parts[1]) if len(parts) > 2 and parts[0] == "blocks" else None


@torch.no_grad()
def native_shared(model, batch, layers):
    _logits, _shared, native = model(batch["input_ids"], batch["attention_mask"], batch["position_ids"], batch["modality_ids"], batch["route_ids"], return_shared=True, return_shared_native_by_module=True)
    allowed = set(layers); return {name: value for name, value in native.items() if layer(name) in allowed}


@torch.no_grad()
def generate(model, target, replacements, tokenizer, steps):
    ids = target["input_ids"].clone(); eligible = target["eligible_mask"]; ids[eligible] = tokenizer.mask_id; remaining = eligible.clone()
    for step in range(max(1, steps)):
        with shared_replacement_context(replacements):
            logits = model(ids, target["attention_mask"], target["position_ids"], target["modality_ids"], target["route_ids"])
        # Only ordinary lexical/punctuation tokens may fill caption-content slots.
        probs = logits[..., len(tokenizer.SPECIAL_TOKENS):len(tokenizer)].softmax(-1)
        proposal = probs.argmax(-1) + len(tokenizer.SPECIAL_TOKENS); confidence = probs.amax(-1)
        for row in range(len(ids)):
            candidates = remaining[row].nonzero().flatten()
            if len(candidates):
                n = min(len(candidates), math.ceil(len(candidates) / max(1, steps - step)))
                chosen = candidates[confidence[row, candidates].topk(n).indices]
                ids[row, chosen] = proposal[row, chosen]; remaining[row, chosen] = False
        if not remaining.any(): break
    return ids


@torch.no_grad()
def nll(parser, ids, attention, count, attr):
    logits_c, logits_a = parser(ids, attention)
    return F.cross_entropy(logits_c, count, reduction="none") + F.binary_cross_entropy_with_logits(logits_a, attr, reduction="none").mean(-1)


def evaluate(model, parser, dataset, collator, labels_, tokenizer, args):
    target_idx, source_idx = pairs(min(args.swap_samples, len(dataset)), args.seed)
    values = {key: [] for key in ("self_target_nll", "swap_source_nll", "swap_target_nll", "swap_source_win")}; samples = []
    for t, s in zip(target_idx.split(args.swap_batch_size), source_idx.split(args.swap_batch_size)):
        # Collate jointly so clean source deltas and target tensors share the same padded sequence length.
        all_batch = only_text(collator([dataset[int(i)] for i in t] + [dataset[int(i)] for i in s]), args.device)
        target = select(all_batch, torch.arange(len(t), device=args.device)); source = select(all_batch, torch.arange(len(t), 2 * len(t), device=args.device))
        source_shared, target_shared = native_shared(model, source, args.layers), native_shared(model, target, args.layers)
        self_ids = generate(model, target, target_shared, tokenizer, args.generation_steps); swap_ids = generate(model, target, source_shared, tokenizer, args.generation_steps)
        tc, ta = labels_[0][t].to(args.device), labels_[1][t].to(args.device); sc, sa = labels_[0][s].to(args.device), labels_[1][s].to(args.device)
        self_score, source_score, target_score = nll(parser, self_ids, target["attention_mask"], tc, ta), nll(parser, swap_ids, target["attention_mask"], sc, sa), nll(parser, swap_ids, target["attention_mask"], tc, ta)
        for key, value in zip(values, (self_score, source_score, target_score, (source_score < target_score).float())): values[key].append(value.cpu())
        if len(samples) < args.sample_count:
            for row in range(min(len(t), args.sample_count - len(samples))):
                clean = target["input_ids"][row][target["eligible_mask"][row]].tolist(); source_clean = source["input_ids"][row][source["eligible_mask"][row]].tolist()
                samples.append({"target_B": tokenizer.decode(clean), "source_A": tokenizer.decode(source_clean), "self": tokenizer.decode(self_ids[row][target["eligible_mask"][row]].tolist()), "swap": tokenizer.decode(swap_ids[row][target["eligible_mask"][row]].tolist())})
    return ({key: float(torch.cat(value).mean()) for key, value in values.items()}, samples)


def main():
    args = cli_args(); torch.manual_seed(args.seed); specs = [item.split("=", 1) for item in args.checkpoint]
    first = torch.load(specs[0][1], map_location="cpu", weights_only=False); model_args = checkpoint_args(first); tokenizer = ClevrTextTokenizer(first["text_vocabulary"])
    cache = Path(args.token_cache); train_cache = cache / "train_tokens.pt" if cache.is_dir() else cache; val_cache = cache / "val_tokens.pt" if cache.is_dir() else cache
    train_set = ClevrMultimodalDataset(args.train_dir, train_cache, "paired", args.train_manifest, args.caption_field); val_set = ClevrMultimodalDataset(args.val_dir, val_cache, "paired", args.val_manifest, args.caption_field)
    train_rows, val_rows = records(args.train_manifest, args.parser_train_samples), records(args.val_manifest, max(args.parser_val_samples, args.swap_samples)); train_y, val_y = labels(train_rows), labels(val_rows)
    train_x = text_arrays(train_set, tokenizer, model_args.max_text_length, len(train_rows)); val_x = text_arrays(val_set, tokenizer, model_args.max_text_length, len(val_rows))
    parser_path = args.parser_checkpoint or str(Path(args.output).with_name("frozen_text_scene_parser.pt"))
    parser, parser_eval, parser_source = train_or_load_parser(args, (*train_x, *train_y), (val_x[0][:args.parser_val_samples], val_x[1][:args.parser_val_samples], val_y[0][:args.parser_val_samples], val_y[1][:args.parser_val_samples]), len(tokenizer), int(max(train_y[0].max(), val_y[0].max()).item()) + 1, parser_path)
    collator = MultimodalCollator(tokenizer, model_args.num_image_codes, model_args.max_text_length)
    report = {"protocol": {"modality": "text", "layers": args.layers, "swap_samples": args.swap_samples, "generation_steps": args.generation_steps, "source": "clean A shared deltas replace fully masked B shared deltas; B text-private deltas remain active"}, "parser": {"source": parser_source, "checkpoint": parser_path, "metrics": parser_eval}, "models": {}, "samples": {}}
    for name, path in specs:
        payload = torch.load(path, map_location="cpu", weights_only=False); current_args = checkpoint_args(payload); model, _ = build_model(current_args, len(tokenizer)); model.load_state_dict(payload["model"]); model.to(args.device).eval()
        report["models"][name], report["samples"][name] = evaluate(model, parser, val_set, collator, val_y, tokenizer, args); del model
        if args.device.startswith("cuda"): torch.cuda.empty_cache()
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Text causal shared/private swap", "", "Shared source deltas are replaced on **every one of the 64 generation forwards**. Parser NLL is evaluated after all target content positions have been unmasked.", "", "| Model | Self B NLL | Swap A NLL | Swap B NLL | Source advantage | Source win |", "|---|---:|---:|---:|---:|---:|"]
    for name, v in report["models"].items(): lines.append(f"| {name} | {v['self_target_nll']:.3f} | {v['swap_source_nll']:.3f} | {v['swap_target_nll']:.3f} | {v['swap_target_nll']-v['swap_source_nll']:.3f} | {v['swap_source_win']:.3f} |")
    output.with_name("REPORT.md").write_text("\n".join(lines) + "\n")

if __name__ == "__main__": main()
