#!/usr/bin/env python3
"""Hard retrieval with an external pretrained model, as a check on the metric.

The task is only meaningful if a model that clearly understands English scores
well above chance on it.  This runs the *identical* items -- the ``texts.jsonl``
and ``tasks.jsonl`` written by ``evaluate_hard_retrieval.py`` -- through a
Hugging Face model instead of a checkpoint from this repo, with the same
pooling, the same cosine ranking, and the same expected-rank tie handling, and
writes its results beside the others so they share one table.

Every layer is scored, under two readouts: ``mean`` over the real tokens (what
this repo's evaluators use) and ``last``/``cls``, the pooled vector a causal or
encoder model would ordinarily be read out with.

The candidate captions are token-multiset-identical under this repo's word
tokenizer.  Whether they stay identical under the external model's subword
tokenizer is checked and reported, because if they do not, that model could in
principle exploit a purely lexical difference.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_hard_retrieval import score


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", action="append", default=[], metavar="LABEL=HF_NAME")
    p.add_argument("--items-dir", default="outputs/hard_retrieval_eval_all_sublayers",
                   help="Directory holding texts.jsonl and tasks.jsonl from the main evaluator.")
    p.add_argument("--output-dir", default=None, help="Defaults to --items-dir.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def read_items(items_dir: Path):
    texts = [json.loads(line)["caption"] for line in (items_dir / "texts.jsonl").read_text().splitlines()]
    tasks = [json.loads(line) for line in (items_dir / "tasks.jsonl").read_text().splitlines()]
    return texts, tasks


def subword_multisets_match(tokenizer, texts, tasks) -> dict:
    """Are the candidates still lexically identical under this tokenizer?"""
    mismatched = 0
    for task in tasks:
        bags = [Counter(tokenizer.tokenize(texts[i])) for i in task["candidates"]]
        if any(bag != bags[0] for bag in bags):
            mismatched += 1
    return {"tasks": len(tasks), "candidate_sets_not_multiset_identical": mismatched}


@torch.no_grad()
def encode(model, tokenizer, texts, batch_size, device):
    features: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start:start + batch_size], return_tensors="pt", padding=True, truncation=True,
        ).to(device)
        output = model(**batch, output_hidden_states=True)
        mask = batch["attention_mask"]
        special = torch.zeros_like(mask, dtype=torch.bool)
        for identifier in tokenizer.all_special_ids:
            special |= batch["input_ids"].eq(identifier)
        content = (mask.bool() & ~special).to(mask.dtype)
        # A sequence of only special tokens cannot happen here, but guard anyway.
        weights = content.unsqueeze(-1).to(output.hidden_states[0].dtype)
        last_index = mask.sum(dim=1) - 1
        for layer, hidden in enumerate(output.hidden_states):
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
            features.setdefault(f"L{layer}.mean", []).append(
                F.normalize(pooled.double(), dim=1).cpu()
            )
            last = hidden[torch.arange(hidden.size(0), device=device), last_index]
            features.setdefault(f"L{layer}.last", []).append(
                F.normalize(last.double(), dim=1).cpu()
            )
            if tokenizer.cls_token_id is not None:
                features.setdefault(f"L{layer}.cls", []).append(
                    F.normalize(hidden[:, 0].double(), dim=1).cpu()
                )
    return {name: torch.cat(parts).numpy() for name, parts in features.items()}


def main():
    args = arguments()
    from transformers import AutoModel, AutoTokenizer

    items_dir = Path(args.items_dir)
    out = Path(args.output_dir or args.items_dir)
    out.mkdir(parents=True, exist_ok=True)
    texts, tasks = read_items(items_dir)
    print(f"items: {len(texts)} captions, {len(tasks)} tasks", flush=True)
    device = torch.device(args.device)

    for spec in args.model:
        label, name = spec.split("=", 1)
        destination = out / f"{label}.json"
        if destination.exists():
            print(f"{label}: exists, skipping", flush=True)
            continue
        tokenizer = AutoTokenizer.from_pretrained(name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(name).to(device).eval()
        lexical = subword_multisets_match(tokenizer, texts, tasks)
        features = encode(model, tokenizer, texts, args.batch_size, device)
        metrics = {n: score(v, tasks, args.bootstrap, args.seed) for n, v in features.items()}
        destination.write_text(json.dumps({
            "protocol": {"model": name, "train_mode": "external",
                         "layers": int(model.config.num_hidden_layers),
                         "subword_lexical_check": lexical},
            "metrics": metrics,
        }, indent=2) + "\n")
        best = max(metrics, key=lambda k: metrics[k]["attribute|reworded"]["R@1"]["value"])
        show = lambda k, t: 100 * metrics[k][t]["R@1"]["value"]
        print(f"{label} ({model.config.num_hidden_layers} layers): best attribute feature {best} "
              f"{show(best, 'attribute|reworded'):.1f}% | relation {show(best, 'relation|reworded'):.1f}% "
              f"| candidates not multiset-identical under its tokenizer: "
              f"{lexical['candidate_sets_not_multiset_identical']}/{lexical['tasks']}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
