#!/usr/bin/env python3
"""Binding (attribute, relation) and retrieval for every layer, sublayer, and
branch-decomposed residual stream.

For no-base Tri-LoRA every sublayer output is shared update + private update +
bias, so the residual stream after block L decomposes exactly:

    h_L = h0 + sum_{i<=L} (b_att,i + ds_att,i + dp_att,i + b_mlp,i + ds_mlp,i + dp_mlp,i)

Features (content-token mean, L2-normalized):

* ``emb.token`` / ``emb.h0``: token embeddings / h0 = token + position embedding.
* ``L{l}.residual``: h_l.  ``L{l}.writes``: h_l - h0 (every sublayer write).
* ``L{l}.shared_writes`` / ``L{l}.private_writes``: cumulative shared /
  private updates of attention and MLP over blocks <= l (biases excluded:
  they are constant over tokens).  ``L{l}.h0+shared_writes`` etc. add h0.
* For l in --sublayer-layers: ``L{l}.attn_out`` / ``L{l}.mlp_out`` (whole
  sublayer outputs), ``L{l}.attn_plus_residual`` (the residual stream after
  the attention sublayer, h_(l-1) + attention output), and ``L{l}.attn_shared``, ``attn_private``,
  ``mlp_shared``, ``mlp_private`` (the four native updates).

The decomposition is verified numerically for every model before scoring.
Captions and minimal pairs are read from an existing binding-swap output
directory, and the retrieval gallery from an existing ``variants.jsonl``, so
results are directly comparable with those evaluations.
"""

from __future__ import annotations

from clevr_paths import CLEVR_DATA_ROOT, CLEVR_GEN_ROOT
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import binding_swap_captions as captions_module
from data import MultimodalCollator
from models import (
    IMAGE_PRIVATE_ONLY_ROUTE_ID,
    IMAGE_ROUTE_ID,
    TEXT_PRIVATE_ONLY_ROUTE_ID,
)
from evaluate_binding_swap import fit_probe, label_matrix, read_human, TIE_TOLERANCE
from evaluate_layer_retrieval_sweep import per_query_cross_pattern
from evaluate_shared_private_retrieval import load_model

ATTRIBUTE_TYPES = ("color", "shape", "material", "size")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--binding-dir", default="outputs/binding_swap_eval")
    parser.add_argument("--gallery", default="outputs/shared_private_semantic_template_retrieval/plain_lora_epoch003/variants.jsonl")
    parser.add_argument("--manifest", default=CLEVR_DATA_ROOT + "/platonic_text_only_v1_1m/val_text_only_human.jsonl")
    parser.add_argument("--train-manifest", default=CLEVR_DATA_ROOT + "/platonic_text_only_v1_2m/train_text_only_human.jsonl")
    parser.add_argument("--probe-train-samples", type=int, default=20000)
    parser.add_argument("--probe-val-offset", type=int, default=10000)
    parser.add_argument("--probe-val-samples", type=int, default=2048)
    parser.add_argument("--min-label-count", type=int, default=50)
    parser.add_argument("--sublayer-layers", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


@torch.no_grad()
def _routed(route_ids, ablate_private, ablate_shared):
    if ablate_private:
        return torch.full_like(route_ids, -1)
    if ablate_shared:
        return torch.where(
            route_ids.eq(IMAGE_ROUTE_ID),
            torch.full_like(route_ids, IMAGE_PRIVATE_ONLY_ROUTE_ID),
            torch.full_like(route_ids, TEXT_PRIVATE_ONLY_ROUTE_ID),
        )
    return route_ids


@torch.no_grad()
def encode(model, tokenizer, model_args, texts, batch_size, device, sublayer_layers, check=False,
           ablate_private=False, ablate_shared=False):
    collator = MultimodalCollator(tokenizer, model_args.num_image_codes, model_args.max_text_length)
    lora = model_args.train_mode in {"lora", "dense_private"}
    # Route ids decide which branches contribute. -1 matches no private branch,
    # leaving the shared path alone; TEXT_PRIVATE_ONLY_ROUTE_ID zeroes the shared
    # write instead and keeps only the private one. Either way the *inputs* to
    # later blocks are also branch-free, which a post-hoc decomposition of a
    # mixed forward pass cannot give.
    if ablate_private or ablate_shared:
        lora = False
    if ablate_private and ablate_shared:
        raise ValueError("ablate_private and ablate_shared are exclusive")
    n = len(model.blocks)
    captured, hooks = {}, []
    # Captures are copied so later code can never alias them.  The embedding
    # input is stored as "embed_out" and block outputs as "block{i}", so the
    # two can never share a key (a "h0" / "h{i}" naming collided at i = 0).
    hooks.append(model.token_embed.register_forward_hook(lambda _m, _i, o: captured.__setitem__("token", o.detach().clone())))
    hooks.append(model.blocks[0].register_forward_pre_hook(lambda _m, args: captured.__setitem__("embed_out", args[0].detach().clone())))
    for i, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(lambda _m, _i, o, i=i: captured.__setitem__(f"block{i}", o.detach().clone())))
        hooks.append(block.attn.out_proj.register_forward_hook(lambda _m, _i, o, i=i: captured.__setitem__(f"attn{i}", o.detach().clone())))
        hooks.append(block.mlp[3].register_forward_hook(lambda _m, _i, o, i=i: captured.__setitem__(f"mlp{i}", o.detach().clone())))
    values: dict[str, list[torch.Tensor]] = {}
    worst = 0.0
    try:
        for start in range(0, len(texts), batch_size):
            batch = collator([{"kind": "text", "text": t, "pair_index": start + k}
                              for k, t in enumerate(texts[start:start + batch_size])])
            batch = {k: v.to(device) for k, v in batch.items()}
            captured.clear()
            out = model(batch["input_ids"], batch["attention_mask"], batch["position_ids"],
                        batch["modality_ids"],
                        _routed(batch["route_ids"], ablate_private, ablate_shared),
                        return_shared=lora,
                        return_shared_private_native_by_module=lora)
            h0 = captured["embed_out"].double()
            feats = {"emb.token": captured["token"].double(), "emb.h0": h0}
            if lora:
                _, _, shared, private = out
            shared_sum = torch.zeros_like(h0)
            private_sum = torch.zeros_like(h0)
            for i in range(n):
                h = captured[f"block{i}"].double()
                feats[f"L{i}.residual"] = h
                feats[f"L{i}.writes"] = h - h0
                if lora:
                    for kind, name in (("attn", f"blocks.{i}.attn.out_proj"), ("mlp", f"blocks.{i}.mlp.3")):
                        ds, dp = shared[name].double(), private[name].double()
                        shared_sum = shared_sum + ds
                        private_sum = private_sum + dp
                        if i in sublayer_layers:
                            feats[f"L{i}.{kind}_shared"] = ds
                            feats[f"L{i}.{kind}_private"] = dp
                    feats[f"L{i}.shared_writes"] = shared_sum.clone()
                    feats[f"L{i}.private_writes"] = private_sum.clone()
                    feats[f"L{i}.h0+shared_writes"] = h0 + shared_sum
                    feats[f"L{i}.h0+private_writes"] = h0 + private_sum
                if i in sublayer_layers:
                    feats[f"L{i}.attn_out"] = captured[f"attn{i}"].double()
                    feats[f"L{i}.mlp_out"] = captured[f"mlp{i}"].double()
                    # Residual stream between the two sublayers: h_(i-1) + attention output.
                    previous = h0 if i == 0 else captured[f"block{i - 1}"].double()
                    feats[f"L{i}.attn_plus_residual"] = previous + captured[f"attn{i}"].double()
            if check and start == 0:
                # Exactness of the decomposition on real data, at real tokens
                # (route masks zero branch outputs at padding positions).
                real = batch["attention_mask"].bool()
                bias = torch.zeros_like(h0)
                for i in range(n):
                    rebuilt_sub = h0 + sum((captured[f"attn{j}"].double() + captured[f"mlp{j}"].double()) for j in range(i + 1))
                    worst = max(worst, float((rebuilt_sub - captured[f"block{i}"].double())[real].abs().max()))
                    if lora:
                        for mod in (model.blocks[i].attn.out_proj, model.blocks[i].mlp[3]):
                            # `shared_bias` is the affine offset that replaces a
                            # deleted base weight. In dense_private the base is
                            # kept, so its bias is already inside the recorded
                            # shared write and there is nothing to add here.
                            if mod.shared_bias is not None:
                                bias = bias + mod.shared_bias.double()
                        rebuilt = h0 + feats[f"L{i}.shared_writes"] + feats[f"L{i}.private_writes"] + bias
                        worst = max(worst, float((rebuilt - captured[f"block{i}"].double())[real].abs().max()))
            weights = batch["eligible_mask"].double().unsqueeze(-1)
            for name, tensor in feats.items():
                pooled = (tensor * weights).sum(1) / weights.sum(1).clamp_min(1)
                values.setdefault(name, []).append(F.normalize(pooled, dim=1).float().cpu())
    finally:
        for hook in hooks:
            hook.remove()
    return {k: torch.cat(v).numpy() for k, v in values.items()}, worst


def bootstrap_groups(scores, groups, world_of, repetitions, seed):
    rng = np.random.default_rng(seed)
    result = {}
    for name, (mask, weights) in groups.items():
        values, worlds, wts = scores[mask], world_of[mask], weights[mask]
        unique = np.unique(worlds)
        by_world = {w: np.flatnonzero(worlds == w) for w in unique}
        draws = np.empty(repetitions)
        for r in range(repetitions):
            picked = np.concatenate([by_world[w] for w in rng.choice(unique, len(unique))])
            draws[r] = (wts[picked] * values[picked]).sum() / wts[picked].sum()
        result[name] = {"value": float((wts * values).sum() / wts.sum()),
                        "ci95": [float(x) for x in np.quantile(draws, (0.025, 0.975))], "pairs": int(mask.sum())}
    return result


def main():
    args = arguments()
    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bdir = Path(args.binding_dir)
    captions = [json.loads(l) for l in (bdir / "captions.jsonl").read_text().splitlines()]
    items = [json.loads(l) for l in (bdir / "items.jsonl").read_text().splitlines()]
    gallery = [json.loads(l) for l in Path(args.gallery).read_text().splitlines()]
    train_texts, train_worlds = read_human(args.train_manifest, 0, args.probe_train_samples)
    val_texts, val_worlds = read_human(args.manifest, args.probe_val_offset, args.probe_val_samples)
    labels = captions_module.all_conjunction_labels()
    train_y = label_matrix(train_worlds, labels)
    keep = (train_y.sum(0) >= args.min_label_count) & ((1 - train_y).sum(0) >= args.min_label_count)
    labels = [l for l, k in zip(labels, keep) if k]
    train_y, val_y = train_y[:, keep], label_matrix(val_worlds, labels)
    position = {l: k for k, l in enumerate(labels)}
    pairs, seen = [], set()
    for item in items:
        kind = item["negative_type"]
        if kind not in captions_module.SWAP_TYPES or (item["query"], item["negative"]) in seen:
            continue
        seen.add((item["query"], item["negative"]))
        ql = captions_module.conjunction_labels(captions[item["query"]]["world"])
        nl = captions_module.conjunction_labels(captions[item["negative"]]["world"])
        flipped = sorted((ql ^ nl) & set(labels), key=position.get)
        if flipped:
            pairs.append((item["query"], item["negative"], np.asarray([position[l] for l in flipped]),
                          np.asarray([1.0 if l in ql else -1.0 for l in flipped]), kind, item["world_index"]))
    kinds = np.asarray([p[4] for p in pairs])
    world_of_pair = np.asarray([p[5] for p in pairs])
    groups = {}
    for gname, members in (("attribute", ATTRIBUTE_TYPES), ("relation", ("relation",)), ("overall", captions_module.SWAP_TYPES)):
        mask = np.isin(kinds, members)
        weights = np.zeros(len(kinds))
        for t in members:
            sel = kinds == t
            if sel.any():
                weights[sel] = 1.0 / sel.sum()
        groups[gname] = (mask, weights)
    gw = np.asarray([r["world_id"] for r in gallery])
    gwi = np.asarray([r["world_index"] for r in gallery])
    gp = np.asarray([r["pattern_id"] for r in gallery])
    for spec in args.checkpoint:
        label, path = spec.split("=", 1)
        dest = out / f"{label}.json"
        if dest.exists():
            print(f"skip {label}", flush=True)
            continue
        model, tokenizer, model_args = load_model(path, device)
        test, worst = encode(model, tokenizer, model_args, [c["caption"] for c in captions], args.batch_size,
                             device, set(args.sublayer_layers), check=True)
        if worst > 1e-3:
            raise RuntimeError(f"{label}: residual decomposition error {worst:.2e}")
        train, _ = encode(model, tokenizer, model_args, train_texts, args.batch_size, device, set(args.sublayer_layers))
        val, _ = encode(model, tokenizer, model_args, val_texts, args.batch_size, device, set(args.sublayer_layers))
        gal, _ = encode(model, tokenizer, model_args, [g["caption"] for g in gallery], args.batch_size, device, set(args.sublayer_layers))
        metrics = {}
        for name in test:
            logits = fit_probe(train[name], train_y, device)
            vl = logits(val[name]) > 0
            balanced = [0.5 * (vl[val_y[:, k] == 1, k].mean() + (~vl[val_y[:, k] == 0, k]).mean())
                        for k in range(val_y.shape[1]) if 0 < val_y[:, k].sum() < len(val_y)]
            tl = logits(test[name])
            scores = np.empty(len(pairs))
            for j, (q, neg, flips, signs, _, _) in enumerate(pairs):
                d = (tl[q, flips] - tl[neg, flips]) * signs
                scores[j] = np.where(np.abs(d) <= TIE_TOLERANCE, 0.5, (d > 0).astype(float)).mean()
            per_query = per_query_cross_pattern(gal[name], gw, gp, device)
            by_world = np.stack([per_query[gwi == w].mean(0) for w in np.unique(gwi)])
            rng = np.random.default_rng(args.seed)
            draws = by_world[rng.integers(0, len(by_world), (args.bootstrap, len(by_world)))].mean(1)
            lo, hi = np.quantile(draws, (0.025, 0.975), axis=0)
            metrics[name] = {
                "binding": bootstrap_groups(scores, groups, world_of_pair, args.bootstrap, args.seed),
                "ordinary_probe_balanced_accuracy": float(np.mean(balanced)),
                "retrieval": {k: {"value": float(per_query[:, i].mean()), "ci95": [float(lo[i]), float(hi[i])]}
                              for i, k in enumerate(("R@1", "R@5", "R@10", "MRR"))},
            }
        dest.write_text(json.dumps({"protocol": {"checkpoint": str(Path(path).resolve()), "train_mode": model_args.train_mode,
                                                  "decomposition_max_abs_error": worst, "pairs": len(pairs), "labels": len(labels),
                                                  "pairs_by_type": Counter(kinds.tolist()), "bootstrap": args.bootstrap},
                                     "metrics": metrics}, indent=2) + "\n")
        m = metrics
        pick = lambda f: (f"{100*m[f]['binding']['attribute']['value']:.1f}/{100*m[f]['binding']['relation']['value']:.1f}/"
                          f"{100*m[f]['retrieval']['R@10']['value']:.1f}") if f in m else "-"
        print(f"{label}: decomposition error {worst:.1e} | attr/rel/R@10  L4.residual {pick('L4.residual')} "
              f"L4.shared_writes {pick('L4.shared_writes')} L4.private_writes {pick('L4.private_writes')} L4.mlp_out {pick('L4.mlp_out')}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
