#!/usr/bin/env python3
"""Binding-swap evaluation: does a representation encode *which* object has
*which* attribute, beyond the bag of attribute words?

Primary metric -- minimal-pair conjunction probe.  Linear probes are trained on
real training captions to detect binding-dependent facts ("some object is red
and a cube", "a cube is right of a sphere").  They are tested on held-out pairs
of captions with identical tokens whose scenes differ by one binding swap, so
those facts flip.  The score is how often the probe ranks the true caption
above its swapped twin for each flipped fact.  A bag of words or any
order-free pooling of token embeddings gives the two captions identical
inputs and scores exactly 50%.

Secondary diagnostic -- cosine preference, described below.  With learned
absolute positions, pooled cosine similarity is dominated by word order, so it
measures "binding versus order" rather than binding alone.

For every held-out world there is a query caption and, per item:

* a positive: the same scene, either
  - ``reorder``: the same words in a different object/relation order, so its
    token multiset is identical to the query's; or
  - ``reword``: different synonyms, phrase styles, and sentence order;
* a negative: the query with exactly one binding exchanged (two objects swap
  a color / shape / material / size, or a relation's subject and anchor swap),
  rendered with the query's words, so its token multiset is identical to the
  query's; plus a ``different_scene`` sanity negative.

A bag of words, or any order-free pooling of context-free token embeddings,
is exactly tied on ``reorder`` items and prefers the negative on ``reword``
items.  A score above 50% therefore requires information about bindings.

Metric per feature: preference rate, the fraction of items where
cos(query, positive) > cos(query, negative), ties counted as one half, with a
world bootstrap; and a normalized margin, the mean cosine difference divided by
the standard deviation of cosines between queries of different worlds.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import binding_swap_captions as captions_module
from data import MultimodalCollator
from evaluate_shared_private_retrieval import load_model

TIE_TOLERANCE = 1e-9


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--manifest", default="/home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_1m/val_text_only_human.jsonl")
    parser.add_argument("--caption-generator", default="/home/zd25e122/clevr-dataset-gen_clone/image_generation/generate_human_captions.py")
    parser.add_argument("--num-worlds", type=int, default=512)
    parser.add_argument("--train-manifest", default="/home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_2m/train_text_only_human.jsonl")
    parser.add_argument("--probe-train-samples", type=int, default=20000)
    parser.add_argument("--probe-val-offset", type=int, default=10000,
                        help="Held-out human captions for probe sanity start here, disjoint from the test worlds.")
    parser.add_argument("--probe-val-samples", type=int, default=2048)
    parser.add_argument("--min-label-count", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=190)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--controls", action="store_true", help="Also score model-free lexical and random controls.")
    return parser.parse_args()


@torch.no_grad()
def encode(model, tokenizer, model_args, texts, batch_size, device):
    collator = MultimodalCollator(tokenizer, model_args.num_image_codes, model_args.max_text_length)
    captured, hooks = {}, []

    def keep(name):
        return lambda _m, _i, output: captured.__setitem__(name, output)

    hooks.append(model.token_embed.register_forward_hook(keep("embedding")))
    for index, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(keep(f"block{index}")))
        hooks.append(block.mlp[3].register_forward_hook(keep(f"mlp{index}")))
        hooks.append(block.attn.out_proj.register_forward_hook(keep(f"attn{index}")))
    # dense_private carries the same shared/private routes, with the dense
    # weight as the shared one.
    lora = model_args.train_mode in {"lora", "dense_private"}
    values: dict[str, list[torch.Tensor]] = {}
    try:
        for start in range(0, len(texts), batch_size):
            batch = collator([{"kind": "text", "text": text, "pair_index": start + k}
                              for k, text in enumerate(texts[start:start + batch_size])])
            batch = {name: value.to(device) for name, value in batch.items()}
            captured.clear()
            output = model(batch["input_ids"], batch["attention_mask"], batch["position_ids"],
                           batch["modality_ids"], batch["route_ids"], return_shared=lora,
                           return_shared_private_native_by_module=lora)
            tensors = dict(captured)
            if lora:
                _, _, shared, private = output
                for name in shared:
                    short = name.removeprefix("blocks.").replace(".attn.out_proj", ".out_proj")
                    if name.endswith(("attn.out_proj", "mlp.3")):
                        tensors[f"shared.{short}"] = shared[name]
                        tensors[f"private.{short}"] = private[name]
            weights = batch["eligible_mask"].to(torch.float32).unsqueeze(-1)
            for name, tensor in tensors.items():
                # float64 pooling: permuted identical tokens must give the same vector.
                pooled = (tensor.double() * weights.double()).sum(1) / weights.double().sum(1).clamp_min(1)
                values.setdefault(name, []).append(F.normalize(pooled, dim=1).cpu())
    finally:
        for hook in hooks:
            hook.remove()
    return {name: torch.cat(parts).numpy() for name, parts in values.items()}


def score_feature(features, items, query_pairs, world_of_item, repetitions, seed):
    q = np.asarray([i["query"] for i in items])
    p = np.asarray([i["positive"] for i in items])
    n = np.asarray([i["negative"] for i in items])
    cos_pos = (features[q] * features[p]).sum(1)
    cos_neg = (features[q] * features[n]).sum(1)
    diff = cos_pos - cos_neg
    win = np.where(np.abs(diff) <= TIE_TOLERANCE, 0.5, (diff > 0).astype(float))
    reference = (features[query_pairs[:, 0]] * features[query_pairs[:, 1]]).sum(1).std()
    margin = diff / max(reference, 1e-12)
    groups = {}
    for key in sorted({(i["negative_type"], i["positive_type"]) for i in items}):
        mask = np.asarray([(i["negative_type"], i["positive_type"]) == key for i in items])
        groups[f"{key[0]}|{key[1]}"] = mask
    for positive_type in sorted({i["positive_type"] for i in items}):
        groups[f"binding_mean|{positive_type}"] = np.asarray(
            [i["positive_type"] == positive_type and i["negative_type"] in captions_module.SWAP_TYPES for i in items])
    rng = np.random.default_rng(seed)
    result = {"different_world_cosine_sd": float(reference)}
    for name, mask in groups.items():
        worlds = world_of_item[mask]
        unique = np.unique(worlds)
        by_world = {w: np.flatnonzero(worlds == w) for w in unique}
        w_win, w_margin = win[mask], margin[mask]
        if name.startswith("binding_mean"):
            # Equal weight per swap type, so types with more valid items do not dominate.
            types = np.asarray([i["negative_type"] for i, m in zip(items, mask) if m])
            weights = np.zeros(len(types))
            for t in np.unique(types):
                weights[types == t] = 1.0 / (types == t).sum()
            weights /= weights.sum()
        else:
            weights = np.full(mask.sum(), 1.0 / mask.sum())
        point = (float((weights * w_win).sum()), float((weights * w_margin).sum()))
        draws = np.empty((repetitions, 2))
        for r in range(repetitions):
            picked = np.concatenate([by_world[w] for w in rng.choice(unique, len(unique))])
            wt = weights[picked] / weights[picked].sum()
            draws[r] = ((wt * w_win[picked]).sum(), (wt * w_margin[picked]).sum())
        low, high = np.quantile(draws, (0.025, 0.975), axis=0)
        result[name] = {
            "items": int(mask.sum()),
            "preference": {"value": point[0], "ci95": [float(low[0]), float(high[0])]},
            "normalized_margin": {"value": point[1], "ci95": [float(low[1]), float(high[1])]},
        }
    return result


def read_human(manifest: str, offset: int, count: int):
    texts, worlds = [], []
    with open(manifest) as handle:
        for index, line in enumerate(handle):
            if index < offset:
                continue
            if len(texts) == count:
                break
            row = json.loads(line)
            texts.append(row["caption_human"])
            worlds.append(row["world"])
    return texts, worlds


def label_matrix(worlds, labels):
    position = {label: k for k, label in enumerate(labels)}
    matrix = np.zeros((len(worlds), len(labels)), dtype=np.float32)
    for row, world in enumerate(worlds):
        for label in captions_module.conjunction_labels(world):
            if label in position:
                matrix[row, position[label]] = 1.0
    return matrix


def fit_probe(train_x, train_y, device, steps=300, lr=0.05, weight_decay=1e-4):
    """Class-balanced multi-label logistic regression on standardized features."""
    x = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    y = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    mean, scale = x.mean(0), x.std(0).clamp_min(1e-6)
    x = (x - mean) / scale
    positive = y.mean(0).clamp(1e-4, 1 - 1e-4)
    weight = torch.zeros(x.shape[1], y.shape[1], device=device, requires_grad=True)
    bias = torch.zeros(y.shape[1], device=device, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x @ weight + bias, y, pos_weight=(1 - positive) / positive)
        (loss + weight_decay * weight.square().sum()).backward()
        optimizer.step()
    mean, scale, weight, bias = (t.detach().double() for t in (mean, scale, weight, bias))

    def logits(features):
        z = torch.as_tensor(features, dtype=torch.float64, device=device)
        return (((z - mean) / scale) @ weight + bias).cpu().numpy()
    return logits


def probe_feature(train_x, train_y, val_x, val_y, test_x, pair_data, world_of_pair, type_of_pair,
                  device, repetitions, seed):
    logits = fit_probe(train_x, train_y, device)
    val_logits = logits(val_x)
    predicted = val_logits > 0
    balanced = []
    for k in range(val_y.shape[1]):
        positives, negatives = val_y[:, k] == 1, val_y[:, k] == 0
        if positives.sum() and negatives.sum():
            balanced.append(0.5 * (predicted[positives, k].mean() + (~predicted[negatives, k]).mean()))
    test_logits = logits(test_x)
    scores = np.empty(len(pair_data))
    for n, (q, neg, flips, signs) in enumerate(pair_data):
        difference = (test_logits[q, flips] - test_logits[neg, flips]) * signs
        scores[n] = np.where(np.abs(difference) <= TIE_TOLERANCE, 0.5, (difference > 0).astype(float)).mean()
    rng = np.random.default_rng(seed)
    result = {"clean_val_balanced_accuracy": float(np.mean(balanced))}
    groups = {kind: type_of_pair == kind for kind in np.unique(type_of_pair)}
    groups["binding_mean"] = np.ones(len(scores), dtype=bool)
    for name, mask in groups.items():
        types = type_of_pair[mask]
        weights = np.zeros(mask.sum())
        for t in np.unique(types):
            weights[types == t] = 1.0 / (types == t).sum()
        weights /= weights.sum()
        values, worlds = scores[mask], world_of_pair[mask]
        unique = np.unique(worlds)
        by_world = {w: np.flatnonzero(worlds == w) for w in unique}
        draws = np.empty(repetitions)
        for r in range(repetitions):
            picked = np.concatenate([by_world[w] for w in rng.choice(unique, len(unique))])
            wt = weights[picked] / weights[picked].sum()
            draws[r] = (wt * values[picked]).sum()
        result[name] = {"pairs": int(mask.sum()), "pair_ranking_accuracy": {
            "value": float((weights * values).sum()), "ci95": [float(x) for x in np.quantile(draws, (0.025, 0.975))]}}
    return result


def main():
    args = arguments()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generator = captions_module.load_generator(args.caption_generator)
    device = torch.device(args.device)
    cache = {}

    def dataset(tokenizer):
        if "items" not in cache:
            worlds, captions, items, stats = captions_module.build_items(
                generator, tokenizer, args.manifest, args.num_worlds, args.seed, args.max_tokens)
            # Guarantee, not assumption: reorder positives and binding-swap
            # negatives have the query's exact token multiset.
            for item in items:
                bag = lambda k: Counter(tokenizer.tokenize(captions[k]["caption"]))
                if item["negative_type"] in captions_module.SWAP_TYPES and bag(item["negative"]) != bag(item["query"]):
                    raise RuntimeError(f"lexical match violated for negative {item}")
                if item["positive_type"] == "reorder" and bag(item["positive"]) != bag(item["query"]):
                    raise RuntimeError(f"lexical match violated for positive {item}")
            query_ids = sorted({i["query"] for i in items})
            rng = np.random.default_rng(args.seed)
            world = {c["index"]: c["world_index"] for c in captions}
            pairs = []
            while len(pairs) < 20000:
                a, b = rng.choice(query_ids, 2, replace=False)
                if world[a] != world[b]:
                    pairs.append((a, b))
            cache.update(items=items, captions=captions, stats=stats, pairs=np.asarray(pairs),
                         world_of_item=np.asarray([i["world_index"] for i in items]))
            train_texts, train_worlds = read_human(args.train_manifest, 0, args.probe_train_samples)
            val_texts, val_worlds = read_human(args.manifest, args.probe_val_offset, args.probe_val_samples)
            test_ids = {c["world_index"] for c in captions}
            labels = captions_module.all_conjunction_labels()
            train_y = label_matrix(train_worlds, labels)
            keep = (train_y.sum(0) >= args.min_label_count) & ((1 - train_y).sum(0) >= args.min_label_count)
            labels = [label for label, k in zip(labels, keep) if k]
            train_y, val_y = train_y[:, keep], label_matrix(val_worlds, labels)
            position = {label: k for k, label in enumerate(labels)}
            pair_data, world_of_pair, type_of_pair, seen = [], [], [], set()
            skipped = Counter()
            for item in items:
                kind = item["negative_type"]
                if kind not in captions_module.SWAP_TYPES or (item["query"], item["negative"]) in seen:
                    continue
                seen.add((item["query"], item["negative"]))
                query_labels = captions_module.conjunction_labels(captions[item["query"]]["world"])
                negative_labels = captions_module.conjunction_labels(captions[item["negative"]]["world"])
                flipped = sorted((query_labels ^ negative_labels) & set(labels), key=position.get)
                if not flipped:
                    skipped[kind] += 1
                    continue
                pair_data.append((item["query"], item["negative"], np.asarray([position[l] for l in flipped]),
                                  np.asarray([1.0 if l in query_labels else -1.0 for l in flipped])))
                world_of_pair.append(item["world_index"])
                type_of_pair.append(kind)
            cache.update(train_texts=train_texts, val_texts=val_texts, train_y=train_y, val_y=val_y,
                         labels=labels, pair_data=pair_data, world_of_pair=np.asarray(world_of_pair),
                         type_of_pair=np.asarray(type_of_pair), pairs_skipped=skipped)
            (out / "captions.jsonl").write_text("".join(json.dumps(c) + "\n" for c in captions))
            (out / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
            (out / "construction.json").write_text(json.dumps({
                "num_worlds": len(worlds), "num_captions": len(captions), "num_items": len(items),
                "items_by_type": Counter(f"{i['negative_type']}|{i['positive_type']}" for i in items),
                "dropped": stats, "seed": args.seed,
                "probe": {"train_manifest": args.train_manifest, "train_samples": len(train_texts),
                          "val_offset": args.probe_val_offset, "val_samples": len(val_texts),
                          "labels_kept": len(labels), "minimal_pairs": len(pair_data),
                          "minimal_pairs_by_type": Counter(type_of_pair),
                          "pairs_without_flipped_label": skipped},
            }, indent=2, default=int) + "\n")
        return cache

    for spec in args.checkpoint:
        label, path = spec.split("=", 1)
        destination = out / f"{label}.json"
        if destination.exists():
            print(f"skip {label}", flush=True)
            continue
        model, tokenizer, model_args = load_model(path, device)
        data = dataset(tokenizer)
        texts = [c["caption"] for c in data["captions"]]
        features = encode(model, tokenizer, model_args, texts, args.batch_size, device)
        train_features = encode(model, tokenizer, model_args, data["train_texts"], args.batch_size, device)
        val_features = encode(model, tokenizer, model_args, data["val_texts"], args.batch_size, device)
        metrics = {}
        for name, value in features.items():
            metrics[name] = {
                "probe": probe_feature(train_features[name], data["train_y"], val_features[name], data["val_y"],
                                       value, data["pair_data"], data["world_of_pair"], data["type_of_pair"],
                                       device, args.bootstrap, args.seed),
                "cosine": score_feature(value, data["items"], data["pairs"], data["world_of_item"],
                                        args.bootstrap, args.seed),
            }
        destination.write_text(json.dumps({
            "protocol": {"checkpoint": str(Path(path).resolve()), "train_mode": model_args.train_mode,
                         "pooling": "content-token mean, L2 normalization", "tie_tolerance": TIE_TOLERANCE,
                         "bootstrap": args.bootstrap, "seed": args.seed},
            "metrics": metrics,
        }, indent=2) + "\n")
        pick = lambda f: metrics[f]["probe"]["binding_mean"]["pair_ranking_accuracy"]["value"]
        extra = (f" shared.4.mlp.3={pick('shared.4.mlp.3'):.3f} private.4.mlp.3={pick('private.4.mlp.3'):.3f}"
                 if "shared.4.mlp.3" in metrics else "")
        print(f"{label}: probe binding emb={pick('embedding'):.3f} block2={pick('block2'):.3f} "
              f"block4={pick('block4'):.3f} block7={pick('block7'):.3f}{extra}", flush=True)
        del model
        torch.cuda.empty_cache()

    if args.controls and not (out / "controls.json").exists():
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        from sklearn.preprocessing import normalize
        tokenizer = load_model(args.checkpoint[0].split("=", 1)[1], torch.device("cpu"))[1]
        data = dataset(tokenizer)
        texts = [c["caption"] for c in data["captions"]]
        controls = {}
        all_texts = data["train_texts"] + data["val_texts"] + texts
        splits = np.cumsum([len(data["train_texts"]), len(data["val_texts"])])
        for name, vectorizer in (("word_counts", CountVectorizer(token_pattern=r"[a-z]+")),
                                 ("tfidf_words", TfidfVectorizer(token_pattern=r"[a-z]+", sublinear_tf=True))):
            vectorizer.fit(data["train_texts"])
            controls[name] = normalize(vectorizer.transform(all_texts)).toarray()
        rng = np.random.default_rng(args.seed)
        random_features = rng.normal(size=(len(all_texts), 384))
        controls["random"] = random_features / np.linalg.norm(random_features, axis=1, keepdims=True)
        result = {}
        for name, value in controls.items():
            train_x, val_x, test_x = np.split(value, splits)
            result[name] = {
                "probe": probe_feature(train_x, data["train_y"], val_x, data["val_y"], test_x, data["pair_data"],
                                       data["world_of_pair"], data["type_of_pair"], device, args.bootstrap, args.seed),
                "cosine": score_feature(test_x, data["items"], data["pairs"], data["world_of_item"],
                                        args.bootstrap, args.seed),
            }
        (out / "controls.json").write_text(json.dumps(result, indent=2) + "\n")
        print("controls written", flush=True)


if __name__ == "__main__":
    main()
