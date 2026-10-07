#!/usr/bin/env python3
"""Semantic effect sizes: is a representation about the scene or about the words?

A representation is semantic to the extent that it is *invariant* to how a
scene is worded and *sensitive* to which scene is described.  Preference rates
("prefers the paraphrase 87% of the time") saturate and compress model
differences; an effect size does not.  For every feature this reports two
signed, normalized quantities, both computed from cosines between pooled
representations:

``d_semantic``
    (paraphrase vs. template) -- the same scene worded by a different phrasing
    plan, against a *different* scene worded by the query's own plan, so the
    negative shares the template and differs in content.  Large when the
    representation follows the scene rather than the sentence pattern.

``d_binding``
    (true scene vs. binding swap) -- both candidates use one phrasing plan that
    differs from the query's, and the negative exchanges one binding between
    two objects, so the two candidates have the *same words in the same
    template* and differ only in which object has which attribute.  This is
    the hard-retrieval comparison expressed as an effect size, which gives it a
    dynamic range that R@1 near chance does not have.

d' = (mean_positive - mean_negative) / sqrt((var_positive + var_negative) / 2),
paired per scene, with a bootstrap over scenes.  The paired preference rate is
reported alongside for continuity with the earlier tables.

Both are reported for model-free controls too, because they are not equally
hard: word content alone separates a paraphrase from a different scene, so
``d_semantic`` must be read against its bag-of-words control, while
``d_binding`` is exactly zero for any order-free function of the words.
"""

from __future__ import annotations

from clevr_paths import CLEVR_DATA_ROOT, CLEVR_GEN_ROOT
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import binding_swap_captions as C
from evaluate_decomposed_representations import encode
from evaluate_hard_retrieval import attribute_swaps, relation_swaps
from evaluate_shared_private_retrieval import load_model


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH")
    p.add_argument("--untrained-from", default=None, metavar="PATH")
    p.add_argument(
        "--ablate-private", action="store_true",
        help="Suppress every modality-private adapter, so the forward pass is the shared "
             "trunk alone. For a frozen-trunk stage-2 model this must reproduce its stage-1 "
             "parent exactly.",
    )
    p.add_argument(
        "--ablate-shared", action="store_true",
        help="The mirror: zero the shared write and keep only the modality-private branch, "
             "so the private route is measured on its own.",
    )
    p.add_argument(
        "--hf-model", action="append", default=[], metavar="LABEL=HF_NAME",
        help="Score a Hugging Face model on the same items, every layer, mean and cls/last "
             "pooling. Needs --tokenizer-from when no checkpoint is given.",
    )
    p.add_argument(
        "--tokenizer-from", default=None, metavar="PATH",
        help="Checkpoint supplying this repo's tokenizer, which builds the items.",
    )
    p.add_argument("--manifest", default=CLEVR_DATA_ROOT + "/platonic_text_only_v1_1m/val_text_only_human.jsonl")
    p.add_argument("--caption-generator", default=CLEVR_GEN_ROOT + "/generate_human_captions.py")
    p.add_argument("--num-worlds", type=int, default=2000)
    p.add_argument("--max-tokens", type=int, default=190)
    p.add_argument("--sublayer-layers", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lexicon-matched", action="store_true",
                   help="Give the paraphrase the query's attribute vocabulary, so it differs "
                        "from the query in sentence structure only. Isolates invariance to "
                        "phrasing from knowledge that 'tube' and 'cylinder' name one shape.")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--delete-checkpoint-when-done", action="store_true",
        help="Delete checkpoint paths after they have been scored (periodic-probe mode).",
    )
    return p.parse_args()


def build(args, tokenizer):
    """Four captions per scene: query, paraphrase, template twin, binding swap."""
    generator = C.load_generator(args.caption_generator)
    worlds = []
    with open(args.manifest) as handle:
        for line in handle:
            world = json.loads(line)["world"]
            if not C.has_twins(world) and len(world["objects"]) >= 3:
                worlds.append(world)
            if len(worlds) == args.num_worlds:
                break
    texts, index = [], {}

    def add(text):
        if text not in index:
            index[text] = len(texts)
            texts.append(text)
        return index[text]

    items, stats = [], Counter()
    for position, world in enumerate(worlds):
        rng = random.Random(args.seed * 1_000_003 + position)
        objects, relations = len(world["objects"]), len(world.get("relations", []))
        plan = C.make_plan(generator, rng.randrange(2**31), objects, relations)
        other = C.make_plan(generator, rng.randrange(2**31), objects, relations,
                            lexicon=plan if args.lexicon_matched else None)
        order = lambda k: rng.sample(range(k), k)
        pattern = rng.randrange(2)
        query = C.render(generator, world, plan, order(objects), order(relations), pattern)
        # Same scene, different phrasing plan.
        paraphrase = C.render(generator, world, other, order(objects), order(relations), rng.randrange(2))
        # A different scene in the query's own template: same plan, same object
        # and relation order, same sentence pattern, every value changed.
        twin = None
        for candidate in worlds[position + 1:] + worlds[:position]:
            if (len(candidate["objects"]), len(candidate.get("relations", []))) == (objects, relations) \
                    and C.scene_signature(candidate) != C.scene_signature(world):
                twin = C.render(generator, candidate, plan, list(range(objects)), list(range(relations)), pattern)
                break
        # One binding exchanged, worded by the paraphrase's plan so that it is
        # word-for-word comparable with the paraphrase.
        pool = attribute_swaps(world) + relation_swaps(world)
        rng.shuffle(pool)
        swap = None
        for scene in pool:
            if C.scene_signature(scene) != C.scene_signature(world):
                swap = C.render(generator, scene, other, order(objects), order(relations), rng.randrange(2))
                break
        if twin is None or swap is None:
            stats["no_twin_or_swap"] += 1
            continue
        if paraphrase == query:
            # Possible once the lexicon is shared: the structural resample can
            # land back on the query's own wording.
            stats["paraphrase_equals_query"] += 1
            continue
        lengths = [len(tokenizer.tokenize(t)) for t in (query, paraphrase, twin, swap)]
        if max(lengths) > args.max_tokens:
            stats["too_long"] += 1
            continue
        if Counter(tokenizer.tokenize(paraphrase)) != Counter(tokenizer.tokenize(swap)):
            # The binding comparison is only fair when the two share their words.
            stats["swap_not_lexically_matched"] += 1
            continue
        items.append({"world": position, "query": add(query), "paraphrase": add(paraphrase),
                      "twin": add(twin), "swap": add(swap)})
    return texts, items, stats


def effect_size(positive, negative, worlds, repetitions, seed):
    positive, negative = np.asarray(positive), np.asarray(negative)
    def statistic(picked):
        p, n = positive[picked], negative[picked]
        spread = np.sqrt((p.var() + n.var()) / 2)
        return (p.mean() - n.mean()) / max(spread, 1e-12), float((p > n).mean())
    order = {world: np.flatnonzero(worlds == world) for world in np.unique(worlds)}
    keys = list(order)
    rng = np.random.default_rng(seed)
    point = statistic(np.arange(len(positive)))
    draws = np.asarray([
        statistic(np.concatenate([order[w] for w in rng.choice(keys, len(keys))]))
        for _ in range(repetitions)
    ])
    low, high = np.quantile(draws, (0.025, 0.975), axis=0)
    return {
        "d": {"value": float(point[0]), "ci95": [float(low[0]), float(high[0])]},
        "preference": {"value": float(point[1]), "ci95": [float(low[1]), float(high[1])]},
        "mean_positive": float(positive.mean()), "mean_negative": float(negative.mean()),
        "pairs": int(len(positive)),
    }


def score(features, items, repetitions, seed):
    worlds = np.asarray([item["world"] for item in items])
    query = np.asarray([item["query"] for item in items])
    paraphrase = np.asarray([item["paraphrase"] for item in items])
    twin = np.asarray([item["twin"] for item in items])
    swap = np.asarray([item["swap"] for item in items])
    cos = lambda a, b: (features[a].astype(np.float64) * features[b].astype(np.float64)).sum(1)
    return {
        "d_semantic": effect_size(cos(query, paraphrase), cos(query, twin), worlds, repetitions, seed),
        "d_binding": effect_size(cos(query, paraphrase), cos(query, swap), worlds, repetitions, seed),
    }


def lexical_features(texts, tokenizer):
    """Bag-of-words control: an order-free function of the words."""
    vocabulary = sorted({token for text in texts for token in tokenizer.tokenize(text)})
    position = {token: k for k, token in enumerate(vocabulary)}
    matrix = np.zeros((len(texts), len(vocabulary)))
    for row, text in enumerate(texts):
        for token, count in Counter(tokenizer.tokenize(text)).items():
            matrix[row, position[token]] = count
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-12)


def main():
    args = arguments()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = [spec.split("=", 1) for spec in args.checkpoint]
    source = specs[0][1] if specs else (args.untrained_from or args.tokenizer_from)
    if source is None:
        raise SystemExit("give --checkpoint, --untrained-from or --tokenizer-from")
    _, tokenizer, _ = load_model(source, torch.device("cpu"))
    texts, items, stats = build(args, tokenizer)
    print(f"items: {len(items)} scenes, {len(texts)} captions, dropped {dict(stats)}", flush=True)
    (out / "texts.jsonl").write_text("".join(json.dumps({"i": i, "caption": t}) + "\n" for i, t in enumerate(texts)))
    (out / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))

    controls = out / "controls.json"
    if not controls.exists():
        bag = lexical_features(texts, tokenizer)
        controls.write_text(json.dumps({"word_counts": score(bag, items, args.bootstrap, args.seed)}, indent=2) + "\n")
        word = score(bag, items, args.bootstrap, args.seed)
        print(f"word counts: d_semantic {word['d_semantic']['d']['value']:+.3f} "
              f"d_binding {word['d_binding']['d']['value']:+.3f}", flush=True)

    device = torch.device(args.device)
    for spec in args.hf_model:
        label, name = spec.split("=", 1)
        destination = out / f"{label}.json"
        if destination.exists():
            print(f"{label}: exists, skipping", flush=True)
            continue
        from evaluate_hard_retrieval_external import encode as encode_external, subword_multisets_match
        from transformers import AutoModel, AutoTokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(name)
        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token
        hf_model = AutoModel.from_pretrained(name).to(device).eval()
        # The binding comparison needs the swap and the paraphrase to share
        # their words under *this* tokenizer too, so report whether they do.
        pseudo = [{"candidates": [item["paraphrase"], item["swap"]]} for item in items]
        lexical = subword_multisets_match(hf_tokenizer, texts, pseudo)
        features = encode_external(hf_model, hf_tokenizer, texts, args.batch_size, device)
        metrics = {n: score(v, items, args.bootstrap, args.seed) for n, v in features.items()}
        destination.write_text(json.dumps({
            "protocol": {"model": name, "train_mode": "external",
                         "layers": int(hf_model.config.num_hidden_layers),
                         "subword_lexical_check": lexical},
            "metrics": metrics,
        }, indent=2) + "\n")
        best_semantic = max(metrics, key=lambda k: metrics[k]["d_semantic"]["d"]["value"])
        best_binding = max(metrics, key=lambda k: metrics[k]["d_binding"]["d"]["value"])
        print(f"{label}: best d_semantic {best_semantic} "
              f"{metrics[best_semantic]['d_semantic']['d']['value']:+.3f} | best d_binding "
              f"{best_binding} {metrics[best_binding]['d_binding']['d']['value']:+.3f} | "
              f"binding pairs not word-matched under its tokenizer: "
              f"{lexical['candidate_sets_not_multiset_identical']}/{lexical['tasks']}", flush=True)
        del hf_model
        torch.cuda.empty_cache()

    from train_multimodal import build_model
    from analyze_paired_representations import checkpoint_args
    for label, path in specs + ([("untrained", args.untrained_from)] if args.untrained_from else []):
        destination = out / f"{label}.json"
        if destination.exists():
            print(f"{label}: exists, skipping", flush=True)
            continue
        if label == "untrained":
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model_args = checkpoint_args(payload)
            torch.manual_seed(args.seed)
            model, _ = build_model(model_args, len(tokenizer))
            model.to(device).eval()
        else:
            model, tokenizer, model_args = load_model(path, device)
        features, worst = encode(model, tokenizer, model_args, texts, args.batch_size, device,
                                 set(args.sublayer_layers),
                                 check=not (args.ablate_private or args.ablate_shared),
                                 ablate_private=args.ablate_private,
                                 ablate_shared=args.ablate_shared)
        if worst > 1e-3:
            raise RuntimeError(f"{label}: decomposition error {worst:.2e}")
        metrics = {name: score(value, items, args.bootstrap, args.seed) for name, value in features.items()}
        destination.write_text(json.dumps({
            "protocol": {"model": str(path), "train_mode": model_args.train_mode,
                         "lexicon_matched": bool(args.lexicon_matched), "scenes": len(items)},
            "metrics": metrics,
        }, indent=2) + "\n")
        pick = lambda feature, key: metrics[feature][key]["d"]["value"] if feature in metrics else float("nan")
        print(f"{label}: L7.residual d_semantic {pick('L7.residual','d_semantic'):+.3f} "
              f"d_binding {pick('L7.residual','d_binding'):+.3f} | "
              f"best d_binding {max(metrics, key=lambda k: metrics[k]['d_binding']['d']['value'])} "
              f"{max(metrics[k]['d_binding']['d']['value'] for k in metrics):+.3f}", flush=True)
        del model
        torch.cuda.empty_cache()
        if args.delete_checkpoint_when_done and label != "untrained":
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
