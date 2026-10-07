#!/usr/bin/env python3
"""Do a caption and its image land in the same place, and where?

On genuinely paired data -- the same CLEVR scene as a caption and as VQ image
tokens -- this asks how similar the two modalities' representations are, for a
model that encodes both, and separately for the shared route of a Tri-LoRA
model.  The comparison the project cares about: is the shared route more
modality-agnostic than a dense model that saw both modalities, or than two dense
models trained separately?

Within one model both modalities live in the *same* space, so direct measures
apply:

``modality_gap``      distance between the two modality centroids (0 = same location)
``modality_auc``      how separable the two modalities are along that direction
                      (0.5 = indistinguishable, 1.0 = perfectly separated)
``paired_r1``         retrieving a scene's image representation from its caption
                      representation among all scenes (chance = 1/N)
``matched_cosine``    cosine between a scene's own caption and image
``mismatched_cosine`` cosine between different scenes' caption and image

Across two separately trained models the spaces are unrelated, so only
rotation-invariant structure comparisons are meaningful:

``cka``               linear centered kernel alignment
``rsa``               Spearman between the two pairwise-distance structures

Both sets are reported for every feature, which for a Tri-LoRA model includes
the accumulated shared-write and private-write streams.
"""

from __future__ import annotations

from clevr_paths import CLEVR_DATA_ROOT, CLEVR_GEN_ROOT
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import MultimodalCollator
from evaluate_shared_private_retrieval import load_model


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", action="append", default=[], metavar="LABEL=PATH",
                   help="A model that encodes both modalities.")
    p.add_argument("--text-model", action="append", default=[], metavar="LABEL=PATH",
                   help="Text side of a separately-trained pair.")
    p.add_argument("--image-model", action="append", default=[], metavar="LABEL=PATH",
                   help="Image side of a separately-trained pair.")
    p.add_argument("--manifest", default=CLEVR_DATA_ROOT + "/platonic_clevr_v1_100k_gpu_visible_s20260825/val_pairs_diverse_human.jsonl")
    p.add_argument("--image-cache", default="outputs/platonic_token_cache_correct_dataset_visible_check/val_tokens.pt")
    p.add_argument("--caption-field", default="caption_human")
    p.add_argument("--num-scenes", type=int, default=2000)
    p.add_argument("--rsa-pairs", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def load_pairs(args):
    cache = torch.load(args.image_cache, map_location="cpu", weights_only=False)
    tokens = cache["tokens"] if isinstance(cache, dict) else cache
    captions, rows = [], []
    with open(args.manifest) as handle:
        for row, line in enumerate(handle):
            if row >= tokens.shape[0] or len(captions) == args.num_scenes:
                break
            captions.append(json.loads(line)[args.caption_field])
            rows.append(row)
    # Token caches are stored as uint16, which torch cannot index directly;
    # numpy can, and the gathered slice is small enough to widen afterwards.
    return captions, torch.from_numpy(tokens.numpy()[rows].astype("int64"))


@torch.no_grad()
def encode(model, tokenizer, model_args, device, batch_size, *, captions=None, image_tokens=None):
    """Pooled residual stream per block, plus shared/private streams for Tri-LoRA."""
    lora = model_args.train_mode in {"lora", "dense_private"}
    modality = "text" if captions is not None else "image"
    collator = MultimodalCollator(tokenizer, model_args.num_image_codes, model_args.max_text_length)
    captured, hooks = {}, []
    hooks.append(model.blocks[0].register_forward_pre_hook(
        lambda _m, inputs: captured.__setitem__("h0", inputs[0].detach().clone())))
    for index, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(
            lambda _m, _i, output, index=index: captured.__setitem__(f"L{index}", output.detach().clone())))
        # The two sublayer writes, so CKA can be read per module and not only
        # on the residual stream.
        hooks.append(block.attn.out_proj.register_forward_hook(
            lambda _m, _i, output, index=index:
                captured.__setitem__(f"L{index}.attn_out", output.detach().clone())))
        hooks.append(block.mlp[3].register_forward_hook(
            lambda _m, _i, output, index=index:
                captured.__setitem__(f"L{index}.mlp_out", output.detach().clone())))
    count = len(captions) if captions is not None else image_tokens.shape[0]
    values: dict[str, list[torch.Tensor]] = {}
    try:
        for start in range(0, count, batch_size):
            if captions is not None:
                examples = [{"kind": "text", "text": text, "pair_index": start + k}
                            for k, text in enumerate(captions[start:start + batch_size])]
            else:
                examples = [{"kind": "image", "image_tokens": chunk, "pair_index": start + k}
                            for k, chunk in enumerate(image_tokens[start:start + batch_size])]
            batch = {name: value.to(device) for name, value in collator(examples).items()}
            captured.clear()
            output = model(batch["input_ids"], batch["attention_mask"], batch["position_ids"],
                           batch["modality_ids"], batch["route_ids"], return_shared=lora,
                           return_shared_private_native_by_module=lora, private_branch=modality)
            features = {name: tensor.double() for name, tensor in captured.items()}
            if lora:
                _, _, shared, private = output
                shared_sum = torch.zeros_like(features["h0"])
                private_sum = torch.zeros_like(features["h0"])
                for index in range(len(model.blocks)):
                    for name in (f"blocks.{index}.attn.out_proj", f"blocks.{index}.mlp.3"):
                        shared_sum = shared_sum + shared[name].double()
                        private_sum = private_sum + private[name].double()
                    features[f"L{index}.shared"] = features["h0"] + shared_sum
                    features[f"L{index}.private"] = features["h0"] + private_sum
            weights = batch["eligible_mask"].double().unsqueeze(-1)
            for name, tensor in features.items():
                pooled = (tensor * weights).sum(1) / weights.sum(1).clamp_min(1)
                values.setdefault(name, []).append(F.normalize(pooled, dim=1).float().cpu())
    finally:
        for hook in hooks:
            hook.remove()
    return {name: torch.cat(parts).numpy() for name, parts in values.items()}


def spearman(a, b):
    rank = lambda x: np.argsort(np.argsort(x)).astype(np.float64)
    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / max(np.sqrt((ra * ra).sum() * (rb * rb).sum()), 1e-12))


def linear_cka(x, y):
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    return float(np.linalg.norm(y.T @ x, ord="fro") ** 2
                 / max(np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro"), 1e-12))


def same_space_metrics(text, image, pairs, rng, retrieval_size=1000):
    """Both modalities in one space: gap, separability, and paired retrieval."""
    gap = float(np.linalg.norm(text.mean(0) - image.mean(0)))
    direction = text.mean(0) - image.mean(0)
    norm = np.linalg.norm(direction)
    if norm > 1e-12:
        projected_text, projected_image = text @ (direction / norm), image @ (direction / norm)
        # AUC of separating the two modalities along the centroid direction.
        order = np.argsort(np.concatenate([projected_text, projected_image]))
        labels = np.concatenate([np.ones(len(text)), np.zeros(len(image))])[order]
        ranks = np.arange(1, len(labels) + 1)
        positives, negatives = labels.sum(), (1 - labels).sum()
        auc = float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / max(positives * negatives, 1))
        auc = max(auc, 1 - auc)
    else:
        auc = 0.5
    size = min(retrieval_size, len(text))
    picked = rng.choice(len(text), size=size, replace=False)
    scores = text[picked] @ image[picked].T
    rank_of_true = (scores > scores[np.arange(size), np.arange(size)][:, None]).sum(1)
    matched = scores[np.arange(size), np.arange(size)]
    off = ~np.eye(size, dtype=bool)
    return {
        "modality_gap": gap,
        "modality_auc": auc,
        "paired_r1": float((rank_of_true == 0).mean()),
        "paired_chance": 1.0 / size,
        "matched_cosine": float(matched.mean()),
        "mismatched_cosine": float(scores[off].mean()),
        "cka": linear_cka(text, image),
        "rsa": spearman(1.0 - (text[pairs[:, 0]] * text[pairs[:, 1]]).sum(1),
                        1.0 - (image[pairs[:, 0]] * image[pairs[:, 1]]).sum(1)),
    }


def main():
    args = arguments()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    first = (args.model or args.text_model or args.image_model)[0].split("=", 1)[1]
    _, tokenizer, _ = load_model(first, torch.device("cpu"))
    captions, image_tokens = load_pairs(args)
    pairs = rng.integers(0, len(captions), size=(args.rsa_pairs, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    print(f"paired scenes: {len(captions)}", flush=True)

    results, sides = {}, {}
    for spec in args.model:
        label, path = spec.split("=", 1)
        model, tokenizer, model_args = load_model(path, device)
        text = encode(model, tokenizer, model_args, device, args.batch_size, captions=captions)
        image = encode(model, tokenizer, model_args, device, args.batch_size, image_tokens=image_tokens)
        metrics = {name: same_space_metrics(text[name], image[name], pairs, rng)
                   for name in text if name in image}
        results[label] = {"kind": "one model, both modalities",
                          "train_mode": model_args.train_mode, "metrics": metrics}
        shown = "L7.shared" if "L7.shared" in metrics else "L7"
        print(f"{label}: {shown} gap {metrics[shown]['modality_gap']:.3f} "
              f"auc {metrics[shown]['modality_auc']:.3f} paired_r1 {metrics[shown]['paired_r1']*100:.1f}% "
              f"cka {metrics[shown]['cka']:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    for spec in args.text_model:
        label, path = spec.split("=", 1)
        model, tok, model_args = load_model(path, device)
        sides[("text", label)] = encode(model, tok, model_args, device, args.batch_size, captions=captions)
        del model; torch.cuda.empty_cache()
    for spec in args.image_model:
        label, path = spec.split("=", 1)
        model, tok, model_args = load_model(path, device)
        sides[("image", label)] = encode(model, tok, model_args, device, args.batch_size, image_tokens=image_tokens)
        del model; torch.cuda.empty_cache()
    for (ma, la), fa in sides.items():
        for (mb, lb), fb in sides.items():
            if ma != "text" or mb != "image":
                continue
            metrics = {}
            for name in fa:
                if name not in fb:
                    continue
                # Separate models: unrelated spaces, so only structure compares.
                metrics[name] = {
                    "cka": linear_cka(fa[name], fb[name]),
                    "rsa": spearman(1.0 - (fa[name][pairs[:, 0]] * fa[name][pairs[:, 1]]).sum(1),
                                    1.0 - (fb[name][pairs[:, 0]] * fb[name][pairs[:, 1]]).sum(1)),
                }
            results[f"{la}+{lb}"] = {"kind": "two separately trained models", "metrics": metrics}
            print(f"{la}+{lb}: L7 cka {metrics['L7']['cka']:.3f} rsa {metrics['L7']['rsa']:+.3f}", flush=True)
    # Merge into any existing file: runs are usually added a few models at a
    # time, and rewriting would drop everything measured earlier.
    destination = out / "alignment.json"
    merged = {}
    if destination.exists():
        merged = json.loads(destination.read_text())
    merged.update(results)
    destination.write_text(json.dumps(merged, indent=2) + "\n")


if __name__ == "__main__":
    main()
