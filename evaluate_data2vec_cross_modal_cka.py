#!/usr/bin/env python3
"""Does data2vec pretraining make image and text representations more alike?

Paired COCO val2017 data (an image and one of its captions) is encoded by a
unimodal image encoder and a unimodal text encoder, and the two
representation geometries are compared with rotation-invariant measures.  The
question: is the image/text similarity higher when *both* encoders are data2vec
than for the baselines data2vec compares against (MAE / BEiT / DINO for vision,
RoBERTa / BERT for text)?

Every image model x text model combination is measured over the full
layer x layer grid:

``cka``            linear CKA on mean-pooled, L2-normalised features (same
                   convention as evaluate_paired_modality_alignment.py)
``cka_debiased``   linear CKA with the unbiased HSIC estimator (removes the
                   finite-sample upward bias of plain CKA)
``mknn``           mutual k-nearest-neighbour overlap (Huh et al. 2024,
                   "platonic representation"), k=10

Controls: randomly initialised encoders of the same architectures, and a
shuffled-pairing floor.  The best layer pair is selected on one half of the
images and reported on the other half, with bootstrap CIs, so "best layer"
numbers are not inflated by selection.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoConfig, AutoImageProcessor, AutoModel, AutoTokenizer)

COCO = Path("/home/cvg/data/yusuf/coco")

# label -> (hub id, family, note).  All vision encoders are ViT-B/16, all text
# encoders are 12-layer / 768-d, all self-supervised only (no fine-tuning)
# unless the label says otherwise.
IMAGE_MODELS = {
    "data2vec-vision": ("facebook/data2vec-vision-base", "ssl", "IN-1k, data2vec"),
    "mae": ("facebook/vit-mae-base", "ssl", "IN-1k, masked pixel reconstruction"),
    "beit": ("microsoft/beit-base-patch16-224-pt22k", "ssl", "IN-22k, masked visual-token prediction"),
    "dino": ("facebook/dino-vitb16", "ssl", "IN-1k, self-distillation"),
    "data2vec-vision-ft1k": ("facebook/data2vec-vision-base-ft1k", "reference", "data2vec + supervised IN-1k fine-tune"),
    "random-vit": ("facebook/data2vec-vision-base", "control", "same architecture, random init"),
}
TEXT_MODELS = {
    "data2vec-text": ("facebook/data2vec-text-base", "ssl", "Books+Wiki, data2vec"),
    "roberta": ("roberta-base", "ssl", "160GB text, MLM"),
    "bert": ("google-bert/bert-base-uncased", "ssl", "Books+Wiki, MLM+NSP"),
    "random-roberta": ("roberta-base", "control", "same architecture, random init"),
}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-images", type=int, default=5000)
    p.add_argument("--image-models", default=",".join(IMAGE_MODELS))
    p.add_argument("--text-models", default=",".join(TEXT_MODELS))
    p.add_argument("--pooling", default="mean,cls")
    p.add_argument("--normalization", default="l2,zscore",
                   help="l2: unit-norm rows (project convention); zscore: standardise each dimension, "
                        "which stops a few rogue high-magnitude dimensions from dominating CKA")
    p.add_argument("--knn", type=int, default=10)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="outputs/data2vec_cross_modal_cka")
    return p.parse_args()


# ----------------------------------------------------------------------------- data

def load_pairs(num_images: int, seed: int):
    """One (image path, caption) per COCO val2017 image; caption drawn with a fixed seed."""
    ann = json.loads((COCO / "annotations/captions_val2017.json").read_text())
    files = {im["id"]: im["file_name"] for im in ann["images"]}
    captions: dict[int, list[str]] = {}
    for a in sorted(ann["annotations"], key=lambda a: a["id"]):
        captions.setdefault(a["image_id"], []).append(a["caption"].strip())
    rng = random.Random(seed)
    ids = sorted(captions)
    rng.shuffle(ids)
    ids = sorted(ids[:num_images])
    return [(str(COCO / "val2017" / files[i]), rng.choice(captions[i]), i) for i in ids]


class Images(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return Image.open(self.paths[index]).convert("RGB")


# ----------------------------------------------------------------------------- encoders

def load(hub_id: str, random_init: bool, **config_overrides):
    config = AutoConfig.from_pretrained(hub_id, **config_overrides)
    if random_init:
        torch.manual_seed(0)
        return AutoModel.from_config(config)
    return AutoModel.from_pretrained(hub_id, config=config)


def pool(hidden_states, weights, poolings, prefix_tokens: int):
    """Per layer: mean over content tokens and the first (CLS / <s>) token."""
    out = {}
    for layer, states in enumerate(hidden_states):
        states = states.float()
        if "mean" in poolings:
            out[("mean", layer)] = (states * weights).sum(1) / weights.sum(1).clamp_min(1)
        # Layer 0 CLS is the same vector for every input, so it carries nothing.
        if "cls" in poolings and prefix_tokens and layer > 0:
            out[("cls", layer)] = states[:, 0]
    return out


def collect(store, pooled):
    for key, value in pooled.items():
        store.setdefault(key, []).append(value.cpu())


@torch.inference_mode()
def encode_images(label, paths, args, device):
    hub_id, family, _ = IMAGE_MODELS[label]
    overrides = {"mask_ratio": 0.0} if "mae" in hub_id else {}  # HF ViTMAE masks 75% by default
    model = load(hub_id, family == "control", **overrides).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(hub_id)
    loader = DataLoader(Images(paths), batch_size=args.batch_size, num_workers=8,
                        collate_fn=lambda ims: processor(images=ims, return_tensors="pt"))
    store = {}
    for batch in loader:
        pixels = batch["pixel_values"].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(pixel_values=pixels, output_hidden_states=True)
        states = output.hidden_states
        # Every ViT here puts CLS at position 0; the patch tokens follow.
        weights = torch.ones(states[0].shape[:2], device=device)
        weights[:, 0] = 0
        collect(store, pool(states, weights.unsqueeze(-1), args.poolings, prefix_tokens=1))
    del model
    torch.cuda.empty_cache()
    return {key: torch.cat(parts) for key, parts in store.items()}


@torch.inference_mode()
def encode_texts(label, captions, args, device):
    hub_id, family, _ = TEXT_MODELS[label]
    model = load(hub_id, family == "control").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(hub_id)
    store = {}
    for start in range(0, len(captions), args.batch_size):
        batch = tokenizer(captions[start:start + args.batch_size], padding=True, truncation=True,
                          max_length=128, return_special_tokens_mask=True, return_tensors="pt")
        special = batch.pop("special_tokens_mask").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(**batch, output_hidden_states=True)
        content = (batch["attention_mask"].bool() & ~special.bool()).float().unsqueeze(-1)
        collect(store, pool(output.hidden_states, content, args.poolings, prefix_tokens=1))
    del model
    torch.cuda.empty_cache()
    return {key: torch.cat(parts) for key, parts in store.items()}


# ----------------------------------------------------------------------------- metrics

def prepare(x, normalization):
    """Centred features, either row-L2-normalised first or per-dimension standardised."""
    x = x.double()
    if normalization == "l2":
        x = F.normalize(x, dim=1)
        return x - x.mean(0, keepdim=True)
    x = x - x.mean(0, keepdim=True)
    return x / x.std(0, keepdim=True).clamp_min(1e-6)


def prepare_f32(x, normalization):
    # Normalise in float64, compare in float32: n x n Gram matrices in fp64 are
    # ~30x slower on consumer GPUs and the extra precision is not needed.
    return prepare(x, normalization).float()


def linear_cka(x, y):
    """x, y already centred.  Batched over index subsets via the caller."""
    return float((y.T @ x).norm() ** 2 / ((x.T @ x).norm() * (y.T @ y).norm()).clamp_min(1e-12))


def hsic_unbiased(k, l):
    n = k.shape[0]
    k = k.clone(); l = l.clone()
    k.fill_diagonal_(0); l.fill_diagonal_(0)
    kl = (k * l).sum()
    return (kl + k.sum() * l.sum() / ((n - 1) * (n - 2)) - 2 * (k.sum(0) @ l.sum(0)) / (n - 2)) / (n * (n - 3))


def debiased_cka(x, y):
    k, l = x @ x.T, y @ y.T
    return float(hsic_unbiased(k, l) / (hsic_unbiased(k, k) * hsic_unbiased(l, l)).clamp_min(1e-24).sqrt())


def mutual_knn(x, y, k):
    def neighbours(z):
        z = F.normalize(z, dim=1)
        s = z @ z.T
        s.fill_diagonal_(-float("inf"))
        return s.topk(k, dim=1).indices
    a, b = neighbours(x), neighbours(y)
    n = x.shape[0]
    ma = torch.zeros(n, n, dtype=torch.bool, device=x.device).scatter_(1, a, True)
    mb = torch.zeros(n, n, dtype=torch.bool, device=x.device).scatter_(1, b, True)
    return float((ma & mb).sum(1).double().mean() / k)


def all_metrics(x, y, k):
    return {"cka": linear_cka(x, y), "cka_debiased": debiased_cka(x, y), "mknn": mutual_knn(x, y, k)}


# ----------------------------------------------------------------------------- main

def main():
    args = arguments()
    args.poolings = args.pooling.split(",")
    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(args.num_images, args.seed)
    paths, captions = [p for p, _, _ in pairs], [c for _, c, _ in pairs]
    n = len(pairs)
    print(f"{n} COCO val2017 image/caption pairs", flush=True)

    cache = out / "features"
    cache.mkdir(exist_ok=True)

    def features(kind, label):
        path = cache / f"{kind}_{label}.pt"
        if path.exists():
            return torch.load(path)
        feats = (encode_images(label, paths, args, device) if kind == "image"
                 else encode_texts(label, captions, args, device))
        torch.save(feats, path)
        print(f"encoded {kind} {label}", flush=True)
        return feats

    image_labels = args.image_models.split(",")
    text_labels = args.text_models.split(",")
    image_feats = {m: features("image", m) for m in image_labels}
    text_feats = {m: features("text", m) for m in text_labels}

    g = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(n, generator=g)
    select_idx, report_idx = order[: n // 2].to(device), order[n // 2:].to(device)
    shuffle = torch.randperm(n, generator=g).to(device)
    boot = torch.randint(0, len(report_idx), (args.bootstrap, len(report_idx)), generator=g).to(device)

    results = {"protocol": {
        "pairs": n, "dataset": "COCO val2017, one random caption per image",
        "features": "hidden states per layer (0 = embeddings), mean over content/patch tokens or CLS "
                    "(CLS from layer 1), then l2 (unit rows, centred) or zscore (per-dim standardised)",
        "selection": f"best layer pair by the metric on {len(select_idx)} images, reported on the other {len(report_idx)}",
        "bootstrap": args.bootstrap, "knn": args.knn,
        "image_models": {m: IMAGE_MODELS[m] for m in image_labels},
        "text_models": {m: TEXT_MODELS[m] for m in text_labels},
    }, "pairs": {}}

    for normalization, pooling in [(z, p) for z in args.normalization.split(",") for p in args.poolings]:
        for im in image_labels:
            for tm in text_labels:
                xi = {l: prepare_f32(v, normalization).to(device)
                      for (p, l), v in image_feats[im].items() if p == pooling}
                xt = {l: prepare_f32(v, normalization).to(device)
                      for (p, l), v in text_feats[tm].items() if p == pooling}
                if not xi or not xt:
                    continue
                grid = {}
                for li, a in xi.items():
                    for lt, b in xt.items():
                        grid[(li, lt)] = all_metrics(a, b, args.knn)
                entry = {"grid": {f"{li},{lt}": m for (li, lt), m in grid.items()}}
                last = (max(xi), max(xt))
                for metric in ("cka", "cka_debiased", "mknn"):
                    # Held-out best pair: choose on one half, measure on the other.
                    def sub(idx, key):
                        a = xi[key[0]][idx]; b = xt[key[1]][idx]
                        a = a - a.mean(0, keepdim=True); b = b - b.mean(0, keepdim=True)
                        return a, b
                    fn = {"cka": linear_cka, "cka_debiased": debiased_cka,
                          "mknn": lambda a, b: mutual_knn(a, b, args.knn)}[metric]
                    best = max(grid, key=lambda key: fn(*sub(select_idx, key)))
                    report = {}
                    for name, key in (("best_heldout", best), ("last_layer", last)):
                        a, b = sub(report_idx, key)
                        value = fn(a, b)
                        samples = [] if metric == "mknn" else [
                            fn(*(z - z.mean(0, keepdim=True) for z in (a[i], b[i]))) for i in boot]
                        report[name] = {"layers": list(key), "value": value,
                                        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
                                        if samples else None}
                    a, b = xi[best[0]], xt[best[1]][shuffle]
                    report["shuffled_floor"] = fn(a, b - b.mean(0, keepdim=True))
                    entry[metric] = report
                results["pairs"][f"{normalization}|{pooling}|{im}|{tm}"] = entry
                r = entry["cka"]["best_heldout"]
                print(f"[{normalization}/{pooling}] {im:>22} x {tm:<15} CKA best {r['value']:.3f} "
                      f"(L{r['layers'][0]},L{r['layers'][1]}) last {entry['cka']['last_layer']['value']:.3f} "
                      f"dCKA {entry['cka_debiased']['best_heldout']['value']:.3f} "
                      f"mKNN {entry['mknn']['best_heldout']['value']:.3f} "
                      f"floor {entry['cka']['shuffled_floor']:.3f}/{entry['cka_debiased']['shuffled_floor']:.3f}"
                      f"/{entry['mknn']['shuffled_floor']:.3f}", flush=True)

    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
