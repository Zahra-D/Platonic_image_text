"""Does a JEPA predict scene-specific targets, or only the shared direction?

For a data2vec / I-JEPA checkpoint (EMA teacher saved in the checkpoint) or a
token-level LeJEPA checkpoint (teacherless: the model itself gives the clean
targets), on validation data masked exactly as in training:

    C+    = mean cos(prediction_i,p , target_i,p)        own scene, masked token p
    C-    = mean cos(prediction_i,p , target_j,p)        same position, another scene
    Delta = C+ - C-

reported raw and after dataset centring (prediction and target each minus its mean
over all evaluated masked tokens of that layer and modality). With a strong shared
direction raw C+ can be ~1 while Delta ~ 0; a large centred Delta is the evidence
that the predictor carries instance-specific information.

  python3 evaluate_jepa_prediction.py --checkpoint LABEL=PATH [...] --output-dir DIR
"""
from __future__ import annotations
from clevr_paths import CLEVR_DATA_ROOT
import argparse, json
from pathlib import Path

import torch
import torch.nn.functional as F

from analyze_paired_representations import checkpoint_args
from data import ClevrTextTokenizer, MultimodalCollator
from data.multimodal_dataset import read_jsonl
from multimodal_diffusion import corrupt_batch
from train_multimodal import build_model, data2vec_selected_layers


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--text-manifest", default=CLEVR_DATA_ROOT + "/platonic_text_only_v1_1m/val_text_only_human.jsonl")
    p.add_argument("--image-cache", default="outputs/image_only_2_5m_token_cache/val_tokens.pt")
    p.add_argument("--num-scenes", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=20261007)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def load_student_teacher(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    margs = checkpoint_args(payload)
    if not getattr(margs, "data2vec_hidden", False):
        raise ValueError(f"{path}: not a data2vec / token-level JEPA checkpoint")
    tokenizer = ClevrTextTokenizer(payload["text_vocabulary"])
    student, _ = build_model(margs, len(tokenizer))
    student.load_state_dict(payload["model"])
    student.to(device).eval()
    if getattr(margs, "data2vec_teacherless", False):
        teacher = student
    else:
        if "shared_jepa_ema_teacher" not in payload:
            raise ValueError(f"{path}: no EMA teacher saved in the checkpoint")
        teacher, _ = build_model(margs, len(tokenizer))
        teacher.load_state_dict(payload["shared_jepa_ema_teacher"])
        teacher.to(device).eval()
    return student, teacher, tokenizer, margs


def modality_batches(margs, tokenizer, args):
    collator = MultimodalCollator(tokenizer, margs.num_image_codes, margs.max_text_length)
    mods = {"text_only": ["text"], "image_only": ["image"]}.get(margs.data_mode, ["text", "image"])
    out = {}
    if "text" in mods:
        texts = [r["caption_human"] for r in read_jsonl(args.text_manifest)[: args.num_scenes]]
        out["text"] = [collator([{"kind": "text", "text": t, "pair_index": s + k}
                                 for k, t in enumerate(texts[s:s + args.batch_size])])
                       for s in range(0, len(texts), args.batch_size)]
    if "image" in mods:
        payload = torch.load(args.image_cache, map_location="cpu", weights_only=False)
        tokens = (payload["tokens"] if isinstance(payload, dict) else payload)[: args.num_scenes]
        out["image"] = [collator([{"kind": "image", "image_tokens": t.long(), "pair_index": s + k}
                                  for k, t in enumerate(tokens[s:s + args.batch_size])])
                        for s in range(0, len(tokens), args.batch_size)]
    return out


def corrupt_like_training(batch, modality, margs, tokenizer):
    """Same masking as train_multimodal's training call (per-modality blocks / spans / t)."""
    # older checkpoints predate some masking options: fall back to the trainer defaults
    block_2d = getattr(margs, "mask_block_2d", False)
    span_min, span_max = getattr(margs, "mask_span_min", None), getattr(margs, "mask_span_max", None)
    both = block_2d and span_min is not None
    blocks = block_2d and (not both or modality == "image")
    spans = span_min is not None and (not both or modality == "text")
    fixed = {"text": getattr(margs, "train_fixed_t_text", None),
             "image": getattr(margs, "train_fixed_t_image", None)}.get(modality)
    return corrupt_batch(
        batch["input_ids"], batch["eligible_mask"], batch["modality_ids"], tokenizer.mask_id,
        margs.eps, modality, fixed_t=fixed if fixed is not None else getattr(margs, "train_fixed_t", None),
        mask_span_min=span_min if spans else None, mask_span_max=span_max if spans else None,
        mask_block_grid=tuple(margs.grid_size) if blocks else None,
        mask_block_scale=tuple(getattr(margs, "mask_block_scale", (0.15, 0.2))),
        mask_block_aspect=tuple(getattr(margs, "mask_block_aspect", (0.75, 1.5))),
    )


@torch.no_grad()
def predictions_and_targets(student, teacher, margs, tokenizer, batch, modality, seed, device):
    """Per supervised layer: (pred [M,D], target [M,D], mismatched target [M,D]) at masked tokens."""
    b = {k: v.to(device) for k, v in batch.items()}
    torch.manual_seed(seed)
    corrupted, masked, _ = corrupt_like_training(b, modality, margs, tokenizer)
    routes = torch.full_like(b["route_ids"], -1) if getattr(margs, "jepa_trunk_only", False) else b["route_ids"]
    ffn = getattr(margs, "data2vec_target_type", "block_residual") == "ffn_output"
    shared = getattr(margs, "data2vec_share_input_encoder", False)
    clean_emb = student.input_embeddings(b["input_ids"], b["position_ids"], b["modality_ids"]) if shared else None
    if ffn:
        _, student_hidden, _ = student(corrupted, b["attention_mask"], b["position_ids"], b["modality_ids"],
                                       routes, return_data2vec_by_layer=True)
        _, _, teacher_hidden = teacher(b["input_ids"], b["attention_mask"], b["position_ids"], b["modality_ids"],
                                       routes, return_data2vec_by_layer=True, input_embeddings=clean_emb)
    else:
        _, student_hidden = student(corrupted, b["attention_mask"], b["position_ids"], b["modality_ids"],
                                    routes, return_hidden_by_layer=True)
        _, teacher_hidden = teacher(b["input_ids"], b["attention_mask"], b["position_ids"], b["modality_ids"],
                                    routes, return_hidden_by_layer=True, input_embeddings=clean_emb)
    normalize = not getattr(margs, "data2vec_teacherless", False)
    width = next(iter(teacher_hidden.values())).size(-1)
    target_of = lambda layer: (F.layer_norm(teacher_hidden[layer].float(), (width,)) if normalize
                               else teacher_hidden[layer].float())
    layers = data2vec_selected_layers(margs)
    mode = getattr(margs, "data2vec_mode", "average")
    # a mismatched partner: the next scene in the batch, at the same token position,
    # kept only where that position is a content token there too
    partner = torch.roll(torch.arange(masked.size(0), device=device), 1)
    valid = masked & b["eligible_mask"][partner]
    out = {}
    if mode == "average":
        target = torch.stack([target_of(l) for l in layers]).mean(0)
        pred = student.predict_data2vec(student_hidden[max(student_hidden)]).float()
        out["avg"] = (pred[valid], target[valid], target[partner][valid])
    else:
        fixed = mode == "fixed_target"
        for layer in layers:
            target = target_of(margs.data2vec_target_layer if fixed else layer)
            pred = student.predict_data2vec(student_hidden[layer], layer=layer).float()
            out[f"L{layer}"] = (pred[valid], target[valid], target[partner][valid])
    return out


def main():
    args = arguments()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for spec in args.checkpoint:
        label, path = spec.split("=", 1)
        dest = out_dir / f"{label}.json"
        if dest.exists():
            print(f"{label}: exists, skipping", flush=True); continue
        student, teacher, tokenizer, margs = load_student_teacher(path, device)
        report = {}
        for modality, batches in modality_batches(margs, tokenizer, args).items():
            # pass 1: dataset means of predictions and targets per layer (same seeded masks as pass 2)
            sums, counts = {}, {}
            for i, batch in enumerate(batches):
                for layer, (pred, tgt, _) in predictions_and_targets(student, teacher, margs, tokenizer, batch,
                                                                     modality, args.seed + i, device).items():
                    s = sums.setdefault(layer, [torch.zeros(pred.size(1), device=device, dtype=torch.float64)] * 2)
                    sums[layer] = [s[0] + pred.double().sum(0), s[1] + tgt.double().sum(0)]
                    counts[layer] = counts.get(layer, 0) + pred.size(0)
            means = {l: (s[0] / counts[l], s[1] / counts[l]) for l, s in sums.items()}
            acc = {}
            for i, batch in enumerate(batches):
                for layer, (pred, tgt, mis) in predictions_and_targets(student, teacher, margs, tokenizer, batch,
                                                                       modality, args.seed + i, device).items():
                    mp, mt = (m.float() for m in means[layer])
                    a = acc.setdefault(layer, {k: 0.0 for k in ("raw+", "raw-", "cen+", "cen-")})
                    a["raw+"] += F.cosine_similarity(pred, tgt, dim=-1).sum().item()
                    a["raw-"] += F.cosine_similarity(pred, mis, dim=-1).sum().item()
                    a["cen+"] += F.cosine_similarity(pred - mp, tgt - mt, dim=-1).sum().item()
                    a["cen-"] += F.cosine_similarity(pred - mp, mis - mt, dim=-1).sum().item()
            per_layer = {}
            for layer, a in acc.items():
                n = counts[layer]
                per_layer[layer] = {
                    "raw": {"C_plus": a["raw+"] / n, "C_minus": a["raw-"] / n, "delta": (a["raw+"] - a["raw-"]) / n},
                    "centred": {"C_plus": a["cen+"] / n, "C_minus": a["cen-"] / n, "delta": (a["cen+"] - a["cen-"]) / n},
                    "tokens": n,
                }
            report[modality] = per_layer
            mean = lambda g, k: sum(v[g][k] for v in per_layer.values()) / len(per_layer)
            print(f"{label} [{modality}] mean over {len(per_layer)} layers: raw C+ {mean('raw','C_plus'):.3f} "
                  f"C- {mean('raw','C_minus'):.3f} Δ {mean('raw','delta'):.3f} | centred C+ {mean('centred','C_plus'):.3f} "
                  f"C- {mean('centred','C_minus'):.3f} Δ {mean('centred','delta'):.3f}", flush=True)
        dest.write_text(json.dumps({"protocol": {"model": path, "scenes": args.num_scenes, "seed": args.seed,
                                                 "trunk_only": bool(getattr(margs, "jepa_trunk_only", False)),
                                                 "teacherless": bool(getattr(margs, "data2vec_teacherless", False))},
                                    "metrics": report}, indent=2) + "\n")
        del student, teacher
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
