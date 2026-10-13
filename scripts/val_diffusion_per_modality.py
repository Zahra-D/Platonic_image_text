"""Masked-diffusion validation loss per modality, on identical data and masks for every model.

For each checkpoint and each modality it can handle (text-only, image-only, or both),
on the first --n validation captions / image grids:
  * masked-token CE and accuracy at fixed mask rates (default 0.15 0.5 0.75);
  * the diffusion NELBO per token, estimated as the mean masked CE over a uniform
    t grid (0.05 ... 0.95): with i.i.d. masking at rate t, E[(1/t) * sum_masked CE / L]
    equals the mean CE over masked tokens, so its average over t ~ U(0,1) is the
    quantity the 1/t-weighted training loss estimates.
Masks are seeded per (batch, t), so every model sees exactly the same corruption.
TRUNK:path scores a dense_private model with its private LoRA switched off (route -1);
a plain path to a dense_private model uses its normal routes.

  python3 scripts/val_diffusion_per_modality.py --checkpoint LABEL=PATH [...] --output OUT.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clevr_paths import CLEVR_DATA_ROOT
from data import MultimodalCollator
from data.multimodal_dataset import read_jsonl
from evaluate_shared_private_retrieval import load_model
from multimodal_diffusion import corrupt_batch

GRID = [0.05 + 0.1 * k for k in range(10)]


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--text-manifest", default=CLEVR_DATA_ROOT + "/platonic_text_only_v1_1m/val_text_only_human.jsonl")
    p.add_argument("--image-cache", default="outputs/image_only_2_5m_token_cache/val_tokens.pt")
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--rates", type=float, nargs="+", default=[0.15, 0.5, 0.75])
    p.add_argument("--output", required=True)
    return p.parse_args()


def modalities(margs):
    return {"text_only": ["text"], "image_only": ["image"]}.get(margs.data_mode, ["text", "image"])


@torch.no_grad()
def score(model, tokenizer, margs, args, trunk, device):
    collator = MultimodalCollator(tokenizer, margs.num_image_codes, margs.max_text_length)
    out = {}
    for modality in modalities(margs):
        if modality == "text":
            items = [{"kind": "text", "text": r["caption_human"]} for r in read_jsonl(args.text_manifest)[: args.n]]
        else:
            payload = torch.load(args.image_cache, map_location="cpu", weights_only=False)
            grids = (payload["tokens"] if isinstance(payload, dict) else payload)[: args.n]
            items = [{"kind": "image", "image_tokens": g.long()} for g in grids]
        sums = {}
        for bi, s in enumerate(range(0, len(items), args.batch_size)):
            chunk = [dict(it, pair_index=s + k) for k, it in enumerate(items[s:s + args.batch_size])]
            b = {k: v.to(device) for k, v in collator(chunk).items()}
            routes = torch.full_like(b["route_ids"], -1) if trunk else b["route_ids"]
            for t in sorted(set(args.rates) | set(GRID)):
                torch.manual_seed(1000 * bi + int(round(t * 100)))
                corrupted, masked, _ = corrupt_batch(b["input_ids"], b["eligible_mask"], b["modality_ids"],
                                                     tokenizer.mask_id, 1e-3, modality, fixed_t=t)
                logits = model(corrupted, b["attention_mask"], b["position_ids"], b["modality_ids"], routes)
                target = b["input_ids"][masked]
                lg = logits[masked].float()
                a = sums.setdefault(round(t, 2), [0.0, 0.0, 0])
                a[0] += F.cross_entropy(lg, target, reduction="sum").item()
                a[1] += lg.argmax(-1).eq(target).sum().item()
                a[2] += target.numel()
        per_t = {t: {"ce": v[0] / v[2], "acc": v[1] / v[2], "tokens": v[2]} for t, v in sums.items()}
        out[modality] = {"rates": {str(t): per_t[round(t, 2)] for t in args.rates},
                         "nelbo_per_token": sum(per_t[round(t, 2)]["ce"] for t in GRID) / len(GRID)}
    return out


def main():
    args = arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dest = Path(args.output)
    res = json.loads(dest.read_text()) if dest.exists() else {}
    for spec in args.checkpoint:
        label, path = spec.split("=", 1)
        if label in res:
            print(f"{label}: exists, skipping", flush=True); continue
        trunk = path.startswith("TRUNK:")
        model, tokenizer, margs = load_model(path[len("TRUNK:"):] if trunk else path, device)
        model.eval()
        res[label] = {"path": path, "vocab": len(tokenizer), **score(model, tokenizer, margs, args, trunk, device)}
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(res, indent=2) + "\n")
        print(label, {m: round(v["nelbo_per_token"], 4) for m, v in res[label].items() if isinstance(v, dict) and "nelbo_per_token" in v}, flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
