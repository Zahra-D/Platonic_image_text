"""How much does each modality rely on its private LoRA in a dense_private model?

For text and image inputs separately (val data, full model unless stated):
  1. write ratio   ||private delta|| / ||trunk (base) output|| per block and module,
                   averaged over content tokens -- how much of each linear layer's
                   output the private adapter supplies;
  2. diffusion     masked-token CE and accuracy with the private LoRA on (normal
                   routes) vs off (trunk only), at fixed mask rates;
  3. divergence    per block, cosine between the pooled residual stream of the
                   full model and of the trunk-only forward on the same clean input
                   (1 = the private LoRA barely changes the computation).

  python3 scripts/diagnose_private_reliance.py --checkpoint LABEL=PATH [...] --output OUT.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clevr_paths import CLEVR_DATA_ROOT
from data import MultimodalCollator
from data.multimodal_dataset import read_jsonl
from evaluate_shared_private_retrieval import load_model
from models.lora import TriLoRALinear
from multimodal_diffusion import corrupt_batch


def batches(args, tokenizer, model_args):
    collator = MultimodalCollator(tokenizer, model_args.num_image_codes, model_args.max_text_length)
    texts = [r["caption_human"] for r in read_jsonl(args.text_manifest)[: args.n]]
    payload = torch.load(args.image_cache, map_location="cpu", weights_only=False)
    tokens = (payload["tokens"] if isinstance(payload, dict) else payload)[: args.n]
    out = {"text": [], "image": []}
    for s in range(0, args.n, args.batch_size):
        out["text"].append(collator([{"kind": "text", "text": t, "pair_index": s + k}
                                     for k, t in enumerate(texts[s:s + args.batch_size])]))
        out["image"].append(collator([{"kind": "image", "image_tokens": t.long(), "pair_index": s + k}
                                      for k, t in enumerate(tokens[s:s + args.batch_size])]))
    return out


@torch.no_grad()
def diagnose(model, tokenizer, data, device, rates):
    lora = [(n, m) for n, m in model.named_modules() if isinstance(m, TriLoRALinear)]
    res = {}
    for modality in ("text", "image"):
        ratios, state = {}, {"record": False, "mask": None}
        def hook(module, inputs, _output, name):
            if not state["record"]:
                return
            x, keep = inputs[0], state["mask"]
            base = module.base(x)[keep].float().norm(dim=-1)
            delta = module._delta(x, modality)[keep].float().norm(dim=-1)
            ratios.setdefault(name, []).append((delta / base.clamp_min(1e-6)).mean().item())
        handles = [m.register_forward_hook(lambda mod, i, o, name=n: hook(mod, i, o, name)) for n, m in lora]
        div, loss = {}, {}
        try:
            for b in data[modality]:
                b = {k: v.to(device) for k, v in b.items()}
                off = torch.full_like(b["route_ids"], -1)
                state.update(record=True, mask=b["eligible_mask"])
                _, full = model(b["input_ids"], b["attention_mask"], b["position_ids"], b["modality_ids"],
                                b["route_ids"], return_hidden_by_layer=True)
                state["record"] = False
                _, trunk = model(b["input_ids"], b["attention_mask"], b["position_ids"], b["modality_ids"],
                                 off, return_hidden_by_layer=True)
                w = b["eligible_mask"].float().unsqueeze(-1)
                for layer in full:
                    pf = (full[layer].float() * w).sum(1) / w.sum(1)
                    pt = (trunk[layer].float() * w).sum(1) / w.sum(1)
                    div.setdefault(layer, []).append(F.cosine_similarity(pf, pt, dim=-1).mean().item())
                for t in rates:
                    torch.manual_seed(1234)  # identical masks for both routings
                    corrupted, masked, _ = corrupt_batch(b["input_ids"], b["eligible_mask"], b["modality_ids"],
                                                         tokenizer.mask_id, 1e-3, modality, fixed_t=t)
                    for mode, routes in (("private_on", b["route_ids"]), ("trunk_only", off)):
                        logits = model(corrupted, b["attention_mask"], b["position_ids"], b["modality_ids"], routes)
                        ce = F.cross_entropy(logits[masked].float(), b["input_ids"][masked]).item()
                        acc = logits[masked].argmax(-1).eq(b["input_ids"][masked]).float().mean().item()
                        loss.setdefault(f"t={t}", {}).setdefault(mode, []).append((ce, acc))
        finally:
            for h in handles:
                h.remove()
        mean = lambda v: sum(v) / len(v)
        res[modality] = {
            "write_ratio": {n: mean(v) for n, v in ratios.items()},
            "residual_cosine_full_vs_trunk": {f"L{l}": mean(v) for l, v in sorted(div.items())},
            "diffusion": {t: {mode: {"ce": mean([c for c, _ in v]), "acc": mean([a for _, a in v])}
                              for mode, v in modes.items()} for t, modes in loss.items()},
        }
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--text-manifest", default=CLEVR_DATA_ROOT + "/platonic_text_only_v1_1m/val_text_only_human.jsonl")
    p.add_argument("--image-cache", default="outputs/image_only_2_5m_token_cache/val_tokens.pt")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--rates", type=float, nargs="+", default=[0.15, 0.5])
    p.add_argument("--output", required=True)
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {}
    for spec in a.checkpoint:
        label, path = spec.split("=", 1)
        model, tokenizer, margs = load_model(path, torch.device(device))
        data = batches(a, tokenizer, margs)
        report[label] = diagnose(model, tokenizer, data, device, a.rates)
        r = report[label]
        for mod in ("text", "image"):
            wr = r[mod]["write_ratio"]; by_layer = {}
            for n, v in wr.items():
                by_layer.setdefault(int(n.split(".")[1]), []).append(v)
            layers = " ".join(f"L{l}:{sum(v)/len(v):.2f}" for l, v in sorted(by_layer.items()))
            d = r[mod]["diffusion"]
            dl = " | ".join(f"{t} CE {m['private_on']['ce']:.2f}->{m['trunk_only']['ce']:.2f} "
                            f"acc {m['private_on']['acc']:.3f}->{m['trunk_only']['acc']:.3f}" for t, m in d.items())
            cos = r[mod]["residual_cosine_full_vs_trunk"]
            print(f"{label} {mod:5s} | private/trunk write ratio {layers} | {dl} | "
                  f"residual cos full~trunk L0 {cos['L0']:.3f} L3 {cos['L3']:.3f} L7 {cos['L7']:.3f}", flush=True)
        del model; torch.cuda.empty_cache()
    Path(a.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
