"""d' on generated image triples: anchor / paraphrase / binding-swap.

The image analogue of evaluate_semantic_dprime. The paraphrase shows the same
scene from a different camera; the swap shows two objects' colours exchanged
from the *paraphrase's* camera, so both candidates differ from the anchor by
the same viewpoint change and only the binding separates them.

    d_binding = d'( cos(anchor, paraphrase) , cos(anchor, swap) )
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from data import MultimodalCollator
from evaluate_shared_private_retrieval import load_model


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--triples", default="outputs/image_eval_triples/triples_tokens.pt")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261005)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


@torch.no_grad()
def encode(model, tokenizer, margs, tokens, batch_size, device):
    """Pooled, L2-normalised features at every residual block and sublayer."""
    collator = MultimodalCollator(tokenizer, margs.num_image_codes, margs.max_text_length)
    captured, hooks = {}, []
    hooks.append(model.blocks[0].register_forward_pre_hook(
        lambda _m, inp: captured.__setitem__("embedding", inp[0].detach())))
    for i, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(
            lambda _m, _i, o, i=i: captured.__setitem__(f"L{i}", o.detach())))
        hooks.append(block.attn.out_proj.register_forward_hook(
            lambda _m, _i, o, i=i: captured.__setitem__(f"L{i}.attn_out", o.detach())))
        hooks.append(block.mlp[3].register_forward_hook(
            lambda _m, _i, o, i=i: captured.__setitem__(f"L{i}.mlp_out", o.detach())))
    values: dict[str, list] = {}
    try:
        for s in range(0, tokens.shape[0], batch_size):
            chunk = tokens[s:s + batch_size]
            examples = [{"kind": "image", "image_tokens": t, "pair_index": s + k}
                        for k, t in enumerate(chunk)]
            batch = {k: v.to(device) for k, v in collator(examples).items()}
            captured.clear()
            model(batch["input_ids"], batch["attention_mask"], batch["position_ids"],
                  batch["modality_ids"], batch["route_ids"])
            w = batch["eligible_mask"].double().unsqueeze(-1)
            for name, t in captured.items():
                pooled = (t.double() * w).sum(1) / w.sum(1).clamp_min(1)
                values.setdefault(name, []).append(F.normalize(pooled, dim=1).float().cpu())
    finally:
        for h in hooks:
            h.remove()
    return {k: torch.cat(v).numpy().astype(np.float64) for k, v in values.items()}


def _unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def effect_size(pos, neg, repetitions, seed):
    pos, neg = np.asarray(pos), np.asarray(neg)
    def stat(pick):
        p, n = pos[pick], neg[pick]
        spread = np.sqrt((p.var() + n.var()) / 2)
        return (p.mean() - n.mean()) / max(spread, 1e-12), float((p > n).mean())
    rng = np.random.default_rng(seed)
    point = stat(np.arange(len(pos)))
    draws = np.asarray([stat(rng.integers(0, len(pos), len(pos))) for _ in range(repetitions)])
    lo, hi = np.quantile(draws, (0.025, 0.975), axis=0)
    return {"d": {"value": float(point[0]), "ci95": [float(lo[0]), float(hi[0])]},
            "preference": {"value": float(point[1]), "ci95": [float(lo[1]), float(hi[1])]},
            "mean_positive": float(pos.mean()), "mean_negative": float(neg.mean()),
            "pairs": int(len(pos))}


def main():
    a = arguments()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    payload = torch.load(a.triples, map_location="cpu", weights_only=False)
    tok = payload["tokens"]
    need = ("anchor", "para", "swap")
    missing = [s for s in need if s not in tok]
    if missing:
        raise SystemExit(f"triples file is missing splits: {missing}")
    as_long = lambda t: torch.from_numpy(t.numpy().astype("int64"))
    anchor, para, swap = (as_long(tok[s]) for s in need)
    print(f"{anchor.shape[0]} triples", flush=True)

    device = torch.device(a.device)
    for spec in a.checkpoint:
        label, path = spec.split("=", 1)
        dest = out / f"{label}.json"
        if dest.exists():
            print(f"{label}: exists, skipping", flush=True); continue
        model, tokenizer, margs = load_model(path, device)
        feats = {s: encode(model, tokenizer, margs, t, a.batch_size, device)
                 for s, t in zip(need, (anchor, para, swap))}
        metrics = {}
        for name in feats["anchor"]:
            cos = lambda x, y: (feats[x][name] * feats[y][name]).sum(1)
            # Raw cosines sit at ~0.999 for every pair because the pooled
            # representation is ~99% a direction shared by all images.  The
            # centred variant removes that common component -- computed over
            # the anchors, then applied to all three splits so the three stay
            # in one frame -- and is the number to read.
            mu = feats["anchor"][name].mean(0, keepdims=True)
            cen = {s_: _unit(feats[s_][name] - mu) for s_ in feats}
            ccos = lambda x, y: (cen[x] * cen[y]).sum(1)
            metrics[name] = {
                "d_binding": effect_size(cos("anchor", "para"),
                                         cos("anchor", "swap"),
                                         a.bootstrap, a.seed),
                "d_binding_centred": effect_size(ccos("anchor", "para"),
                                                 ccos("anchor", "swap"),
                                                 a.bootstrap, a.seed)}
        best = max(metrics, key=lambda k: metrics[k]["d_binding_centred"]["d"]["value"])
        dest.write_text(json.dumps({
            "protocol": {"model": str(path), "triples": int(anchor.shape[0]),
                         "train_mode": margs.train_mode},
            "metrics": metrics}, indent=2) + "\n")
        print(f"{label}: L7 raw {metrics['L7']['d_binding']['d']['value']:+.3f} "
              f"centred {metrics['L7']['d_binding_centred']['d']['value']:+.3f} | "
              f"best centred {best} {metrics[best]['d_binding_centred']['d']['value']:+.3f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
