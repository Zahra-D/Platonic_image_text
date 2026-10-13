"""Image features pooled over object tokens only (and, for contrast, background only).

The usual image feature is the mean over all 384 VQ tokens, ~91% of which are
floor and background.  Here each token of the 16x24 grid (4x4 px of the 96x64
render) is marked foreground when its centre lies within an object's projected
radius, r = 90 * (0.7 large | 0.35 small) / depth + 1 px, from the full CLEVR
scene file (pixel_coords).  Hidden states at the chosen readouts are pooled over
those tokens and L2-normalised, and saved in the xret feature format as
<label>_FG.npz (and <label>_BG.npz), for the 8000 retrieval scenes, so every
existing evaluation can use them unchanged.

  python3 scripts/foreground_pooled_features.py --checkpoint LABEL=PATH [...]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clevr_paths import CLEVR_DATA_ROOT
from data import MultimodalCollator
from evaluate_shared_private_retrieval import load_model

READOUTS = ["L4", "L5", "L6", "L7", "L4.mlp_out", "L5.mlp_out", "L6.mlp_out", "L7.mlp_out"]


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--image-manifest", default=CLEVR_DATA_ROOT + "/platonic_clevr_v1_5M_train_gpu_visible/val_image_only.jsonl")
    p.add_argument("--image-cache", default="outputs/image_only_2_5m_token_cache/val_tokens.pt")
    p.add_argument("--num-scenes", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output-dir", default="outputs/eval_all/xret/features")
    return p.parse_args()


def foreground_masks(manifest, n):
    root = Path(manifest).parent
    yy, xx = np.mgrid[0:16, 0:24]
    cy, cx = yy * 4 + 2, xx * 4 + 2
    out = []
    for _, line in zip(range(n), open(manifest)):
        row = json.loads(line)
        sc = json.load(open(root / row["image_path"].replace("images/", "scenes/", 1).replace(".png", ".json")))
        m = np.zeros((16, 24), bool)
        for o in sc["objects"]:
            x, y, d = o["pixel_coords"]
            r = 90 * (0.7 if o["size"] == "large" else 0.35) / d + 1.0
            m |= (cx - x) ** 2 + (cy - y) ** 2 <= r * r
        out.append(m.ravel())
    return np.stack(out)


@torch.no_grad()
def encode(model, tokenizer, margs, tokens, fg, batch_size, device):
    collator = MultimodalCollator(tokenizer, margs.num_image_codes, margs.max_text_length)
    captured, hooks = {}, []
    for i, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(lambda _m, _i, o, i=i: captured.__setitem__(f"L{i}", o.detach())))
        hooks.append(block.mlp[3].register_forward_hook(lambda _m, _i, o, i=i: captured.__setitem__(f"L{i}.mlp_out", o.detach())))
    out = {f"{r}|{part}": [] for r in READOUTS for part in ("fg", "bg")}
    try:
        for s in range(0, len(tokens), batch_size):
            chunk = tokens[s:s + batch_size]
            batch = collator([{"kind": "image", "image_tokens": g, "pair_index": s + k} for k, g in enumerate(chunk)])
            batch = {k: v.to(device) for k, v in batch.items()}
            captured.clear()
            model(batch["input_ids"], batch["attention_mask"], batch["position_ids"], batch["modality_ids"], batch["route_ids"])
            elig = batch["eligible_mask"]                                   # [B, L], 384 True per row, grid order
            m = torch.as_tensor(fg[s:s + batch_size], device=device)
            fgw = torch.zeros_like(elig, dtype=torch.float32); bgw = torch.zeros_like(fgw)
            fgw[elig] = m.reshape(-1).float(); bgw[elig] = (~m).reshape(-1).float()
            for r in READOUTS:
                h = captured[r].float()
                for part, w in (("fg", fgw), ("bg", bgw)):
                    pooled = (h * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp_min(1)
                    out[f"{r}|{part}"].append(F.normalize(pooled, dim=1).cpu())
    finally:
        for h in hooks:
            h.remove()
    return {k: torch.cat(v).numpy().astype(np.float32) for k, v in out.items()}


def main():
    a = arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = torch.load(a.image_cache, map_location="cpu", weights_only=False)["tokens"]
    tokens = torch.from_numpy(cache.numpy()[:a.num_scenes].astype("int64"))
    fg = foreground_masks(a.image_manifest, a.num_scenes)
    print(f"foreground fraction {fg.mean():.3f}", flush=True)
    for spec in a.checkpoint:
        label, path = spec.split("=", 1)
        model, tokenizer, margs = load_model(path, device)
        model.eval()
        feats = encode(model, tokenizer, margs, tokens, fg, a.batch_size, device)
        for part in ("fg", "bg"):
            np.savez(Path(a.output_dir) / f"{label}_{part.upper()}.npz",
                     **{r: feats[f"{r}|{part}"] for r in READOUTS})
        print(f"{label}: saved _FG / _BG", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
