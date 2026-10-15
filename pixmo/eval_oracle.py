"""Oracle-W alignment of a checkpoint's shared trunk on PixMo-Cap: are image and caption features a rotation apart?

The checkpoint is loaded trunk-only (as "TRUNK:" in the CLEVR evals: a dense_private model rebuilt as a plain dense
model from its base weights, every private LoRA tensor dropped; pass --full to keep the private routes). For every
readout -- the token embedding, the residual stream after each block (L<i>) and each block's MLP output before it is
added to the stream (L<i>.mlp_out) -- image and caption features are mean-pooled over content tokens, and:

  oracle    orthogonal Procrustes W fitted on the probe pairs (10k train images with their own, never-trained
            captions), scored on the 5k test pairs: the best any rotation can do
  identity  the same features compared with no map (each modality centred on its probe mean): does the trunk
            already put an image and its caption in the same place?

Scores: FOSCTTM (0 perfect, 0.5 chance), R@1 / R@10 image->text and text->image over the 5k x 5k test pool.
"score" = mean of the two oracle R@1 at the best readout (the stage-2 convergence signal of pixmo/pipeline.py).

    python pixmo/eval_oracle.py --checkpoint runs/stage2/epoch_003.pt --out runs/stage2/eval/epoch_003.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data import ClevrTextTokenizer, MultimodalCollator  # noqa: E402
from train_multimodal import build_model  # noqa: E402

IGNORED = ("modulewise_jepa_predictors.", "shared_jepa_predictors.", "data2vec_heads.")
PRIVATE = ("text_A", "text_B", "image_A", "image_B")


def load(checkpoint: str, device: str, full: bool):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    values = dict(payload["args"])
    values.setdefault("lora_architecture", "tri")
    values.setdefault("delete_base_weights", values.get("train_mode") == "lora")
    values.setdefault("modality_adversarial", False)
    values.setdefault("modality_discriminator_hidden", None)
    args = Namespace(**values)
    state = payload["model"]
    if not full and args.train_mode == "dense_private":
        args.train_mode = "dense"
        state = {re.sub(r"\.base\.(weight|bias)$", r".\1", k): v for k, v in state.items() if not k.endswith(PRIVATE)}
    tokenizer = ClevrTextTokenizer(payload["text_vocabulary"])
    model, _ = build_model(args, len(tokenizer))
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith(IGNORED)]
    unexpected = [k for k in result.unexpected_keys if not k.startswith(IGNORED)]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint/model mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.to(device).eval(), tokenizer, args, int(payload.get("epoch", -1))


@torch.no_grad()
def encode(model, tokenizer, args, device, batch_size, captions=None, image_tokens=None) -> dict[str, torch.Tensor]:
    """Mean over content tokens of the embedding, each block output and each block's MLP output."""
    collator = MultimodalCollator(tokenizer, args.num_image_codes, args.max_text_length)
    captured, hooks = {}, []
    hooks.append(model.blocks[0].register_forward_pre_hook(lambda _m, inputs: captured.__setitem__("embedding", inputs[0])))
    for i, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(lambda _m, _i, out, i=i: captured.__setitem__(f"L{i:02d}", out)))
        hooks.append(block.mlp[3].register_forward_hook(lambda _m, _i, out, i=i: captured.__setitem__(f"L{i:02d}.mlp_out", out)))
    count = len(captions) if captions is not None else len(image_tokens)
    values: dict[str, list[torch.Tensor]] = {}
    try:
        for start in range(0, count, batch_size):
            if captions is not None:
                examples = [{"kind": "text", "text": t, "pair_index": start + k} for k, t in enumerate(captions[start:start + batch_size])]
            else:
                examples = [{"kind": "image", "image_tokens": t.long(), "pair_index": start + k}
                            for k, t in enumerate(image_tokens[start:start + batch_size])]
            batch = {k: v.to(device) for k, v in collator(examples).items()}
            captured.clear()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                model(batch["input_ids"], batch["attention_mask"], batch["position_ids"], batch["modality_ids"], batch["route_ids"])
            weights = batch["eligible_mask"].float().unsqueeze(-1)
            for name, tensor in captured.items():
                values.setdefault(name, []).append(((tensor.float() * weights).sum(1) / weights.sum(1).clamp_min(1)).cpu())
    finally:
        for hook in hooks:
            hook.remove()
    return {name: torch.cat(parts) for name, parts in values.items()}


def retrieval(sim: torch.Tensor) -> dict[str, float]:
    """sim[i, j] = image i vs caption j; the partner of image i is caption i."""
    n = len(sim)
    true = sim.diagonal()
    target = torch.arange(n, device=sim.device)
    i2t, t2i = sim.topk(10, dim=1).indices, sim.T.topk(10, dim=1).indices
    return {"foscttm": ((sim > true[:, None]).float().mean() + (sim > true[None, :]).float().mean()).item() / 2,
            "i2t_R@1": (i2t[:, 0] == target).float().mean().item(), "i2t_R@10": (i2t == target[:, None]).any(1).float().mean().item(),
            "t2i_R@1": (t2i[:, 0] == target).float().mean().item(), "t2i_R@10": (t2i == target[:, None]).any(1).float().mean().item()}


def align(probe_img, probe_txt, test_img, test_txt) -> dict[str, dict]:
    mi, mt = probe_img.mean(0), probe_txt.mean(0)
    fi, ft = F.normalize(probe_img - mi, dim=1), F.normalize(probe_txt - mt, dim=1)
    ti, tt = F.normalize(test_img - mi, dim=1), F.normalize(test_txt - mt, dim=1)
    u, _, vt = torch.linalg.svd(fi.T @ ft, full_matrices=False)
    return {"oracle": retrieval((ti @ (u @ vt)) @ tt.T), "identity": retrieval(ti @ tt.T)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=os.environ.get("PIXMO_DATA", str(Path.home() / "platonic_pixmo_data")))
    p.add_argument("--full", action="store_true", help="keep the private LoRA routes (default: trunk only)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    pairs = torch.load(Path(a.data) / "eval_pairs.pt", weights_only=False)
    model, tokenizer, args, epoch = load(a.checkpoint, a.device, a.full)
    feats = {name: (encode(model, tokenizer, args, a.device, a.batch_size, image_tokens=s["image_tokens"]),
                    encode(model, tokenizer, args, a.device, a.batch_size, captions=s["captions"]))
             for name, s in pairs.items()}
    (pi, pt), (ti, tt) = feats["probe"], feats["test"]
    readouts = {k: align(pi[k].to(a.device), pt[k].to(a.device), ti[k].to(a.device), tt[k].to(a.device)) for k in ti}
    mean_r1 = lambda r: (r["i2t_R@1"] + r["t2i_R@1"]) / 2
    best = max(readouts, key=lambda k: mean_r1(readouts[k]["oracle"]))
    result = {"checkpoint": a.checkpoint, "epoch": epoch, "trunk_only": not a.full,
              "num_probe": len(pairs["probe"]["captions"]), "num_test": len(pairs["test"]["captions"]),
              "best_readout": best, "score": mean_r1(readouts[best]["oracle"]),
              "best": readouts[best], "readouts": readouts}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1))
    o, i = readouts[best]["oracle"], readouts[best]["identity"]
    print(f"epoch {epoch}: best readout {best}: oracle FOSCTTM {o['foscttm']:.4f} R@1 {100 * o['i2t_R@1']:.1f}/{100 * o['t2i_R@1']:.1f} "
          f"R@10 {100 * o['i2t_R@10']:.1f}/{100 * o['t2i_R@10']:.1f} | no W: FOSCTTM {i['foscttm']:.4f} "
          f"R@1 {100 * i['i2t_R@1']:.1f}/{100 * i['t2i_R@1']:.1f}", flush=True)


if __name__ == "__main__":
    main()
