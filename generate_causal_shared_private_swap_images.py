"""Save visual self-versus-swap samples for the image causal-swap evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from analyze_paired_representations import checkpoint_args
from data import ClevrMultimodalDataset, ClevrTextTokenizer, MultimodalCollator
from evaluate_causal_shared_private_swap import generate_with_shared_source, native_shared, only_image, pairs
from models.vqvae import VQVAE
from train_multimodal import build_model


def save_grid(images, path, nrow):
    """Dependency-free CHW tensor grid writer; images are in [0, 1]."""
    images = images.detach().cpu().clamp(0, 1)
    count, _channels, height, width = images.shape
    rows = (count + nrow - 1) // nrow
    grid = torch.ones(3, rows * height, nrow * width)
    for index, image in enumerate(images):
        row, column = divmod(index, nrow)
        grid[:, row * height:(row + 1) * height, column * width:(column + 1) * width] = image
    array = (grid.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    Image.fromarray(array).save(path)


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--val-dir", required=True); p.add_argument("--val-manifest", required=True); p.add_argument("--token-cache", required=True)
    p.add_argument("--caption-field", default="caption_human"); p.add_argument("--vqvae", default="outputs/vqvae_training_bs128/best.pt")
    p.add_argument("--layers", nargs="+", type=int, default=[2, 3, 4]); p.add_argument("--samples", type=int, default=4); p.add_argument("--steps", type=int, default=64); p.add_argument("--seed", type=int, default=20260915); p.add_argument("--output", required=True); p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    cli = args(); specs = [item.split("=", 1) for item in cli.checkpoint]
    first = torch.load(specs[0][1], map_location="cpu", weights_only=False); cfg = checkpoint_args(first); tokenizer = ClevrTextTokenizer(first["text_vocabulary"])
    cache = Path(cli.token_cache); cache = cache / "val_tokens.pt" if cache.is_dir() else cache
    dataset = ClevrMultimodalDataset(cli.val_dir, cache, "paired", cli.val_manifest, cli.caption_field); collator = MultimodalCollator(tokenizer, cfg.num_image_codes, cfg.max_text_length)
    vq_ckpt = torch.load(cli.vqvae, map_location="cpu", weights_only=False); vq_cfg = vq_ckpt.get("args", {}); vqvae = VQVAE(embed_dim=vq_cfg.get("embed_dim", 64), num_codes=vq_cfg.get("num_codes", 512)).to(cli.device); vqvae.load_state_dict(vq_ckpt["model"]); vqvae.eval()
    target_idx, source_idx = pairs(cli.samples, cli.seed); target = only_image(collator([dataset[int(i)] for i in target_idx]), cli.device); source = only_image(collator([dataset[int(i)] for i in source_idx]), cli.device)
    output = Path(cli.output); output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for name, path in specs:
            payload = torch.load(path, map_location="cpu", weights_only=False); model_cfg = checkpoint_args(payload); model, _ = build_model(model_cfg, len(tokenizer)); model.load_state_dict(payload["model"]); model.to(cli.device).eval()
            source_shared = native_shared(model, source, cli.layers); target_shared = native_shared(model, target, cli.layers)
            self_codes = generate_with_shared_source(model, target, target_shared, len(tokenizer), model_cfg.num_image_codes, tokenizer.mask_id, cli.steps).reshape(cli.samples, *model_cfg.grid_size)
            swap_codes = generate_with_shared_source(model, target, source_shared, len(tokenizer), model_cfg.num_image_codes, tokenizer.mask_id, cli.steps).reshape(cli.samples, *model_cfg.grid_size)
            clean_target = target["input_ids"][target["eligible_mask"]].reshape(cli.samples, *model_cfg.grid_size) - len(tokenizer)
            clean_source = source["input_ids"][source["eligible_mask"]].reshape(cli.samples, *model_cfg.grid_size) - len(tokenizer)
            # For each pair: source clean, target clean, self(B-shared), swap(A-shared).
            decoded = vqvae.decode_from_indices(torch.stack((clean_source, clean_target, self_codes, swap_codes), 1).reshape(-1, *model_cfg.grid_size))
            save_grid((decoded.clamp(-1, 1) + 1) / 2, output / f"{name}.png", nrow=4)
            print(f"wrote {output / (name + '.png')}", flush=True); del model
            if cli.device.startswith("cuda"): torch.cuda.empty_cache()
    (output / "README.md").write_text("# Causal image swap samples\n\nEach PNG has one row per held-out source/target pair and four columns: **clean source A**, **clean target B**, **self** $G(\\Delta_s(B_{clean}),P_B)$, and **swap** $G(\\Delta_s(A_{clean}),P_B)$. All targets were generated from full mask over 64 confidence steps; source shared deltas are substituted at layers 2–4 on every generation forward pass.\n")

if __name__ == "__main__": main()
