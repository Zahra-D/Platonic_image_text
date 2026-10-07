import argparse
import sys
import time
from pathlib import Path

import torch
import yaml
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.vqvae import VQVAE
from models.transformer import MaskedTokenTransformer
from diffusion import generate

DEFAULT_CONFIG = str(Path(__file__).resolve().parent / "configs" / "d3pm.yaml")


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=DEFAULT_CONFIG,
                      help="YAML file supplying defaults for every flag below.")
    pre_args, _ = pre.parse_known_args()

    cfg = {}
    if pre_args.config and Path(pre_args.config).exists():
        with open(pre_args.config) as f:
            cfg = yaml.safe_load(f) or {}
    gen_cfg = cfg.get("generate", {})
    wandb_cfg = cfg.get("wandb", {})

    p = argparse.ArgumentParser(
        parents=[pre],
        description="Stage-2 gate: generate CLEVR token grids from pure noise (all-MASK) "
        "through a trained Transformer, decode via the frozen VQ-VAE, and save a contact "
        "sheet for visual inspection.",
    )
    p.add_argument("--checkpoint", required=True, help="Trained Transformer checkpoint.")
    p.add_argument("--num-samples", type=int, default=gen_cfg.get("num_samples", 8))
    p.add_argument("--num-steps", type=int, default=gen_cfg.get("num_steps", 50))
    p.add_argument("--temperature", type=float, default=gen_cfg.get("temperature", 1.0))
    p.add_argument("--reveal-order", choices=["confidence", "random"],
                    default=gen_cfg.get("reveal_order", "confidence"))
    p.add_argument("--output-dir", default=gen_cfg.get("output_dir", "outputs/d3pm_eval"))
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction,
                    default=wandb_cfg.get("enabled", True),
                    help="Log the sample grid to Weights & Biases.")
    p.add_argument("--wandb-project",
                    default=wandb_cfg.get("project", "clevr-discrete-diffusion"))
    p.add_argument("--wandb-entity", default=wandb_cfg.get("entity"))
    p.add_argument("--wandb-group", default=wandb_cfg.get("group", "d3pm_training"))
    p.add_argument("--wandb-run-name", default=wandb_cfg.get("run_name"),
                    help="Defaults to '<checkpoint-run-name>-generate-<timestamp>'.")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    wandb_run = None
    if args.wandb:
        import wandb

        run_base = Path(ckpt_args.get("output_dir", "run")).name
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name or f"{run_base}-generate-{time.strftime('%m%d-%H%M')}",
            job_type="generate",
            config={"checkpoint": args.checkpoint, **ckpt_args},
        )
        wandb_run.save(args.config, policy="now")

    tokenizer_checkpoint = ckpt_args["tokenizer_checkpoint"]
    tok_ckpt = torch.load(tokenizer_checkpoint, map_location=device, weights_only=False)
    tok_ckpt_args = tok_ckpt.get("args", {})
    vqvae = VQVAE(
        embed_dim=tok_ckpt_args.get("embed_dim", 64),
        num_codes=tok_ckpt_args.get("num_codes", 512),
    ).to(device)
    vqvae.load_state_dict(tok_ckpt["model"])
    vqvae.eval()

    model = MaskedTokenTransformer(
        vocab_size=ckpt_args.get("num_codes", 512),
        grid_size=tuple(ckpt_args.get("grid_size", [16, 24])),
        d_model=ckpt_args.get("d_model", 384),
        n_layers=ckpt_args.get("n_layers", 8),
        n_heads=ckpt_args.get("n_heads", 6),
        mlp_ratio=ckpt_args.get("mlp_ratio", 4),
        dropout=ckpt_args.get("dropout", 0.1),
        pos_embed_type=ckpt_args.get("pos_embed_type", "absolute"),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"generating {args.num_samples} samples from {args.checkpoint} "
          f"(tokenizer: {tokenizer_checkpoint}), {args.num_steps} unmasking steps")

    tokens = generate(
        model, seq_len=model.seq_len, mask_id=model.mask_id,
        num_steps=args.num_steps, batch_size=args.num_samples,
        device=device, temperature=args.temperature, reveal_order=args.reveal_order,
    )
    grid_h, grid_w = model.grid_h, model.grid_w
    tokens = tokens.view(-1, grid_h, grid_w)
    images = vqvae.decode_from_indices(tokens)

    grid_path = out_dir / "generated_samples.png"
    save_image(images, grid_path, nrow=args.num_samples, normalize=True, value_range=(-1, 1))
    print(f"generated samples: {grid_path}")
    print(
        "\nNow eyeball the grid: do these look like plausible CLEVR scenes "
        "(recognizable shapes/colors, sensible object counts) even though none of them "
        "correspond to a real image?"
    )

    if wandb_run:
        wandb_run.log({"generate/samples": wandb.Image(str(grid_path))})
        wandb_run.finish()


if __name__ == "__main__":
    main()
