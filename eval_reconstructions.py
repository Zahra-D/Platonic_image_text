import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.clevr_dataset import ClevrImageFolder
from models.vqvae import VQVAE

DEFAULT_CONFIG = str(Path(__file__).resolve().parent / "configs" / "vqvae.yaml")
DEFAULT_VAL_DIR = str(Path.home() / "Omni/data/clevr/val")


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=DEFAULT_CONFIG,
                      help="YAML file supplying defaults for every flag below.")
    pre_args, _ = pre.parse_known_args()

    cfg = {}
    if pre_args.config and Path(pre_args.config).exists():
        with open(pre_args.config) as f:
            cfg = yaml.safe_load(f) or {}
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("eval", {})
    wandb_cfg = cfg.get("wandb", {})

    p = argparse.ArgumentParser(
        parents=[pre],
        description="Stage-1 gate: reconstruct CLEVR images through a trained VQ-VAE "
        "and report fidelity metrics + a contact sheet for visual inspection.",
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val-dir", default=data_cfg.get("val_dir", DEFAULT_VAL_DIR))
    p.add_argument("--image-size", type=int, nargs=2,
                    default=data_cfg.get("image_size", [64, 96]), metavar=("H", "W"))
    p.add_argument("--batch-size", type=int, default=eval_cfg.get("batch_size", 128))
    p.add_argument("--num-workers", type=int, default=eval_cfg.get("num_workers", 8))
    p.add_argument("--max-samples", type=int, default=eval_cfg.get("max_samples", 2000),
                    help="Cap how many val images to score (full val is 15k).")
    p.add_argument("--grid-samples", type=int, default=eval_cfg.get("grid_samples", 32),
                    help="How many images to render in the contact sheet.")
    p.add_argument("--output-dir", default=eval_cfg.get("output_dir", "outputs/eval"))
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction,
                    default=wandb_cfg.get("enabled", True),
                    help="Log the gate metrics + contact sheet to Weights & Biases.")
    p.add_argument("--wandb-project",
                    default=wandb_cfg.get("project", "clevr-discrete-diffusion"))
    p.add_argument("--wandb-entity", default=wandb_cfg.get("entity"))
    p.add_argument("--wandb-group", default=wandb_cfg.get("group", "vqvae_training"))
    p.add_argument("--wandb-run-name", default=wandb_cfg.get("run_name"),
                    help="Defaults to '<checkpoint-run-name>-eval-<timestamp>'.")
    return p.parse_args()


def psnr(recon, target):
    mse = F.mse_loss(recon, target, reduction="none").mean(dim=[1, 2, 3])
    mse = mse.clamp_min(1e-10)
    return (20 * torch.log10(torch.tensor(2.0)) - 10 * torch.log10(mse)).mean().item()


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})

    wandb_run = None
    if args.wandb:
        import wandb

        run_base = Path(ckpt_args.get("output_dir", "run")).name
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name or f"{run_base}-eval-{time.strftime('%m%d-%H%M')}",
            job_type="eval",
            config={"checkpoint": args.checkpoint, **ckpt_args},
        )
        wandb_run.save(args.config, policy="now")
    model = VQVAE(
        embed_dim=ckpt_args.get("embed_dim", 64),
        num_codes=ckpt_args.get("num_codes", 512),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    num_codes = model.quantizer.num_codes

    val_ds = ClevrImageFolder(args.val_dir, image_size=tuple(args.image_size))
    if args.max_samples is not None:
        val_ds.paths = val_ds.paths[: args.max_samples]
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    total_l1, total_psnr, n_batches = 0.0, 0.0, 0
    code_counts = torch.zeros(num_codes, device=device)

    for x in val_loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        recon = out["recon"]
        total_l1 += F.l1_loss(recon, x).item()
        total_psnr += psnr(recon, x)
        n_batches += 1
        code_counts += torch.bincount(out["indices"].flatten(), minlength=num_codes)

    mean_l1 = total_l1 / n_batches
    mean_psnr = total_psnr / n_batches
    probs = code_counts / code_counts.sum().clamp_min(1)
    perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum()).item()
    dead_codes = int((code_counts == 0).sum().item())

    print(f"scored {len(val_ds)} val images from {args.checkpoint}")
    print(f"mean L1:        {mean_l1:.4f}")
    print(f"mean PSNR:      {mean_psnr:.2f} dB")
    print(f"perplexity:     {perplexity:.1f} / {num_codes}")
    print(f"dead codes:     {dead_codes} / {num_codes}")

    n = min(args.grid_samples, len(val_ds))
    imgs = torch.stack([val_ds[i] for i in range(n)]).to(device)
    recon = model(imgs)["recon"]
    ncols = math.ceil(math.sqrt(n))
    grid = torch.cat([imgs, recon], dim=0)
    grid_path = out_dir / "contact_sheet.png"
    save_image(grid, grid_path, nrow=n, normalize=True, value_range=(-1, 1))
    print(f"contact sheet (top=original, bottom=reconstruction): {grid_path}")
    print(
        "\nNow eyeball the contact sheet: object count, shape, color, material, size, "
        "and position should all match before starting Stage 2 (D3PM Transformer)."
    )

    if wandb_run:
        wandb_run.log(
            {
                "eval/l1": mean_l1,
                "eval/psnr": mean_psnr,
                "eval/perplexity": perplexity,
                "eval/dead_codes": dead_codes,
                "eval/contact_sheet": wandb.Image(str(grid_path)),
            }
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
