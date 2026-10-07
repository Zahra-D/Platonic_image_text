import argparse
import logging
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
DEFAULT_TRAIN_DIR = str(Path.home() / "Omni/data/clevr/train")
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
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    wandb_cfg = cfg.get("wandb", {})

    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument("--train-dir", default=data_cfg.get("train_dir", DEFAULT_TRAIN_DIR))
    p.add_argument("--val-dir", default=data_cfg.get("val_dir", DEFAULT_VAL_DIR))
    p.add_argument("--image-size", type=int, nargs=2,
                    default=data_cfg.get("image_size", [64, 96]), metavar=("H", "W"))
    p.add_argument("--batch-size", type=int, default=train_cfg.get("batch_size", 128))
    p.add_argument("--epochs", type=int, default=train_cfg.get("epochs", 100))
    p.add_argument("--lr", type=float, default=train_cfg.get("lr", 3e-4))
    p.add_argument("--num-workers", type=int, default=train_cfg.get("num_workers", 8))
    p.add_argument("--amp", choices=["none", "bf16"], default=train_cfg.get("amp", "bf16"))
    p.add_argument("--resume", default=train_cfg.get("resume"))
    p.add_argument("--output-dir", default=train_cfg.get("output_dir", "outputs/run1"))
    p.add_argument("--log-every", type=int, default=train_cfg.get("log_every", 50))
    p.add_argument("--eval-every", type=int, default=train_cfg.get("eval_every", 1))
    p.add_argument("--save-every", type=int, default=train_cfg.get("save_every", 1))
    p.add_argument("--num-codes", type=int, default=model_cfg.get("num_codes", 512))
    p.add_argument("--embed-dim", type=int, default=model_cfg.get("embed_dim", 64))
    p.add_argument("--commitment-cost", type=float,
                    default=model_cfg.get("commitment_cost", 0.25))
    p.add_argument("--ema-decay", type=float, default=model_cfg.get("ema_decay", 0.99))
    p.add_argument("--max-train-samples", type=int,
                    default=train_cfg.get("max_train_samples"),
                    help="Truncate the training set (for smoke tests).")
    p.add_argument("--max-steps", type=int, default=train_cfg.get("max_steps"),
                    help="Stop after this many optimizer steps (for smoke tests).")
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction,
                    default=wandb_cfg.get("enabled", True),
                    help="Log to Weights & Biases.")
    p.add_argument("--wandb-project",
                    default=wandb_cfg.get("project", "clevr-discrete-diffusion"))
    p.add_argument("--wandb-entity", default=wandb_cfg.get("entity"))
    p.add_argument("--wandb-group", default=wandb_cfg.get("group", "vqvae_training"))
    p.add_argument("--wandb-run-name", default=wandb_cfg.get("run_name"),
                    help="Defaults to an auto-generated name from key hyperparams + timestamp.")
    return p.parse_args()


def default_run_name(args):
    return f"k{args.num_codes}-bs{args.batch_size}-{args.amp}-{time.strftime('%m%d-%H%M')}"


def setup_logging(out_dir, resume):
    logger = logging.getLogger("train_vqvae")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(out_dir / "train.log", mode="a" if resume else "w")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger


def make_loaders(args):
    train_ds = ClevrImageFolder(args.train_dir, image_size=tuple(args.image_size))
    val_ds = ClevrImageFolder(args.val_dir, image_size=tuple(args.image_size))

    if args.max_train_samples is not None:
        train_ds.paths = train_ds.paths[: args.max_train_samples]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    return train_loader, val_loader


@torch.no_grad()
def save_recon_grid(model, val_ds, device, out_path, n=8):
    model.eval()
    imgs = torch.stack([val_ds[i] for i in range(n)]).to(device)
    recon = model(imgs)["recon"]
    grid = torch.cat([imgs, recon], dim=0)  # top: original, bottom: recon
    save_image(grid, out_path, nrow=n, normalize=True, value_range=(-1, 1))


@torch.no_grad()
def evaluate(model, val_loader, device, amp_dtype):
    model.eval()
    total_l1, total_commit, total_perp, n_batches = 0.0, 0.0, 0.0, 0
    for x in val_loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(x)
            l1 = F.l1_loss(out["recon"], x)
        total_l1 += l1.item()
        total_commit += out["commitment_loss"].item()
        total_perp += out["perplexity"].item()
        n_batches += 1
    return total_l1 / n_batches, total_commit / n_batches, total_perp / n_batches


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else None

    out_dir = Path(args.output_dir)
    (out_dir / "reconstructions").mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir, args.resume)
    log.info(f"config: {vars(args)}")

    wandb_run = None
    if args.wandb:
        import wandb

        resume_id = None
        if args.resume and Path(args.resume).exists():
            try:
                peek = torch.load(args.resume, map_location="cpu", weights_only=False)
                resume_id = peek.get("wandb_run_id")
            except Exception:
                resume_id = None
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name or default_run_name(args),
            config=vars(args),
            id=resume_id,
            resume="allow" if resume_id else None,
        )
        wandb_run.save(args.config, policy="now")

    train_loader, val_loader = make_loaders(args)
    val_ds = val_loader.dataset

    model = VQVAE(
        embed_dim=args.embed_dim, num_codes=args.num_codes,
        commitment_cost=args.commitment_cost, decay=args.ema_decay,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=args.lr
    )

    start_epoch = 0
    global_step = 0
    best_val_l1 = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        best_val_l1 = ckpt["best_val_l1"]
        log.info(f"resumed from {args.resume} at epoch {start_epoch}, step {global_step}")

    n_images_seen = 0
    t_window_start = time.time()
    stop = False

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for x in train_loader:
            x = x.to(device, non_blocking=True)

            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(x)
                recon_loss = F.l1_loss(out["recon"], x)
                loss = recon_loss + out["commitment_loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            n_images_seen += x.size(0)

            if global_step % args.log_every == 0:
                elapsed = time.time() - t_window_start
                imgs_per_sec = n_images_seen / max(elapsed, 1e-6)
                log.info(
                    f"[train] epoch {epoch:03d} step {global_step:06d} | "
                    f"L1 {recon_loss.item():.4f} | commit {out['commitment_loss'].item():.4f} | "
                    f"ppl {out['perplexity'].item():.1f}/{args.num_codes} | "
                    f"{imgs_per_sec:.1f} img/s"
                )
                if wandb_run:
                    wandb_run.log(
                        {
                            "train/l1": recon_loss.item(),
                            "train/commitment": out["commitment_loss"].item(),
                            "train/loss": loss.item(),
                            "train/perplexity": out["perplexity"].item(),
                            "train/imgs_per_sec": imgs_per_sec,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "epoch": epoch,
                        },
                        step=global_step,
                    )
                n_images_seen = 0
                t_window_start = time.time()

            if args.max_steps is not None and global_step >= args.max_steps:
                stop = True
                break
        if stop:
            break

        if (epoch + 1) % args.eval_every == 0:
            val_l1, val_commit, val_ppl = evaluate(model, val_loader, device, amp_dtype)
            log.info(
                f"[val]   epoch {epoch:03d} | L1 {val_l1:.4f} | commit {val_commit:.4f} | "
                f"ppl {val_ppl:.1f}/{args.num_codes}"
            )
            recon_path = out_dir / "reconstructions" / f"epoch_{epoch:03d}.png"
            save_recon_grid(model, val_ds, device, recon_path)

            if wandb_run:
                import wandb

                wandb_run.log(
                    {
                        "val/l1": val_l1,
                        "val/commitment": val_commit,
                        "val/perplexity": val_ppl,
                        "val/reconstructions": wandb.Image(str(recon_path)),
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            if val_l1 < best_val_l1:
                best_val_l1 = val_l1
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                        "best_val_l1": best_val_l1,
                        "args": vars(args),
                        "wandb_run_id": wandb_run.id if wandb_run else None,
                    },
                    out_dir / "best.pt",
                )

        if (epoch + 1) % args.save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "best_val_l1": best_val_l1,
                    "args": vars(args),
                    "wandb_run_id": wandb_run.id if wandb_run else None,
                },
                out_dir / "last.pt",
            )

    log.info("training complete")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
