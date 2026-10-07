import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.clevr_dataset import ClevrImageFolder
from data.token_dataset import TokenGridDataset
from models.vqvae import VQVAE
from models.transformer import MaskedTokenTransformer
from diffusion import sample_t, forward_process, masked_diffusion_loss, generate

DEFAULT_CONFIG = str(Path(__file__).resolve().parent / "configs" / "d3pm.yaml")
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
    tok_cfg = cfg.get("tokenizer", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    diff_cfg = cfg.get("diffusion", {})
    train_cfg = cfg.get("train", {})
    gen_cfg = cfg.get("generate", {})
    wandb_cfg = cfg.get("wandb", {})

    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument("--tokenizer-checkpoint",
                    default=tok_cfg.get("checkpoint", "outputs/vqvae_training/best.pt"))
    p.add_argument("--num-codes", type=int, default=tok_cfg.get("num_codes", 512))
    p.add_argument("--tok-embed-dim", type=int, default=tok_cfg.get("embed_dim", 64))
    p.add_argument("--train-dir", default=data_cfg.get("train_dir", DEFAULT_TRAIN_DIR))
    p.add_argument("--val-dir", default=data_cfg.get("val_dir", DEFAULT_VAL_DIR))
    p.add_argument("--image-size", type=int, nargs=2,
                    default=data_cfg.get("image_size", [64, 96]), metavar=("H", "W"))
    p.add_argument("--grid-size", type=int, nargs=2,
                    default=model_cfg.get("grid_size", [16, 24]), metavar=("H", "W"))
    p.add_argument("--token-cache-dir", default=data_cfg.get("token_cache_dir"),
                    help="Directory containing train_tokens.pt and val_tokens.pt. "
                         "Create it with pretokenize_clevr.py.")
    p.add_argument("--no-token-cache", dest="token_cache_dir", action="store_const", const=None,
                    help="Use image files and run the frozen VQ-VAE on every batch (benchmark only).")
    p.add_argument("--d-model", type=int, default=model_cfg.get("d_model", 384))
    p.add_argument("--n-layers", type=int, default=model_cfg.get("n_layers", 8))
    p.add_argument("--n-heads", type=int, default=model_cfg.get("n_heads", 6))
    p.add_argument("--mlp-ratio", type=int, default=model_cfg.get("mlp_ratio", 4))
    p.add_argument("--dropout", type=float, default=model_cfg.get("dropout", 0.1))
    p.add_argument("--pos-embed-type", choices=["absolute", "rope"],
                    default=model_cfg.get("pos_embed_type", "absolute"))
    p.add_argument("--eps", type=float, default=diff_cfg.get("eps", 1e-3))
    p.add_argument("--weight-by-t", action=argparse.BooleanOptionalAction,
                    default=diff_cfg.get("weight_by_t", True),
                    help="1/t-weighted loss (default) vs plain unweighted CE.")
    p.add_argument("--batch-size", type=int, default=train_cfg.get("batch_size", 256))
    p.add_argument("--epochs", type=int, default=train_cfg.get("epochs", 100))
    p.add_argument("--lr", type=float, default=train_cfg.get("lr", 3e-4))
    p.add_argument("--num-workers", type=int, default=train_cfg.get("num_workers", 8))
    p.add_argument("--amp", choices=["none", "bf16"], default=train_cfg.get("amp", "bf16"))
    p.add_argument("--resume", default=train_cfg.get("resume"))
    p.add_argument("--output-dir", default=train_cfg.get("output_dir", "outputs/d3pm_training"))
    p.add_argument("--log-every", type=int, default=train_cfg.get("log_every", 20))
    p.add_argument("--eval-every", type=int, default=train_cfg.get("eval_every", 1))
    p.add_argument("--val-max-samples", type=int, default=train_cfg.get("val_max_samples", 2048),
                    help="Number of validation examples used per fixed-noise evaluation.")
    p.add_argument("--val-timesteps", type=float, nargs="+",
                    default=train_cfg.get("val_timesteps", [0.5, 0.75, 0.9, 1.0]),
                    help="Fixed masking probabilities used for validation.")
    p.add_argument("--save-every", type=int, default=train_cfg.get("save_every", 1))
    p.add_argument("--max-train-samples", type=int,
                    default=train_cfg.get("max_train_samples"),
                    help="Truncate the training set (for smoke tests).")
    p.add_argument("--max-steps", type=int, default=train_cfg.get("max_steps"),
                    help="Stop after this many optimizer steps (for smoke tests).")
    p.add_argument("--gen-num-samples", type=int, default=gen_cfg.get("num_samples", 8))
    p.add_argument("--gen-num-steps", type=int, default=gen_cfg.get("num_steps", 50))
    p.add_argument("--gen-temperature", type=float, default=gen_cfg.get("temperature", 1.0))
    p.add_argument("--gen-reveal-order", choices=["confidence", "random"],
                    default=gen_cfg.get("reveal_order", "confidence"))
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction,
                    default=wandb_cfg.get("enabled", True),
                    help="Log to Weights & Biases.")
    p.add_argument("--wandb-project",
                    default=wandb_cfg.get("project", "clevr-discrete-diffusion"))
    p.add_argument("--wandb-entity", default=wandb_cfg.get("entity"))
    p.add_argument("--wandb-group", default=wandb_cfg.get("group", "d3pm_training"))
    p.add_argument("--wandb-run-name", default=wandb_cfg.get("run_name"),
                    help="Defaults to an auto-generated name from key hyperparams + timestamp.")
    return p.parse_args()


def default_run_name(args):
    weight_tag = "w1t" if args.weight_by_t else "noweight"
    return (
        f"d{args.d_model}-L{args.n_layers}-{args.pos_embed_type}-{weight_tag}-"
        f"bs{args.batch_size}-{args.amp}-{time.strftime('%m%d-%H%M')}"
    )


def setup_logging(out_dir, resume):
    logger = logging.getLogger("train_d3pm")
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


def load_frozen_vqvae(args, device):
    ckpt = torch.load(args.tokenizer_checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    vqvae = VQVAE(
        embed_dim=ckpt_args.get("embed_dim", args.tok_embed_dim),
        num_codes=ckpt_args.get("num_codes", args.num_codes),
    ).to(device)
    vqvae.load_state_dict(ckpt["model"])
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    return vqvae


def make_loaders(args):
    cached_tokens = args.token_cache_dir is not None
    if cached_tokens:
        cache_dir = Path(args.token_cache_dir).expanduser()
        train_ds = TokenGridDataset(cache_dir / "train_tokens.pt")
        val_ds = TokenGridDataset(cache_dir / "val_tokens.pt")
        if train_ds.grid_size != tuple(args.grid_size) or val_ds.grid_size != tuple(args.grid_size):
            raise ValueError(
                f"Cache grids train={train_ds.grid_size}, val={val_ds.grid_size} do not match "
                f"--grid-size {tuple(args.grid_size)}"
            )
    else:
        train_ds = ClevrImageFolder(args.train_dir, image_size=tuple(args.image_size))
        val_ds = ClevrImageFolder(args.val_dir, image_size=tuple(args.image_size))

    if args.max_train_samples is not None:
        train_ds = Subset(train_ds, range(min(args.max_train_samples, len(train_ds))))
    if args.val_max_samples is not None:
        val_ds = Subset(val_ds, range(min(args.val_max_samples, len(val_ds))))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0 if cached_tokens else args.num_workers,
        drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0 if cached_tokens else args.num_workers, pin_memory=True,
    )
    return train_loader, val_loader, cached_tokens


@torch.no_grad()
def tokenize(vqvae, images):
    """[B,3,H,W] -> [B, seq_len] flattened token ids."""
    indices = vqvae.encode_to_indices(images)  # [B, gh, gw]
    return indices.view(indices.size(0), -1)


@torch.no_grad()
def batch_to_tokens(batch, vqvae, cached_tokens, device):
    if cached_tokens:
        return batch.to(device, dtype=torch.long, non_blocking=True).flatten(1)
    images = batch.to(device, non_blocking=True)
    return tokenize(vqvae, images)


@torch.no_grad()
def evaluate_fixed_noise(model, vqvae, val_loader, device, amp_dtype, cached_tokens, timesteps):
    """Measure denoising in the high-noise regimes used by generation.

    Random-t, 1/t-weighted validation can improve while all-MASK generation
    gets worse.  Fixed t values make the model selection signal stable and
    expose the all-MASK starting condition directly.
    """
    model.eval()
    results = {}
    for t_value in timesteps:
        total_ce, total_correct, total_masked = 0.0, 0, 0
        for batch in val_loader:
            x0 = batch_to_tokens(batch, vqvae, cached_tokens, device)
            if t_value >= 1.0:
                masked = torch.ones_like(x0, dtype=torch.bool)
            else:
                masked = torch.rand_like(x0, dtype=torch.float32) < t_value
            xt = torch.where(masked, torch.full_like(x0, model.mask_id), x0)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(xt)
            masked_logits = logits[masked].float()
            masked_targets = x0[masked]
            total_ce += torch.nn.functional.cross_entropy(
                masked_logits, masked_targets, reduction="sum"
            ).item()
            total_correct += (masked_logits.argmax(dim=-1) == masked_targets).sum().item()
            total_masked += masked_targets.numel()
        suffix = f"{t_value:.2f}"
        results[f"val/ce_t{suffix}"] = total_ce / total_masked
        results[f"val/acc_t{suffix}"] = total_correct / total_masked
    results["val/high_noise_ce"] = sum(
        results[f"val/ce_t{t_value:.2f}"] for t_value in timesteps
    ) / len(timesteps)
    return results


@torch.no_grad()
def save_sample_grid(model, vqvae, args, device, out_path):
    model.eval()
    tokens = generate(
        model, seq_len=model.seq_len, mask_id=model.mask_id,
        num_steps=args.gen_num_steps, batch_size=args.gen_num_samples,
        device=device, temperature=args.gen_temperature, reveal_order=args.gen_reveal_order,
    )
    grid_h, grid_w = args.grid_size
    tokens = tokens.view(-1, grid_h, grid_w)
    images = vqvae.decode_from_indices(tokens)
    save_image(images, out_path, nrow=args.gen_num_samples, normalize=True, value_range=(-1, 1))


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else None

    out_dir = Path(args.output_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
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

    vqvae = load_frozen_vqvae(args, device)
    log.info(f"loaded frozen tokenizer from {args.tokenizer_checkpoint}")

    train_loader, val_loader, cached_tokens = make_loaders(args)
    log.info("using cached VQ tokens" if cached_tokens else "tokenizing images on the fly")

    model = MaskedTokenTransformer(
        vocab_size=args.num_codes, grid_size=tuple(args.grid_size),
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio, dropout=args.dropout,
        pos_embed_type=args.pos_embed_type,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"transformer params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        best_val_loss = ckpt["best_val_loss"]
        log.info(f"resumed from {args.resume} at epoch {start_epoch}, step {global_step}")

    n_images_seen = 0
    t_window_start = time.time()
    stop = False

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for images in train_loader:
            x0 = batch_to_tokens(images, vqvae, cached_tokens, device)
            t = sample_t(x0.size(0), device, args.eps)
            xt, masked = forward_process(x0, t, model.mask_id)

            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(xt)
                loss, n_masked = masked_diffusion_loss(logits, x0, masked, t, args.weight_by_t)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            n_images_seen += x0.size(0)

            if global_step % args.log_every == 0:
                elapsed = time.time() - t_window_start
                imgs_per_sec = n_images_seen / max(elapsed, 1e-6)
                log.info(
                    f"[train] epoch {epoch:03d} step {global_step:06d} | "
                    f"loss {loss.item():.4f} | masked {masked.float().mean().item():.3f} | "
                    f"{imgs_per_sec:.1f} img/s"
                )
                if wandb_run:
                    wandb_run.log(
                        {
                            "train/loss": loss.item(),
                            "train/masked_frac": masked.float().mean().item(),
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
            val_metrics = evaluate_fixed_noise(
                model, vqvae, val_loader, device, amp_dtype, cached_tokens, args.val_timesteps
            )
            val_loss = val_metrics["val/high_noise_ce"]
            metric_text = " | ".join(
                f"t={t:.2f}: {val_metrics[f'val/ce_t{t:.2f}']:.3f}"
                for t in args.val_timesteps
            )
            log.info(f"[val]   epoch {epoch:03d} | high-noise CE {val_loss:.4f} | {metric_text}")

            sample_path = out_dir / "samples" / f"epoch_{epoch:03d}.png"
            save_sample_grid(model, vqvae, args, device, sample_path)

            if wandb_run:
                import wandb

                wandb_run.log(
                    {
                        **val_metrics,
                        "val/samples": wandb.Image(str(sample_path)),
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                        "best_val_loss": best_val_loss,
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
                    "best_val_loss": best_val_loss,
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
