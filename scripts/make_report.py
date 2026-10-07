import argparse
import base64
import io
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import to_pil_image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.clevr_dataset import ClevrImageFolder
from models.vqvae import VQVAE


def parse_args():
    p = argparse.ArgumentParser(
        description="Reconstruct CLEVR val images through one or more VQ-VAE "
        "checkpoints and render a side-by-side HTML comparison report."
    )
    p.add_argument("--checkpoints", nargs="+", required=True,
                    help="name=path pairs, e.g. bs1024=outputs/vqvae_training/last.pt")
    p.add_argument("--val-dir", default=str(Path.home() / "Omni/data/clevr/val"))
    p.add_argument("--image-size", type=int, nargs=2, default=[64, 96], metavar=("H", "W"))
    p.add_argument("--num-samples", type=int, default=10,
                    help="How many images to show side-by-side in the visual comparison.")
    p.add_argument("--metrics-samples", type=int, default=2000,
                    help="How many val images to score for the metrics table.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--output", default="report/vqvae_checkpoint_comparison.html")
    return p.parse_args()


def tensor_to_base64_png(chw):
    img01 = (chw.clamp(-1, 1) + 1) / 2
    pil = to_pil_image(img01.cpu())
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def psnr(recon, target):
    mse = F.mse_loss(recon, target, reduction="none").mean(dim=[1, 2, 3]).clamp_min(1e-10)
    return (20 * torch.log10(torch.tensor(2.0)) - 10 * torch.log10(mse)).mean().item()


@torch.no_grad()
def evaluate_checkpoint(name, path, val_ds, args, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    model = VQVAE(
        embed_dim=ckpt_args.get("embed_dim", 64), num_codes=ckpt_args.get("num_codes", 512)
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    num_codes = model.quantizer.num_codes

    sample_imgs = torch.stack([val_ds[i] for i in range(args.num_samples)]).to(device)
    sample_recon = model(sample_imgs)["recon"]

    n = min(args.metrics_samples, len(val_ds))
    loader = DataLoader(
        Subset(val_ds, range(n)), batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
    )
    total_l1, total_psnr, n_batches = 0.0, 0.0, 0
    code_counts = torch.zeros(num_codes, device=device)
    for x in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        recon = out["recon"]
        total_l1 += F.l1_loss(recon, x).item()
        total_psnr += psnr(recon, x)
        n_batches += 1
        code_counts += torch.bincount(out["indices"].flatten(), minlength=num_codes)

    probs = code_counts / code_counts.sum().clamp_min(1)
    perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum()).item()

    print(f"[{name}] epoch {ckpt.get('epoch')} step {ckpt.get('step')} | "
          f"L1 {total_l1 / n_batches:.4f} | PSNR {total_psnr / n_batches:.2f} | "
          f"ppl {perplexity:.1f}/{num_codes}")

    return {
        "name": name,
        "checkpoint": str(path),
        "epoch": ckpt.get("epoch"),
        "step": ckpt.get("step"),
        "best_val_l1": ckpt.get("best_val_l1"),
        "config": ckpt_args,
        "sample_recon": sample_recon,
        "metrics": {
            "l1": total_l1 / n_batches,
            "psnr": total_psnr / n_batches,
            "perplexity": perplexity,
            "num_codes": num_codes,
            "dead_codes": int((code_counts == 0).sum().item()),
            "n_scored": n,
        },
    }


def render_html(results, sample_originals, out_path):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_samples = sample_originals.size(0)

    metric_cols = "".join(f"<th>{r['name']}</th>" for r in results)
    metric_rows = ""
    for label, key, fmt in [
        ("Epoch reached", "epoch", "{}"),
        ("Step", "step", "{}"),
        ("Batch size", "batch_size_cfg", "{}"),
        ("Learning rate", "lr_cfg", "{}"),
        ("Val L1 (checkpoint)", "best_val_l1", "{:.4f}"),
        ("Eval L1 (this report)", "l1", "{:.4f}"),
        ("Eval PSNR (dB)", "psnr", "{:.2f}"),
        ("Codebook perplexity", "perplexity_str", "{}"),
        ("Dead codes", "dead_codes_str", "{}"),
        ("Images scored", "n_scored", "{}"),
    ]:
        cells = ""
        for r in results:
            if key == "batch_size_cfg":
                val = r["config"].get("batch_size")
            elif key == "lr_cfg":
                val = r["config"].get("lr")
            elif key == "perplexity_str":
                val = f"{r['metrics']['perplexity']:.1f} / {r['metrics']['num_codes']}"
            elif key == "dead_codes_str":
                val = f"{r['metrics']['dead_codes']} / {r['metrics']['num_codes']}"
            elif key in ("epoch", "step", "best_val_l1"):
                val = r.get(key)
            else:
                val = r["metrics"].get(key)
            cells += f"<td>{fmt.format(val) if not isinstance(val, str) else val}</td>"
        metric_rows += f"<tr><th>{label}</th>{cells}</tr>\n"

    sample_rows = ""
    for i in range(n_samples):
        orig_b64 = tensor_to_base64_png(sample_originals[i])
        cells = f'<td><div class="cell-label">original</div><img src="data:image/png;base64,{orig_b64}"></td>'
        for r in results:
            recon_b64 = tensor_to_base64_png(r["sample_recon"][i])
            cells += (
                f'<td><div class="cell-label">{r["name"]}</div>'
                f'<img src="data:image/png;base64,{recon_b64}"></td>'
            )
        sample_rows += f"<tr>{cells}</tr>\n"

    header_cols = "<th>original</th>" + "".join(f"<th>{r['name']}</th>" for r in results)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VQ-VAE Checkpoint Comparison</title>
<style>
  :root {{
    --bg: #0f1216;
    --panel: #171b21;
    --border: #262b33;
    --text: #e7eaee;
    --muted: #9aa4b2;
    --accent: #6ea8fe;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 40px 24px 80px;
  }}
  .wrap {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); margin-bottom: 32px; font-size: 0.9rem; }}
  h2 {{ font-size: 1.1rem; margin: 40px 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  .metrics-table th, .metrics-table td {{
    text-align: left; padding: 8px 14px; border-bottom: 1px solid var(--border); font-size: 0.88rem;
  }}
  .metrics-table th:first-child {{ color: var(--muted); font-weight: 500; }}
  .metrics-table thead th {{ color: var(--accent); font-weight: 600; }}
  .samples-table {{ width: 100%; }}
  .samples-table th {{
    text-align: center; padding: 6px 8px 12px; font-size: 0.8rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.04em;
  }}
  .samples-table td {{ padding: 6px 8px; vertical-align: top; text-align: center; }}
  .samples-table img {{
    width: 100%; max-width: 220px; border-radius: 6px; border: 1px solid var(--border);
    background: #000;
  }}
  .cell-label {{ display: none; }}
  .panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 20px; }}
  code {{ background: #1e232b; padding: 1px 6px; border-radius: 4px; font-size: 0.85em; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>VQ-VAE Checkpoint Comparison</h1>
  <div class="subtitle">Generated {generated_at} &middot; CLEVR val split &middot; {n_samples} sample images shown, {results[0]['metrics']['n_scored']} scored for metrics</div>

  <h2>Metrics</h2>
  <div class="panel">
    <table class="metrics-table">
      <thead><tr><th></th>{metric_cols}</tr></thead>
      <tbody>
      {metric_rows}
      </tbody>
    </table>
  </div>

  <h2>Reconstructions</h2>
  <div class="panel">
    <table class="samples-table">
      <thead><tr>{header_cols}</tr></thead>
      <tbody>
      {sample_rows}
      </tbody>
    </table>
  </div>
</div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"report written to {out_path}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_ds = ClevrImageFolder(args.val_dir, image_size=tuple(args.image_size))
    sample_originals = torch.stack([val_ds[i] for i in range(args.num_samples)])

    results = []
    for pair in args.checkpoints:
        name, path = pair.split("=", 1)
        results.append(evaluate_checkpoint(name, path, val_ds, args, device))

    render_html(results, sample_originals, Path(args.output))


if __name__ == "__main__":
    main()
