import argparse
import base64
import io
import sys
from datetime import datetime
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.vqvae import VQVAE
from models.transformer import MaskedTokenTransformer
from diffusion import generate


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate CLEVR samples from one or more trained Transformer "
        "checkpoints (pure noise -> tokens -> frozen VQ-VAE decode) and render an "
        "HTML report. Rerunnable as a living template: same output path each time, "
        "just rerun after more training to refresh the results."
    )
    p.add_argument("--checkpoints", nargs="+", required=True,
                    help="name=path pairs, optionally name=path:reveal_order to override "
                    "--reveal-order for just that entry, e.g. "
                    "absolute=outputs/d3pm_absolute/best.pt "
                    "absolute-random=outputs/d3pm_absolute/best.pt:random")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=None,
                    help="Unmasking steps. Default: one step per token (seq_len) -- "
                    "the finest-grained schedule, i.e. the practical max.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--reveal-order", choices=["confidence", "random"], default="confidence",
                    help="Default reveal order for entries that don't specify their own.")
    p.add_argument("--output", default="report/d3pm_generation_comparison.html")
    return p.parse_args()


def tensor_to_base64_png(chw):
    img01 = (chw.clamp(-1, 1) + 1) / 2
    pil = to_pil_image(img01.cpu())
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@torch.no_grad()
def run_checkpoint(name, path, reveal_order, args, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

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

    num_steps = args.num_steps or model.seq_len
    tokens = generate(
        model, seq_len=model.seq_len, mask_id=model.mask_id,
        num_steps=num_steps, batch_size=args.num_samples,
        device=device, temperature=args.temperature, reveal_order=reveal_order,
    )
    tokens = tokens.view(-1, model.grid_h, model.grid_w)
    images = vqvae.decode_from_indices(tokens)

    print(f"[{name}] epoch {ckpt.get('epoch')} step {ckpt.get('step')} | "
          f"best_val_loss {ckpt.get('best_val_loss'):.4f} | "
          f"pos_embed_type={ckpt_args.get('pos_embed_type')} | num_steps={num_steps} | "
          f"reveal_order={reveal_order}")

    return {
        "name": name,
        "checkpoint": str(path),
        "epoch": ckpt.get("epoch"),
        "step": ckpt.get("step"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "config": ckpt_args,
        "num_steps": num_steps,
        "reveal_order": reveal_order,
        "images": images,
    }


def loss_type_label(config):
    """The 'best_val_loss' field means different, non-comparable things
    depending on which version of train_d3pm.py produced the checkpoint --
    make that explicit instead of implying one shared formula."""
    if config.get("val_timesteps"):
        ts = ",".join(f"{t:g}" for t in config["val_timesteps"])
        return f"unweighted CE, avg over fixed t={{{ts}}}"
    weighted = config.get("weight_by_t", True)
    return f"{'1/t-weighted' if weighted else 'unweighted'} CE, random t"


def render_html(results, num_samples, out_path):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    metric_cols = "".join(f"<th>{r['name']}</th>" for r in results)
    metric_rows = ""
    for label, key, fmt in [
        ("Positional embedding", "pos_embed_type_cfg", "{}"),
        ("Epoch reached", "epoch", "{}"),
        ("Step", "step", "{}"),
        ("Best val loss", "best_val_loss", "{:.4f}"),
        ("Val loss definition (NOT comparable across rows)", "loss_type", "{}"),
        ("Unmasking steps used", "num_steps", "{}"),
        ("Reveal order", "reveal_order", "{}"),
        ("Tokenizer checkpoint", "tokenizer_checkpoint_cfg", "{}"),
    ]:
        cells = ""
        for r in results:
            if key == "pos_embed_type_cfg":
                val = r["config"].get("pos_embed_type")
            elif key == "tokenizer_checkpoint_cfg":
                val = r["config"].get("tokenizer_checkpoint")
            elif key == "num_steps":
                val = r["num_steps"]
            elif key == "loss_type":
                val = loss_type_label(r["config"])
            else:
                val = r.get(key)
            cells += f"<td>{fmt.format(val) if not isinstance(val, str) else val}</td>"
        metric_rows += f"<tr><th>{label}</th>{cells}</tr>\n"

    section_html = ""
    for r in results:
        imgs_html = "".join(
            f'<div class="sample"><img src="data:image/png;base64,{tensor_to_base64_png(r["images"][i])}"></div>'
            for i in range(r["images"].size(0))
        )
        section_html += f"""
        <h3>{r['name']} <span class="tag">{r['config'].get('pos_embed_type')}</span></h3>
        <div class="sample-grid">{imgs_html}</div>
        """

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>D3PM Generation Report</title>
<style>
  :root {{
    --bg: #0f1216; --panel: #171b21; --border: #262b33;
    --text: #e7eaee; --muted: #9aa4b2; --accent: #6ea8fe;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    margin: 0; padding: 40px 24px 80px;
  }}
  .wrap {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); margin-bottom: 8px; font-size: 0.9rem; }}
  .notice {{
    background: #2a2312; border: 1px solid #55450f; color: #e8cf7a;
    border-radius: 8px; padding: 10px 14px; font-size: 0.85rem; margin-bottom: 32px;
  }}
  h2 {{ font-size: 1.1rem; margin: 40px 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
  h3 {{ font-size: 0.95rem; margin: 24px 0 10px; color: var(--accent); }}
  .tag {{
    font-size: 0.7rem; color: var(--muted); border: 1px solid var(--border);
    border-radius: 4px; padding: 2px 6px; margin-left: 6px; font-weight: 400;
  }}
  table {{ border-collapse: collapse; width: 100%; }}
  .metrics-table th, .metrics-table td {{
    text-align: left; padding: 8px 14px; border-bottom: 1px solid var(--border); font-size: 0.85rem;
    word-break: break-all;
  }}
  .metrics-table th:first-child {{ color: var(--muted); font-weight: 500; white-space: nowrap; }}
  .metrics-table thead th {{ color: var(--accent); font-weight: 600; }}
  .panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 20px; margin-bottom: 20px; }}
  .sample-grid {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .sample img {{
    width: 160px; border-radius: 6px; border: 1px solid var(--border); background: #000; display: block;
  }}
</style>
</head>
<body>
<div class="wrap">
  <h1>D3PM Generation Report</h1>
  <div class="subtitle">Generated {generated_at} &middot; unconditional samples (pure noise &rarr; tokens &rarr; decode), {num_samples} per checkpoint</div>
  <div class="notice">Snapshot during training, not a final result &mdash; rerun <code>scripts/make_generation_report.py</code> against the same checkpoint paths any time to refresh this report as training progresses.</div>

  <h2>Checkpoint status</h2>
  <div class="panel">
    <table class="metrics-table">
      <thead><tr><th></th>{metric_cols}</tr></thead>
      <tbody>
      {metric_rows}
      </tbody>
    </table>
  </div>

  <h2>Generated samples</h2>
  <div class="panel">
    {section_html}
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

    results = []
    for pair in args.checkpoints:
        name, path = pair.split("=", 1)
        reveal_order = args.reveal_order
        if ":" in path:
            path, reveal_order = path.rsplit(":", 1)
        results.append(run_checkpoint(name, path, reveal_order, args, device))

    render_html(results, args.num_samples, Path(args.output))


if __name__ == "__main__":
    main()
