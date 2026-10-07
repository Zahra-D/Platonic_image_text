"""Export W&B JEPA trajectories and combine them with gradient diagnostics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="entity/project/run_id")
    parser.add_argument("--gradient-json", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def window_summary(frame, lower, upper):
    window = frame[(frame["_step"] >= lower) & (frame["_step"] <= upper)]
    result = {
        "start_step": int(window["_step"].min()),
        "end_step": int(window["_step"].max()),
        "rows": len(window),
        "diffusion": float(window["train/loss"].mean()),
        "optimized": float(window["train/optimized_loss"].mean()),
    }
    for modality in ("text", "image"):
        result[f"{modality}_jepa"] = float(window[f"jepa/{modality}_loss"].mean())
        for layer in range(8):
            for quantity in ("loss", "cosine"):
                key = f"jepa/{modality}/layer_{layer:02d}_{quantity}"
                result[f"{modality}_layer_{layer:02d}_{quantity}"] = float(window[key].mean())
    return result


def plot_layer_trajectories(frame, output, quantity, ylabel):
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for axis, modality in zip(axes, ("text", "image")):
        for layer in range(8):
            key = f"jepa/{modality}/layer_{layer:02d}_{quantity}"
            smoothed = frame[key].rolling(50, min_periods=1).mean()
            axis.plot(frame["_step"], smoothed, label=f"L{layer}", linewidth=1.6)
        axis.set_title(modality.capitalize())
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_gradient_ratios(gradient_report, output):
    layers = list(range(8))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, modality in zip(axes, ("text", "image")):
        metrics = gradient_report["metrics"][modality]
        jepa = [metrics[f"layer_{layer:02d}/jepa_to_diffusion_norm_ratio"] for layer in layers]
        sigreg = [metrics[f"layer_{layer:02d}/sigreg_to_diffusion_norm_ratio"] for layer in layers]
        axis.plot(layers, jepa, marker="o", label="weighted JEPA / diffusion")
        axis.plot(layers, sigreg, marker="o", label="weighted SIGReg / diffusion")
        axis.axvspan(1.5, 4.5, color="grey", alpha=0.12, label="proposed L2–L4")
        axis.set_title(modality.capitalize())
        axis.set_xlabel("Transformer layer")
        axis.set_xticks(layers)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Shared-gradient norm ratio")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def format_values(values):
    return " | ".join(f"{value:.3f}" for value in values)


def main():
    args = arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.Api(timeout=90).run(args.run)
    keys = [
        "_step", "epoch", "train/loss", "train/optimized_loss",
        "jepa/text_loss", "jepa/image_loss", "sigreg/text_loss", "sigreg/image_loss",
    ] + [
        f"jepa/{modality}/layer_{layer:02d}_{quantity}"
        for modality in ("text", "image")
        for layer in range(8)
        for quantity in ("loss", "cosine")
    ]
    rows = list(run.scan_history(keys=keys, page_size=1000))
    frame = pd.DataFrame(rows).sort_values("_step")
    frame.to_csv(output_dir / "wandb_jepa_history.csv", index=False)
    final_step = int(frame["_step"].max())
    summaries = {
        "early_full_weight": window_summary(frame, 1000, 5000),
        "middle": window_summary(frame, 10000, 15000),
        "recent": window_summary(frame, max(0, final_step - 5000), final_step),
    }
    gradient_report = json.loads(Path(args.gradient_json).read_text())
    plot_layer_trajectories(
        frame, output_dir / "jepa_layer_loss_trajectory.png", "loss", "JEPA MSE (50-log rolling mean)"
    )
    plot_layer_trajectories(
        frame, output_dir / "jepa_layer_cosine_trajectory.png", "cosine", "Prediction/target cosine (50-log rolling mean)"
    )
    plot_gradient_ratios(gradient_report, output_dir / "objective_gradient_ratios.png")

    lines = [
        "# Shared-JEPA pretraining report",
        "",
        f"W&B run: [{run.id}]({run.url}); state at export: **{run.state}**; latest exported step: **{final_step}**.",
        "",
        "This report is provisional while the 70-epoch run is active.",
        "",
        "## Aggregate trajectory",
        "",
        "| Window | Steps | Diffusion | Optimized | Text JEPA | Image JEPA |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary['start_step']}–{summary['end_step']} | "
            f"{summary['diffusion']:.3f} | {summary['optimized']:.3f} | "
            f"{summary['text_jepa']:.3f} | {summary['image_jepa']:.3f} |"
        )
    lines.extend([
        "",
        "The rising MSE does not by itself mean prediction deteriorated. SIGReg increases the "
        "scale/variance of initially near-zero shared latents, so an unnormalized MSE naturally "
        "grows as the target representation grows. Cosine is the scale-independent control.",
        "",
        "## Recent per-layer JEPA behavior",
        "",
        "| Modality / metric | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    recent = summaries["recent"]
    for modality in ("text", "image"):
        losses = [recent[f"{modality}_layer_{layer:02d}_loss"] for layer in range(8)]
        cosines = [recent[f"{modality}_layer_{layer:02d}_cosine"] for layer in range(8)]
        lines.append(f"| {modality} loss | {format_values(losses)} |")
        lines.append(f"| {modality} cosine | {format_values(cosines)} |")

    lines.extend([
        "",
        "Text is the weak side: recent loss grows strongly in L3–L7 and cosine falls to roughly "
        "0.65–0.73 in the deeper layers. Image cosine is healthier (roughly 0.79–0.87 in L2–L7), "
        "although its unnormalized MSE also grows.",
        "",
        "## Objective influence on shared A/B gradients",
        "",
        f"Checkpoint: epoch {gradient_report['checkpoint_epoch']}, step {gradient_report['checkpoint_step']}; "
        f"{gradient_report['num_samples']} examples in batches of {gradient_report['batch_size']}, "
        f"mask ratio {gradient_report['mask_ratio']}.",
        "",
        "| Modality / ratio | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for modality in ("text", "image"):
        metrics = gradient_report["metrics"][modality]
        jepa = [metrics[f"layer_{layer:02d}/jepa_to_diffusion_norm_ratio"] for layer in range(8)]
        sigreg = [metrics[f"layer_{layer:02d}/sigreg_to_diffusion_norm_ratio"] for layer in range(8)]
        lines.append(f"| {modality} JEPA/diffusion | {format_values(jepa)} |")
        lines.append(f"| {modality} SIGReg/diffusion | {format_values(sigreg)} |")

    lines.extend([
        "",
        "For the proposed L2–L4 trunk, JEPA/diffusion gradient ratios are only 2.9%, 4.2%, "
        "7.5% for text and 0.8%, 1.0%, 1.8% for image. JEPA is therefore a weak optimizer signal, "
        "especially for images. Weighted SIGReg is substantially stronger.",
        "",
        "Diffusion–JEPA gradient cosines are close to zero, so JEPA generally supplies an "
        "independent direction rather than reinforcing or opposing token prediction. JEPA–SIGReg "
        "cosines are also mostly near zero, except a moderate image conflict at L3.",
        "",
        "SIGReg is batch-size dependent because its statistic multiplies by sample count. This "
        "offline measurement uses batch 64 versus training's approximate per-modality batch 128; "
        "the reported SIGReg influence is conservative. JEPA and diffusion are mean losses and "
        "their ratio is much less sensitive to this difference.",
        "",
        "## Conclusion",
        "",
        "The present JEPA objective is not convincingly shaping L2–L4, particularly for images. "
        "Its increasing raw MSE is partly a scale artifact, but the direct gradient measurement "
        "shows that its weighted signal is genuinely small. The next ablation should restrict "
        "JEPA/SIGReg to L2–L4 and replace or supplement raw MSE with a scale-normalized objective "
        "(normalized MSE or cosine loss). Coefficients should then be tuned using gradient ratios, "
        "not raw loss magnitudes.",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    (output_dir / "window_summaries.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"wrote report to {output_dir}")


if __name__ == "__main__":
    main()
