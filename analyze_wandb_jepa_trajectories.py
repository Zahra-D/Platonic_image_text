"""Summarize JEPA trajectory and representation diagnostics from W&B."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import wandb


ENTITY = "zahra-delbari-university-of-bern"
PROJECT = "Platonic_CLEVR_shared_lora_alignment"
RUN_IDS = ("ttztvt9n", "s6159f54", "pz54tjy5", "hv18e0fd")
OUTPUT = Path("outputs/jepa_wandb_trajectory_report")
BIN_SIZE = 5000


def finite(values):
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def mean(values):
    values = finite(values)
    return statistics.fmean(values) if values else None


def fmt(value):
    return "—" if value is None else f"{value:.4f}"


def correlation(left, right):
    pairs = [
        (float(a), float(b))
        for a, b in zip(left, right)
        if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in pairs)
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator > 0 else None


def main():
    api = wandb.Api(timeout=120)
    report = {}
    metric_keys = [
        "_step", "epoch", "jepa/text_loss", "jepa/image_loss",
        "train/loss_unweighted", "train/text_loss_unweighted", "train/image_loss_unweighted",
        "sigreg/text_mean_abs", "sigreg/image_mean_abs",
        "sigreg/text_variance_mean", "sigreg/image_variance_mean",
        "sigreg/text_variance_error", "sigreg/image_variance_error",
    ]
    for modality in ("text", "image"):
        for layer in range(8):
            metric_keys.extend([
                f"jepa/{modality}/layer_{layer:02d}_loss",
                f"jepa/{modality}/layer_{layer:02d}_cosine",
                f"objective_grad/{modality}/layer_{layer:02d}/jepa_to_diffusion_norm_ratio",
                f"objective_grad/{modality}/layer_{layer:02d}/diffusion_jepa_cosine",
            ])

    for run_id in RUN_IDS:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        # ``scan_history`` can temporarily omit unsynced rows for active runs.
        # All current runs log every 10 steps and remain below 10,000 metric
        # rows, so this sampled endpoint returns their complete trajectories.
        # Do not request an explicit union of keys here: older runs lack the
        # objective-gradient fields, and W&B can return no sampled rows when a
        # requested key never existed in that run.
        rows = run.history(samples=10000, pandas=True).to_dict("records")
        train_rows = [row for row in rows if row.get("jepa/text_loss") is not None]
        max_step = max((int(row["_step"]) for row in train_rows), default=0)
        bins = []
        for start in range(0, max_step + 1, BIN_SIZE):
            selected = [row for row in train_rows if start < int(row["_step"]) <= start + BIN_SIZE]
            if not selected:
                continue
            item = {"start": start + 1, "end": start + BIN_SIZE, "count": len(selected)}
            for modality in ("text", "image"):
                item[f"{modality}_jepa"] = mean(row.get(f"jepa/{modality}_loss") for row in selected)
                item[f"{modality}_variance"] = mean(row.get(f"sigreg/{modality}_variance_mean") for row in selected)
                item[f"{modality}_mean_abs"] = mean(row.get(f"sigreg/{modality}_mean_abs") for row in selected)
                for layer in range(8):
                    item[f"{modality}_l{layer}_loss"] = mean(
                        row.get(f"jepa/{modality}/layer_{layer:02d}_loss") for row in selected
                    )
                    item[f"{modality}_l{layer}_cos"] = mean(
                        row.get(f"jepa/{modality}/layer_{layer:02d}_cosine") for row in selected
                    )
            item["diffusion_unweighted"] = mean(row.get("train/loss_unweighted") for row in selected)
            bins.append(item)

        after_warmup = [row for row in train_rows if int(row["_step"]) >= 1000]
        objective_rows = [
            row for row in rows
            if any("jepa_to_diffusion_norm_ratio" in key and value is not None for key, value in row.items())
        ]
        objective_summary = {}
        for modality in ("text", "image"):
            for layer in range(8):
                ratio_key = f"objective_grad/{modality}/layer_{layer:02d}/jepa_to_diffusion_norm_ratio"
                cosine_key = f"objective_grad/{modality}/layer_{layer:02d}/diffusion_jepa_cosine"
                values = [(int(row["_step"]), row.get(ratio_key), row.get(cosine_key)) for row in objective_rows if row.get(ratio_key) is not None]
                if values:
                    objective_summary[f"{modality}_l{layer}"] = {
                        "mean_ratio": mean(value[1] for value in values),
                        "last_step": values[-1][0],
                        "last_ratio": float(values[-1][1]),
                        "mean_gradient_cosine": mean(value[2] for value in values),
                        "last_gradient_cosine": None if values[-1][2] is None else float(values[-1][2]),
                    }
        correlations = {}
        for modality in ("text", "image"):
            correlations[f"{modality}_loss_vs_variance"] = correlation(
                [row.get(f"jepa/{modality}_loss") for row in after_warmup],
                [row.get(f"sigreg/{modality}_variance_mean") for row in after_warmup],
            )
            correlations[f"{modality}_loss_vs_step"] = correlation(
                [row.get(f"jepa/{modality}_loss") for row in after_warmup],
                [row.get("_step") for row in after_warmup],
            )
        report[run_id] = {
            "name": run.name,
            "state": run.state,
            "max_step": max_step,
            "loss_type": run.config.get("shared_jepa_loss", "mse"),
            "text_weight": run.config.get("shared_jepa_text_weight", run.config.get("shared_jepa_weight")),
            "image_weight": run.config.get("shared_jepa_image_weight", run.config.get("shared_jepa_weight")),
            "bins": bins,
            "correlations": correlations,
            "objective_gradients": objective_summary,
        }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# W&B JEPA trajectory report", ""]
    for run_id, result in report.items():
        lines.extend([
            f"## {run_id}: {result['name']}", "",
            f"State: {result['state']}; last training step: {result['max_step']}; "
            f"loss: {result['loss_type']}; weights text/image: "
            f"{result['text_weight']}/{result['image_weight']}.", "",
            "| Steps | Text JEPA | Image JEPA | Text variance | Image variance | Diffusion CE |", 
            "|---:|---:|---:|---:|---:|---:|",
        ])
        for item in result["bins"]:
            lines.append(
                f"| {item['start']:,}–{item['end']:,} | {fmt(item['text_jepa'])} | "
                f"{fmt(item['image_jepa'])} | {fmt(item['text_variance'])} | "
                f"{fmt(item['image_variance'])} | {fmt(item['diffusion_unweighted'])} |"
            )
        lines.extend(["", "Layer losses in the final bin:", ""])
        final_bin = result["bins"][-1] if result["bins"] else {}
        lines.extend(["| Layer | Text loss | Text cosine | Image loss | Image cosine |", "|---:|---:|---:|---:|---:|"])
        for layer in range(8):
            if final_bin.get(f"text_l{layer}_loss") is not None or final_bin.get(f"image_l{layer}_loss") is not None:
                lines.append(
                    f"| {layer} | {fmt(final_bin.get(f'text_l{layer}_loss'))} | "
                    f"{fmt(final_bin.get(f'text_l{layer}_cos'))} | "
                    f"{fmt(final_bin.get(f'image_l{layer}_loss'))} | "
                    f"{fmt(final_bin.get(f'image_l{layer}_cos'))} |"
                )
        lines.extend(["", f"Correlations after step 1,000: `{result['correlations']}`", ""])
        if result["objective_gradients"]:
            lines.extend(["Latest/mean objective-gradient diagnostics:", ""])
            for key, value in result["objective_gradients"].items():
                lines.append(f"- {key}: `{value}`")
            lines.append("")
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(OUTPUT / "REPORT.md")


if __name__ == "__main__":
    main()
