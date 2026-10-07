"""Plot bidirectional retrieval recall across Transformer block outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text())
    labels = [f"L{i}" for i in range(8)] + ["Final"]
    levels = [f"block_{i:02d}_final" for i in range(8)] + ["final"]
    models = [
        ("dense", "Dense", "#4C78A8"),
        ("lora", "Tri-LoRA", "#F58518"),
        ("lora_adversarial", "Tri-LoRA + adversarial", "#54A24B"),
    ]
    directions = [("text_to_image", "Text → image"), ("image_to_text", "Image → text")]
    recalls = [("r@1", "Recall@1", 1), ("r@5", "Recall@5", 5), ("r@10", "Recall@10", 10)]
    count = payload["num_samples"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for row, (direction, direction_title) in enumerate(directions):
        for column, (recall, recall_title, k) in enumerate(recalls):
            axis = axes[row, column]
            for model_key, model_label, color in models:
                values = [payload["models"][model_key][level][direction][recall] for level in levels]
                axis.plot(labels, values, marker="o", linewidth=2, markersize=5, label=model_label, color=color)
            chance = min(1.0, k / count)
            axis.axhline(chance, color="#777777", linestyle="--", linewidth=1.2, label="Chance" if row == 0 and column == 0 else None)
            axis.set_title(f"{direction_title} — {recall_title}")
            axis.set_ylabel("Recall")
            axis.grid(alpha=0.25)
            axis.set_ylim(bottom=0)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.935),
        ncol=4, frameon=False,
    )
    fig.suptitle(
        f"Cross-modal retrieval across layers ({count} paired CLEVR validation examples)\n"
        "Representations are centered and L2-normalized; dashed line is random-retrieval chance",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
