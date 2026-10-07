#!/usr/bin/env python3
"""Render separate text/image per-attribute tables from LoRA evaluation JSON."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/zd25e122/clevr_discrete_diffusion")
ATTRIBUTES = [
    "gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow",
    "cube", "sphere", "cylinder", "metal", "rubber", "small", "large",
    "left", "right", "front", "behind",
]
SHORT = {
    "gray": "Gry", "red": "Red", "blue": "Blu", "green": "Grn",
    "brown": "Brn", "purple": "Pur", "cyan": "Cyn", "yellow": "Yel",
    "cube": "Cub", "sphere": "Sph", "cylinder": "Cyl", "metal": "Met",
    "rubber": "Rub", "small": "Sml", "large": "Lrg", "left": "Lft",
    "right": "Rgt", "front": "Frt", "behind": "Bhd",
}
FRIENDLY = {
    "plain_lora_no_stage": "Plain LoRA",
    "lora_stage0": "LoRA stage 0",
    "lora_dann": "LoRA DANN",
    "lora_stage0_then_dann": "Stage 0 → DANN",
    "jepa_per_layer": "JEPA per layer",
    "jepa_normalized": "JEPA normalized",
    "jepa_strong_normalized": "JEPA strong normalized",
    "jepa_strong_raw_l2": "JEPA strong raw-L2",
}


def value(x):
    return "—" if x is None else f"{x:.3f}"


def best_row(model):
    layers = model["semantic_clean_train_masked_test"]["t0.8"]
    layer = max(layers, key=lambda k: layers[k]["shared"]["attribute_group_mean"])
    probe = layers[layer]["shared"]
    return layer, probe


def table(report, modality):
    columns = ["Model", "L", "Count", "Attr avg"] + [SHORT[x] for x in ATTRIBUTES]
    lines = [
        f"## {modality.capitalize()}-only evaluation",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for name, model in report["models"].items():
        layer, probe = best_row(model)
        attrs = probe["attribute_balanced_accuracy"]
        row = [
            FRIENDLY.get(name, name), str(layer),
            value(probe["object_count"]["balanced_accuracy"]),
            value(probe["attribute_group_mean"]),
        ] + [value(attrs.get(attribute)) for attribute in ATTRIBUTES]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def summary(report, modality):
    rows = [(name, *best_row(model)) for name, model in report["models"].items()]
    best_count = max(rows, key=lambda row: row[2]["object_count"]["balanced_accuracy"])
    best_attr = max(rows, key=lambda row: row[2]["attribute_group_mean"])
    return (
        f"For {modality}, best count is **{FRIENDLY.get(best_count[0], best_count[0])}** "
        f"at layer {best_count[1]} ({best_count[2]['object_count']['balanced_accuracy']:.3f}); "
        f"best attribute average is **{FRIENDLY.get(best_attr[0], best_attr[0])}** "
        f"at layer {best_attr[1]} ({best_attr[2]['attribute_group_mean']:.3f})."
    )


def main():
    reports = {
        modality: json.loads((ROOT / f"outputs/jepa_modality_eval_lora_family_{modality}/results.json").read_text())
        for modality in ("text", "image")
    }
    lines = [
        "# Multimodal LoRA: separate-modality attribute report",
        "",
        "Each checkpoint was pretrained with both modalities in the same unpaired run. "
        "Evaluation forwards text and image separately, using the final student model.",
        "",
        "All scores use a balanced logistic probe fit on clean features and evaluated on "
        "the same modality at 80% random token masking. Each row selects one layer—the "
        "highest shared attribute-average among layers 2, 3, and 4—so all entries in a row "
        "come from the same layer.",
        "",
        "`Count` is object-count balanced accuracy. `Attr avg` equally averages color, shape, "
        "material, size, and relation group balanced accuracies. `—` means that the label was "
        "degenerate in this evaluation split. Abbreviations: Gry=gray, Blu=blue, Grn=green, "
        "Brn=brown, Pur=purple, Cyn=cyan, Yel=yellow, Cub=cube, Sph=sphere, Cyl=cylinder, "
        "Met=metal, Rub=rubber, Sml=small, Lrg=large, Lft=left, Rgt=right, Frt=front, Bhd=behind.",
        "",
    ]
    for modality, report in reports.items():
        lines += table(report, modality) + ["", summary(report, modality), ""]
    lines += [
        "## Important interpretation",
        "",
        "These are within-modality robustness scores, not cross-modal alignment scores. "
        "DANN can therefore be neutral or harmful here while still changing modality invariance. "
        "Full per-layer, per-mask-ratio, shared/private exact-token, and direct-JEPA metrics are in the two JSON files.",
    ]
    target = ROOT / "outputs/LORA_MULTIMODAL_SEPARATE_MODALITY_ATTRIBUTE_REPORT.md"
    target.write_text("\n".join(lines) + "\n")
    print(target)


if __name__ == "__main__":
    main()
