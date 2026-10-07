#!/usr/bin/env python3
"""Build a linked Markdown catalog of pretraining models and evaluations."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

ROOT = Path("/home/zd25e122/clevr_discrete_diffusion")
DOCS = ROOT / "docs/experiment_catalog"
MODELS = DOCS / "models"
EVALDOCS = DOCS / "evaluations"
PROJECT = "https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_shared_lora_alignment"
SINGLE = "https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_single_modality_ema_jepa"

# Checkpoint catalog slugs versus labels used by the evaluation launchers.
EVALUATION_MODEL_ALIASES = {
    "text_diffusion": "text_diffusion_only",
    "image_diffusion": "image_diffusion_only",
    "text_ema_r010": "text_ema_dynamic_r010",
    "text_ema_r025": "text_ema_dynamic_r025",
    "text_ema_r050": "text_ema_dynamic_r050",
    "image_ema_r010": "image_ema_dynamic_r010",
    "image_ema_r025": "image_ema_dynamic_r025",
    "image_ema_r050": "image_ema_dynamic_r050",
}

# Result labels in the completed causal image-swap report.
CAUSAL_SWAP_MODEL_ALIASES = {
    "image_diffusion": "image_diffusion_only",
    "image_ema_fixed": "image_ema_fixed",
    "image_ema_r010": "image_ema_dynamic_r010",
    "image_ema_r025": "image_ema_dynamic_r025",
    "image_ema_r050": "image_ema_dynamic_r050",
}


def path_link(path: str, label: str | None = None) -> str:
    absolute = ROOT / path
    return f"[{label or absolute.name}]({absolute})" if absolute.exists() else f"`{absolute}` (not present)"


def run_url(project: str, run_id: str | None) -> str:
    return f"[{run_id}]({project}/runs/{run_id})" if run_id else f"[{project.rsplit('/', 1)[-1]}]({project})"


records = [
    # Multimodal runs: text and image were both optimized in the same run.
    ("dense_paired", "Dense paired", "Multimodal", "Dense, true paired data", "configs/pretraining_paired_dense_absolute_70e.yaml", "outputs/pretraining_paired_dense_absolute_70e", PROJECT, "h7u81qb6", "finished"),
    ("lora_paired", "LoRA paired", "Multimodal", "No-base Tri-LoRA, true paired data", "configs/pretraining_paired_lora_absolute_70e.yaml", "outputs/pretraining_paired_lora_absolute_70e", PROJECT, "30fi6whi", "finished"),
    ("dense_unpaired", "Dense unpaired", "Multimodal", "Dense control; independently shuffled text/image", "configs/pretraining_unpaired_dense_absolute_70e.yaml", "outputs/pretraining_unpaired_dense_absolute_70e", PROJECT, "coyf0dx3", "finished"),
    ("plain_lora_no_stage", "Plain LoRA unpaired", "Multimodal", "No-base Tri-LoRA; shared and private branches from epoch 0", "configs/pretraining_unpaired_lora_absolute_no_stage0.yaml", "outputs/pretraining_unpaired_lora_absolute_no_stage0_70e", PROJECT, "zvle4hpq", "finished"),
    ("lora_stage0", "LoRA stage 0", "Multimodal", "Shared-only for epochs 0–9; private branches enabled at epoch 10", "configs/pretraining_unpaired_lora_absolute_stage0_70e.yaml", "outputs/pretraining_unpaired_lora_absolute_stage0_70e", PROJECT, "smua88cz", "finished"),
    ("lora_dann", "LoRA + DANN", "Multimodal", "L2-normalized shared readout; modality adversary from epoch 0", "configs/pretraining_unpaired_lora_absolute_dann_70e.yaml", "outputs/pretraining_unpaired_lora_absolute_dann_70e", PROJECT, "vn430zje", "finished"),
    ("lora_stage0_then_dann", "LoRA stage 0 → DANN", "Multimodal", "Shared-only 10 epochs; private branches and DANN begin at epoch 10", "configs/pretraining_unpaired_lora_absolute_stage0_then_dann_70e.yaml", "outputs/pretraining_unpaired_lora_absolute_stage0_then_dann_70e", PROJECT, "yglwleep", "finished"),
    ("sigreg", "LoRA + SIGReg", "Multimodal", "Global shared-distribution regularization; no stage 0", None, "outputs/pretraining_unpaired_lora_sigreg_absolute_no_stage0_70e", PROJECT, "1ambo2ko", "finished"),
    ("stage0_sigreg", "LoRA stage 0 → SIGReg", "Multimodal", "Stage 0 then global SIGReg", None, "outputs/pretraining_unpaired_lora_absolute_stage0_then_sigreg_70e", PROJECT, "29sd132d", "finished"),
    ("per_layer_sigreg", "LoRA + per-layer SIGReg", "Multimodal", "SIGReg separately at each shared layer", None, "outputs/pretraining_unpaired_lora_sigreg_per_layer_absolute_no_stage0_70e", PROJECT, "ldbop141", "finished"),
    ("stage0_per_layer_sigreg", "LoRA stage 0 → per-layer SIGReg", "Multimodal", "Stage 0 then SIGReg separately per layer", None, "outputs/pretraining_unpaired_lora_absolute_stage0_then_sigreg_per_layer_70e", PROJECT, "t443ovp6", "finished"),
    ("jepa_per_layer", "LoRA + per-layer JEPA/SIGReg", "Multimodal", "Online clean target; shared JEPA and per-layer SIGReg", "configs/pretraining_unpaired_lora_shared_jepa_sigreg_per_layer_absolute_70e.yaml", "outputs/pretraining_unpaired_lora_shared_jepa_sigreg_per_layer_absolute_70e", PROJECT, "ttztvt9n", "finished"),
    ("jepa_normalized", "LoRA + normalized JEPA/SIGReg", "Multimodal", "Normalized-MSE JEPA at layers 2–4 plus per-layer SIGReg", "configs/pretraining_unpaired_lora_middle_l2_l4_normalized_jepa_sigreg_absolute_70e.yaml", "outputs/pretraining_unpaired_lora_middle_l2_l4_normalized_jepa_sigreg_absolute_70e", PROJECT, "s6159f54", "finished"),
    ("jepa_strong_normalized", "LoRA + strong normalized JEPA/SIGReg", "Multimodal", "Larger balanced normalized-MSE JEPA at layers 2–4", "configs/pretraining_unpaired_lora_middle_l2_l4_strong_balanced_normalized_jepa_sigreg_absolute_70e.yaml", "outputs/pretraining_unpaired_lora_middle_l2_l4_strong_balanced_normalized_jepa_sigreg_absolute_70e", PROJECT, "pz54tjy5", "finished"),
    ("jepa_strong_raw_l2", "LoRA + strong raw-L2 JEPA/SIGReg", "Multimodal", "Larger raw squared-L2 JEPA at layers 2–4", "configs/pretraining_unpaired_lora_middle_l2_l4_strong_balanced_raw_l2_jepa_sigreg_absolute_70e.yaml", "outputs/pretraining_unpaired_lora_middle_l2_l4_strong_balanced_raw_l2_jepa_sigreg_absolute_70e", PROJECT, "hv18e0fd", "finished"),
    # These are deliberately single-modality pilots and are labelled as such.
    ("text_diffusion", "Text diffusion-only", "Text-only pilot", "No JEPA control", "configs/text_diffusion_only_ema_jepa_pilot.yaml", "outputs/text_diffusion_only_ema_jepa_pilot", SINGLE, "v1gr53y0", "finished"),
    ("text_ema_fixed", "Text EMA-JEPA fixed", "Text-only pilot", "EMA teacher; normalized-MSE JEPA layers 2–4; fixed λ=.5", "configs/text_ema_jepa_pilot.yaml", "outputs/text_ema_jepa_fixed_lambda050", SINGLE, "kdo2g035", "finished"),
    ("text_ema_r010", "Text EMA-JEPA dynamic .10", "Text-only pilot", "EMA teacher; dynamic JEPA gradient target .10", "configs/text_ema_jepa_pilot.yaml", "outputs/text_ema_jepa_dynamic_r010", SINGLE, "s7ii512c", "W&B crashed; checkpoint/report retained"),
    ("text_ema_r025", "Text EMA-JEPA dynamic .25", "Text-only pilot", "EMA teacher; dynamic JEPA gradient target .25", "configs/text_ema_jepa_pilot.yaml", "outputs/text_ema_jepa_dynamic_r025", SINGLE, "ykppwfrp", "W&B crashed; checkpoint/report retained"),
    ("text_ema_r050", "Text EMA-JEPA dynamic .50", "Text-only pilot", "EMA teacher; dynamic JEPA gradient target .50", "configs/text_ema_jepa_pilot.yaml", "outputs/text_ema_jepa_dynamic_r050", SINGLE, "xlzkx2ev", "W&B crashed; checkpoint/report retained"),
    ("image_diffusion", "Image diffusion-only", "Image-only pilot", "No JEPA control", "configs/image_diffusion_only_ema_jepa_pilot.yaml", "outputs/image_diffusion_only_ema_jepa_pilot", SINGLE, "rp9rbau8", "W&B failed; checkpoint retained"),
    ("image_ema_fixed", "Image EMA-JEPA fixed", "Image-only pilot", "EMA teacher; normalized-MSE JEPA layers 2–4; fixed λ=.5", "configs/image_ema_jepa_pilot.yaml", "outputs/image_ema_jepa_fixed_lambda050_retry", SINGLE, "el5xn2y3", "running"),
    ("image_ema_r010", "Image EMA-JEPA dynamic .10", "Image-only pilot", "EMA teacher; dynamic JEPA gradient target .10", "configs/image_ema_jepa_pilot.yaml", "outputs/image_ema_jepa_dynamic_r010", SINGLE, "6q6cywma", "running"),
    ("image_ema_r025", "Image EMA-JEPA dynamic .25", "Image-only pilot", "EMA teacher; dynamic JEPA gradient target .25", "configs/image_ema_jepa_pilot.yaml", "outputs/image_ema_jepa_dynamic_r025", SINGLE, "d2lp63ha", "running"),
    ("image_ema_r050", "Image EMA-JEPA dynamic .50", "Image-only pilot", "EMA teacher; dynamic JEPA gradient target .50", "configs/image_ema_jepa_pilot.yaml", "outputs/image_ema_jepa_dynamic_r050", SINGLE, "6ncr6t5q", "running"),
]

EVALUATIONS = [
    ("Diffusion training/validation", "train_multimodal.py", "W&B train/*, val/*; weighted and unweighted masked-token cross entropy and accuracy."),
    ("Paired conditional validation", "alignment_evaluation.py", "Matched, shuffled, null, selection, and translation directional losses on reserved true pairs."),
    ("No-base branch ablation", "analyze_branch_ablation_multiratio.py", "Full/shared-only/private-only LoRA branch loss at fixed mask ratios."),
    ("Shared/private retrieval", "analyze_branch_layerwise_recall.py", "Recall@1/5/10 by layer for shared and private branches."),
    ("Semantic clustering and probes", "evaluate_shared_semantics.py", "Frozen shared/private probes and clustering for CLEVR semantics."),
    ("Layer gradient conflict", "analyze_layer_gradient_conflict.py", "Text/image gradient cosine and conflict per layer."),
    ("SIGReg geometry", "analyze_sigreg_shared_distribution.py", "Shared-representation distribution and Gaussianity diagnostics."),
    ("JEPA training trajectories", "analyze_wandb_jepa_trajectories.py", "JEPA loss/cosine trajectory and W&B history."),
    ("JEPA objective checkpoint test", "evaluate_jepa_checkpoints.py", "Fixed-mask masked-position prediction metrics against clean targets."),
    ("Modality-local JEPA/semantic test", "evaluate_jepa_modality.py", "Separate text/image masking stability, semantic probes, per-attribute scores, and exact token detail."),
    ("Causal shared/private swapping", "evaluate_causal_shared_private_swap.py", "Generate B with clean source-A shared deltas substituted at selected layers while preserving B private LoRAs; score output against both source and target scene labels."),
    ("Generation/reconstruction quality", "evaluate_checkpoint_quality.py", "Marginal and conditional reconstruction/generation diagnostics."),
]

# Only evaluations with an organized, comparable result set are shown in the
# user-facing index.  Keep the remaining definitions above so their pages can
# be enabled later without recreating their documentation.
ACTIVE_EVALUATIONS = [
    item for item in EVALUATIONS if item[0] in {
        "Modality-local JEPA/semantic test", "Causal shared/private swapping"
    }
]


# A separate page exists for every evaluation family.  Most evaluations are
# run-specific, therefore their pages deliberately link the durable script and
# the current result artifact instead of copying a number which may belong to
# only one checkpoint.  The modality-local page is richer because the same
# fixed test protocol was run across the LoRA family.
EVAL_DETAILS = {
    "Diffusion training/validation": (
        "diffusion_training_validation",
        "During training, `train_multimodal.py` logs masked-token cross entropy and accuracy to the W&B run for each checkpoint.  It logs both the configured time-weighted loss and the unweighted masked-token loss.",
        "Use the linked pretraining model page to reach the exact W&B history and checkpoint.  This catalog does not merge train curves from different runs into one number.",
        None,
    ),
    "Paired conditional validation": (
        "paired_conditional_validation",
        "Uses reserved true text-image pairs.  It compares matched context, deliberately shuffled context, and null (no-condition) context; lower token loss is better.  Directional text-to-image and image-to-text variants use the corresponding modality as clean condition.",
        "Values are logged per run because the evaluation depends on the checkpoint and its held-out split.",
        None,
    ),
    "No-base branch ablation": (
        "no_base_branch_ablation",
        "For a no-base Tri-LoRA checkpoint, evaluates the normal full routing, shared-only routing, and private-only routing at chosen fixed mask ratios.  This isolates which branch supplies predictive information.",
        "The original per-checkpoint tables are run-specific outputs/W&B logs; no single aggregate value is appropriate.",
        None,
    ),
    "Shared/private retrieval": (
        "shared_private_retrieval",
        "Computes Recall@1, Recall@5, and Recall@10 by Transformer layer.  Full hidden state is formed with both branches, but pooling selects either the shared branch or the private branch, depending on the reported condition.",
        "Values are checkpoint- and layer-specific; retain the generated report next to the evaluation output.",
        None,
    ),
    "Semantic clustering and probes": (
        "semantic_clustering_probes",
        "Fits frozen probes/clusters on representation features to test whether CLEVR object count, color, shape, material, size, and relation are decodable.  These are representation-quality tests, not direct cross-modal alignment tests.",
        "Current report is linked below.",
        "outputs/shared_semantics/REPORT.md",
    ),
    "Layer gradient conflict": (
        "layer_gradient_conflict",
        "Measures the cosine between text-loss and image-loss gradients per layer.  Negative cosine indicates a conflicting update direction; layers with lower conflict are candidates for shared processing.",
        "Current report is linked below.",
        "outputs/layer_gradient_conflict_pretrained_128/REPORT.md",
    ),
    "SIGReg geometry": (
        "sigreg_geometry",
        "Measures the geometry of shared representations and diagnostics related to the Gaussian-reference regularizer (SIGReg).  Similar marginal geometry alone does not establish paired semantic alignment.",
        "Values are stored in run-specific W&B/output artifacts.",
        None,
    ),
    "JEPA training trajectories": (
        "jepa_training_trajectories",
        "Reads W&B JEPA losses and predictor/target diagnostics across training steps.  It is used to detect collapse, late divergence, or a loss that is too small relative to diffusion training.",
        "Current report is linked below.",
        "outputs/jepa_wandb_trajectory_report/REPORT.md",
    ),
    "JEPA objective checkpoint test": (
        "jepa_objective_checkpoint_test",
        "At fixed masked positions, compares the student shared readout (and, when present, its JEPA predictor) with clean stop-gradient or EMA-teacher targets.  This tests the JEPA objective itself, not cross-modal retrieval.",
        "Current report is linked below.",
        "outputs/jepa_evaluation_256/REPORT.md",
    ),
    "Modality-local JEPA/semantic test": (
        "modality_local_jepa_semantic_80pct",
        "Evaluates text and image separately.  A logistic semantic probe is fitted on clean pooled representations and evaluated on the same modality after random token corruption.  It also reports masked-position shared/private stability and exact-token probes.",
        "The dedicated 80%-mask page records the fixed examples, deterministic masks, selected layer rule, model comparison, and raw JSON locations.",
        None,
    ),
    "Causal shared/private swapping": (
        "causal_shared_private_swapping",
        "For clean image source A and masked image target B, records clean A shared LoRA deltas and replaces only layers 2–4 in B during iterative generation. B retains its own private LoRA pathway. An evaluation-only parser scores generated codes against both Y_A and Y_B.",
        "Current result is image-only and same-modality. It establishes causal use of the substituted shared updates, not text-image alignment. The parser's clean count accuracy is limited, so count conclusions are preliminary.",
        "outputs/causal_shared_private_swap_image/REPORT.md",
    ),
    "Generation/reconstruction quality": (
        "generation_reconstruction_quality",
        "Generates/reconstructs marginal and conditional samples.  Marginal evaluation begins with a fully masked modality; conditional evaluation keeps one modality as context and generates the other.",
        "Values and qualitative samples are kept in the run-specific output/W&B artifacts.",
        None,
    ),
}


def pilot_summary_table(results: dict, result_key: str, family: str, branch: str = "shared") -> str:
    """Compact 80%-mask table for the completed single-modality pilots."""
    report = results.get(result_key, {})
    rows = []
    for slug, name, record_family, *_rest in records:
        if record_family != family:
            continue
        item = report.get("models", {}).get(EVALUATION_MODEL_ALIASES.get(slug, slug))
        if not item:
            continue
        layers = item["semantic_clean_train_masked_test"]["t0.8"]
        layer = max(layers, key=lambda key: layers[key][branch]["attribute_group_mean"])
        semantic = layers[layer][branch]
        stability = item["direct_masked_position_stability"]["t0.8"][layer]
        predictor = stability.get("predictor_teacher_cosine") if branch == "shared" else None
        rows.append(
            f"| [{name}](../models/{slug}.md) | {layer} | "
            f"{semantic['object_count']['balanced_accuracy']:.3f} | "
            f"{semantic['attribute_group_mean']:.3f} | "
            f"{stability[f'{branch}_clean_cosine']:.3f} | "
            f"{'—' if predictor is None else f'{predictor:.3f}'} |"
        )
    if not rows:
        return "No completed results are available yet."
    return "\n".join([
        f"| Model | Selected layer | Count | Attribute avg. | {branch.capitalize()}–clean cosine | Predictor–teacher cosine |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
    ])


def modality_local_80pct_page(results: dict) -> str:
    """Document the controlled comparison used for the rows the user asked about."""
    compact_rows = []
    for slug, name, family, *_rest in records:
        if family != "Multimodal":
            continue
        text_item = results.get("text", {}).get("models", {}).get(slug)
        image_item = results.get("image", {}).get("models", {}).get(slug)
        if not text_item or not image_item:
            continue

        def best(item):
            layers = item["semantic_clean_train_masked_test"]["t0.8"]
            layer = max(layers, key=lambda key: layers[key]["shared"]["attribute_group_mean"])
            probe = layers[layer]["shared"]
            return layer, probe["object_count"]["balanced_accuracy"], probe["attribute_group_mean"]

        tl, tc, ta = best(text_item)
        il, ic, ia = best(image_item)
        compact_rows.append(
            f"| [{name}](../models/{slug}.md) | {tl} | {tc:.3f} | {ta:.3f} | {il} | {ic:.3f} | {ia:.3f} |"
        )

    compact_table = "\n".join([
        "| Model | Text L | Text count | Text attr. avg | Image L | Image count | Image attr. avg |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *compact_rows,
    ])
    text_pilot_table = pilot_summary_table(results, "text_ema", "Text-only pilot")
    image_pilot_table = pilot_summary_table(results, "image_ema", "Image-only pilot")
    text_private_pilot_table = pilot_summary_table(results, "text_ema", "Text-only pilot", "private")
    image_private_pilot_table = pilot_summary_table(results, "image_ema", "Image-only pilot", "private")
    return markdown_math(f'''# Modality-local semantic probe at 80% masking

## Question answered

Does a pretrained model retain scene semantics in its **shared** representation when only one modality is presented and 80% of that modality's discrete tokens are randomly masked?

This is a within-modality robustness/semantic-decoding evaluation.  It is **not** a text-image retrieval or paired-alignment score.

## Controlled protocol

| Item | Fixed for every compared checkpoint? | Actual setting |
|---|---|---|
| Training examples for the probe | Yes | First 512 rows of the training manifest, fixed order |
| Held-out test examples | Yes | First 256 rows of the validation manifest, fixed order |
| Evaluated modality | Yes within a table | Text-only or image-only; the other modality is removed |
| Masking rate | Yes | `t=0.8` (80% of eligible discrete tokens) |
| Exact mask positions | Yes | The seed is reset before every validation batch and mask ratio, so every checkpoint sees identical corrupted tokens |
| Semantic labels | Yes | Object count; color, shape, material, size, and relation presence from the CLEVR scene manifest |
| Probe type | Same procedure | L2-normalization + standardization + class-balanced logistic regression; a new probe is fitted per model because features differ |
| Layer selection in summary tables | Same rule | Select L2, L3, or L4 with the highest shared attribute-group average for that model/modality |

The deterministic corruption seed is `seed + batch_index × 1009 + ratio_index × 31 + 500000` on validation.  Models run in `eval()` mode, so they have no dropout/random inference.  Therefore the plain-LoRA and JEPA rows below see exactly the same scenes and the same masked positions.

## What is fitted and scored

1. Run each modality alone through the final **student** checkpoint.
2. Pool eligible-token shared representations from clean training examples.
3. Fit one class-balanced logistic-regression classifier for object count and one binary classifier per attribute.
4. Apply the deterministic 80% token mask to each held-out example.
5. Pool its masked shared representations and score the clean-trained classifiers against ground-truth scene labels.

`Count` is balanced accuracy for number of objects. `Attribute avg` equally averages five group averages: color, shape, material, size, and relation. A value near 0.5 for a binary attribute is chance-level under balanced accuracy.

## Results: every evaluated multimodal LoRA checkpoint

All rows below used the exact same train/held-out examples and deterministic masks within each modality. `L` is the selected shared layer, chosen independently for every row as the one with the largest attribute average among layers 2, 3, and 4.

{compact_table}

For example, plain LoRA versus strong normalized JEPA is `0.253 → 0.335` in image count and `0.547 → 0.602` in image attribute average.  This difference cannot be explained by a changed held-out split or easier mask realization.  It says the selected masked image shared representation supports more robust semantic decoding.  It does **not**, by itself, prove text-image representations are aligned.

## Results: single-modality EMA-JEPA pilots

These models are trained and evaluated on one modality only.  The same clean-to-masked, 80%-mask probe protocol is used.  `Shared–clean cosine` measures the masked student shared readout against its clean target at masked positions; `Predictor–teacher cosine` is present only for JEPA models.

### Text-only pilots

{text_pilot_table}

### Image-only pilots

{image_pilot_table}

## Results: private-LoRA readout, single-modality EMA-JEPA pilots

This uses the same completed forward passes, examples, labels, mask ratios, and exact mask positions as the shared table above.  The only change is the readout: each selected module contributes its private-LoRA update $\Delta_T$ for text or $\Delta_I$ for image, rather than $\Delta_s$.  The selected layer is now the layer with the highest **private** attribute average.

`Predictor–teacher cosine` is not applicable: the JEPA predictor and target are defined only for the shared branch.

### Text-only private branch

{text_private_pilot_table}

### Image-only private branch

{image_private_pilot_table}

## Implementation and raw artifacts

- Evaluation implementation: [{ROOT / 'evaluate_jepa_modality.py'}]({ROOT / 'evaluate_jepa_modality.py'})
- Image raw results: [{ROOT / 'outputs/jepa_modality_eval_lora_family_image/results.json'}]({ROOT / 'outputs/jepa_modality_eval_lora_family_image/results.json'})
- Text raw results: [{ROOT / 'outputs/jepa_modality_eval_lora_family_text/results.json'}]({ROOT / 'outputs/jepa_modality_eval_lora_family_text/results.json'})
- Text-only EMA-pilot raw results: [{ROOT / 'outputs/jepa_modality_eval_text_ema/results.json'}]({ROOT / 'outputs/jepa_modality_eval_text_ema/results.json'})
- Image-only EMA-pilot raw results: [{ROOT / 'outputs/jepa_modality_eval_image_ema/results.json'}]({ROOT / 'outputs/jepa_modality_eval_image_ema/results.json'})
- Full per-attribute table: [{ROOT / 'outputs/LORA_MULTIMODAL_SEPARATE_MODALITY_ATTRIBUTE_REPORT.md'}]({ROOT / 'outputs/LORA_MULTIMODAL_SEPARATE_MODALITY_ATTRIBUTE_REPORT.md'})
- Pretraining and checkpoint links: [model catalog]({ROOT / 'docs/experiment_catalog/README.md'})
''')


def generic_evaluation_page(title: str, script: str, description: str) -> str:
    slug, protocol, value_note, artifact = EVAL_DETAILS[title]
    lines = [f"# {title}", "", "## Purpose", "", description, "", "## Protocol", "", protocol, "", "## Values and artifacts", "", value_note, ""]
    if artifact:
        lines += [f"- Current report: {path_link(artifact)}", ""]
    lines += ["## Implementation", "", f"- Script: {path_link(script)}", f"- Related model/checkpoint pages: [experiment catalog]({DOCS / 'README.md'})", ""]
    return markdown_math("\n".join(lines))


def causal_swap_page() -> str:
    """Document the completed same-modality causal intervention run."""
    result_path = ROOT / "outputs/causal_shared_private_swap_image/results.json"
    if not result_path.exists():
        return generic_evaluation_page(
            "Causal shared/private swapping", "evaluate_causal_shared_private_swap.py",
            EVAL_DETAILS["Causal shared/private swapping"][1],
        )
    report = json.loads(result_path.read_text())
    protocol = report["protocol"]
    parser = report["parser"]["metrics"]
    layers = ", ".join(str(value) for value in protocol["layers"])
    row_to_slug = {value: key for key, value in CAUSAL_SWAP_MODEL_ALIASES.items()}
    rows = []
    for name, values in report["models"].items():
        advantage = values["swap_target_nll"] - values["swap_source_nll"]
        model_label = f"[{name}](../models/{row_to_slug[name]}.md)" if name in row_to_slug else name
        rows.append(
            f"| {model_label} | {values['self_target_nll']:.3f} | {values['swap_source_nll']:.3f} | "
            f"{values['swap_target_nll']:.3f} | {advantage:.3f} | {values['swap_source_win']:.3f} |"
        )
    return markdown_math(f'''# Causal shared/private swapping

## Question

Does the learned shared LoRA route causally control image semantics when the private route is held fixed? This is deliberately an **image-only** test; it does not yet test text-image alignment.

## Fixed protocol

For each cyclic source/target pair $(A,B)$, source image $A$ is clean and target image $B$ is fully masked. At selected Transformer layers $l\\in\\{{{layers}\\}}$, the native shared update of every selected adapter module is captured from $A$ and substitutes for the corresponding shared update while $B$ is generated for {protocol['generation_steps']} confidence-based steps:

$$G_{{swap}}=G(\\Delta_s(A),P_B),\\qquad G_{{self}}=G(\\Delta_s(B),P_B).$$

$P_B$ remains B's own image-private LoRA route; the masked/generation state of B and unselected shared layers also remain B's. Thus this is a direct intervention on **only** the selected native shared adapter deltas, not a replacement of B's hidden state or private branch.

There are {protocol['swap_samples']} held-out cyclic pairs. An evaluation-only VQ-code scene parser is trained on clean training images, then frozen. It scores each generated code grid against both complete scene labels $Y_A$ and $Y_B$.

## Parser quality and limitation

| Held-out clean parser metric | Value |
|---|---:|
| Count accuracy | {parser['count_accuracy']:.3f} |
| Attribute accuracy (ordinary, not balanced) | {parser['attribute_accuracy']:.3f} |

The count parser is only moderately accurate. Therefore this is a strong *intervention diagnostic*, but not yet a paper-ready semantic score: repeat it with a calibrated, class-balanced scene parser before making a definitive claim.

## Results

Lower NLL is better. “Source advantage” is $\\operatorname{{NLL}}(Y_B)-\\operatorname{{NLL}}(Y_A)$ for the swapped sample. A positive value and a source-win rate above 0.5 indicate source-like generated semantics.

| Model | Self NLL vs B | Swapped NLL vs A | Swapped NLL vs B | Source advantage | Source win rate |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Interpretation

All four EMA-JEPA variants have a large source advantage (12.2–15.4) and source-win rate (0.812–0.859). The diffusion-only control is near chance by source-win rate (0.453). Within this test, that is evidence that the selected middle-layer shared updates causally affect generated image semantics rather than being ignored.

The near equality of self NLL vs B and swapped NLL vs A for the EMA-JEPA variants is compatible with successful cyclic source transfer: after swapping, the output can be as source-like as self generation is target-like. It does not mean the self and swap samples were reused.

## Detailed specification

- [Detailed protocol, formulas, parser loss, and all model-linked results](causal_shared_private_swap_detailed.md)

## Artifacts

- Implementation: {path_link('evaluate_causal_shared_private_swap.py')}
- Raw results: {path_link('outputs/causal_shared_private_swap_image/results.json')}
- Full report: {path_link('outputs/causal_shared_private_swap_image/REPORT.md')}
- Frozen parser checkpoint: {path_link('outputs/causal_shared_private_swap_image/frozen_image_code_scene_parser.pt')}
- Relevant model/checkpoint pages: [experiment catalog]({DOCS / 'README.md'})
''')


def modality_results():
    out = {}
    for label, relative in {
        "text": "outputs/jepa_modality_eval_lora_family_text/results.json",
        "image": "outputs/jepa_modality_eval_lora_family_image/results.json",
        "text_ema": "outputs/jepa_modality_eval_text_ema/results.json",
        "image_ema": "outputs/jepa_modality_eval_image_ema/results.json",
    }.items():
        path = ROOT / relative
        if path.exists():
            out[label] = json.loads(path.read_text())
    return out


def saved_runtime_args(output: str) -> dict:
    """Read the exact argparse values saved in the final checkpoint.

    mmap keeps this catalog build inexpensive: we only read the small `args`
    metadata object, not model tensors into ordinary CPU memory.
    """
    checkpoint = ROOT / output / "last.pt"
    if not checkpoint.exists():
        return {}
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        args = dict(payload.get("args", {}))
        del payload
        return args
    except Exception as exc:  # A missing/corrupt checkpoint must not block docs.
        return {"_catalog_metadata_error": str(exc)}


def flag(args: dict, name: str) -> bool:
    return bool(args.get(name, False))


def number(value) -> str:
    if value is None:
        return "not set"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def markdown_math(text: str) -> str:
    """Use dollar-delimited math throughout; this renders in the Markdown
    preview used for the catalog, unlike mixed `\\(...\\)` delimiters."""
    return (
        text.replace(r"\[", "$$").replace(r"\]", "$$")
        .replace(r"\(", "$").replace(r"\)", "$")
    )


def mathematics_block(args: dict) -> list[str]:
    """Consistent forward and objective notation, parameterized by saved args."""
    if not args:
        return ["## Exact forward pass and optimized objective", "", "The checkpoint metadata needed to reconstruct exact coefficients is unavailable.", ""]
    mode = args.get("train_mode", "dense")
    use_modality = flag(args, "use_modality_embeddings")
    input_formula = r"\(h_i^0=E_{tok}(\tilde x_i)+E_{pos}(p_i)\)"
    if use_modality:
        input_formula += r" \(+E_{mod}(m_i)\)"
    objective = args.get("objective", "both")
    modalities = {"both": r"\(q\in\{T,I\}\)", "text": r"\(q=T\)", "image": r"\(q=I\)"}.get(objective, objective)
    time_weight = r"1/t_b" if not flag(args, "unweighting") else "1"
    indexed_time_weight = r"1/t_{b(i)}" if not flag(args, "unweighting") else "1"
    eps = number(args.get("eps", 1e-3))
    full_mask = number(args.get("full_mask_probability", 0.0))
    lines = [
        "## Exact forward pass and optimized objective", "",
        "### Shared notation", "",
        r"- \(x_i\) is the clean discrete token at position \(i\); \(\tilde x_i\) is its corrupted value (the mask token when \(i\in M\)).",
        "- \(p_i\) is the within-modality position; \(m_i\) is the modality identifier; \(\ell_i\) is the output-logit vector.",
        f"- Input: {input_formula}.  " + ("A modality embedding is present." if use_modality else "There is **no modality embedding**."),
        f"- A mask rate \(t_b\sim U({eps},1)\) is drawn per row; each eligible token is independently included in \(M\) with probability \(t_b\), with at least one eligible token forced into \(M\).  Full-mask probability: **{full_mask}**.",
        "",
        "### Forward pass", "",
    ]
    if mode == "lora":
        rank = int(args.get("lora_rank", 0) or 0)
        alpha = number(args.get("lora_alpha"))
        nominal_shared = max(1, (2 * rank) // 3)
        private = max(1, rank - nominal_shared) if rank else "unknown"
        targets = ", ".join(args.get("lora_targets", [])) or "saved target list unavailable"
        lines += [
            "Every selected linear map has **no frozen dense base weight**.  For branch \(r\in\{s,T,I\}\):",
            "",
            r"$$\Delta_r^{(m)}(z)=\frac{\alpha}{\rho_s}\cdot B_r^{(m)}\cdot A_r^{(m)}\cdot D(z),\qquad \rho_s=\max(1,\lfloor2R/3\rfloor).$$",
            "",
            f"Here $D$ is adapter dropout.  Its configured probability is **{number(args.get('lora_dropout'))}**, so "
            + ("$D(z)=z$ in this run." if float(args.get("lora_dropout", 0.0) or 0.0) == 0.0 else "$D(z)$ is the stochastic dropout-transformed input during training and $D(z)=z$ in evaluation."),
            r"For ordinary text and image routing, respectively, \(y_T=b_s+\Delta_s(z)+\Delta_T(z)\) and \(y_I=b_s+\Delta_s(z)+\Delta_I(z)\).  \(b_s\) is a learned shared bias; there is no \(Wz\) term.",
            f"- Requested rank \(R\): **{rank}**; shared rank \(\\rho_s\): **{nominal_shared}**; text-private rank: **{private}**; image-private rank: **{private}**; \(\\alpha\): **{alpha}**; adapter dropout: **{number(args.get('lora_dropout'))}**.",
            "- Thus the shared and private branches do **not** have equal rank.  The requested rank determines roughly two-thirds shared and one-third **for each** private branch; consequently, the three branch ranks together sum to roughly $4R/3$, not $R$.",
            f"- Replaced linear modules: `{targets}`.",
        ]
        stage = int(args.get("shared_only_epochs", 0) or 0)
        if stage:
            lines.append(f"- Stage 0 (epochs 0–{stage - 1}): private \(A_T,B_T,A_I,B_I\) are frozen, so \(y=b_s+\\Delta_s(z)\).  From epoch {stage}: full modality routing above is trainable.")
    else:
        lines += [
            r"Each linear map is dense: \(y^{(m)}=W^{(m)}z+b^{(m)}\).  There are no shared/private LoRA branches in this model.",
        ]
    lines += [
        "",
        r"The Transformer maps the corrupted sequence to logits: \(\ell=F_\theta(\tilde x,p,m)\).",
        "",
        "### Diffusion/reconstruction loss", "",
        r"$$\mathcal L_{\mathrm{diff}}^{(q)}=\frac{1}{|M_q|}\sum_{i\in M_q} w_{b(i)}\cdot\operatorname{CE}(\ell_i,x_i).$$",
        "",
        f"where \(w_{{b(i)}}={indexed_time_weight}\).  `unweighting={flag(args, 'unweighting')}`; the diffusion coefficient is **1.0**.  The optimized step averages active modality sub-losses ({modalities}).",
    ]
    additions = []
    if flag(args, "shared_jepa"):
        selected = args.get("shared_jepa_layers") or list(range(int(args.get("n_layers", 0))))
        selected_text = ", ".join(map(str, selected))
        loss_type = args.get("shared_jepa_loss", "mse")
        if loss_type == "normalized_mse":
            jepa_formula = r"\(J_l=\operatorname{MSE}(\sqrt d\,\widehat{P_l(s_l^{mask})},\sqrt d\,\widehat{\operatorname{sg}(s_l^{clean})})\)"
        elif loss_type == "mse":
            jepa_formula = r"\(J_l=\operatorname{MSE}(P_l(s_l^{mask}),\operatorname{sg}(s_l^{clean}))\)"
        else:
            jepa_formula = r"\(J_l=1-\cos(P_l(s_l^{mask}),\operatorname{sg}(s_l^{clean}))\)"
        target = "EMA teacher" if flag(args, "shared_jepa_ema") else "online clean stop-gradient student"
        coefficient = args.get("shared_jepa_weight")
        dynamic = flag(args, "shared_jepa_dynamic_weight")
        if dynamic:
            coefficient_text = (
                r"\(\lambda_J=\operatorname{clip}(r\,\operatorname{EMA}\|g_{diff}\|/\operatorname{EMA}\|g_J\|,"
                f" {number(args.get('shared_jepa_dynamic_min_weight'))}, {number(args.get('shared_jepa_dynamic_max_weight'))})\)"
            )
        else:
            coefficient_text = f"\(\\lambda_J={number(coefficient)}\)"
        additions += [
            "### JEPA term", "",
            f"At direct target layers **{selected_text}**, {jepa_formula}, averaged over selected layers and exactly masked positions.  \(P_l\) is the learned JEPA predictor and \(\\widehat{{v}}=v/(\\lVert v\\rVert_2+10^{{-6}})\).",
            f"- Teacher: **{target}**" + (f" (EMA decay **{number(args.get('shared_jepa_ema_decay'))}**)." if flag(args, "shared_jepa_ema") else "."),
            f"- Coefficient: {coefficient_text}; warm-up: **{number(args.get('shared_jepa_warmup_steps'))}** steps beginning at epoch **{number(args.get('shared_jepa_start_epoch'))}**.  With local post-start step \(u\), \(r_J=\\min(1,(u+1)/{number(args.get('shared_jepa_warmup_steps'))})\) (or \(1\) when warm-up is zero).",
        ]
        if dynamic:
            additions.append(f"- Dynamic-gradient target ratio \(r\): **{number(args.get('shared_jepa_gradient_ratio'))}**; coefficient EMA decay: **{number(args.get('shared_jepa_dynamic_ema_decay'))}**.")
        text_w = args.get("shared_jepa_text_weight")
        image_w = args.get("shared_jepa_image_weight")
        if text_w is not None or image_w is not None:
            additions.append(f"- Per-modality coefficient overrides: text **{number(text_w if text_w is not None else coefficient)}**, image **{number(image_w if image_w is not None else coefficient)}**.")
    if flag(args, "sigreg"):
        per_layer = flag(args, "sigreg_per_layer")
        selected = args.get("sigreg_layers")
        target_layers = ", ".join(map(str, selected)) if selected else ("all layers" if per_layer else "pooled shared readout")
        sig_formula = r"R(z)=\frac1S\sum_{j=1}^S N\sum_k w_k[(\overline{\cos(t_k u_j^\top z)}-e^{-t_k^2/2})^2+\overline{\sin(t_k u_j^\top z)}^2]"
        additions += [
            "### SIGReg term", "",
            f"$${sig_formula}$$", "",
            "Random unit directions \(u_j\) compare the empirical projected characteristic function with \(N(0,1).\)",
            f"- Applied to **{target_layers}**" + ("; the selected layer losses are averaged." if per_layer else "."),
            f"- Coefficient \(\\lambda_S\): **{number(args.get('sigreg_weight'))}**; warm-up: **{number(args.get('sigreg_warmup_steps'))}** steps from epoch **{number(args.get('sigreg_start_epoch'))}**, with \(r_S=\\min(1,(u+1)/{number(args.get('sigreg_warmup_steps'))})\) (or \(1\) when warm-up is zero); slices **{number(args.get('sigreg_num_slices'))}**, points **{number(args.get('sigreg_num_points'))}**, \(t_{{max}}\) **{number(args.get('sigreg_t_max'))}**.",
        ]
    if flag(args, "modality_adversarial"):
        normalization = args.get("modality_adversarial_representation_normalization", "none")
        warmup = int(args.get("modality_adversarial_warmup_steps", 0) or 0)
        weight = number(args.get("modality_adversarial_weight"))
        start_epoch = number(args.get("modality_adversarial_start_epoch"))
        grl = number(args.get("modality_adversarial_grl_lambda"))
        hidden = args.get("modality_discriminator_hidden") or args.get("d_model")
        normalization_detail = (
            r"$\bar s=s/(\lVert s\rVert_2+10^{-6})$ independently for each row.  Therefore the discriminator cannot classify the modality from the length of $s$; it can only use its direction."
            if normalization == "l2" else
            "No normalization is applied, so the discriminator receives both the length and direction of $s$."
        )
        warmup_detail = (
            f"At its first active step, $r_D=1/{warmup}$ and the effective loss coefficient is ${weight}/{warmup}$; after {warmup} active steps, $r_D=1$ and the coefficient is ${weight}$."
            if warmup else f"There is no warm-up: $r_D=1$ and the coefficient is ${weight}$ immediately."
        )
        additions += [
            "### DANN term", "",
            "For a block $l$ and its shared adapter module $a$, let $\Delta^{(l,a)}_{s,b,j}$ be the shared adapter output for example $b$ and token $j$.  Let $P_{l,a}$ reduce its output width to $d$ when necessary (adaptive average pooling), and let $A_{b,j}\in\{0,1\}$ be the attention mask.",
            r"$$\Delta^{(l,a)}_{s,b,j}=\frac{\alpha}{\rho_s}\,B_s^{(l,a)}A_s^{(l,a)}D\!\left(z^{(l,a)}_{b,j}\right).$$",
            "This is the **raw additive shared-LoRA update of that one linear module**.  It is not the full Transformer-block output, not the hidden state $x^l$, and not a separate forward pass in which private LoRAs are disabled or subtracted.",
            "For example, $z^{(l,\mathrm{qkv})}=\operatorname{LN}_1(x^{l-1})$; $z^{(l,\mathrm{out\_proj})}$ is the attention output; $z^{(l,\mathrm{mlp.0})}=\operatorname{LN}_2(x^{l-1}+\operatorname{Attn}(\cdot))$; and $z^{(l,\mathrm{mlp.3})}$ is the GELU/dropout output of `mlp.0`.",
            "Consequently, later module inputs can contain effects caused by private branches in earlier modules.  However, the recorder uses $\operatorname{stopgrad}(z^{(l,a)})$ when forming this DANN readout, so the DANN gradient cannot flow back through those inputs into private branches or earlier blocks.",
            r"$$r^{(l,a)}_{s,b}=\frac{\sum_j A_{b,j}\,P_{l,a}\!\left(\Delta^{(l,a)}_{s,b,j}\right)}{\sum_j A_{b,j}},\qquad h^l_{s,b}=\frac1{|\mathcal A_l|}\sum_{a\in\mathcal A_l}r^{(l,a)}_{s,b},\qquad s_b=\frac1L\sum_{l=0}^{L-1}h^l_{s,b}.$$",
            "Here $\mathcal A_l=\{\mathrm{qkv},\mathrm{out\_proj},\mathrm{mlp.0},\mathrm{mlp.3}\}$ and $L=8$ in these runs.  Thus $h^l_s$ is the named per-layer shared readout, while $s$ is their global average.",
            r"$$\bar s_b=\operatorname{norm}(s_b),\qquad \mathcal L_{\mathrm{DANN}}^{(q)}=\frac1B\sum_{b=1}^B\operatorname{CE}\!\left(D\!\left(\operatorname{GRL}_{\gamma}(\bar s_b)\right),q\right).$$", "",
            "There is **one** DANN cross-entropy per homogeneous text or image sub-batch, not one independent CE per layer.  Because $s$ averages all $h_s^l$, this one loss sends a direct gradient to every recorded shared adapter in every layer.",
            f"- L2 input normalization: {normalization_detail}",
            f"- The discriminator is an MLP $D: d\\rightarrow {number(hidden)}\\rightarrow 2$ with GELU; its target $q$ is text (0) or image (1).",
            f"- Loss strength: $\\lambda_D={weight}$.  DANN turns on at epoch **{start_epoch}**.  With $u=0,1,\ldots$ counting optimizer steps since activation, $r_D=\\min(1,(u+1)/{warmup})$.",
            f"  {warmup_detail}",
            f"- Gradient reversal: $\\gamma={grl}$.  The discriminator parameters receive the ordinary CE gradient and learn to identify the modality.  The shared branch receives that gradient multiplied by $-{grl}$, so it learns to make the modality harder to identify.  $\\gamma$ does not reverse the discriminator's own update.",
            "- Exact gradient path: the recorder recomputes each $\Delta_s^{(l,a)}$ from a detached adapter input.  Therefore DANN updates the shared $A_s^{(l,a)},B_s^{(l,a)}$ parameters in every recorded layer and the discriminator, but it does not backpropagate through this DANN path into earlier Transformer activations, private LoRAs, shared biases, embeddings, or the output head.",
        ]
    lines += additions
    terms = [r"\mathcal L_{\mathrm{diff}}^{(q)}"]
    if flag(args, "shared_jepa"):
        terms.append(r"\lambda_J\cdot r_J\cdot\mathcal L_{\mathrm{JEPA}}^{(q)}")
    if flag(args, "sigreg"):
        terms.append(r"\lambda_S\cdot r_S\cdot\mathcal L_{\mathrm{SIGReg}}^{(q)}")
    if flag(args, "modality_adversarial"):
        terms.append(r"\lambda_D\cdot r_D\cdot\mathcal L_{\mathrm{DANN}}^{(q)}")
    lines += ["", "### Final optimized objective", "", r"$$\boxed{\mathcal L_{\mathrm{step}}=\frac1{|\mathcal Q|}\sum_{q\in\mathcal Q}\left[" + "+".join(terms) + r"\right]}$$", "", f"$\\mathcal Q$ is the set of active modality sub-batches: {modalities}.  This outer average is the code's `len(sub_batches)` division before gradient accumulation.", ""]
    return lines


def detail_block(slug: str, results: dict) -> list[str]:
    lines = ["## Available evaluated values", ""]
    found = False
    for modality, report in results.items():
        item = report["models"].get(EVALUATION_MODEL_ALIASES.get(slug, slug))
        if not item:
            continue
        layers = item["semantic_clean_train_masked_test"]["t0.8"]
        layer = max(layers, key=lambda key: layers[key]["shared"]["attribute_group_mean"])
        probe = layers[layer]["shared"]
        attrs = probe.get("attribute_balanced_accuracy", {})
        attribute_text = (
            "; ".join(f"{name}={score:.3f}" for name, score in attrs.items())
            if attrs else "individual attributes were not retained in this earlier result JSON"
        )
        lines += [
            f"### { {'text': 'Text', 'image': 'Image', 'text_ema': 'Text EMA pilot', 'image_ema': 'Image EMA pilot'}[modality] } only, 80% masking", "",
            f"Selected shared layer: **{layer}** (highest attribute average among L2–L4).",
            "",
            "| Count | Attribute avg | Attributes |",
            "|---:|---:|---|",
            f"| {probe['object_count']['balanced_accuracy']:.3f} | {probe['attribute_group_mean']:.3f} | "
            + attribute_text + " |",
            "",
        ]
        found = True
    if not found:
        lines += ["No modality-local value is available for this model yet.", ""]
    return lines


def causal_swap_detail(slug: str) -> list[str]:
    """Add the run-specific causal-swap result to participating model pages."""
    label = CAUSAL_SWAP_MODEL_ALIASES.get(slug)
    result_path = ROOT / "outputs/causal_shared_private_swap_image/results.json"
    if label is None or not result_path.exists():
        return []
    values = json.loads(result_path.read_text())["models"].get(label)
    if values is None:
        return []
    advantage = values["swap_target_nll"] - values["swap_source_nll"]
    return [
        "## Causal shared/private swap result", "",
        "Image-only, layers 2–4; clean source shared deltas replace the masked target's selected shared deltas while its image-private route remains active.",
        "",
        "| Self NLL vs B | Swapped NLL vs A | Swapped NLL vs B | Source advantage | Source win rate |",
        "|---:|---:|---:|---:|---:|",
        f"| {values['self_target_nll']:.3f} | {values['swap_source_nll']:.3f} | {values['swap_target_nll']:.3f} | {advantage:.3f} | {values['swap_source_win']:.3f} |",
        "",
        "The parser's count accuracy is modest (0.434), so see the complete protocol and limitation before interpreting this as a final semantic result.",
        "",
        "- [Detailed causal-swap protocol and comparison](../evaluations/causal_shared_private_swap_detailed.md)",
        "",
    ]


def model_page(record, results, runtime_args):
    slug, name, family, summary, config, output, project, run_id, status = record
    lines = [f"# {name}", "", f"**Family:** {family}", "", f"**Purpose:** {summary}", "", "## Artifacts", ""]
    lines += [
        f"- W&B: {run_url(project, run_id)} — `{status}`",
        f"- Configuration: {path_link(config) if config else 'Configuration YAML was not retained; use W&B config.'}",
        f"- Best checkpoint: {path_link(output + '/best.pt', 'best.pt')}",
        f"- Latest/resume checkpoint: {path_link(output + '/last.pt', 'last.pt')}",
        f"- Output directory: `{ROOT / output}`",
        "",
    ]
    if config and (ROOT / config).exists():
        cfg = yaml.safe_load((ROOT / config).read_text())
        lines += ["## Saved configuration summary", ""]
        for section, keys in {
            "data": ("mode", "balanced_modalities"),
            "model": ("d_model", "n_layers", "use_modality_embeddings"),
            "lora": ("train_mode", "rank", "target_modules"),
            "train": ("epochs", "shared_only_epochs", "batch_size", "gradient_accumulation_steps"),
            "alignment": ("modality_adversarial_enabled", "modality_adversarial_start_epoch", "sigreg_enabled", "shared_jepa_enabled", "shared_jepa_layers", "shared_jepa_loss", "shared_jepa_ema_enabled"),
        }.items():
            present = {key: cfg.get(section, {}).get(key) for key in keys if key in cfg.get(section, {})}
            if present:
                lines.append(f"- `{section}`: `{present}`")
        lines.append("")
        alignment = cfg.get("alignment", {})
        if alignment.get("shared_jepa_enabled"):
            n_layers = cfg.get("model", {}).get("n_layers")
            selected = alignment.get("shared_jepa_layers")
            selected_text = ", ".join(str(layer) for layer in selected) if selected else "all layers"
            if selected and n_layers is not None:
                direct_excluded = [layer for layer in range(n_layers) if layer not in set(selected)]
                excluded_text = ", ".join(str(layer) for layer in direct_excluded) or "none"
            else:
                excluded_text = "none"
            teacher = "clean EMA teacher" if alignment.get("shared_jepa_ema_enabled") else "clean stop-gradient online target"
            lines += [
                "## JEPA placement", "",
                f"- Direct JEPA target layers: **{selected_text}** (Transformer layer indices).",
                f"- Layers without their own direct JEPA target: **{excluded_text}**.",
                f"- Target: {teacher}; student input is the masked modality.",
                f"- Loss: `{alignment.get('shared_jepa_loss', 'mse')}` on the shared representation, through a JEPA predictor.",
                "- The regular diffusion/reconstruction loss trains all layers.  A JEPA loss at a selected layer also backpropagates through earlier layers that produced its representation, but those earlier layers have no separate JEPA target unless listed above.",
                "",
            ]
    lines += mathematics_block(runtime_args)
    lines += detail_block(slug, results)
    lines += causal_swap_detail(slug)
    lines += [
        "## Evaluation pointers", "",
        f"- [Evaluation catalog](../evaluations.md)",
        f"- [Multimodal separate-modality attribute table]({ROOT / 'outputs/LORA_MULTIMODAL_SEPARATE_MODALITY_ATTRIBUTE_REPORT.md'})",
        f"- [Multimodal LoRA experiment notebook]({ROOT / 'notebooks/multimodal_lora_experiment_record.ipynb'})",
    ]
    return markdown_math("\n".join(lines) + "\n")


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    EVALDOCS.mkdir(parents=True, exist_ok=True)
    results = modality_results()
    for record in records:
        runtime_args = saved_runtime_args(record[5])
        (MODELS / f"{record[0]}.md").write_text(model_page(record, results, runtime_args))
    index = [
        "# CLEVR discrete diffusion experiment catalog", "",
        "This is the start page for every pretrained model, checkpoint, W&B run, evaluation protocol, and result table.", "",
        "## Start here", "",
        "- [Evaluation index: active evaluations, scripts, and result reports](evaluations.md)",
        "- [Standalone evaluation-report index](evaluations/README.md)",
        "- [80% modality-local semantic evaluation: controlled protocol and example comparison](evaluations/modality_local_jepa_semantic_80pct.md)",
        "- [Detailed causal shared/private swapping protocol and results](evaluations/causal_shared_private_swap_detailed.md)",
        f"- [All multimodal-LoRA 80% semantic values: text and image, every attribute]({ROOT / 'outputs/LORA_MULTIMODAL_SEPARATE_MODALITY_ATTRIBUTE_REPORT.md'})",
        "- [Multimodal LoRA experiment notebook](%s)" % (ROOT / "notebooks/multimodal_lora_experiment_record.ipynb"),
        "",
        "## Pretraining models", "", "| Model | Family | W&B | Checkpoint |", "|---|---|---|---|",
    ]
    for slug, name, family, _summary, _config, output, project, run_id, status in records:
        index.append(f"| [{name}](models/{slug}.md) | {family} | {run_url(project, run_id)} ({status}) | {path_link(output + '/best.pt', 'best.pt')} |")
    index += ["", "## Navigation", "", "- [Evaluation catalog](evaluations.md)", "- [Standalone evaluation reports](evaluations/README.md)", "- [80% modality-local semantic evaluation](evaluations/modality_local_jepa_semantic_80pct.md)", "- [Detailed causal shared/private swap evaluation](evaluations/causal_shared_private_swap_detailed.md)", "- [Per-attribute multimodal LoRA report](%s)" % (ROOT / "outputs/LORA_MULTIMODAL_SEPARATE_MODALITY_ATTRIBUTE_REPORT.md"), ""]
    (DOCS / "README.md").write_text("\n".join(index))
    ev = ["# Evaluation index", "", "This index lists evaluations that currently have an organized, comparable result report.  Add new evaluations here once their script, protocol, and results are ready.", "", "| Evaluation | Script | What it measures | Results |", "|---|---|---|---|"]
    artifacts = {
        "Semantic clustering and probes": "outputs/shared_semantics/REPORT.md",
        "Layer gradient conflict": "outputs/layer_gradient_conflict_pretrained_128/REPORT.md",
        "JEPA training trajectories": "outputs/jepa_wandb_trajectory_report/REPORT.md",
        "JEPA objective checkpoint test": "outputs/jepa_evaluation_256/REPORT.md",
        "Modality-local JEPA/semantic test": "outputs/LORA_MULTIMODAL_SEPARATE_MODALITY_ATTRIBUTE_REPORT.md",
    }
    evaluation_index = ["# Standalone evaluation reports", "", "Each page records the goal, protocol, implementation, result locations, and how to interpret its values.", "", "## Available now", ""]
    for title, script, description in ACTIVE_EVALUATIONS:
        artifact = artifacts.get(title)
        slug = EVAL_DETAILS[title][0]
        ev.append(f"| [{title}](evaluations/{slug}.md) | {path_link(script)} | {description} | [Protocol and all current results](evaluations/{slug}.md) |")
        evaluation_index.append(f"- [{title}]({slug}.md) — {description}")
        page = (
            modality_local_80pct_page(results) if title == "Modality-local JEPA/semantic test"
            else causal_swap_page() if title == "Causal shared/private swapping"
            else generic_evaluation_page(title, script, description)
        )
        (EVALDOCS / f"{slug}.md").write_text(page)
    ev += ["", "## Adding the next evaluation", "", "For each new evaluation, add one entry with: its script, a fixed protocol, a result table, raw-output links, and links back to the relevant model/checkpoint pages.", ""]
    (DOCS / "evaluations.md").write_text("\n".join(ev))
    (EVALDOCS / "README.md").write_text("\n".join(evaluation_index) + "\n")
    print(DOCS / "README.md")


if __name__ == "__main__":
    main()
