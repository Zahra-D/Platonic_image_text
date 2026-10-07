# Every experiment, what it is, and what it cost

> **Type:** master index · **Status:** 62 trained runs in 9 families · **Updated:** 2026-09-24
> **Menu:** [experiment catalog](README.md) · **Regenerate:** `python3 scripts/build_all_experiments.py`

Grouped by **method family**, simplest first. Modality is a column, so a family
that exists in both text and images is read in one place.

**This page is generated.** Every number comes from a file on disk — the run's
config, its `train.log`, and the evaluation JSONs under `outputs/`. Edit
`docs/experiment_catalog/_all_experiments_preamble.md` for the prose and re-run
the generator for the tables; do not hand-edit the tables.

## Contents

1. [Baselines: diffusion only](#1-baselines-diffusion-only--6-runs) — dense and Tri-LoRA, no JEPA
2. [Tri-LoRA with JEPA on adapter writes](#2-tri-lora-with-jepa-on-adapter-writes--10-runs) — the modulewise family
3. [data2vec on hidden states, from a pretrained trunk](#3-data2vec-on-hidden-states-from-a-pretrained-trunk--9-runs) — the one that works
4. [data2vec on hidden states, from scratch](#4-data2vec-on-hidden-states-from-scratch--11-runs) — the one that does not
5. [SIGReg / LeJEPA instead of an EMA teacher](#5-sigreg--lejepa-instead-of-an-ema-teacher--4-runs) — all four collapse
6. [Stage 2: dense shared route + private LoRA](#6-stage-2-dense-shared-route--private-lora--4-runs) — the shared/private test
7. [Multimodal: paired and unpaired](#7-multimodal-paired-and-unpaired--5-runs) — one model, both modalities
8. [Other corpora (MS-COCO)](#8-other-corpora-ms-coco--2-runs)
9. [Pilots and ablations (not yet evaluated)](#9-pilots-and-ablations-not-yet-evaluated--11-runs)

## Reading the columns

| Column | Meaning |
|---|---|
| **Mod** | text, image, or both (paired or unpaired multimodal) |
| **What it is** | the JEPA objective in full: target design, **whether an MLP predictor is used** and how many heads, EMA teacher or SIGReg, the loss, the adapter ranks and the masking |
| **From** | initialization; `scratch` means random |
| **Ep** | epochs with a saved checkpoint |
| **Own** | tokens this run consumed |
| **Total** | tokens including every stage it was built on |
| **Val** | final validation loss — *meaningless for any `no diffusion` run* |
| **d_sem / d_bind full** | [semantic effect sizes](evaluations/semantic_dprime.md) of the model as it runs; `d_sem` at `L7.residual`, `d_bind` at the model's best feature; bag-of-words control is −2.53 / 0.000 |
| **… shared** | the same, with every modality-private adapter suppressed, so only the shared route runs |
| **… private** | the mirror: the shared write zeroed, only the private branch |
| **text probe / img probe** | linear scene-fact readout at L7, [cross-modal structure](evaluations/cross_modal_structure.md) |
| **img bind** | [image binding](evaluations/image_binding.md) on content-matched pairs, chance 0.500 |

Tokens per epoch: text 2M captions × 91.7 content tokens = **183M**; images
1.2M × 384 = **461M**; multimodal sees both = **571M**; the old 90k paired set
42.8M; MS-COCO 118k captions ≈ 1.5M.

## Three traps when comparing rows

**1. Match the token budget.** The 4-epoch dense baseline (0.73B) is
undertrained as a reference: its binding effect size rises 0.244 → 0.443 between
4 and 6 epochs. A from-dense JEPA run costs 1.10B total, so it must be compared
against dense at **6** epochs. Read the **Total** column, not **Own**. An earlier
version of this catalog compared against the 4-epoch number and reported that
JEPA "more than doubles" binding; the correct figure is **+19%**.

**2. Ignore Val for `no diffusion` runs.** Nothing trains the output head, so
~8.8 is uniform guessing and 1.4–2.0 only measures how far a JEPA phase drifted
the trunk away from a frozen head. It says nothing about representation quality.

**3. Route isolation means different things in different families.** The shared
and private columns come from an actual forward pass with the other route's
write zeroed by route id, not from decomposing a mixed pass, so later blocks
also receive a branch-free input.

This is meaningful for **dense shared + private LoRA** (family 5), where the
trunk alone is a complete network: the frozen-trunk run read trunk-only
reproduces its stage-1 parent to three decimals (−0.763 / +0.293 against
−0.76 / +0.293), which is the strongest end-to-end check the evaluation code
has.

It is **not** a clean read-out for **Tri-LoRA** (families 1–2), where each
adapted linear is `shared_delta + private_delta` with no base weight: removing
either branch does not isolate a route, it breaks the layer, because the
remaining half was never trained to work alone. Plain Tri-LoRA scores +0.009
private-only and +0.019 shared-only against +0.293 intact — binding exists only
in the sum. For that family the per-module `*_shared` / `*_private` **write**
features in [semantic effect sizes](evaluations/semantic_dprime.md) remain the
meaningful comparison.

## The families

**1. Baselines** — mask tokens, predict them. Dense, or Tri-LoRA where every
adapted linear is `shared_delta + private_delta` with no frozen base (rank 384
= shared 256 + private 128, one private branch per modality).

**2. Tri-LoRA + JEPA on adapter writes** — an EMA-teacher JEPA applied to the
*adapter writes* of selected modules, averaged or layerwise across blocks,
optionally with the gradient routed end-to-end into the shared A/B tensors
("gated"), optionally with HSIC penalizing shared/private dependence. Trained
together with the diffusion loss.

**3–4. data2vec on hidden states** — predict the EMA teacher's block outputs at
masked positions, either **averaged** (one target, `t = mean_k LN(h_k^EMA)`, one
predictor on the final block) or **layerwise** (each block predicts its own
target through its own MLP head; 8 heads = 4.72M parameters). The split between
families 3 and 4 is the single most decisive result in the study: **from a
pretrained trunk it produces the best representations measured; from scratch it
collapses to a surface representation, in both modalities, at every masking
rate and both anti-collapse mechanisms tried.**

**5. SIGReg / LeJEPA** — the same layerwise hidden-state objective with the EMA
teacher, the stop-gradient and the target LayerNorm all removed, replaced by the
sliced Epps–Pulley isotropic-Gaussian test of
[LeJEPA](https://arxiv.org/abs/2511.08544) (1024 slices, 17 evaluation points),
combined as `(1 − λ)·L_pred + λ·SIGReg` at λ = 0.05. Four runs — text from
scratch, text from the stage-1 dense trunk, multimodal from the unpaired dense
trunk, multimodal from scratch. **All four collapse**, and the from-dense run
ends up *below* the bag-of-words control on `d_semantic` (−2.68). λ was not
swept; that is the obvious open question.

**6. Stage 2** — keep the dense weight as the shared route
(`y = Wx + (α/ρ_p)·B_p A_p x + b`) and rank-limit only the private branch
(128 = d/3), training diffusion + HSIC with the trunk frozen or free. Read
trunk-only, this family contains the best text model in the study.

**7. Multimodal** — one model, both modalities. **Paired** carries a caption
with its own image; **unpaired** never does (a derangement guarantees it).
Paired dense is the reference target on every cross-modal measure.

**8–9. Other corpora and pilots** — MS-COCO transfer, and short exploratory runs
(EMA decay, target fidelity, masking style). Not evaluated.


## 1. Baselines: diffusion only — 6 runs

| Run | Mod | What it is (objective, predictor, teacher, loss) | From | Ep | Own | Total | Val | d_sem full | d_bind full | d_sem shared | d_bind shared | d_sem private | d_bind private | text probe | img probe | img bind |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `image_dense_diffusion_1_2m_4e` | image | plain diffusion | scratch | 5 | 2.30B | **2.30B** | 3.6340 | — | — | — | — | — | — | — | 80.7% | 0.802 |
| `image_lora_diffusion_1_2m_12e_continued` | image | Tri-LoRA 256+128 | lora_diffusion_1_2m_4e | 4/12 ⏳ | 1.84B | **3.69B** | 3.6214 | — | — | — | — | — | — | — | 77.8% | 0.761 |
| `image_lora_diffusion_1_2m_4e` | image | Tri-LoRA 256+128 | scratch | 4 | 1.84B | **1.84B** | 3.9252 | — | — | — | — | — | — | — | 75.1% | 0.703 |
| `text_dense_diffusion_2m_4e_matched` | text | plain diffusion | scratch | 4 | 0.73B | **0.73B** | 1.0758 | +0.38 | +0.244 | — | — | — | — | 91.2% | — | — |
| `text_dense_diffusion_2m_8e_continued` | text | plain diffusion | dense_diffusion_2m_4e_matche | 4/8 ⏳ | 0.73B | **1.47B** | 1.0475 | +0.34 | +0.438 | — | — | — | — | — | — | — |
| `text_lora_diffusion_2m_4e_matched` | text | Tri-LoRA 256+128 | scratch | 4 | 0.73B | **0.73B** | 1.1638 | +0.47 | +0.293 | -0.98 | +0.019 | +0.27 | +0.009 | — | — | — |

## 2. Tri-LoRA with JEPA on adapter writes — 10 runs

| Run | Mod | What it is (objective, predictor, teacher, loss) | From | Ep | Own | Total | Val | d_sem full | d_bind full | d_sem shared | d_bind shared | d_sem private | d_bind private | text probe | img probe | img bind |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `text_data2vec_avg_gated_shared_calibrated_jepa_hsic_2m_4e` | text | modulewise on adapter writes, data2vec_average, **no predictor (direct regression)**, EMA teacher, normalized_mse; HSIC; Tri-LoRA 256+128 | scratch | 4 | 0.73B | **0.73B** | 1.1571 | -0.02 | +0.111 | — | — | — | — | — | — | — |
| `text_data2vec_noavg_gated_shared_calibrated_jepa_hsic_2m_4e` | text | modulewise on adapter writes, data2vec_no_average, **no predictor (direct regression)**, EMA teacher, normalized_mse; HSIC; Tri-LoRA 256+128 | scratch | 4 | 0.73B | **0.73B** | 1.1932 | +0.39 | +0.179 | — | — | — | — | — | — | — |
| `text_module_jepa_avg_hsic_2m_4e` | text | modulewise on adapter writes, average, **MLP predictor per module/layer**, EMA teacher, normalized_mse; HSIC; Tri-LoRA 256+128 | scratch | 4 | 0.73B | **0.73B** | 1.1613 | +0.14 | +0.108 | — | — | — | — | — | — | — |
| `text_module_jepa_avg_hsic_calibrated_2m_4e` | text | modulewise on adapter writes, average, **MLP predictor per module/layer**, EMA teacher, normalized_mse; HSIC; Tri-LoRA 256+128 | scratch | 4 | 0.73B | **0.73B** | 1.1489 | +0.09 | +0.102 | — | — | — | — | — | — | — |
| `text_module_jepa_avg_no_hsic_2m_4e` | text | modulewise on adapter writes, average, **MLP predictor per module/layer**, EMA teacher, normalized_mse; Tri-LoRA 256+128 | scratch | 4 | 0.73B | **0.73B** | 1.1435 | -0.07 | +0.147 | — | — | — | — | — | — | — |
| `text_module_jepa_layerwise_gated_hsic_2m_4e` | text | gated modulewise on adapter writes, layerwise, **MLP predictor per module/layer**, EMA teacher, normalized_mse; HSIC; Tri-LoRA 256+128 | scratch | 4 | 0.73B | **0.73B** | 1.1778 | +0.18 | +0.148 | -0.52 | +0.036 | -0.56 | +0.007 | — | — | — |
| `text_module_jepa_layerwise_gated_hsic_all_layers_2m_4e` | text | gated modulewise on adapter writes, layerwise, **MLP predictor per module/layer**, EMA teacher, normalized_mse; HSIC; Tri-LoRA 256+128 | scratch | 4 | 0.73B | **0.73B** | 1.1679 | -0.23 | +0.143 | — | — | — | — | — | — | — |
| `text_module_jepa_layerwise_gated_no_hsic_2m_4e` | text | gated modulewise on adapter writes, layerwise, **MLP predictor per module/layer**, EMA teacher, normalized_mse; Tri-LoRA 256+128 | scratch | 4 | 0.73B | **0.73B** | 1.1705 | -0.49 | +0.074 | — | — | — | — | — | — | — |
| `text_module_jepa_layerwise_hsic_2m_4e` | text | modulewise on adapter writes, layerwise, **MLP predictor per module/layer**, EMA teacher, normalized_mse; HSIC; Tri-LoRA 256+128 | scratch | 4 | 0.73B | **0.73B** | 1.1677 | +0.05 | +0.224 | -1.12 | +0.023 | -0.24 | +0.019 | — | — | — |
| `text_module_jepa_layerwise_no_hsic_2m_4e` | text | modulewise on adapter writes, layerwise, **MLP predictor per module/layer**, EMA teacher, normalized_mse; Tri-LoRA 256+128 | scratch | 4 | 0.73B | **0.73B** | 1.1595 | +0.03 | +0.117 | — | — | — | — | — | — | — |

## 3. data2vec on hidden states, from a pretrained trunk — 9 runs

| Run | Mod | What it is (objective, predictor, teacher, loss) | From | Ep | Own | Total | Val | d_sem full | d_bind full | d_sem shared | d_bind shared | d_sem private | d_bind private | text probe | img probe | img bind |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `image_data2vec_from_dense_avg_all_randt_1_2m_2e` | image | hidden-state data2vec, average, blocks 0-7, **MLP predictor ×1**, EMA teacher, SmoothL1 β=2.0; no diffusion | dense_diffusion_1_2m_4e | 2 | 0.92B | **3.23B** | 4.6375 | — | — | — | — | — | — | — | 78.2% | 0.710 |
| `image_data2vec_from_dense_avg_l4to7_randt_1_2m_2e` | image | hidden-state data2vec, average, blocks 4-7, **MLP predictor ×1**, EMA teacher, SmoothL1 β=2.0; no diffusion | dense_diffusion_1_2m_4e | 2 | 0.92B | **3.23B** | 4.9132 | — | — | — | — | — | — | — | 78.5% | 0.744 |
| `image_data2vec_from_dense_layerwise_all_block2d_1_2m_2e` | image | hidden-state data2vec, layerwise, blocks 0-7, **MLP predictor ×8**, EMA teacher, SmoothL1 β=2.0; no diffusion; 2D blocks | dense_diffusion_1_2m_4e | 2 | 0.92B | **3.23B** | 4.8729 | — | — | — | — | — | — | — | 81.2% | 0.822 |
| `image_data2vec_from_dense_layerwise_all_randt_1_2m_2e` | image | hidden-state data2vec, layerwise, blocks 0-7, **MLP predictor ×8**, EMA teacher, SmoothL1 β=2.0; no diffusion | dense_diffusion_1_2m_4e | 2 | 0.92B | **3.23B** | 4.8051 | — | — | — | — | — | — | — | 81.1% | 0.816 |
| `image_data2vec_from_dense_layerwise_l4to7_randt_1_2m_2e` | image | hidden-state data2vec, layerwise, blocks 4-7, **MLP predictor ×4**, EMA teacher, SmoothL1 β=2.0; no diffusion | dense_diffusion_1_2m_4e | 2 | 0.92B | **3.23B** | 4.8177 | — | — | — | — | — | — | — | 81.0% | 0.814 |
| `text_data2vec_from_dense_2m_2e` | text | hidden-state data2vec, average, top 8, **MLP predictor ×1**, EMA teacher, SmoothL1 β=2.0; no diffusion | dense_diffusion_2m_4e_matche | 2 | 0.37B | **1.10B** | 1.4488 | -0.76 | +0.293 | — | — | — | — | — | — | — |
| `text_data2vec_from_dense_avg_l4to7_2m_2e` | text | hidden-state data2vec, average, blocks 4-7, **MLP predictor ×1**, EMA teacher, SmoothL1 β=2.0; no diffusion | dense_diffusion_2m_4e_matche | 2 | 0.37B | **1.10B** | 1.5243 | +0.11 | +0.395 | — | — | — | — | — | — | — |
| `text_data2vec_from_dense_layerwise_all_2m_2e` | text | hidden-state data2vec, layerwise, blocks 0-7, **MLP predictor ×8**, EMA teacher, SmoothL1 β=2.0; no diffusion | dense_diffusion_2m_4e_matche | 2 | 0.37B | **1.10B** | 1.9656 | +0.75 | +0.527 | — | — | — | — | 89.4% | — | — |
| `text_data2vec_from_dense_layerwise_l4to7_2m_2e` | text | hidden-state data2vec, layerwise, blocks 4-7, **MLP predictor ×4**, EMA teacher, SmoothL1 β=2.0; no diffusion | dense_diffusion_2m_4e_matche | 2 | 0.37B | **1.10B** | 1.9885 | +0.66 | +0.445 | — | — | — | — | — | — | — |

## 4. data2vec on hidden states, from scratch — 11 runs

| Run | Mod | What it is (objective, predictor, teacher, loss) | From | Ep | Own | Total | Val | d_sem full | d_bind full | d_sem shared | d_bind shared | d_sem private | d_bind private | text probe | img probe | img bind |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `image_data2vec_scratch_block2d_30pct_avg_l4to7_1_2m_4e` | image | hidden-state data2vec, average, blocks 4-7, **MLP predictor ×1**, EMA teacher, SmoothL1 β=2.0; no diffusion; 2D blocks | scratch | 4 | 1.84B | **1.84B** | 8.5352 | — | — | — | — | — | — | — | 61.6% | 0.552 |
| `image_data2vec_scratch_block2d_30pct_layerwise_l4to7_1_2m_4e` | image | hidden-state data2vec, layerwise, blocks 4-7, **MLP predictor ×4**, EMA teacher, SmoothL1 β=2.0; no diffusion; 2D blocks | scratch | 4 | 1.84B | **1.84B** | 8.5394 | — | — | — | — | — | — | — | 60.4% | 0.535 |
| `image_data2vec_scratch_block2d_65pct_avg_l4to7_1_2m_4e` | image | hidden-state data2vec, average, blocks 4-7, **MLP predictor ×1**, EMA teacher, SmoothL1 β=2.0; no diffusion; 2D blocks | scratch | 2/4 ⏳ | 0.92B | **0.92B** | 8.5454 | — | — | — | — | — | — | — | — | — |
| `image_data2vec_scratch_block2d_65pct_layerwise_l4to7_1_2m_4e` | image | hidden-state data2vec, layerwise, blocks 4-7, **MLP predictor ×4**, EMA teacher, SmoothL1 β=2.0; no diffusion; 2D blocks | scratch | 1/4 ⏳ | 0.46B | **0.46B** | 8.5434 | — | — | — | — | — | — | — | — | — |
| `text_data2vec_scratch_layerwise_all_randt_2m_4e` | text | hidden-state data2vec, layerwise, blocks 0-7, **MLP predictor ×8**, EMA teacher, SmoothL1 β=2.0; no diffusion | scratch | 4 | 0.73B | **0.73B** | 8.9516 | -1.64 | +0.022 | — | — | — | — | 62.8% | — | — |
| `text_data2vec_scratch_window12_24_30pct_avg_l4to7_2m_4e` | text | hidden-state data2vec, average, blocks 4-7, **MLP predictor ×1**, EMA teacher, SmoothL1 β=2.0; no diffusion; windows 12+ | scratch | 4 | 0.73B | **0.73B** | 8.7964 | +0.42 | +0.198 | — | — | — | — | 85.8% | — | — |
| `text_data2vec_scratch_window12_24_30pct_layerwise_l4to7_2m_4e` | text | hidden-state data2vec, layerwise, blocks 4-7, **MLP predictor ×4**, EMA teacher, SmoothL1 β=2.0; no diffusion; windows 12+ | scratch | 4 | 0.73B | **0.73B** | 8.9335 | -1.72 | +0.085 | — | — | — | — | — | — | — |
| `text_data2vec_scratch_window12_24_60pct_avg_l4to7_2m_4e` | text | hidden-state data2vec, average, blocks 4-7, **MLP predictor ×1**, EMA teacher, SmoothL1 β=2.0; no diffusion; windows 12+ | scratch | 2/4 ⏳ | 0.37B | **0.37B** | 8.8103 | — | — | — | — | — | — | — | — | — |
| `text_data2vec_scratch_window12_24_60pct_layerwise_l4to7_2m_4e` | text | hidden-state data2vec, layerwise, blocks 4-7, **MLP predictor ×4**, EMA teacher, SmoothL1 β=2.0; no diffusion; windows 12+ | scratch | 2/4 ⏳ | 0.37B | **0.37B** | 8.9296 | — | — | — | — | — | — | — | — | — |
| `text_data2vec_scratch_window4_8_15pct_avg_l4to7_2m_4e` | text | hidden-state data2vec, average, blocks 4-7, **MLP predictor ×1**, EMA teacher, SmoothL1 β=2.0; no diffusion; windows 4+ | scratch | 4 | 0.73B | **0.73B** | 8.8497 | -0.53 | +0.248 | — | — | — | — | — | — | — |
| `text_data2vec_scratch_window4_8_15pct_layerwise_l4to7_2m_4e` | text | hidden-state data2vec, layerwise, blocks 4-7, **MLP predictor ×4**, EMA teacher, SmoothL1 β=2.0; no diffusion; windows 4+ | scratch | 4 | 0.73B | **0.73B** | 8.9406 | -1.27 | +0.081 | — | — | — | — | — | — | — |

## 5. SIGReg / LeJEPA instead of an EMA teacher — 4 runs

| Run | Mod | What it is (objective, predictor, teacher, loss) | From | Ep | Own | Total | Val | d_sem full | d_bind full | d_sem shared | d_bind shared | d_sem private | d_bind private | text probe | img probe | img bind |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `text_lejepa_from_dense_layerwise_all_2m_2e` | text | hidden-state data2vec, layerwise, blocks 0-7, **MLP predictor ×8**, **SIGReg** (no teacher, no stop-grad), SmoothL1 β=2.0; SIGReg λ=0.05; no diffusion | dense_diffusion_2m_4e_matche | 2 | 0.37B | **1.10B** | 7.2492 | -2.68 | +0.013 | — | — | — | — | — | — | — |
| `text_lejepa_scratch_layerwise_all_2m_4e` | text | hidden-state data2vec, layerwise, blocks 0-7, **MLP predictor ×8**, **SIGReg** (no teacher, no stop-grad), SmoothL1 β=2.0; SIGReg λ=0.05; no diffusion | scratch | 4 | 0.73B | **0.73B** | 8.8516 | -0.82 | +0.016 | — | — | — | — | — | — | — |
| `multimodal_unpaired_d2v_from_unpaired_dense_sigreg_1_2m_2e` | both | hidden-state data2vec, layerwise, blocks 0-7, **MLP predictor ×8**, **SIGReg** (no teacher, no stop-grad), SmoothL1 β=2.0; SIGReg λ=0.05; no diffusion | multimodal_unpaired_dense_1_ | 2 | 1.14B | **3.43B** | 8.4424 | -2.14 | +0.022 | — | — | — | — | 55.7% | 57.2% | — |
| `multimodal_unpaired_lejepa_scratch_1_2m_4e` | both | hidden-state data2vec, layerwise, blocks 0-7, **MLP predictor ×8**, **SIGReg** (no teacher, no stop-grad), SmoothL1 β=2.0; SIGReg λ=0.05; no diffusion | scratch | 4 | 2.28B | **2.28B** | 8.7888 | — | — | — | — | — | — | 54.7% | 53.5% | — |

## 6. Stage 2: dense shared route + private LoRA — 4 runs

| Run | Mod | What it is (objective, predictor, teacher, loss) | From | Ep | Own | Total | Val | d_sem full | d_bind full | d_sem shared | d_bind shared | d_sem private | d_bind private | text probe | img probe | img bind |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `text_dense_private_frozen_hsic_2m_2e` | text | HSIC; private rank 128, frozen trunk | data2vec_from_dense_2m_2e | 2 | 0.37B | **1.47B** | 1.0677 | -0.51 | +0.347 | -0.76 | +0.293 | -1.08 | +0.003 | 88.8% | — | — |
| `text_dense_private_frozen_hsic_from_d2v_lw_all_2m_2e` | text | HSIC; private rank 128, frozen trunk | data2vec_from_dense_layerwis | 2 | 0.37B | **1.47B** | 1.0617 | +0.10 | +0.529 | +0.75 | +0.527 | -1.54 | +0.024 | — | — | — |
| `text_dense_private_trainable_hsic_2m_2e` | text | HSIC; private rank 128, trunk trains | data2vec_from_dense_2m_2e | 2 | 0.37B | **1.47B** | 1.0648 | -0.24 | +0.396 | +0.36 | +0.433 | -1.74 | +0.014 | — | — | — |
| `text_dense_private_trainable_hsic_from_d2v_lw_all_2m_2e` | text | HSIC; private rank 128, trunk trains | data2vec_from_dense_layerwis | 2 | 0.37B | **1.47B** | 1.0553 | — | — | +0.63 | +0.555 | — | — | — | — | — |

## 7. Multimodal: paired and unpaired — 5 runs

| Run | Mod | What it is (objective, predictor, teacher, loss) | From | Ep | Own | Total | Val | d_sem full | d_bind full | d_sem shared | d_bind shared | d_sem private | d_bind private | text probe | img probe | img bind |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `multimodal_paired_dense_1_2m_4e` | both | plain diffusion | scratch | 4 | 2.28B | **2.28B** | 2.2596 | — | — | — | — | — | — | 86.0% | 82.4% | 0.792 |
| `multimodal_unpaired_d2v_from_image_dense_1_2m_2e` | both | hidden-state data2vec, layerwise, blocks 0-7, **MLP predictor ×8**, EMA teacher, SmoothL1 β=2.0; no diffusion | dense_diffusion_1_2m_4e | 2 | 1.14B | **3.45B** | 7.1726 | -0.93 | +0.010 | — | — | — | — | 58.6% | 80.3% | — |
| `multimodal_unpaired_d2v_from_text_dense_1_2m_2e` | both | hidden-state data2vec, layerwise, blocks 0-7, **MLP predictor ×8**, EMA teacher, SmoothL1 β=2.0; no diffusion | dense_diffusion_2m_4e_matche | 2 | 1.14B | **1.88B** | 9.4818 | +0.30 | +0.381 | — | — | — | — | 89.9% | 59.2% | — |
| `multimodal_unpaired_d2v_from_unpaired_dense_ema_1_2m_2e` | both | hidden-state data2vec, layerwise, blocks 0-7, **MLP predictor ×8**, EMA teacher, SmoothL1 β=2.0; no diffusion | multimodal_unpaired_dense_1_ | 2 | 1.14B | **3.43B** | 3.2485 | -3.19 | +0.021 | — | — | — | — | 68.1% | 78.7% | — |
| `multimodal_unpaired_dense_1_2m_4e` | both | plain diffusion | scratch | 4 | 2.28B | **2.28B** | 2.1990 | -1.46 | +0.007 | — | — | — | — | 73.2% | 79.8% | — |

## 8. Other corpora (MS-COCO) — 2 runs

| Run | Mod | What it is (objective, predictor, teacher, loss) | From | Ep | Own | Total | Val | d_sem full | d_bind full | d_sem shared | d_bind shared | d_sem private | d_bind private | text probe | img probe | img bind |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `text_diffusion_coco_one_caption_scratch_20e` | text | plain diffusion | scratch | 20 | 0.03B | **0.03B** | 5.6658 | — | — | — | — | — | — | — | — | — |
| `text_jepa_coco_one_caption_scratch` | text | hidden-state data2vec, average, blocks 4-7, **MLP predictor ×1**, EMA teacher, SmoothL1 β=4.0; no diffusion; windows 1+ | scratch | 8 | 0.01B | **0.01B** | 13.0522 | — | — | — | — | — | — | — | — | — |

## 9. Pilots and ablations (not yet evaluated) — 11 runs

| Run | Mod | What it is (objective, predictor, teacher, loss) | From | Ep | Own | Total | Val | d_sem full | d_bind full | d_sem shared | d_bind shared | d_sem private | d_bind private | text probe | img probe | img bind |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `text_jepa_scratch_faithful_bert15_k4` | text | hidden-state data2vec, average, blocks 4-7, **MLP predictor ×1**, EMA teacher, SmoothL1 β=4.0; no diffusion; windows 1+ | scratch | 2 | 0.37B | **0.37B** | 8.8749 | — | — | — | — | — | — | — | — | — |
| `text_jepa_scratch_faithful_bert15_k8` | text | hidden-state data2vec, average, blocks 0-7, **MLP predictor ×1**, EMA teacher, SmoothL1 β=4.0; no diffusion; windows 1+ | scratch | 2 | 0.37B | **0.37B** | 8.8886 | — | — | — | — | — | — | — | — | — |
| `text_jepa_scratch_faithful_span4_k8` | text | hidden-state data2vec, average, blocks 0-7, **MLP predictor ×1**, EMA teacher, SmoothL1 β=4.0; no diffusion; windows 4+ | scratch | 2 | 0.37B | **0.37B** | 8.8972 | — | — | — | — | — | — | — | — | — |
| `text_jepa_scratch_faithful_window30_k8` | text | hidden-state data2vec, average, blocks 0-7, **MLP predictor ×1**, EMA teacher, SmoothL1 β=4.0; no diffusion; windows 12+ | scratch | 2 | 0.37B | **0.37B** | 8.8831 | — | — | — | — | — | — | — | — | — |
| `text_jepa_scratch_fidelity_dprime_bert15_k4` | text | hidden-state data2vec, average, blocks 4-7, **MLP predictor ×1**, EMA teacher, SmoothL1 β=4.0; no diffusion; windows 1+ | scratch | 2 | 0.37B | **0.37B** | 8.8741 | — | — | — | — | — | — | — | — | — |
| `text_jepa_scratch_fidelity_dprime_bert15_k4_2m_4e` | text | hidden-state data2vec, average, blocks 4-7, **MLP predictor ×1**, EMA teacher, SmoothL1 β=4.0; no diffusion; windows 1+ | scratch | 4 | 0.73B | **0.73B** | 8.9042 | — | — | — | — | — | — | — | — | — |
| `text_jepa_scratch_fidelity_dprime_bert15_k8` | text | hidden-state data2vec, average, blocks 0-7, **MLP predictor ×1**, EMA teacher, SmoothL1 β=4.0; no diffusion; windows 1+ | scratch | 2 | 0.37B | **0.37B** | 8.9745 | — | — | — | — | — | — | — | — | — |
| `text_jepa_scratch_pilot_ema990` | text | hidden-state data2vec, average, blocks 4-7, **MLP predictor ×1**, EMA teacher, SmoothL1 β=2.0; no diffusion; windows 12+ | scratch | 2 | 0.37B | **0.37B** | 8.9060 | — | — | — | — | — | — | — | — | — |
| `text_jepa_scratch_pilot_ema990_global` | text | hidden-state data2vec, average, blocks 4-7, **MLP predictor ×1**, EMA teacher, SmoothL1 β=2.0; no diffusion; windows 12+ | scratch | 2 | 0.37B | **0.37B** | 8.9104 | — | — | — | — | — | — | — | — | — |
| `text_jepa_scratch_pilot_ema990_global_token025` | text | hidden-state data2vec, average, blocks 4-7, **MLP predictor ×1**, EMA teacher, SmoothL1 β=2.0; no diffusion; windows 12+ | scratch | 2 | 0.37B | **0.37B** | 8.9115 | — | — | — | — | — | — | — | — | — |
| `text_jepa_scratch_pilot_ema999` | text | hidden-state data2vec, average, blocks 4-7, **MLP predictor ×1**, EMA teacher, SmoothL1 β=2.0; no diffusion; windows 12+ | scratch | 2 | 0.37B | **0.37B** | 8.9122 | — | — | — | — | — | — | — | — | — |
