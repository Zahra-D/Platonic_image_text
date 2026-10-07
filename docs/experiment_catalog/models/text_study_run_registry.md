# Text-only study: run registry

> **Type:** model registry · **Status:** 43 text runs with checkpoints, 4 stopped without · **Updated:** 2026-09-24  
> **Menu:** [experiment catalog](../README.md)

Every run in the text-only shared/private study. **For the generated
all-families index with token budgets and every evaluation, see
[every experiment](../all_experiments.md);** this page keeps the W&B links,
config paths and per-family notes. Shared setup: 8-block
Transformer, width 384, learned absolute positions, no modality embeddings;
Tri-LoRA runs use no frozen base weight, rank 384 (shared 256, text-private
128), adapters on `qkv`, `out_proj`, `mlp.0`, `mlp.3`. Data: 2M human-style
CLEVR captions, 4 epochs, microbatch 64 × 4 accumulation, seed 20260915. All
runs log to the W&B project
[Platonic_CLEVR_modulewise_ema_jepa_hsic](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic).

Validation loss is masked-token loss at fixed mask ratio t = 0.75 on 2,048
held-out captions, from the latest completed epoch. Checkpoints are in each
output directory as `epoch_00N.pt`, `best.pt`, and `last.pt`.

## Controls

| Run | What differs | Status | Epochs done | Validation loss | W&B | Config | Output |
|---|---|---|---:|---:|---|---|---|
| Dense | Dense Transformer, diffusion only | finished | 4 / 4 | 1.0758 (epoch 3) | [4v57nzhs](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/4v57nzhs) | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_dense_diffusion_2m_4e_matched.yaml) | `outputs/text_dense_diffusion_2m_4e_matched/` |
| Plain Tri-LoRA | No-base Tri-LoRA, diffusion only | finished | 4 / 4 | 1.1638 (epoch 3) | [4q137e90](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/4q137e90) | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_lora_diffusion_2m_4e_matched.yaml) | `outputs/text_lora_diffusion_2m_4e_matched/` |

## Original modulewise JEPA sweep ([plan](../plans/modulewise_jepa_hsic_four_run_plan.md))

| Run | What differs | Status | Epochs done | Validation loss | W&B | Config | Output |
|---|---|---|---:|---:|---|---|---|
| Average JEPA, no HSIC | Predictor; layer-averaged target; layer-4 student | finished | 4 / 4 | 1.1435 (epoch 3) | [xzskwqhp](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/xzskwqhp) | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_module_jepa_avg_no_hsic_2m_4e.yaml) | `outputs/text_module_jepa_avg_no_hsic_2m_4e/` |
| Average JEPA + HSIC | As above + adaptive HSIC (max weight 1) | finished | 4 / 4 | 1.1613 (epoch 3) | [6nhfkmyl](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/6nhfkmyl) | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_module_jepa_avg_hsic_2m_4e.yaml) | `outputs/text_module_jepa_avg_hsic_2m_4e/` |
| Layerwise JEPA, no HSIC | Predictor; same-layer target at blocks 2–4 | finished | 4 / 4 | 1.1595 (epoch 3) | [muq3kc5f](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/muq3kc5f) | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_module_jepa_layerwise_no_hsic_2m_4e.yaml) | `outputs/text_module_jepa_layerwise_no_hsic_2m_4e/` |
| Layerwise JEPA + HSIC | As above + adaptive HSIC (max weight 1) | finished | 4 / 4 | 1.1677 (epoch 3) | [sxpwb1cb](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/sxpwb1cb) | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_module_jepa_layerwise_hsic_2m_4e.yaml) | `outputs/text_module_jepa_layerwise_hsic_2m_4e/` |
| Calibrated average JEPA + HSIC | Average JEPA + HSIC, max HSIC weight 50 | finished | 4 / 4 | 1.1489 (epoch 3) | [ayxc635a](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/ayxc635a) | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_module_jepa_avg_hsic_calibrated_2m_4e.yaml) | `outputs/text_module_jepa_avg_hsic_calibrated_2m_4e/` |

## All-shared-module JEPA ([plan](../plans/modulewise_jepa_all_shared_correction.md))

| Run | What differs | Status | Epochs done | Validation loss | W&B | Config | Output |
|---|---|---|---:|---:|---|---|---|
| All-shared average JEPA + HSIC | JEPA on all four adapter types | stopped (W&B: crashed) | 0 / 4 | — | [s5e1xnia](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/s5e1xnia) | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_module_jepa_avg_all_shared_hsic_calibrated_2m_4e.yaml) | `outputs/text_module_jepa_avg_all_shared_hsic_calibrated_2m_4e/` |
| All-shared layerwise JEPA + HSIC | JEPA on all four adapter types | stopped (W&B: crashed) | 0 / 4 | — | [ooqd8dg4](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/ooqd8dg4) | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_module_jepa_layerwise_all_shared_hsic_calibrated_2m_4e.yaml) | `outputs/text_module_jepa_layerwise_all_shared_hsic_calibrated_2m_4e/` |

## Gated data2vec ([plan](../plans/data2vec_gated_shared_lora_plan.md))

| Run | What differs | Status | Epochs done | Validation loss | W&B | Config | Output |
|---|---|---|---:|---:|---|---|---|
| Gated data2vec, average (fixed coefficient) | No predictor; Smooth-L1; end-to-end gradient | stopped (W&B: crashed) | 0 / 4 | — | [kln1evt2](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/kln1evt2) | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_data2vec_avg_gated_shared_hsic_2m_4e.yaml) | `outputs/text_data2vec_avg_gated_shared_hsic_2m_4e/` |
| Gated data2vec, final layer (fixed coefficient) | No predictor; Smooth-L1; end-to-end gradient | stopped (W&B: crashed) | 0 / 4 | — | [dk7bvhm5](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/dk7bvhm5) | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_data2vec_noavg_gated_shared_hsic_2m_4e.yaml) | `outputs/text_data2vec_noavg_gated_shared_hsic_2m_4e/` |
| Gated data2vec, average target | Calibrated JEPA and HSIC coefficients | finished, **collapsed** | 4 / 4 | 1.1571 (epoch 3) | [iw7140ry](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/iw7140ry) | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_data2vec_avg_gated_shared_calibrated_jepa_hsic_2m_4e.yaml) | `outputs/text_data2vec_avg_gated_shared_calibrated_jepa_hsic_2m_4e/` |
| Gated data2vec, final-layer target | Calibrated JEPA and HSIC coefficients | finished, **collapsed** | 4 / 4 | 1.1932 (epoch 3) | [rdb0athg](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/rdb0athg) | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_data2vec_noavg_gated_shared_calibrated_jepa_hsic_2m_4e.yaml) | `outputs/text_data2vec_noavg_gated_shared_calibrated_jepa_hsic_2m_4e/` |

## Gated-predictor layerwise JEPA ([plan](../plans/gated_predictor_layerwise_jepa.md))

| Run | What differs | Status | Epochs done | Validation loss | W&B | Config | Output |
|---|---|---|---:|---:|---|---|---|
| Gated layerwise JEPA, no HSIC | Predictor; same-layer target; end-to-end gradient | finished | 4 / 4 | 1.1705 (epoch 3) | [itm48bxb](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/itm48bxb) | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_module_jepa_layerwise_gated_no_hsic_2m_4e.yaml) | `outputs/text_module_jepa_layerwise_gated_no_hsic_2m_4e/` |
| Gated layerwise JEPA + HSIC | As above + adaptive HSIC | finished, **best JEPA variant** | 4 / 4 | 1.1778 (epoch 3) | [y0eklen2](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/y0eklen2) | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_module_jepa_layerwise_gated_hsic_2m_4e.yaml) | `outputs/text_module_jepa_layerwise_gated_hsic_2m_4e/` |

## Dense shared route, rank-128 private LoRA ([plan](../plans/dense_shared_private_lora_plan.md))

These runs leave the dense weight in place as the shared route and rank-limit
only the private branch, so their parameterization differs from every run
above. All three train on the same 2M captions, for 2 epochs, starting from the
dense diffusion checkpoint; W&B project
[Platonic_CLEVR_data2vec_faithful](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_data2vec_faithful).

| Run | What differs | JEPA mode | Status | Epochs done | Validation loss | Config | Output |
|---|---|---|---|---:|---:|---|---|
| Stage 1: faithful data2vec | Dense; data2vec only, `diffusion.weight: 0`; init from the dense diffusion checkpoint | averaged at the end (final block predicts the mean of all 8 LayerNorm-ed teacher blocks) | finished; no binding gain over dense | 2 / 2 | 1.3232 (epoch 0), 1.4488 (epoch 1) | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_data2vec_from_dense_2m_2e.yaml) | `outputs/text_data2vec_from_dense_2m_2e/` |
| Stage 2A: frozen trunk | Dense trunk frozen; rank-128 private adapters; diffusion + HSIC | none (frozen trunk) | finished | 2 / 2 | **1.0433** (epoch 0) | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_dense_private_frozen_hsic_2m_2e.yaml) | `outputs/text_dense_private_frozen_hsic_2m_2e/` |
| Stage 2B: trainable trunk | As 2A but the dense trunk keeps training | none | finished | 2 / 2 | 1.0480 (epoch 0) | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_dense_private_trainable_hsic_2m_2e.yaml) | `outputs/text_dense_private_trainable_hsic_2m_2e/` |

### Stage-1 variants (averaged vs. layerwise targets, [results](../plans/dense_shared_private_lora_plan.md#stage-1-variants-averaged-vs-layerwise-targets))

| Run | Target | Blocks | Status | Validation loss | Config | Output |
|---|---|---|---|---:|---|---|
| avg 4-7 | average | 4-7 | finished; no better than all-8 | 1.3625 / 1.5243 | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_data2vec_from_dense_avg_l4to7_2m_2e.yaml) | `outputs/text_data2vec_from_dense_avg_l4to7_2m_2e/` |
| layerwise 4-7 | each block's own | 4-7 | finished; **best binding of the stage-1 variants** | 1.6409 / 1.9885 | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_data2vec_from_dense_layerwise_l4to7_2m_2e.yaml) | `outputs/text_data2vec_from_dense_layerwise_l4to7_2m_2e/` |
| layerwise all | each block's own | 0-7 | running | 1.6065 (epoch 0) | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_data2vec_from_dense_layerwise_all_2m_2e.yaml) | `outputs/text_data2vec_from_dense_layerwise_all_2m_2e/` |

### From-scratch runs (no pretrained initialization, JEPA only)

All start from random weights with `diffusion.weight: 0`, so their validation
loss is the uniform-guess level (ln(682)/0.75 = 8.70) and means nothing: with no
reconstruction term the output head never receives gradient. Judge them by
[d′](../evaluations/semantic_dprime.md).

| Run | Target | Masking | Epochs | `d_semantic` (L7) | best `d_binding` |
|---|---|---|---:|---:|---:|
| `text_data2vec_scratch_window12_24_30pct_avg_l4to7_2m_4e` | averaged 4-7 | windows 12-24, 30% | 4 | **+0.42** | +0.198 |
| `text_data2vec_scratch_window12_24_30pct_layerwise_l4to7_2m_4e` | layerwise 4-7 | windows 12-24, 30% | 4 | −1.72 | +0.085 |
| `text_data2vec_scratch_window4_8_15pct_avg_l4to7_2m_4e` | averaged 4-7 | windows 4-8, 15% | 4 | −0.53 | +0.248 |
| `text_data2vec_scratch_window4_8_15pct_layerwise_l4to7_2m_4e` | layerwise 4-7 | windows 4-8, 15% | 4 | −1.27 | +0.081 |
| `text_data2vec_scratch_layerwise_all_randt_2m_4e` | layerwise 0-7 | random-t (matches the from-dense runs) | 4 | −1.64 | +0.022 |

The last row is the control that isolates initialization: it uses the *identical*
masking of the best from-dense run, and still collapses. Masking was not the
variable.

### Stage 2: dense trunk as the shared route, rank-128 private LoRA

Diffusion loss plus HSIC between each module's dense (shared) write and its
private write at `out_proj` and `mlp.3` in all 8 blocks, adaptive coefficient
calibrated to a 5% gradient ratio. Frozen runs train **only** the private
adapters: 12,582,912 of 27,452,458 parameters, one optimizer group.

| Run | Stage-1 trunk | Trunk | Val loss (ep0 / ep1) | `d_semantic` (L7) | best `d_binding` |
|---|---|---|---:|---:|---:|
| `text_dense_private_frozen_hsic_2m_2e` | averaged all-8 | frozen | **1.0433** / 1.0677 | −0.33 | `L6.mlp_shared` +0.356 |
| `text_dense_private_trainable_hsic_2m_2e` | averaged all-8 | trains | 1.0480 / 1.0648 | +0.20 | `L6.mlp_shared` +0.426 |
| `text_dense_private_frozen_hsic_from_d2v_lw_all_2m_2e` | **layerwise all-8** | frozen | 1.0440 / 1.0617 | **+0.41** | `L5.mlp_shared` **+0.508** |

The third run is the combination worth keeping: the trunk with the best
semantics, a clean shared-over-private separation, and reconstruction equal to
the others.

### Compute-matched control

| Run | Total epochs | Val loss |
|---|---:|---:|
| `text_dense_diffusion_2m_4e_matched` | 4 | 1.0758 |
| `text_dense_diffusion_2m_8e_continued` | 8 | **1.0475** (epochs 5-8: 1.0713, 1.0520, 1.0555, 1.0475) |

Stage 2 has seen eight epochs of data (four as the dense baseline, two of stage-1
data2vec, two of stage 2), so comparing it against the four-epoch baseline
overstates it. Continued to eight epochs the dense model reaches 1.0475 against
stage 2's best of 1.0433 — a 0.004 difference rather than 0.032. The surviving
claim is that a frozen trunk plus a rank-128 private branch *matches* full dense
training while updating 46% of the parameters.

## data2vec on hidden states, from the dense trunk ([plan](../plans/dense_shared_private_lora_plan.md#stage-1-variants-averaged-vs-layerwise-targets))

Stage 1: take the trained dense checkpoint and continue with the data2vec
hidden-state objective and **no diffusion loss**, for 2 epochs (0.37B tokens on
top of the parent's 0.73B = 1.10B total). Four target designs.

| Run | Target design | Predictor heads | Val | best `d_bind` |
|---|---|---:|---:|---|
| `text_data2vec_from_dense_2m_2e` | averaged, top 8 | 1 | 1.4488 | +0.293 |
| `text_data2vec_from_dense_avg_l4to7_2m_2e` | averaged, blocks 4–7 | 1 | 1.5243 | +0.395 |
| `text_data2vec_from_dense_layerwise_l4to7_2m_2e` | layerwise, blocks 4–7 | 4 | 1.9885 | +0.445 |
| **`text_data2vec_from_dense_layerwise_all_2m_2e`** | **layerwise, blocks 0–7** | **8** | 1.9656 | **+0.527** |

Validation loss is *worse* for the better representations — these runs have
`diffusion.weight: 0`, so the number measures drift from a frozen output head,
not quality. Budget-matched dense (6 epochs, 1.10B) reaches +0.443, so the best
row is a **+19%** gain, not the "more than double" figure an earlier version of
this catalog reported against the 4-epoch baseline.

## data2vec from scratch — five runs, all collapsed

No pretrained initialization, `diffusion.weight: 0`, 4 epochs (0.73B).

| Run | Masking | Target | best `d_bind` |
|---|---|---|---|
| `text_data2vec_scratch_layerwise_all_randt_2m_4e` | random-t (identical to the from-dense run) | layerwise ×8 | +0.022 |
| `text_data2vec_scratch_window4_8_15pct_avg_l4to7_2m_4e` | 1-D windows 4–8, 15% | averaged | +0.248 |
| `text_data2vec_scratch_window4_8_15pct_layerwise_l4to7_2m_4e` | 1-D windows 4–8, 15% | layerwise ×4 | +0.081 |
| `text_data2vec_scratch_window12_24_30pct_avg_l4to7_2m_4e` | 1-D windows 12–24, 30% | averaged | +0.198 |
| `text_data2vec_scratch_window12_24_30pct_layerwise_l4to7_2m_4e` | 1-D windows 12–24, 30% | layerwise ×4 | +0.085 |

Untrained floor is +0.004. **Masking was not the variable** — the random-t run
uses exactly the masking of the from-dense run that reaches +0.527 — and the
averaged/layerwise ordering *reverses* from scratch, which is the clearest sign
that these runs are not doing the same thing as their from-dense counterparts.

Two harder-masking runs are in flight at ~61% realized masking
(`text_data2vec_scratch_window12_24_60pct_{avg,layerwise}_l4to7_2m_4e`), testing
the diagnosis that the objective converges in one epoch because it is too easy.

## SIGReg / LeJEPA ([survey](../jepa_model_survey.md#sigreg--lejepa))

| Run | From | Budget | Val | `d_sem` L7 | best `d_bind` |
|---|---|---:|---:|---:|---|
| `text_lejepa_from_dense_layerwise_all_2m_2e` | dense | 1.10B | 7.2492 | **−2.68** | +0.013 |
| `text_lejepa_scratch_layerwise_all_2m_4e` | scratch | 0.73B | 8.8516 | −0.82 | +0.016 |

Both collapse at λ = 0.05; the from-dense run ends below the bag-of-words
control (−2.53) on `d_semantic`, i.e. the penalty destroyed the trunk it started
from. λ was never swept.

## Where each run is evaluated

| Evaluation | Final checkpoints evaluated | Epoch 0 evaluated |
|---|---|---|
| [Binding-swap evaluation](../evaluations/binding_swap_evaluation.md) | dense, plain, all five sweep runs, both calibrated gated data2vec, both gated-predictor runs | dense, plain, both layerwise baselines, both gated-predictor runs |
| [Cross-pattern semantic retrieval](../evaluations/cross_pattern_semantic_retrieval.md) | dense, plain, all five sweep runs, both calibrated gated data2vec, both gated-predictor runs | plain, all five sweep runs, both gated-predictor runs |
| [Exact-template counterfactual](../evaluations/exact_template_counterfactual.md) | plain, all five sweep runs, both calibrated gated data2vec, both gated-predictor runs | — |
| [Gated data2vec collapse diagnosis](../evaluations/data2vec_gated_collapse_diagnosis.md) | plain, layerwise and average JEPA, both calibrated gated data2vec | — |
| **[Semantic effect sizes (d′)](../evaluations/semantic_dprime.md)** | dense (4/6/8 ep), plain Tri-LoRA, all five sweep runs, both gated data2vec, both gated-predictor runs, all four from-dense data2vec, all five from-scratch, both LeJEPA, all four stage-2 cells, six external pretrained models | stage-2 runs at both epochs |
| **[d′ with the private route suppressed](../evaluations/semantic_dprime.md#stage-2-dense-trunk-as-shared-route--private-lora)** | all four stage-2 cells, plain Tri-LoRA, gated layerwise + HSIC, layerwise + HSIC | — |
| **[d′ with the shared route suppressed](../evaluations/semantic_dprime.md#stage-2-dense-trunk-as-shared-route--private-lora)** | the same six models | — |
| [Cross-modal structure](../evaluations/cross_modal_structure.md) | dense, `d2v_lw_all`, stage-2 frozen, two from-scratch runs | — |
