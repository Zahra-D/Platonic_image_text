# Gated data2vec runs: results and collapse diagnosis

> **Type:** evaluation, text-only study · **Status:** complete (both runs finished; final `epoch_003.pt`)  
> **Question:** what did removing the JEPA predictor and routing the JEPA gradient end-to-end do to the shared and private representations?  
> **Runs:** [plan and configurations](../plans/data2vec_gated_shared_lora_plan.md) · W&B [iw7140ry](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/iw7140ry) (average target), [rdb0athg](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/rdb0athg) (final-layer target)  
> **Menu:** [experiment catalog](../README.md)

## Summary

Both runs **collapsed in the one module the JEPA loss regresses**: the shared
LoRA map of `blocks.4.mlp.3` fell to effective rank 1.8 and 1.1 (healthy
layerwise JEPA: 45.7), and the private map at the same module collapsed with
it (4.0 and 1.7). The final-layer-target run also collapsed at block 4
`out_proj` and block 2 `mlp.3`. Every evaluation that reads that module failed: retrieval
fell to 3–11% R@10, binding to chance, and attribute probes far below healthy
models. The information is gone, not hidden: removing dominant directions
does not bring it back.

The rest of the network routed around the damaged modules. Blocks outside the
JEPA window (0–1 and 5–7) kept normal rank, the block-7 residual stream still
retrieves and binds like a healthy model, and diffusion validation loss was unaffected
(1.1571 for the average-target run vs. 1.1595 for healthy layerwise JEPA). A
falling diffusion loss therefore says nothing about the shared branch.

The apparent shared-semantic / private-template split in the
[counterfactual control](exact_template_counterfactual.md) is a consequence of
this collapse, not a success.

## Contents

[What changed](#what-changed-relative-to-the-original-modulewise-jepa) · [Headline](#headline-results) · [Collapse evidence](#evidence-of-collapse) · [Probes](#probes-on-the-collapsed-module) · [Why diffusion was unaffected](#why-diffusion-loss-was-unaffected) · [Why data2vec avoids this](#why-data2vec-itself-does-not-collapse) · [Follow-up](#follow-up) · [Artifacts](#artifacts)

## What changed relative to the original modulewise JEPA

| | Original modulewise JEPA | Gated data2vec |
|---|---|---|
| Loss function | `modulewise_shared_jepa_loss` | `modulewise_data2vec_loss` |
| Predictor | one MLP per (module, layer) | **none**: the student update itself is regressed |
| Loss | `normalized_mse` (direction only, = 2(1 − cos)) | **Smooth-L1, β = 4**, raw student vs. LayerNorm-ed teacher target |
| Student | every supervised layer 2–4 | layer 4 only |
| Target | same-layer or layer-averaged clean EMA teacher | mean of LayerNorm-ed teacher updates, layers 2–4 (or layer 4 only) |
| Gradient reach | layer-local: the recorder detaches the adapter input | **end-to-end** into shared `qkv`, `out_proj`, `mlp.0`, `mlp.3` of layers 2–4 |

Both runs also used calibrated HSIC (5% gradient ratio) and an adaptive JEPA
coefficient (5% ratio). Code: [shared_jepa.py](/home/zd25e122/clevr_discrete_diffusion/shared_jepa.py),
[train_multimodal.py](/home/zd25e122/clevr_discrete_diffusion/train_multimodal.py).

## Headline results

| Model | Validation loss | Retrieval R@10 shared / private | Binding shared / private (L4 `mlp.3`) | Residual block 7: retrieval R@1 / binding |
|---|---:|---:|---:|---:|
| Plain Tri-LoRA | 1.1638 | 22.9% / 25.5% | 62.7% / 62.6% | 63.8% / 78.5% |
| Layerwise JEPA, no HSIC | 1.1595 | 35.4% / 30.5% | 63.2% / 59.0% | 49.6% / 77.5% |
| **Gated data2vec, average target** | 1.1571 | 10.8% / 6.4% | 45.2% / 44.6% | 50.0% / 78.3% |
| **Gated data2vec, final-layer target** | 1.1932 | 3.2% / 4.3% | 50.8% / 51.0% | 59.9% / 76.6% |

Healthy JEPA reaches 35.4% shared R@10 and 63.2% shared binding at the same
module; the collapsed runs are at or near the random floor (3.0% R@10, 50%
binding). The average-target shared binding of 45.2% is significantly below
chance: its probe relies on a cue that reverses under the swap.

## Evidence of collapse

### 1. Rank of the LoRA maps

Entropy effective rank of `B·A`, **shared / private**, read directly from the
final checkpoints (maximum 256 shared, 128 private):

| Model | L2 `out_proj` | L2 `mlp.3` | L3 `out_proj` | L3 `mlp.3` | L4 `out_proj` | L4 `mlp.3` |
|---|---|---|---|---|---|---|
| Plain Tri-LoRA | 18.2 / 22.7 | 6.9 / 9.4 | 36.4 / 32.1 | 29.5 / 29.1 | 29.8 / 29.3 | 25.5 / 28.2 |
| Layerwise JEPA, no HSIC | 31.2 / 18.3 | 16.2 / 12.0 | 44.6 / 31.4 | 47.1 / 31.0 | 47.2 / 35.5 | 45.7 / 27.8 |
| Average JEPA, no HSIC | 21.8 / 26.0 | 30.2 / 28.9 | 34.7 / 33.6 | 29.2 / 30.9 | 55.7 / 28.7 | 87.3 / 29.0 |
| **Gated data2vec, average target** | 13.2 / 21.1 | 17.5 / 24.6 | 18.0 / 24.9 | 16.2 / 25.2 | 9.9 / 30.9 | 1.8 / 4.0 |
| **Gated data2vec, final-layer target** | 4.7 / 19.4 | 1.7 / 2.3 | 21.9 / 26.8 | 18.4 / 24.5 | 1.0 / 20.9 | 1.1 / 1.7 |

Across all eight blocks, `mlp.3`, shared / private:

| Model | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---|---|---|---|---|---|---|
| Layerwise JEPA, no HSIC | 20.1 / 25.7 | 9.6 / 12.2 | 16.2 / 12.0 | 47.1 / 31.0 | 45.7 / 27.8 | 26.1 / 24.6 | 31.8 / 36.6 | 19.7 / 16.4 |
| **Gated data2vec, average target** | 18.0 / 17.7 | 20.5 / 25.5 | 17.5 / 24.6 | 16.2 / 25.2 | 1.8 / 4.0 | 28.3 / 32.6 | 25.1 / 25.7 | 19.1 / 18.1 |
| **Gated data2vec, final-layer target** | 25.9 / 32.8 | 10.6 / 15.0 | 1.7 / 2.3 | 18.4 / 24.5 | 1.1 / 1.7 | 28.0 / 26.7 | 26.8 / 27.9 | 21.2 / 20.0 |

Collapse is confined to the JEPA window (blocks 2–4). The average-target run
collapsed only at block 4 `mlp.3` (shared 1.8, private 4.0). The final-layer
run collapsed more widely: block 4 `out_proj` and `mlp.3` (shared 1.0 and
1.1) and also block 2 `mlp.3` (shared 1.7, private 2.3). In every case the
private branch of a collapsed module collapsed with it. Blocks 0–1 and 5–7
are normal in both runs.

### 2. Geometry of the pooled features

On the 2,048-caption retrieval gallery (content-token mean, L2-normalized
`blocks.4.mlp.3` update):

| Model | Shared mean cosine (sd) | Shared effective rank | Private mean cosine (sd) | Private effective rank |
|---|---:|---:|---:|---:|
| Plain Tri-LoRA | 0.883 (0.061) | 14.9 | 0.887 (0.053) | 17.3 |
| Layerwise JEPA, no HSIC | 0.794 (0.105) | 16.8 | 0.890 (0.057) | 17.3 |
| Average JEPA, no HSIC | 0.690 (0.115) | 27.7 | 0.780 (0.090) | 20.8 |
| **Gated data2vec, average target** | 0.989 (0.013) | 2.9 | 0.642 (0.378) | 7.2 |
| **Gated data2vec, final-layer target** | 0.156 (0.827) | 1.8 | 0.361 (0.391) | 5.5 |

The average-target shared update has nearly one direction for every caption
(mean cosine 0.989). The final-layer-target shared update is also rank ≈ 2,
but split into two opposite groups, so its mean cosine is near 0 with a very
large spread.

### 3. The information is gone, not hidden

Cross-pattern retrieval MRR after centering and removing dominant principal
components:

| Model | Route | Centered | Centered, top PC removed | Centered, top 2 PCs removed |
|---|---|---:|---:|---:|
| Plain Tri-LoRA | shared | 0.111 | 0.105 | 0.111 |
|  | private | 0.136 | 0.099 | 0.100 |
| Layerwise JEPA, no HSIC | shared | 0.195 | 0.239 | 0.208 |
|  | private | 0.165 | 0.154 | 0.130 |
| Average JEPA, no HSIC | shared | 0.156 | 0.134 | 0.116 |
|  | private | 0.102 | 0.118 | 0.097 |
| **Gated data2vec, average target** | shared | 0.056 | 0.052 | 0.039 |
|  | private | 0.039 | 0.038 | 0.037 |
| **Gated data2vec, final-layer target** | shared | 0.021 | 0.022 | 0.025 |
|  | private | 0.023 | 0.024 | 0.022 |

Removing the top component helps a healthy model (layerwise JEPA shared:
0.195 → 0.239) but does nothing for the collapsed runs.

## Probes on the collapsed module

### Native-update probe, clean held-out captions

Linear probes on each native update, trained on 4,096 clean training captions
and scored on 2,048 clean validation captions (balanced accuracy; count is
8-way, attributes average five binary groups):

| Model | Shared count | Shared attributes | Private count | Private attributes |
|---|---:|---:|---:|---:|
| Plain Tri-LoRA | 0.611 | 0.883 | 0.614 | 0.873 |
| Layerwise JEPA, no HSIC | 0.682 | 0.928 | 0.598 | 0.899 |
| Layerwise JEPA + HSIC | 0.609 | 0.909 | 0.578 | 0.892 |
| **Gated data2vec, average target** | 0.558 | 0.697 | 0.544 | 0.694 |
| **Gated data2vec, final-layer target** | 0.279 | 0.601 | 0.261 | 0.613 |
| Random features | 0.159 | 0.506 | — | — |

Shared attribute probes at the neighbouring modules of the same runs, where
nothing collapsed:

| Model | Shared L2 `mlp.3` | Shared L3 `mlp.3` | Shared L4 `out_proj` | Shared L4 `mlp.3` |
|---|---:|---:|---:|---:|
| Plain Tri-LoRA | 0.823 | 0.890 | 0.882 | 0.883 |
| Layerwise JEPA, no HSIC | 0.855 | 0.918 | 0.865 | 0.928 |
| Layerwise JEPA + HSIC | 0.827 | 0.917 | 0.868 | 0.909 |
| **Gated data2vec, average target** | 0.898 | 0.902 | 0.847 | 0.697 |
| **Gated data2vec, final-layer target** | 0.830 | 0.882 | 0.881 | 0.601 |

The damage is local: block 2 `mlp.3` of the average-target run is the best of
all models (0.898). Note that attribute probes stay above chance even on the
collapsed module; a standardized linear probe can amplify tiny residual
variation that cosine retrieval cannot use. Ordinary attribute probes are also
largely solvable from word content (see the
[binding-swap evaluation](binding_swap_evaluation.md)), which is why the
binding result above, at chance, is the more decisive one.

### Standard modality probe (per-layer average of all four shared deltas)

Block 4; clean in-sample fit, 80%-masked held-out test, and masked-vs-clean
cosine stability:

| Model | Clean fit, shared / private | 80%-masked held-out, shared / private | Masked–clean cosine, shared / private |
|---|---|---|---|
| Plain Tri-LoRA | 0.960 / 0.961 | 0.573 / 0.577 | 0.707 / 0.722 |
| Layerwise JEPA, no HSIC | 0.980 / 0.977 | 0.592 / 0.616 | 0.782 / 0.637 |
| Layerwise JEPA + HSIC | 0.972 / 0.970 | 0.546 / 0.569 | 0.770 / 0.607 |
| **Gated data2vec, average target** | 0.936 / 0.939 | 0.573 / 0.572 | 0.997 / 0.999 |
| **Gated data2vec, final-layer target** | 0.923 / 0.929 | 0.548 / 0.603 | 0.993 / 0.999 |

This standard probe **hides the collapse** and even rewards it. Averaging all
four adapters keeps the clean fit high because `qkv` and `mlp.0` did not
collapse, the 80%-masked test is near chance for every model, and a constant
output looks perfectly stable (masked–clean cosine 0.997–0.999). Do not use
masked–clean stability as evidence of successful JEPA training.

Script: [evaluate_native_update_probes.py](/home/zd25e122/clevr_discrete_diffusion/evaluate_native_update_probes.py);
standard probe: [evaluate_jepa_modality.py](/home/zd25e122/clevr_discrete_diffusion/evaluate_jepa_modality.py).

## Why diffusion loss was unaffected

- Tri-LoRA here has **no frozen base weight**: each adapted linear layer
  outputs shared + private + bias. Where both branches collapsed, that layer
  adds almost nothing.
- The residual stream carries everything around it. Blocks 5–7 are intact,
  and the block-7 residual stream still retrieves at 50.0–59.9% R@1 (healthy
  layerwise JEPA 49.6%, plain Tri-LoRA 63.8%) and binds at 76.6–78.3% (77.5%
  and 78.5%).
- Nothing in the denoising objective requires that particular module, so its
  collapse is free and invisible in the training loss.

## Why data2vec itself does not collapse

data2vec regresses the model's main representation, which the network cannot
route around; it keeps a prediction head; and its target normalization is
designed to discourage constant outputs. Here the regressed quantity is one
additive LoRA update among 32 adapted layers, the predictor and the
direction-only loss were both removed, and the target was normalized per token
over features, which still allows every token to share one vector. HSIC adds
pressure in the same direction, because HSIC(shared, private) is exactly zero
when shared is constant.

## Follow-up

The [gated-predictor layerwise JEPA runs](../plans/gated_predictor_layerwise_jepa.md)
keep the end-to-end gradient routing but restore the predictor and the
`normalized_mse` loss, use same-layer targets, and log the effective rank of
every supervised adapter every 250 steps so that collapse is visible early.

## Artifacts

- Adapter ranks, geometry, and principal-component tests are regenerated from
  the checkpoints and `outputs/shared_private_semantic_template_retrieval/<label>/features.pt`.
- Native-update probes: `outputs/native_update_probes_final/results.json`
- Standard modality probe: `outputs/collapsed_shared_final_modality_probe/results.json`
- Retrieval: [cross-pattern semantic retrieval](cross_pattern_semantic_retrieval.md);
  binding: [binding-swap evaluation](binding_swap_evaluation.md);
  counterfactual: [exact-template counterfactual](exact_template_counterfactual.md)
