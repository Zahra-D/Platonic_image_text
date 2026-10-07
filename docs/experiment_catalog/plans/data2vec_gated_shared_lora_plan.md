# Direct data2vec-style EMA-JEPA with gated shared-LoRA gradients

> **Type:** experiment plan, text-only study · **Status:** complete. The two calibrated runs finished four epochs; the two fixed-coefficient runs were stopped without checkpoints.  
> **Outcome:** both finished runs **collapsed** at the supervised module. See [results and collapse diagnosis](../evaluations/data2vec_gated_collapse_diagnosis.md).  
> **Menu:** [experiment catalog](../README.md) · **Run registry:** [text-study runs](../models/text_study_run_registry.md)

## Objective

This is the corrected implementation of the requested objective. It is
data2vec-style, but applies it to native shared LoRA updates rather than whole
Transformer hidden states.

For each masked token, the clean EMA teacher supplies native shared updates at
`out_proj` and `mlp.3`. The masked student uses the corresponding native update
at final selected layer 4 directly: there is **no JEPA predictor**, projection,
or MLP between student and target.

The average-target condition constructs, separately for each module (m),

\[
y_m=\frac{1}{3}\sum_{l=2}^{4}\operatorname{LayerNorm}
 (\Delta^{\mathrm{EMA}}_{s,l,m}(x_{\mathrm{clean}})).
\]

It minimizes masked-position Smooth-L1 loss between this target and
\(\Delta_{s,4,m}(x_{\mathrm{masked}})\). The no-average control instead uses
only the clean teacher update from layer 4.

## Gradient contract

The student native update retains its real forward graph. The training loop
calculates the JEPA gradient and applies it only to shared `A`/`B` LoRA matrices
for `qkv`, `out_proj`, `mlp.0`, and `mlp.3` at layers 2--4. Thus layer-4 JEPA
signal reaches all selected lower shared LoRAs end-to-end, while private LoRAs,
embeddings, backbone parameters, and the EMA teacher receive **no JEPA
gradient**. Ordinary diffusion and HSIC retain their own stated routing.

HSIC remains on native shared/text-private `out_proj` and `mlp.3` updates at
layers 2--4, calibrated to a 5% gradient ratio with a maximum multiplier of 50.

## Matched runs

| Condition | Teacher target | W&B | Output |
|---|---|---|---|
| Average (stopped, no checkpoint) | Mean of normalized clean EMA layers 2--4 | [kln1evt2](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/kln1evt2) | `outputs/text_data2vec_avg_gated_shared_hsic_2m_4e/` |
| No-average (stopped, no checkpoint) | Normalized clean EMA layer 4 only | [dk7bvhm5](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/dk7bvhm5) | `outputs/text_data2vec_noavg_gated_shared_hsic_2m_4e/` |

- Average config: [text_data2vec_avg_gated_shared_hsic_2m_4e.yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_data2vec_avg_gated_shared_hsic_2m_4e.yaml)
- No-average config: [text_data2vec_noavg_gated_shared_hsic_2m_4e.yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_data2vec_noavg_gated_shared_hsic_2m_4e.yaml)

The previously launched all-module direct-prediction jobs were stopped before
completion because they did not implement this end-to-end routing. The older
`out_proj`/`mlp.3` checkpoints remain narrow module-local ablations.

## Calibrated-JEPA replacements

The first direct runs used a fixed JEPA coefficient of 0.5. Their average-target
JEPA gradient was too weak relative to diffusion, so they were stopped without
checkpoints and replaced by the following matched adaptive-coefficient runs.
They target a pre-warmup JEPA/shared-diffusion gradient ratio of 5%, with a
0.95 EMA, bounds `[1e-4, 50]`, and the same 1,000-step warmup.

| Condition | W&B | Output |
|---|---|---|
| Average teacher target, calibrated JEPA (finished; final loss 1.1571) | [iw7140ry](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/iw7140ry) | `outputs/text_data2vec_avg_gated_shared_calibrated_jepa_hsic_2m_4e/` |
| Final-teacher-layer control, calibrated JEPA (finished; final loss 1.1932) | [rdb0athg](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/rdb0athg) | `outputs/text_data2vec_noavg_gated_shared_calibrated_jepa_hsic_2m_4e/` |

At step 250, when JEPA warmup is 25%, the average run's pre-warmup calibrated
ratio is 4.5%; the no-average run is temporarily 8.0% while its coefficient
EMA settles. Per-layer effective ratios are expected to differ because one
coefficient calibrates the combined selected shared parameter set.

## Results

Both calibrated runs collapsed in the module the JEPA loss regresses
(`blocks.4.mlp.3` shared effective rank 1.8 and 1.1; healthy layerwise JEPA
45.7). Retrieval, binding, and attribute probes at that module fell to or near
chance, while diffusion validation loss and the block-7 residual stream were
unaffected. Full results, evidence, and explanation:
[gated data2vec results and collapse diagnosis](../evaluations/data2vec_gated_collapse_diagnosis.md).

The follow-up keeps the gated gradient but restores the predictor and the
direction-only loss: [gated-predictor layerwise JEPA](gated_predictor_layerwise_jepa.md).
