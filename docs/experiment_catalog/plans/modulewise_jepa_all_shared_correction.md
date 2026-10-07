# All-shared-LoRA modulewise EMA-JEPA + HSIC runs

> **Type:** experiment plan, text-only study · **Status:** not completed. Both runs stopped (W&B state `crashed`) before writing any checkpoint, so there are no results.  
> **Menu:** [experiment catalog](../README.md) · **Run registry:** [text-study runs](../models/text_study_run_registry.md)

## Why this follow-up exists

The completed four-cell modulewise sweep and the first calibrated-HSIC follow-up
supervised only `attn.out_proj` and `mlp.3`. They remain valid **narrow-module
ablations**, but they do not test the intended claim that JEPA shapes the full
shared LoRA route in layers 2--4.

This corrected run uses the same data, seed, EMA teacher, normalized-MSE JEPA,
average-target construction, HSIC calibration, and four-epoch budget as the
calibrated follow-up. The only substantive correction is the JEPA/HSIC module
set: `qkv`, `out_proj`, `mlp.0`, `mlp.3` at Transformer layers 2, 3, and 4.

## Exact gradient contract

For every one of the 12 `(layer, module)` cells, the masked student native
shared update is passed through its own predictor and compared with a
stop-gradient EMA-teacher target.

- In `average` mode, every cell predicts the normalized mean clean update of
  that **same module** over layers 2--4. Therefore average-target JEPA still
  supplies a direct gradient at each selected layer.
- The recorder detaches the adapter input. JEPA updates only the selected
  `shared_A`, `shared_B`, and JEPA predictor; it cannot update private LoRAs,
  embeddings, other adapter branches, or the EMA teacher through that loss.
- Ordinary masked-token diffusion trains shared and text-private LoRAs
  normally. HSIC is separate and updates selected shared **and** text-private
  A/B matrices, because it minimizes their dependence.

## Run record

| Condition | Teacher target | W&B | Output |
|---|---|---|---|
| All-shared average | Per module, normalized mean clean update over layers 2--4 | [s5e1xnia](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/s5e1xnia) | `outputs/text_module_jepa_avg_all_shared_hsic_calibrated_2m_4e/` |
| All-shared layerwise | Per module, same-layer clean update | [ooqd8dg4](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/ooqd8dg4) | `outputs/text_module_jepa_layerwise_all_shared_hsic_calibrated_2m_4e/` |

- Average configuration: [text_module_jepa_avg_all_shared_hsic_calibrated_2m_4e.yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_module_jepa_avg_all_shared_hsic_calibrated_2m_4e.yaml)
- Layerwise configuration: [text_module_jepa_layerwise_all_shared_hsic_calibrated_2m_4e.yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_module_jepa_layerwise_all_shared_hsic_calibrated_2m_4e.yaml)

The earlier `out_proj`/`mlp.3` checkpoints must not be presented as a test of
this full-shared-route condition. Because these runs never produced a
checkpoint, the question of supervising all four adapter types remains open.
