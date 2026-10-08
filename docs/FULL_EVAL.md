# Full evaluation (last checkpoint of every eligible run, all layers)

Built by `scripts/build_full_eval_report.py` from `outputs/eval_all` (`scripts/run_full_eval.py`, Oct 7–8 2026). Shared-trunk (`dense_private`) models are scored **trunk only**. Every value at every layer is in `outputs/eval_all/all_metrics_long.csv`; metric definitions in `EVALUATIONS.md`. `ep` = epochs of the run's own last checkpoint (`epoch_NNN` + 1).

## 1. Cross-modal retrieval (text × image of the same model; dense = two separately trained models)

Ridge map chosen on val, scored on 1000 test scenes. Chance R@1 0.001, MRR 0.0075. Identity = no map (trunk models).

| pair | best cell | i→t R@1 | i→t MRR | t→i R@1 | t→i MRR | Procrustes i→t MRR | identity i→t MRR |
|---|---|---|---|---|---|---|---|
| `denseT_e040|denseI_e040` | L5|L7 | 0.548 | 0.666 | 0.695 | 0.783 | 0.215 | 0.010 |
| `denseT_e020|denseI_e020` | L5|L7 | 0.550 | 0.662 | 0.692 | 0.776 | 0.181 | 0.011 |
| `denseT_e007|denseI_e007` | L6|L7 | 0.468 | 0.590 | 0.634 | 0.734 | 0.143 | 0.010 |
| `denseT_e010|denseI_e010` | L5|L7 | 0.470 | 0.592 | 0.621 | 0.723 | 0.167 | 0.009 |
| `mm_paired_stage2_trainable_1_2m_2e__e001|mm_paired_stage2_trainable_1_2m_2e__e001` | L5|L7 | 0.374 | 0.514 | 0.596 | 0.705 | 0.146 | 0.019 |
| `denseT_e004|denseI_e004` | L6|L7 | 0.293 | 0.416 | 0.436 | 0.566 | 0.126 | 0.009 |
| `mm_paired_stage1_jepa_lw_all_1_2m_2e__e001|mm_paired_stage1_jepa_lw_all_1_2m_2e__e001` | L5|L6 | 0.205 | 0.326 | 0.421 | 0.556 | 0.280 | 0.137 |
| `mm_paired_stage2_frozen_1_2m_2e__e001|mm_paired_stage2_frozen_1_2m_2e__e001` | L5|L6 | 0.205 | 0.326 | 0.421 | 0.556 | 0.280 | 0.137 |
| `multimodal_paired_dense_1_2m_4e__e003|multimodal_paired_dense_1_2m_4e__e003` | L5|L7 | 0.168 | 0.276 | 0.316 | 0.451 | 0.207 | 0.011 |
| `denseT_e002|denseI_e002` | L6|L7 | 0.129 | 0.219 | 0.175 | 0.287 | 0.073 | 0.010 |
| `denseT_e001|denseI_e001` | embedding|L7 | 0.070 | 0.144 | 0.109 | 0.207 | 0.066 | 0.009 |
| `mm_unpaired_moments_only_1_2m_2e__e001|mm_unpaired_moments_only_1_2m_2e__e001` | embedding|L7 | 0.079 | 0.152 | 0.107 | 0.199 | 0.056 | 0.013 |
| `mm_unpaired_dense_8e_continued__e007|mm_unpaired_dense_8e_continued__e007` | embedding|L7 | 0.083 | 0.155 | 0.099 | 0.197 | 0.052 | 0.012 |
| `mm_unpaired_stage2_trainable_1_2m_2e__e001|mm_unpaired_stage2_trainable_1_2m_2e__e001` | embedding|L7 | 0.077 | 0.149 | 0.099 | 0.195 | 0.054 | 0.012 |
| `mm_unpaired_dense_private_lora_r128_2m_40e__e012|mm_unpaired_dense_private_lora_r128_2m_40e__e012` | embedding|L7 | 0.085 | 0.158 | 0.098 | 0.191 | 0.072 | 0.010 |
| `multimodal_unpaired_dense_1_2m_4e__e003|multimodal_unpaired_dense_1_2m_4e__e003` | embedding|L7 | 0.084 | 0.154 | 0.095 | 0.185 | 0.057 | 0.013 |
| `mm_trunk_ijepa_private_diff_from_ep6_4e__e003|mm_trunk_ijepa_private_diff_from_ep6_4e__e003` | embedding|L7 | 0.076 | 0.150 | 0.108 | 0.184 | 0.088 | 0.008 |
| `mm_unpaired_private_dropout_gradbal_2m_40e__e000|mm_unpaired_private_dropout_gradbal_2m_40e__e000` | embedding|L7 | 0.058 | 0.134 | 0.081 | 0.180 | 0.080 | 0.011 |
| `mm_trunk_ijepa_from_ep6_4e__e003|mm_trunk_ijepa_from_ep6_4e__e003` | embedding|L7 | 0.079 | 0.148 | 0.095 | 0.177 | 0.074 | 0.009 |
| `multimodal_unpaired_d2v_from_unpaired_dense_ema_1_2m_2e__e001|multimodal_unpaired_d2v_from_unpaired_dense_ema_1_2m_2e__e001` | embedding|L7 | 0.065 | 0.129 | 0.079 | 0.163 | 0.070 | 0.012 |
| `mm_unpaired_jepa_adversarial_1_2m_2e__e001|mm_unpaired_jepa_adversarial_1_2m_2e__e001` | embedding|L7 | 0.066 | 0.129 | 0.079 | 0.163 | 0.070 | 0.012 |
| `mm_unpaired_jepa_fixedL6_1_2m_2e__e001|mm_unpaired_jepa_fixedL6_1_2m_2e__e001` | embedding|L7 | 0.058 | 0.115 | 0.067 | 0.151 | 0.056 | 0.009 |
| `multimodal_unpaired_d2v_from_image_dense_1_2m_2e__e001|multimodal_unpaired_d2v_from_image_dense_1_2m_2e__e001` | embedding|L7 | 0.061 | 0.123 | 0.062 | 0.146 | 0.061 | 0.010 |
| `multimodal_unpaired_d2v_from_text_dense_1_2m_2e__e001|multimodal_unpaired_d2v_from_text_dense_1_2m_2e__e001` | L5.mlp_out|embedding | 0.059 | 0.121 | 0.058 | 0.123 | 0.044 | 0.009 |
| `mm_unpaired_private_gradbal_2m_40e__e000|mm_unpaired_private_gradbal_2m_40e__e000` | embedding|L5.mlp_out | 0.039 | 0.103 | 0.053 | 0.114 | 0.044 | 0.009 |
| `mm_trunk_ijepa_sigreg_from_ep6_4e__e003|mm_trunk_ijepa_sigreg_from_ep6_4e__e003` | embedding|L2 | 0.045 | 0.098 | 0.045 | 0.106 | 0.045 | 0.010 |
| `mm_paired_merged_jepa_trunk_private_diff_1_2m_3e__e001|mm_paired_merged_jepa_trunk_private_diff_1_2m_3e__e001` | embedding|embedding | 0.028 | 0.065 | 0.024 | 0.068 | 0.040 | 0.009 |
| `mm_unpaired_jepa_moments_w10_1_2m_2e__e001|mm_unpaired_jepa_moments_w10_1_2m_2e__e001` | L1|L1 | 0.027 | 0.068 | 0.018 | 0.065 | 0.036 | 0.010 |
| `mm_unpaired_jepa_fixedL6_moments_1_2m_2e__e001|mm_unpaired_jepa_fixedL6_moments_1_2m_2e__e001` | L1|L1 | 0.023 | 0.060 | 0.018 | 0.060 | 0.033 | 0.012 |
| `mm_unpaired_jepa_moments_1_2m_2e__e001|mm_unpaired_jepa_moments_1_2m_2e__e001` | L1|L0 | 0.019 | 0.058 | 0.021 | 0.059 | 0.035 | 0.010 |
| `random|random` | embedding|L0 | 0.033 | 0.067 | 0.024 | 0.059 | 0.026 | 0.010 |
| `mm_unpaired_merged_jepa_trunk_private_diff_1_2m_2e__e001|mm_unpaired_merged_jepa_trunk_private_diff_1_2m_2e__e001` | embedding|embedding | 0.020 | 0.053 | 0.016 | 0.053 | 0.033 | 0.009 |
| `multimodal_unpaired_d2v_from_unpaired_dense_sigreg_1_2m_2e__e001|multimodal_unpaired_d2v_from_unpaired_dense_sigreg_1_2m_2e__e001` | embedding|embedding | 0.028 | 0.053 | 0.017 | 0.049 | 0.030 | 0.011 |
| `mm_unpaired_merged_moments_1_2m_2e__e001|mm_unpaired_merged_moments_1_2m_2e__e001` | embedding|embedding | 0.018 | 0.054 | 0.015 | 0.049 | 0.032 | 0.009 |
| `multimodal_unpaired_lejepa_scratch_1_2m_4e__e003|multimodal_unpaired_lejepa_scratch_1_2m_4e__e003` | embedding|embedding | 0.012 | 0.033 | 0.010 | 0.033 | 0.028 | 0.008 |

## 2. Image binding (397 content-matched pairs; pairwise chance 50%, strict = both scenes classified)

| model | ep | objective | binding L7 | binding best (layer) | strict L7 | strict best | clean probe L7 |
|---|---|---|---|---|---|---|---|
| `mm_paired_stage2_frozen_1_2m_2e` | 2 | diffusion | 89.8 | 91.5 (L6.mlp_out) | 68.1 | 70.8 (L6.mlp_out) | 87.1 |
| `mm_paired_stage1_jepa_lw_all_1_2m_2e` | 2 | I-JEPA/data2vec (EMA) layerwise | 89.8 | 91.5 (L6.mlp_out) | 68.1 | 70.8 (L6.mlp_out) | 87.1 |
| `image_ijepa_sweep_blk_tiny_t45` | 4 | I-JEPA/data2vec (EMA) layerwise | 81.2 | 90.4 (L7.mlp_out) | 58.5 | 69.5 (L7.mlp_out) | 83.2 |
| `image_ijepa_sweep_blk_s05_t60` | 4 | I-JEPA/data2vec (EMA) layerwise | 80.6 | 90.2 (L7.mlp_out) | 57.0 | 70.2 (L7.mlp_out) | 83.0 |
| `image_ijepa_sweep_blk_wide_t45` | 4 | I-JEPA/data2vec (EMA) layerwise | 81.9 | 90.0 (L7.mlp_out) | 57.6 | 70.2 (L7.mlp_out) | 83.0 |
| `image_ijepa_sweep_blk_s15_t60` | 4 | I-JEPA/data2vec (EMA) layerwise | 82.1 | 89.7 (L7.mlp_out) | 57.1 | 70.2 (L7.mlp_out) | 83.2 |
| `image_ijepa_sweep_blk_s15_t45` | 4 | I-JEPA/data2vec (EMA) layerwise | 82.2 | 89.7 (L7.mlp_out) | 57.7 | 70.3 (L7.mlp_out) | 83.2 |
| `image_data2vec_from_dense_layerwise_all_block2d_1_2m_6e_continued` | 6 | I-JEPA/data2vec (EMA) layerwise | 82.3 | 89.7 (L7.mlp_out) | 57.6 | 68.2 (L7.mlp_out) | 83.0 |
| `image_ijepa_sweep_blk_s10_t30` | 4 | I-JEPA/data2vec (EMA) layerwise | 81.7 | 89.6 (L7.mlp_out) | 58.0 | 70.6 (L7.mlp_out) | 83.3 |
| `image_ijepa_sweep_blk_s30_t30` | 4 | I-JEPA/data2vec (EMA) layerwise | 80.9 | 89.5 (L7.mlp_out) | 56.5 | 69.8 (L7.mlp_out) | 82.5 |
| `image_data2vec_from_dense_layerwise_all_ijepa_2m_2e_seed20260930` | 2 | I-JEPA/data2vec (EMA) layerwise | 81.5 | 89.3 (L7.mlp_out) | 57.4 | 66.4 (L7.mlp_out) | 82.3 |
| `image_dense_private_frozen_hsic_from_d2v_lw_block2d_1_2m_2e` | 2 | diffusion + HSIC | 82.3 | 89.2 (L7.mlp_out) | 58.1 | 69.0 (L7.mlp_out) | 82.7 |
| `image_data2vec_from_dense_layerwise_all_block2d_1_2m_2e` | 2 | I-JEPA/data2vec (EMA) layerwise | 82.3 | 89.2 (L7.mlp_out) | 58.1 | 69.0 (L7.mlp_out) | 82.7 |
| `image_ijepa_sweep_blk_s30_t60` | 4 | I-JEPA/data2vec (EMA) layerwise | 80.2 | 89.0 (L7.mlp_out) | 55.4 | 68.0 (L7.mlp_out) | 82.0 |
| `image_stage2_frozen_from_ijepa_1_2m_2e` | 2 | diffusion + HSIC | 81.9 | 88.8 (L7.mlp_out) | 57.3 | 67.0 (L7.mlp_out) | 82.3 |
| `image_data2vec_from_dense_layerwise_all_ijepa_2m_2e` | 2 | I-JEPA/data2vec (EMA) layerwise | 81.9 | 88.8 (L7.mlp_out) | 57.3 | 67.0 (L7.mlp_out) | 82.3 |
| `image_data2vec_from_dense_layerwise_all_ijepa_2m_6e_continued` | 6 | I-JEPA/data2vec (EMA) layerwise | 81.8 | 88.6 (L7.mlp_out) | 57.4 | 67.1 (L7.mlp_out) | 82.7 |
| `image_data2vec_from_dense_layerwise_all_randt_1_2m_2e` | 2 | I-JEPA/data2vec (EMA) layerwise | 81.6 | 88.6 (L7.mlp_out) | 57.4 | 66.4 (L7.mlp_out) | 82.4 |
| `image_ijepa_sweep_rand_t75` | 4 | I-JEPA/data2vec (EMA) layerwise | 82.0 | 88.5 (L7.mlp_out) | 56.7 | 66.4 (L7.mlp_out) | 82.3 |
| `mm_trunk_ijepa_private_diff_from_ep6_4e` | 4 | diffusion (private LoRA only) + I-JEPA/data2vec (EMA) layerwise, trunk-only | 76.9 | 88.3 (L7.mlp_out) | 50.0 | 66.2 (L7.mlp_out) | 80.1 |
| `mm_trunk_ijepa_from_ep6_4e` | 4 | I-JEPA/data2vec (EMA) layerwise, trunk-only | 76.6 | 88.1 (L7.mlp_out) | 50.5 | 67.1 (L7.mlp_out) | 80.3 |
| `image_data2vec_from_dense_layerwise_l4to7_randt_1_2m_2e` | 2 | I-JEPA/data2vec (EMA) layerwise | 81.4 | 87.8 (L7.mlp_out) | 57.4 | 66.9 (L7.mlp_out) | 82.4 |
| `multimodal_unpaired_d2v_from_image_dense_1_2m_2e` | 2 | I-JEPA/data2vec (EMA) layerwise | 80.2 | 87.4 (L7.mlp_out) | 55.3 | 64.9 (L7.mlp_out) | 81.5 |
| `image_ijepa_sweep_rand_t45` | 4 | I-JEPA/data2vec (EMA) layerwise | 80.5 | 86.8 (L7.mlp_out) | 56.7 | 66.2 (L7.mlp_out) | 82.6 |
| `mm_unpaired_jepa_fixedL6_1_2m_2e` | 2 | I-JEPA/data2vec (EMA) fixed_target | 75.3 | 86.3 (L6.mlp_out) | 50.0 | 64.9 (L6.mlp_out) | 79.1 |
| `multimodal_unpaired_d2v_from_unpaired_dense_ema_1_2m_2e` | 2 | I-JEPA/data2vec (EMA) layerwise | 76.4 | 85.7 (L6.mlp_out) | 51.0 | 65.6 (L6.mlp_out) | 80.1 |
| `mm_unpaired_jepa_adversarial_1_2m_2e` | 2 | I-JEPA/data2vec (EMA) layerwise + DANN | 76.7 | 85.6 (L6.mlp_out) | 51.2 | 65.4 (L6.mlp_out) | 80.1 |
| `multimodal_paired_dense_1_2m_4e` | 4 | diffusion | 79.2 | 84.7 (L6.mlp_out) | 50.2 | 59.2 (L6.mlp_out) | 83.8 |
| `image_dense_diffusion_1_2m_6e_continued` | 6 | diffusion | 80.5 | 84.0 (L6.mlp_out) | 56.1 | 60.1 (L6.mlp_out) | 82.9 |
| `image_dense_diffusion_2_5m_40e` | 40 | diffusion | 80.6 | 83.7 (L6.mlp_out) | 54.3 | 59.3 (L5.mlp_out) | 82.7 |
| `mm_unpaired_stage2_trainable_1_2m_2e` | 2 | diffusion | 77.8 | 83.6 (L6.mlp_out) | 52.4 | 58.5 (L6.mlp_out) | 81.8 |
| `mm_unpaired_dense_private_lora_r128_2m_40e` | 13 | diffusion | 73.6 | 83.3 (L6.mlp_out) | 43.3 | 57.7 (L6.mlp_out) | 79.0 |
| `image_dense_diffusion_1_2m_4e` | 4 | diffusion | 80.0 | 83.2 (L6.mlp_out) | 54.2 | 59.4 (L6.mlp_out) | 82.4 |
| `mm_unpaired_dense_8e_continued` | 8 | diffusion | 79.5 | 83.1 (L6.mlp_out) | 52.5 | 58.8 (L6.mlp_out) | 82.5 |
| `mm_unpaired_moments_only_1_2m_2e` | 2 | diffusion + moment matching | 77.9 | 83.0 (L6.mlp_out) | 53.1 | 59.3 (L6.mlp_out) | 81.9 |
| `multimodal_unpaired_dense_1_2m_4e` | 4 | diffusion | 78.5 | 82.7 (L6.mlp_out) | 52.2 | 59.2 (L6.mlp_out) | 81.8 |
| `mm_paired_stage2_trainable_1_2m_2e` | 2 | diffusion | 76.6 | 81.7 (L6.mlp_out) | 48.3 | 56.1 (L6.mlp_out) | 82.6 |
| `mm_unpaired_private_dropout_gradbal_2m_40e` | 1 | diffusion + grad-balance + private-dropout 0.5 | 76.0 | 81.3 (L6.mlp_out) | 47.1 | 53.0 (L6.mlp_out) | 80.1 |
| `image_data2vec_from_dense_avg_l4to7_randt_1_2m_2e` | 2 | I-JEPA/data2vec (EMA) average | 73.8 | 80.9 (L7.mlp_out) | 46.9 | 56.1 (L6.mlp_out) | 79.7 |
| `image_data2vec_from_dense_avg_all_randt_1_2m_2e` | 2 | I-JEPA/data2vec (EMA) average | 70.8 | 79.4 (L6.mlp_out) | 42.1 | 51.0 (L6.mlp_out) | 79.2 |
| `image_lejepa_paper_from_dense_ep12_4e` | 4 | LeJEPA multi-crop (paper) | 64.3 | 77.9 (L4.mlp_out) | 33.3 | 50.4 (L4.mlp_out) | 72.0 |
| `mm_trunk_ijepa_sigreg_from_ep6_4e` | 4 | I-JEPA/data2vec (EMA) layerwise, trunk-only + SIGReg projector | 66.6 | 67.9 (L6) | 35.3 | 36.1 (L6) | 72.5 |
| `mm_unpaired_private_gradbal_2m_40e` | 1 | diffusion + grad-balance | 61.0 | 66.5 (L4.mlp_out) | 30.0 | 35.0 (L4.mlp_out) | 73.1 |
| `mm_unpaired_jepa_fixedL6_moments_1_2m_2e` | 2 | I-JEPA/data2vec (EMA) fixed_target + moment matching | 57.3 | 64.4 (embedding) | 27.7 | 34.2 (L1) | 66.0 |
| `image_lejepa_blk_s15_t45_from_dense_ep12` | 1 | LeJEPA token-level (SIGReg, no teacher) layerwise | 51.1 | 64.2 (embedding) | 25.3 | 33.4 (embedding) | 56.1 |
| `mm_unpaired_jepa_moments_1_2m_2e` | 2 | I-JEPA/data2vec (EMA) layerwise + moment matching | 57.8 | 63.8 (embedding) | 29.3 | 33.8 (embedding) | 66.3 |
| `mm_unpaired_jepa_moments_w10_1_2m_2e` | 2 | I-JEPA/data2vec (EMA) layerwise + moment matching | 57.7 | 63.6 (embedding) | 28.1 | 34.2 (L1) | 64.4 |
| `mm_unpaired_merged_moments_1_2m_2e` | 2 | diffusion (private LoRA only) + I-JEPA/data2vec (EMA) layerwise + moment matching | 52.3 | 63.5 (embedding) | 24.2 | 32.8 (embedding) | 55.0 |
| `multimodal_unpaired_d2v_from_unpaired_dense_sigreg_1_2m_2e` | 2 | LeJEPA token-level (SIGReg, no teacher) layerwise | 50.8 | 63.4 (embedding) | 22.0 | 32.9 (embedding) | 57.1 |
| `mm_paired_merged_jepa_trunk_private_diff_1_2m_3e` | 2 | diffusion (private LoRA only) + I-JEPA/data2vec (EMA) layerwise | 50.6 | 63.4 (embedding) | 24.0 | 33.9 (embedding) | 52.3 |
| `image_data2vec_scratch_block2d_65pct_layerwise_l4to7_1_2m_4e` | 4 | I-JEPA/data2vec (EMA) layerwise | 52.1 | 63.0 (embedding) | 21.8 | 33.0 (embedding) | 59.2 |
| `image_data2vec_scratch_block2d_30pct_avg_l4to7_1_2m_4e` | 4 | I-JEPA/data2vec (EMA) average | 55.3 | 62.6 (embedding) | 23.7 | 33.4 (embedding) | 62.9 |
| `image_data2vec_scratch_block2d_65pct_avg_l4to7_1_2m_4e` | 4 | I-JEPA/data2vec (EMA) average | 53.3 | 62.6 (embedding) | 21.7 | 33.1 (embedding) | 59.8 |
| `mm_unpaired_merged_jepa_trunk_private_diff_1_2m_2e` | 2 | diffusion (private LoRA only) + I-JEPA/data2vec (EMA) layerwise | 51.5 | 62.6 (embedding) | 22.1 | 32.9 (embedding) | 56.0 |
| `multimodal_unpaired_lejepa_scratch_1_2m_4e` | 4 | LeJEPA token-level (SIGReg, no teacher) layerwise | 49.5 | 62.5 (embedding) | 21.4 | 32.7 (embedding) | 54.7 |
| `image_data2vec_scratch_block2d_30pct_layerwise_l4to7_1_2m_4e` | 4 | I-JEPA/data2vec (EMA) layerwise | 53.6 | 62.5 (embedding) | 22.8 | 33.5 (embedding) | 61.2 |
| `multimodal_unpaired_d2v_from_text_dense_1_2m_2e` | 2 | I-JEPA/data2vec (EMA) layerwise | 52.7 | 62.4 (embedding) | 22.9 | 30.5 (embedding) | 59.9 |

## 3. Image summary (L7 and best layer)

| model | ep | probe L7 | probe best | RSA L7 | cos / rank L7 | conj mAP best | conj mAP t=.75 | scene ρ_conj best | triples d′ centred L7 | S L7 raw / centred / z (d_s / d_n centred) |
|---|---|---|---|---|---|---|---|---|---|---|
| `image_data2vec_from_dense_avg_all_randt_1_2m_2e` | 2 | 78.2 | 78.4 (L6.mlp_out) | 0.139 | 0.86 / 18 | 0.548 (L6.mlp_out) | 0.413 | 0.117 (L5.mlp_out) | 0.254 | 0.327 / 0.349 / 0.372 (0.119 / 0.342) |
| `image_data2vec_from_dense_avg_l4to7_randt_1_2m_2e` | 2 | 78.5 | 78.5 (L7) | 0.159 | 0.85 / 28 | 0.573 (L6.mlp_out) | 0.416 | 0.105 (L5.mlp_out) | 0.218 | 0.293 / 0.314 / 0.338 (0.124 / 0.396) |
| `image_data2vec_from_dense_layerwise_all_block2d_1_2m_2e` | 2 | 81.2 | 82.4 (L7.mlp_out) | 0.190 | 0.75 / 31 | 0.696 (L7.mlp_out) | 0.461 | 0.210 (L7) | 0.333 | 0.418 / 0.444 / 0.479 (0.115 / 0.258) |
| `image_data2vec_from_dense_layerwise_all_block2d_1_2m_6e_continued` | 6 | 81.5 | 82.5 (L7.mlp_out) | 0.198 | 0.85 / 32 | 0.713 (L7.mlp_out) | 0.475 | 0.220 (L7) | 0.387 | 0.467 / 0.481 / 0.495 (0.169 / 0.351) |
| `image_data2vec_from_dense_layerwise_all_ijepa_2m_2e` | 2 | 80.9 | 82.2 (L7.mlp_out) | 0.188 | 0.73 / 31 | 0.691 (L7.mlp_out) | 0.465 | 0.208 (L7) | 0.370 | 0.394 / 0.418 / 0.430 (0.127 / 0.303) |
| `image_data2vec_from_dense_layerwise_all_ijepa_2m_2e_seed20260930` | 2 | 80.9 | 82.3 (L7.mlp_out) | 0.188 | 0.73 / 31 | 0.694 (L7.mlp_out) | 0.467 | 0.214 (L7) | 0.338 | 0.390 / 0.402 / 0.431 (0.132 / 0.328) |
| `image_data2vec_from_dense_layerwise_all_ijepa_2m_6e_continued` | 6 | 81.1 | 82.3 (L7.mlp_out) | 0.206 | 0.84 / 31 | 0.706 (L7.mlp_out) | 0.485 | 0.226 (L7) | 0.293 | 0.374 / 0.379 / 0.382 (0.172 / 0.453) |
| `image_data2vec_from_dense_layerwise_all_randt_1_2m_2e` | 2 | 81.1 | 82.0 (L7.mlp_out) | 0.186 | 0.78 / 30 | 0.696 (L7.mlp_out) | 0.489 | 0.182 (L7) | 0.220 | 0.401 / 0.441 / 0.406 (0.093 / 0.210) |
| `image_data2vec_from_dense_layerwise_l4to7_randt_1_2m_2e` | 2 | 81.0 | 82.1 (L7.mlp_out) | 0.181 | 0.78 / 31 | 0.691 (L7.mlp_out) | 0.482 | 0.179 (L7) | 0.325 | 0.412 / 0.423 / 0.430 (0.128 / 0.303) |
| `image_data2vec_scratch_block2d_30pct_avg_l4to7_1_2m_4e` | 4 | 61.6 | 67.9 (embedding) | 0.095 | 0.57 / 22 | 0.405 (embedding) | 0.311 | 0.115 (L3.mlp_out) | 0.101 | 0.127 / 0.187 / 0.235 (0.112 / 0.597) |
| `image_data2vec_scratch_block2d_30pct_layerwise_l4to7_1_2m_4e` | 4 | 60.4 | 68.0 (embedding) | 0.096 | 0.45 / 25 | 0.406 (embedding) | 0.311 | 0.133 (L2.mlp_out) | 0.085 | 0.124 / 0.189 / 0.241 (0.113 / 0.600) |
| `image_data2vec_scratch_block2d_65pct_avg_l4to7_1_2m_4e` | 4 | 59.3 | 67.6 (embedding) | 0.083 | 0.62 / 12 | 0.405 (embedding) | 0.310 | 0.093 (L4.mlp_out) | 0.091 | 0.147 / 0.175 / 0.206 (0.120 / 0.686) |
| `image_data2vec_scratch_block2d_65pct_layerwise_l4to7_1_2m_4e` | 4 | 58.3 | 67.7 (embedding) | 0.054 | 0.53 / 13 | 0.404 (embedding) | 0.310 | 0.116 (L2.mlp_out) | 0.097 | 0.148 / 0.179 / 0.207 (0.114 / 0.637) |
| `image_dense_diffusion_1_2m_4e` | 4 | 80.7 | 80.7 (L7) | 0.104 | 0.85 / 19 | 0.664 (L7.mlp_out) | 0.442 | 0.167 (L7) | 0.411 | 0.583 / 0.584 / 0.557 (0.144 / 0.247) |
| `image_dense_diffusion_1_2m_6e_continued` | 6 | 80.9 | 80.9 (L7) | 0.103 | 0.86 / 18 | 0.656 (L7.mlp_out) | 0.451 | 0.158 (L7) | 0.446 | 0.658 / 0.654 / 0.617 (0.129 / 0.198) |
| `image_dense_diffusion_2_5m_40e` | 40 | 81.2 | 81.2 (L7) | 0.114 | 0.98 / 9 | 0.607 (L6.mlp_out) | 0.458 | 0.118 (L7) | 0.285 | 0.606 / 0.562 / 0.509 (0.135 / 0.239) |
| `image_dense_private_frozen_hsic_from_d2v_lw_block2d_1_2m_2e` | 2 | 81.2 | 82.4 (L7.mlp_out) | 0.190 | 0.75 / 31 | 0.696 (L7.mlp_out) | 0.461 | 0.210 (L7) | 0.333 | 0.418 / 0.444 / 0.479 (0.115 / 0.258) |
| `image_ijepa_sweep_blk_s05_t60` | 4 | 81.6 | 83.4 (L7.mlp_out) | 0.160 | 0.92 / 27 | 0.738 (L7.mlp_out) | 0.513 | 0.174 (L7) | 0.295 | 0.358 / 0.363 / 0.385 (0.143 / 0.395) |
| `image_ijepa_sweep_blk_s10_t30` | 4 | 81.7 | 83.3 (L7.mlp_out) | 0.160 | 0.92 / 29 | 0.727 (L7.mlp_out) | 0.490 | 0.190 (L7) | 0.271 | 0.342 / 0.329 / 0.342 (0.150 / 0.454) |
| `image_ijepa_sweep_blk_s15_t45` | 4 | 81.7 | 83.7 (L7.mlp_out) | 0.162 | 0.90 / 30 | 0.735 (L7.mlp_out) | 0.497 | 0.193 (L7) | 0.190 | 0.240 / 0.255 / 0.281 (0.115 / 0.452) |
| `image_ijepa_sweep_blk_s15_t60` | 4 | 81.8 | 83.6 (L7.mlp_out) | 0.162 | 0.90 / 30 | 0.735 (L7.mlp_out) | 0.498 | 0.193 (L7) | 0.190 | 0.240 / 0.256 / 0.282 (0.116 / 0.453) |
| `image_ijepa_sweep_blk_s30_t30` | 4 | 81.0 | 82.7 (L7.mlp_out) | 0.155 | 0.91 / 30 | 0.707 (L7.mlp_out) | 0.460 | 0.195 (L7) | 0.223 | 0.265 / 0.270 / 0.295 (0.115 / 0.425) |
| `image_ijepa_sweep_blk_s30_t60` | 4 | 80.4 | 82.8 (L7.mlp_out) | 0.156 | 0.90 / 30 | 0.705 (L7.mlp_out) | 0.465 | 0.198 (L7) | 0.296 | 0.322 / 0.335 / 0.344 (0.131 / 0.391) |
| `image_ijepa_sweep_blk_tiny_t45` | 4 | 81.8 | 82.7 (L7.mlp_out) | 0.163 | 0.93 / 25 | 0.721 (L7.mlp_out) | 0.512 | 0.169 (L7) | 0.232 | 0.309 / 0.337 / 0.366 (0.122 / 0.363) |
| `image_ijepa_sweep_blk_wide_t45` | 4 | 81.5 | 83.2 (L7.mlp_out) | 0.156 | 0.91 / 29 | 0.723 (L7.mlp_out) | 0.497 | 0.186 (L7) | 0.248 | 0.294 / 0.301 / 0.308 (0.123 / 0.409) |
| `image_ijepa_sweep_rand_t45` | 4 | 81.2 | 81.2 (L7) | 0.152 | 0.92 / 22 | 0.683 (L7.mlp_out) | 0.479 | 0.168 (L7) | 0.080 | 0.077 / 0.126 / 0.171 (0.063 / 0.500) |
| `image_ijepa_sweep_rand_t75` | 4 | 81.0 | 81.8 (L7.mlp_out) | 0.147 | 0.90 / 29 | 0.701 (L7.mlp_out) | 0.531 | 0.157 (L7) | 0.259 | 0.389 / 0.376 / 0.363 (0.134 / 0.356) |
| `image_lejepa_blk_s15_t45_from_dense_ep12` | 1 | 55.4 | 70.5 (embedding) | 0.039 | 0.02 / 22 | 0.416 (embedding) | 0.338 | 0.026 (L4.mlp_out) | 0.124 | 0.483 / 0.315 / 0.262 (0.109 / 0.346) |
| `image_lejepa_paper_from_dense_ep12_4e` | 4 | 70.5 | 76.8 (L4) | 0.008 | 0.91 / 22 | 0.498 (L4) | 0.342 | 0.072 (L0) | 0.516 | 0.781 / 0.754 / 0.732 (0.137 / 0.181) |
| `image_stage2_frozen_from_ijepa_1_2m_2e` | 2 | 80.9 | 82.2 (L7.mlp_out) | 0.188 | 0.73 / 31 | 0.691 (L7.mlp_out) | 0.465 | 0.208 (L7) | 0.370 | 0.394 / 0.418 / 0.430 (0.127 / 0.303) |
| `mm_paired_merged_jepa_trunk_private_diff_1_2m_3e` | 2 | 51.5 | 69.5 (embedding) | 0.014 | 1.00 / 2 | 0.413 (embedding) | 0.323 | 0.050 (embedding) | 0.034 | 0.457 / 0.426 / 0.411 (0.321 / 0.755) |
| `mm_paired_stage1_jepa_lw_all_1_2m_2e` | 2 | 86.3 | 86.3 (L7) | 0.337 | 0.83 / 22 | 0.761 (L7) | 0.415 | 0.264 (L7) | 0.173 | 0.251 / 0.276 / 0.302 (0.102 / 0.369) |
| `mm_paired_stage2_frozen_1_2m_2e` | 2 | 86.3 | 86.3 (L7) | 0.337 | 0.83 / 22 | 0.761 (L7) | 0.415 | 0.264 (L7) | 0.173 | 0.251 / 0.276 / 0.302 (0.102 / 0.369) |
| `mm_paired_stage2_trainable_1_2m_2e` | 2 | 81.4 | 81.4 (L7) | 0.106 | 0.88 / 15 | 0.633 (L6.mlp_out) | 0.390 | 0.144 (L6.mlp_out) | 0.235 | 0.359 / 0.372 / 0.376 (0.099 / 0.267) |
| `mm_trunk_ijepa_from_ep6_4e` | 4 | 79.3 | 82.0 (L7.mlp_out) | 0.157 | 0.89 / 26 | 0.690 (L7.mlp_out) | 0.475 | 0.163 (L7.mlp_out) | 0.198 | 0.292 / 0.328 / 0.354 (0.117 / 0.358) |
| `mm_trunk_ijepa_private_diff_from_ep6_4e` | 4 | 79.1 | 82.1 (L7.mlp_out) | 0.154 | 0.89 / 26 | 0.694 (L7.mlp_out) | 0.472 | 0.159 (L7) | 0.270 | 0.379 / 0.406 / 0.405 (0.122 / 0.301) |
| `mm_trunk_ijepa_sigreg_from_ep6_4e` | 4 | 70.8 | 72.1 (L1) | 0.069 | 0.97 / 46 | 0.426 (L1) | 0.334 | 0.102 (L1) | 0.201 | 0.317 / 0.283 / 0.289 (0.123 / 0.436) |
| `mm_unpaired_dense_8e_continued` | 8 | 80.8 | 81.4 (L6.mlp_out) | 0.112 | 0.93 / 15 | 0.641 (L7.mlp_out) | 0.446 | 0.145 (L7) | 0.383 | 0.570 / 0.563 / 0.556 (0.128 / 0.228) |
| `mm_unpaired_dense_private_lora_r128_2m_40e` | 13 | 77.8 | 79.8 (L6.mlp_out) | 0.130 | 0.99 / 12 | 0.562 (L6.mlp_out) | 0.356 | 0.144 (L7) | 0.169 | 0.444 / 0.453 / 0.332 (0.088 / 0.195) |
| `mm_unpaired_jepa_adversarial_1_2m_2e` | 2 | 78.7 | 80.8 (L7.mlp_out) | 0.150 | 0.81 / 25 | 0.651 (L7.mlp_out) | 0.467 | 0.157 (L7) | 0.172 | 0.240 / 0.309 / 0.369 (0.103 / 0.333) |
| `mm_unpaired_jepa_fixedL6_1_2m_2e` | 2 | 78.0 | 80.4 (L6.mlp_out) | 0.141 | 0.72 / 26 | 0.630 (L6.mlp_out) | 0.450 | 0.132 (L7.mlp_out) | 0.288 | 0.458 / 0.489 / 0.458 (0.119 / 0.244) |
| `mm_unpaired_jepa_fixedL6_moments_1_2m_2e` | 2 | 62.2 | 69.5 (embedding) | 0.057 | 1.00 / 199 | 0.407 (embedding) | 0.324 | 0.092 (L1.mlp_out) | 0.078 | 0.124 / 0.169 / 0.209 (0.096 / 0.570) |
| `mm_unpaired_jepa_moments_1_2m_2e` | 2 | 62.7 | 69.5 (embedding) | 0.069 | 1.00 / 185 | 0.407 (embedding) | 0.324 | 0.089 (L1.mlp_out) | 0.081 | 0.126 / 0.173 / 0.214 (0.099 / 0.574) |
| `mm_unpaired_jepa_moments_w10_1_2m_2e` | 2 | 61.2 | 69.5 (embedding) | 0.048 | 1.00 / 158 | 0.407 (embedding) | 0.324 | 0.085 (L1.mlp_out) | 0.094 | 0.171 / 0.202 / 0.228 (0.108 / 0.534) |
| `mm_unpaired_merged_jepa_trunk_private_diff_1_2m_2e` | 2 | 55.5 | 69.6 (embedding) | 0.002 | 1.00 / 2 | 0.412 (embedding) | 0.326 | 0.022 (embedding) | 0.047 | 0.407 / 0.497 / 0.475 (0.393 / 0.791) |
| `mm_unpaired_merged_moments_1_2m_2e` | 2 | 54.7 | 69.5 (embedding) | -0.002 | 1.00 / 2 | 0.414 (embedding) | 0.326 | 0.032 (embedding) | 0.038 | 0.247 / 0.338 / 0.318 (0.269 / 0.796) |
| `mm_unpaired_moments_only_1_2m_2e` | 2 | 80.1 | 80.9 (L6.mlp_out) | 0.105 | 0.90 / 16 | 0.625 (L6.mlp_out) | 0.443 | 0.144 (L7) | 0.333 | 0.434 / 0.474 / 0.491 (0.117 / 0.247) |
| `mm_unpaired_private_dropout_gradbal_2m_40e` | 1 | 78.5 | 79.8 (L6.mlp_out) | 0.084 | 0.85 / 16 | 0.581 (L6.mlp_out) | 0.410 | 0.160 (L7) | 0.337 | 0.394 / 0.416 / 0.424 (0.144 / 0.347) |
| `mm_unpaired_private_gradbal_2m_40e` | 1 | 71.8 | 72.6 (L5.mlp_out) | 0.055 | 0.98 / 6 | 0.428 (L5.mlp_out) | 0.322 | 0.102 (L7) | 0.230 | 0.327 / 0.338 / 0.339 (0.107 / 0.316) |
| `mm_unpaired_stage2_trainable_1_2m_2e` | 2 | 80.1 | 81.0 (L6.mlp_out) | 0.105 | 0.90 / 14 | 0.633 (L7.mlp_out) | 0.445 | 0.145 (L7) | 0.247 | 0.301 / 0.345 / 0.383 (0.110 / 0.319) |
| `multimodal_paired_dense_1_2m_4e` | 4 | 82.4 | 84.1 (L6.mlp_out) | 0.241 | 0.74 / 22 | 0.693 (L6.mlp_out) | 0.365 | 0.229 (L7) | 0.297 | 0.371 / 0.382 / 0.395 (0.132 / 0.347) |
| `multimodal_unpaired_d2v_from_image_dense_1_2m_2e` | 2 | 80.4 | 81.9 (L7.mlp_out) | 0.186 | 0.79 / 29 | 0.688 (L7.mlp_out) | 0.483 | 0.188 (L7) | 0.213 | 0.401 / 0.453 / 0.450 (0.081 / 0.180) |
| `multimodal_unpaired_d2v_from_text_dense_1_2m_2e` | 2 | 59.2 | 68.0 (embedding) | 0.020 | 0.90 / 6 | 0.401 (embedding) | 0.313 | 0.078 (L0.mlp_out) | 0.090 | 0.179 / 0.227 / 0.238 (0.135 / 0.592) |
| `multimodal_unpaired_d2v_from_unpaired_dense_ema_1_2m_2e` | 2 | 78.7 | 80.8 (L7.mlp_out) | 0.150 | 0.81 / 25 | 0.651 (L7.mlp_out) | 0.467 | 0.158 (L7) | 0.174 | 0.245 / 0.313 / 0.374 (0.104 / 0.331) |
| `multimodal_unpaired_d2v_from_unpaired_dense_sigreg_1_2m_2e` | 2 | 57.2 | 69.7 (embedding) | 0.059 | 0.00 / 12 | 0.408 (embedding) | 0.327 | 0.053 (L1) | 0.151 | 0.271 / 0.238 / 0.166 (0.102 / 0.429) |
| `multimodal_unpaired_dense_1_2m_4e` | 4 | 79.8 | 81.4 (L6.mlp_out) | 0.092 | 0.86 / 16 | 0.638 (L7.mlp_out) | 0.434 | 0.151 (L7) | 0.209 | 0.241 / 0.295 / 0.344 (0.115 / 0.391) |
| `multimodal_unpaired_lejepa_scratch_1_2m_4e` | 4 | 53.5 | 68.2 (embedding) | 0.020 | 0.00 / 21 | 0.403 (embedding) | 0.316 | 0.044 (L0) | 0.167 | 0.357 / 0.253 / 0.193 (0.135 / 0.534) |
| `random` |  | — | — | — | — | 0.393 (L2) | 0.312 | 0.060 (L4.mlp_out) | — | 0.071 / 0.133 / 0.190 (0.092 / 0.693) |

## 4. Text summary (L7 and best layer)

| model | ep | probe L7 | probe best | RSA L7 | cos / rank L7 | conj mAP best | conj mAP t=.75 | scene ρ_conj best | S_bind L7 raw / centred / z | S_cont L7 raw / centred / z |
|---|---|---|---|---|---|---|---|---|---|---|
| `mm_paired_merged_jepa_trunk_private_diff_1_2m_3e` | 2 | 56.3 | 84.8 (embedding) | -0.006 | 1.00 / 2 | 0.603 (embedding) | 0.353 | 0.129 (embedding) | 0.499 / 0.558 / 0.563 | 0.287 / 0.421 / 0.435 |
| `mm_paired_stage1_jepa_lw_all_1_2m_2e` | 2 | 85.7 | 88.0 (L5.mlp_out) | 0.093 | 0.72 / 30 | 0.694 (L5.mlp_out) | 0.352 | 0.235 (L6.mlp_out) | 0.262 / 0.313 / 0.309 | 1.186 / 1.338 / 1.291 |
| `mm_paired_stage2_frozen_1_2m_2e` | 2 | 85.7 | 88.0 (L5.mlp_out) | 0.093 | 0.72 / 30 | 0.694 (L5.mlp_out) | 0.352 | 0.235 (L6.mlp_out) | 0.262 / 0.313 / 0.309 | 1.186 / 1.338 / 1.291 |
| `mm_paired_stage2_trainable_1_2m_2e` | 2 | 89.3 | 91.7 (L5.mlp_out) | 0.126 | 0.81 / 33 | 0.793 (L5.mlp_out) | 0.352 | 0.181 (L5.mlp_out) | 0.417 / 0.468 / 0.464 | 1.517 / 1.596 / 1.596 |
| `mm_trunk_ijepa_from_ep6_4e` | 4 | 72.8 | 84.9 (embedding) | 0.103 | 0.95 / 11 | 0.616 (embedding) | 0.351 | 0.255 (L2.mlp_out) | 0.416 / 0.496 / 0.454 | 0.824 / 0.777 / 0.810 |
| `mm_trunk_ijepa_private_diff_from_ep6_4e` | 4 | 73.0 | 84.8 (embedding) | 0.104 | 0.95 / 11 | 0.616 (embedding) | 0.351 | 0.226 (L1.mlp_out) | 0.413 / 0.494 / 0.458 | 0.824 / 0.775 / 0.810 |
| `mm_trunk_ijepa_sigreg_from_ep6_4e` | 4 | 65.2 | 84.9 (embedding) | 0.028 | 0.95 / 44 | 0.614 (embedding) | 0.352 | 0.115 (embedding) | 0.583 / 0.591 / 0.595 | 0.581 / 0.579 / 0.583 |
| `mm_unpaired_dense_8e_continued` | 8 | 71.7 | 84.9 (embedding) | 0.089 | 0.93 / 24 | 0.605 (embedding) | 0.352 | 0.126 (embedding) | 0.247 / 0.274 / 0.284 | 0.634 / 0.780 / 0.783 |
| `mm_unpaired_dense_private_lora_r128_2m_40e` | 13 | 77.7 | 84.8 (embedding) | 0.060 | 1.00 / 5 | 0.614 (embedding) | 0.353 | 0.113 (embedding) | 0.191 / 0.226 / 0.233 | 0.516 / 0.551 / 0.571 |
| `mm_unpaired_jepa_adversarial_1_2m_2e` | 2 | 68.1 | 84.8 (embedding) | 0.052 | 0.84 / 25 | 0.606 (embedding) | 0.352 | 0.134 (L2.mlp_out) | 0.138 / 0.152 / 0.151 | 0.563 / 0.606 / 0.582 |
| `mm_unpaired_jepa_fixedL6_1_2m_2e` | 2 | 66.5 | 84.8 (embedding) | 0.044 | 0.76 / 22 | 0.606 (embedding) | 0.352 | 0.114 (embedding) | 0.135 / 0.152 / 0.151 | 0.423 / 0.458 / 0.437 |
| `mm_unpaired_jepa_fixedL6_moments_1_2m_2e` | 2 | 70.6 | 84.8 (embedding) | 0.037 | 1.00 / 217 | 0.603 (embedding) | 0.349 | 0.140 (embedding) | 0.554 / 0.561 / 0.564 | 0.705 / 0.712 / 0.713 |
| `mm_unpaired_jepa_moments_1_2m_2e` | 2 | 70.9 | 84.8 (embedding) | 0.047 | 1.00 / 192 | 0.603 (embedding) | 0.350 | 0.140 (embedding) | 0.547 / 0.557 / 0.560 | 0.706 / 0.711 / 0.712 |
| `mm_unpaired_jepa_moments_w10_1_2m_2e` | 2 | 69.1 | 84.8 (embedding) | 0.028 | 1.00 / 105 | 0.603 (embedding) | 0.350 | 0.142 (embedding) | 0.545 / 0.563 / 0.566 | 0.720 / 0.728 / 0.730 |
| `mm_unpaired_merged_jepa_trunk_private_diff_1_2m_2e` | 2 | 60.7 | 84.8 (embedding) | 0.030 | 1.00 / 2 | 0.604 (embedding) | 0.350 | 0.128 (embedding) | 0.776 / 0.795 / 0.770 | 1.292 / 1.177 / 1.160 |
| `mm_unpaired_merged_moments_1_2m_2e` | 2 | 56.2 | 84.8 (embedding) | 0.002 | 1.00 / 2 | 0.603 (embedding) | 0.350 | 0.127 (embedding) | 0.360 / 0.411 / 0.417 | 0.493 / 0.569 / 0.490 |
| `mm_unpaired_moments_only_1_2m_2e` | 2 | 72.3 | 84.8 (embedding) | 0.071 | 0.92 / 25 | 0.604 (embedding) | 0.351 | 0.128 (embedding) | 0.243 / 0.264 / 0.272 | 0.645 / 0.754 / 0.763 |
| `mm_unpaired_private_dropout_gradbal_2m_40e` | 1 | 82.8 | 84.9 (embedding) | 0.117 | 0.86 / 18 | 0.699 (L7) | 0.352 | 0.139 (embedding) | 0.338 / 0.338 / 0.326 | 1.274 / 1.222 / 1.156 |
| `mm_unpaired_private_gradbal_2m_40e` | 1 | 81.3 | 84.9 (embedding) | 0.102 | 0.89 / 6 | 0.627 (L7) | 0.351 | 0.138 (embedding) | 0.378 / 0.461 / 0.435 | 0.857 / 0.875 / 0.919 |
| `mm_unpaired_stage2_trainable_1_2m_2e` | 2 | 72.7 | 84.9 (embedding) | 0.078 | 0.94 / 24 | 0.604 (embedding) | 0.351 | 0.128 (embedding) | 0.221 / 0.257 / 0.300 | 0.468 / 0.560 / 0.728 |
| `multimodal_paired_dense_1_2m_4e` | 4 | 86.0 | 86.0 (L7) | 0.191 | 0.93 / 26 | 0.603 (embedding) | 0.351 | 0.125 (L5) | 0.517 / 0.533 / 0.533 | 2.161 / 2.018 / 1.822 |
| `multimodal_unpaired_d2v_from_image_dense_1_2m_2e` | 2 | 58.6 | 84.9 (embedding) | 0.000 | 0.89 / 12 | 0.613 (embedding) | 0.349 | 0.115 (embedding) | 0.137 / 0.219 / 0.235 | 0.266 / 0.354 / 0.366 |
| `multimodal_unpaired_d2v_from_text_dense_1_2m_2e` | 2 | 89.9 | 92.9 (L6.mlp_out) | 0.221 | 0.84 / 26 | 0.886 (L6.mlp_out) | 0.359 | 0.359 (L5.mlp_out) | 0.206 / 0.234 / 0.235 | 1.642 / 1.756 / 1.640 |
| `multimodal_unpaired_d2v_from_unpaired_dense_ema_1_2m_2e` | 2 | 68.2 | 84.8 (embedding) | 0.052 | 0.84 / 25 | 0.606 (embedding) | 0.352 | 0.134 (L2.mlp_out) | 0.138 / 0.152 / 0.151 | 0.563 / 0.606 / 0.582 |
| `multimodal_unpaired_d2v_from_unpaired_dense_sigreg_1_2m_2e` | 2 | 55.8 | 84.8 (embedding) | 0.014 | 0.00 / 14 | 0.603 (embedding) | 0.350 | 0.110 (embedding) | 0.147 / 0.185 / 0.079 | 0.168 / 0.151 / 0.353 |
| `multimodal_unpaired_dense_1_2m_4e` | 4 | 73.1 | 84.9 (embedding) | 0.071 | 0.89 / 27 | 0.604 (embedding) | 0.350 | 0.130 (embedding) | 0.242 / 0.259 / 0.271 | 0.624 / 0.705 / 0.721 |
| `multimodal_unpaired_lejepa_scratch_1_2m_4e` | 4 | 54.7 | 84.7 (embedding) | 0.008 | 0.00 / 41 | 0.598 (embedding) | 0.345 | 0.095 (L0.mlp_out) | 0.031 / 0.043 / 0.088 | 0.587 / 0.499 / 0.585 |
| `random` |  | — | — | — | — | 0.613 (embedding) | 0.350 | 0.157 (L0) | 0.037 / 0.040 / 0.042 | 0.905 / 0.913 / 0.908 |
| `ref_text_dense_4e` |  | — | — | — | — | — | — | — | 0.323 / 0.300 / 0.296 | 1.828 / 1.769 / 1.617 |
| `text_data2vec_from_dense_2m_2e` | 2 | 87.1 | 91.0 (L5.mlp_out) | 0.131 | 0.96 / 19 | 0.884 (L5.mlp_out) | 0.378 | 0.213 (L4.mlp_out) | 0.178 / 0.203 / 0.201 | 0.813 / 0.832 / 0.802 |
| `text_data2vec_from_dense_avg_l4to7_2m_2e` | 2 | 87.2 | 91.0 (L5.mlp_out) | 0.211 | 0.94 / 21 | 0.880 (L5.mlp_out) | 0.396 | 0.230 (L4.mlp_out) | 0.254 / 0.280 / 0.277 | 1.355 / 1.331 / 1.261 |
| `text_data2vec_from_dense_fixedL6_all8_2m_2e` | 2 | 90.2 | 93.9 (L5.mlp_out) | 0.245 | 0.85 / 19 | 0.917 (L5.mlp_out) | 0.416 | 0.363 (L2.mlp_out) | 0.268 / 0.278 / 0.280 | 2.866 / 2.806 / 2.543 |
| `text_data2vec_from_dense_fixedL6_l4to7_2m_2e` | 2 | 89.8 | 93.7 (L6.mlp_out) | 0.243 | 0.85 / 19 | 0.917 (L5.mlp_out) | 0.414 | 0.272 (L6) | 0.266 / 0.280 / 0.279 | 2.616 / 2.543 / 2.317 |
| `text_data2vec_from_dense_layerwise_all_2m_2e` | 2 | 89.4 | 92.6 (L6.mlp_out) | 0.194 | 0.89 / 20 | 0.891 (L6.mlp_out) | 0.419 | 0.281 (L4.mlp_out) | 0.244 / 0.256 / 0.256 | 1.963 / 1.959 / 1.839 |
| `text_data2vec_from_dense_layerwise_all_2m_8e_continued` | 8 | 88.5 | 91.9 (L6.mlp_out) | 0.182 | 0.87 / 23 | 0.875 (L5.mlp_out) | 0.430 | 0.263 (L6) | 0.203 / 0.220 / 0.224 | 1.314 / 1.341 / 1.285 |
| `text_data2vec_from_dense_layerwise_all_w12_24_45pct_2m_2e` | 2 | 89.9 | 93.0 (L5.mlp_out) | 0.235 | 0.88 / 21 | 0.923 (L5.mlp_out) | 0.412 | 0.298 (L6) | 0.233 / 0.246 / 0.246 | 2.335 / 2.335 / 2.180 |
| `text_data2vec_from_dense_layerwise_all_w12_24_45pct_2m_2e_seed20260930` | 2 | 89.9 | 93.1 (L5.mlp_out) | 0.229 | 0.87 / 21 | 0.926 (L5.mlp_out) | 0.410 | 0.298 (L6) | 0.249 / 0.261 / 0.262 | 2.294 / 2.282 / 2.122 |
| `text_data2vec_from_dense_layerwise_all_w12_24_45pct_2m_6e_continued` | 6 | 91.4 | 93.9 (L6.mlp_out) | 0.221 | 0.87 / 27 | 0.941 (L5.mlp_out) | 0.426 | 0.331 (L6) | 0.198 / 0.214 / 0.220 | 1.421 / 1.446 / 1.401 |
| `text_data2vec_from_dense_layerwise_all_w4_8_15pct_2m_2e` | 2 | 91.3 | 95.1 (L6.mlp_out) | 0.165 | 0.92 / 23 | 0.945 (L6.mlp_out) | 0.407 | 0.271 (L5.mlp_out) | 0.281 / 0.301 / 0.299 | 1.714 / 1.761 / 1.687 |
| `text_data2vec_from_dense_layerwise_all_w4_8_15pct_2m_2e_seed20260930` | 2 | 91.6 | 95.1 (L6.mlp_out) | 0.168 | 0.91 / 22 | 0.939 (L6.mlp_out) | 0.415 | 0.289 (L5.mlp_out) | 0.279 / 0.301 / 0.300 | 1.708 / 1.758 / 1.686 |
| `text_data2vec_from_dense_layerwise_all_w4_8_15pct_2m_6e_continued` | 6 | 91.6 | 95.9 (L6.mlp_out) | 0.148 | 0.92 / 25 | 0.955 (L5.mlp_out) | 0.419 | 0.231 (L7) | 0.244 / 0.273 / 0.271 | 1.141 / 1.212 / 1.193 |
| `text_data2vec_from_dense_layerwise_l3to6_2m_2e` | 1 | 91.2 | 92.2 (L5.mlp_out) | 0.189 | 0.92 / 23 | 0.929 (L5.mlp_out) | 0.357 | 0.207 (L6) | 0.329 / 0.307 / 0.303 | 1.922 / 1.862 / 1.697 |
| `text_data2vec_from_dense_layerwise_l4to7_2m_2e` | 2 | 89.4 | 92.8 (L6.mlp_out) | 0.189 | 0.89 / 20 | 0.893 (L6.mlp_out) | 0.418 | 0.266 (L4.mlp_out) | 0.243 / 0.262 / 0.262 | 1.823 / 1.824 / 1.723 |
| `text_data2vec_from_ep5_layerwise_all_2m_2e` | 2 | 90.7 | 94.0 (L6.mlp_out) | 0.223 | 0.91 / 23 | 0.908 (L5.mlp_out) | 0.429 | 0.281 (L4.mlp_out) | 0.254 / 0.266 / 0.266 | 2.003 / 1.985 / 1.877 |
| `text_data2vec_scratch_layerwise_all_joint_2m_4e` | 1 | 84.3 | 84.9 (embedding) | 0.122 | 1.00 / 38 | 0.612 (L0) | 0.351 | 0.160 (L1) | 0.030 / 0.033 / 0.035 | 0.879 / 0.900 / 0.867 |
| `text_data2vec_scratch_layerwise_all_randt_2m_4e` | 4 | 62.8 | 84.9 (embedding) | 0.014 | 0.65 / 16 | 0.612 (embedding) | 0.351 | 0.114 (L3.mlp_out) | 0.072 / 0.088 / 0.099 | 0.351 / 0.346 / 0.276 |
| `text_data2vec_scratch_window12_24_30pct_avg_l4to7_2m_4e` | 4 | 85.8 | 89.5 (L5.mlp_out) | 0.225 | 0.79 / 24 | 0.810 (L5.mlp_out) | 0.357 | 0.312 (L7) | 0.125 / 0.132 / 0.134 | 1.669 / 1.645 / 1.541 |
| `text_data2vec_scratch_window12_24_30pct_layerwise_l4to7_2m_4e` | 4 | 80.0 | 85.1 (L3) | 0.052 | 0.67 / 20 | 0.694 (L3) | 0.359 | 0.219 (L6.mlp_out) | 0.090 / 0.102 / 0.114 | 0.452 / 0.479 / 0.485 |
| `text_data2vec_scratch_window12_24_60pct_avg_l4to7_2m_4e` | 4 | 76.7 | 84.9 (embedding) | 0.145 | 0.88 / 20 | 0.613 (embedding) | 0.351 | 0.228 (L0) | 0.155 / 0.171 / 0.175 | 0.968 / 0.930 / 0.850 |
| `text_data2vec_scratch_window12_24_60pct_layerwise_l4to7_2m_4e` | 4 | 75.2 | 84.9 (embedding) | 0.028 | 0.65 / 20 | 0.686 (L3) | 0.351 | 0.218 (L4) | 0.117 / 0.134 / 0.149 | 0.484 / 0.503 / 0.473 |
| `text_data2vec_scratch_window4_8_15pct_avg_l4to7_2m_4e` | 4 | 84.4 | 86.0 (L6) | 0.145 | 0.89 / 21 | 0.731 (L6) | 0.361 | 0.237 (L6) | 0.074 / 0.081 / 0.084 | 0.984 / 1.007 / 0.950 |
| `text_data2vec_scratch_window4_8_15pct_layerwise_l4to7_2m_4e` | 4 | 76.0 | 84.9 (embedding) | 0.067 | 0.73 / 17 | 0.655 (L5) | 0.357 | 0.220 (L6.mlp_out) | 0.056 / 0.065 / 0.070 | 0.570 / 0.588 / 0.549 |
| `text_dense_diffusion_2m_12e_continued` | 12 | 92.0 | 93.9 (L5.mlp_out) | 0.100 | 0.94 / 14 | 0.944 (L5.mlp_out) | 0.356 | 0.220 (L6) | 0.379 / 0.313 / 0.298 | 2.271 / 1.926 / 1.565 |
| `text_dense_diffusion_2m_20e_continued` | 20 | 93.0 | 94.9 (L5.mlp_out) | 0.169 | 0.94 / 25 | 0.962 (L5.mlp_out) | 0.368 | 0.298 (L6) | 0.603 / 0.462 / 0.441 | 1.827 / 1.847 / 1.683 |
| `text_dense_diffusion_2m_40e_continued` | 40 | 93.2 | 94.9 (L6.mlp_out) | 0.113 | 0.95 / 26 | 0.967 (L5.mlp_out) | 0.377 | 0.297 (L6) | 0.707 / 0.480 / 0.451 | 1.705 / 1.660 / 1.483 |
| `text_dense_diffusion_2m_4e_matched` | 4 | 91.2 | 92.0 (L5.mlp_out) | 0.186 | 0.91 / 23 | 0.924 (L5.mlp_out) | 0.358 | 0.207 (L6) | 0.323 / 0.300 / 0.296 | 1.828 / 1.769 / 1.617 |
| `text_dense_diffusion_2m_8e_continued` | 8 | 92.0 | 94.7 (L5.mlp_out) | 0.133 | 0.94 / 26 | 0.958 (L5.mlp_out) | 0.355 | 0.246 (L6) | 0.462 / 0.359 / 0.351 | 1.793 / 1.744 / 1.586 |
| `text_dense_private_frozen_hsic_2m_2e` | 2 | 87.1 | 91.0 (L5.mlp_out) | 0.131 | 0.96 / 19 | 0.884 (L5.mlp_out) | 0.378 | 0.213 (L4.mlp_out) | 0.178 / 0.203 / 0.201 | 0.813 / 0.832 / 0.802 |
| `text_dense_private_frozen_hsic_from_d2v_lw_all_2m_2e` | 2 | 89.4 | 92.6 (L6.mlp_out) | 0.194 | 0.89 / 20 | 0.891 (L6.mlp_out) | 0.419 | 0.281 (L4.mlp_out) | 0.244 / 0.256 / 0.256 | 1.963 / 1.959 / 1.839 |
| `text_dense_private_trainable_hsic_2m_2e` | 2 | 91.9 | 93.3 (L5.mlp_out) | 0.149 | 0.97 / 23 | 0.939 (L5.mlp_out) | 0.371 | 0.237 (L6) | 0.255 / 0.248 / 0.246 | 1.423 / 1.425 / 1.333 |
| `text_dense_private_trainable_hsic_from_d2v_lw_all_2m_2e` | 2 | 92.7 | 95.0 (L5.mlp_out) | 0.185 | 0.96 / 27 | 0.954 (L5.mlp_out) | 0.388 | 0.259 (L5.mlp_out) | 0.320 / 0.291 / 0.288 | 1.758 / 1.728 / 1.599 |
| `text_ijepa_w4_8_from_dense_ep12_2m_6e` | 6 | 93.6 | 96.8 (L6.mlp_out) | 0.173 | 0.95 / 26 | 0.969 (L5.mlp_out) | 0.416 | 0.231 (L6) | 0.286 / 0.320 / 0.317 | 1.411 / 1.497 / 1.477 |
| `text_ijepa_w4_8_from_dense_ep12_2m_full` | 29 | 92.9 | 94.2 (L5.mlp_out) | 0.139 | 0.91 / 38 | 0.912 (L7) | 0.409 | 0.223 (L6) | 0.295 / 0.313 / 0.318 | 0.995 / 1.025 / 1.023 |
| `text_ijepa_w4_8_from_dense_ep16_2m_6e` | 6 | 93.2 | 97.2 (L6.mlp_out) | 0.195 | 0.96 / 28 | 0.981 (L5.mlp_out) | 0.431 | 0.249 (L6) | 0.298 / 0.328 / 0.328 | 1.508 / 1.570 / 1.565 |
| `text_ijepa_w4_8_from_dense_ep16_2m_full` | 25 | 92.7 | 94.8 (L6.mlp_out) | 0.158 | 0.93 / 38 | 0.939 (L5.mlp_out) | 0.423 | 0.225 (L7) | 0.305 / 0.323 / 0.327 | 1.058 / 1.083 / 1.085 |
| `text_ijepa_w4_8_from_dense_ep20_2m_6e` | 6 | 93.3 | 97.2 (L7.mlp_out) | 0.209 | 0.96 / 28 | 0.982 (L5.mlp_out) | 0.434 | 0.285 (L4.mlp_out) | 0.302 / 0.332 / 0.334 | 1.617 / 1.696 / 1.681 |
| `text_ijepa_w4_8_from_dense_ep20_2m_full` | 21 | 93.0 | 95.6 (L6.mlp_out) | 0.173 | 0.94 / 36 | 0.959 (L5.mlp_out) | 0.427 | 0.232 (L7) | 0.305 / 0.325 / 0.332 | 1.099 / 1.133 / 1.142 |
| `text_ijepa_w4_8_from_dense_ep4_2m_6e` | 6 | 91.4 | 96.0 (L6.mlp_out) | 0.148 | 0.92 / 24 | 0.950 (L5.mlp_out) | 0.419 | 0.232 (L5.mlp_out) | 0.245 / 0.275 / 0.272 | 1.137 / 1.209 / 1.190 |
| `text_ijepa_w4_8_from_dense_ep4_2m_full` | 37 | 89.7 | 92.3 (L6.mlp_out) | 0.166 | 0.88 / 40 | 0.893 (L6.mlp_out) | 0.395 | 0.261 (L7) | 0.261 / 0.272 / 0.280 | 0.944 / 0.959 / 0.954 |
| `text_ijepa_w4_8_from_dense_ep8_2m_6e` | 6 | 93.1 | 97.0 (L6.mlp_out) | 0.139 | 0.94 / 26 | 0.965 (L5.mlp_out) | 0.423 | 0.215 (L6) | 0.244 / 0.277 / 0.278 | 1.190 / 1.279 / 1.268 |
| `text_ijepa_w4_8_from_dense_ep8_2m_full` | 33 | 92.1 | 93.5 (L5.mlp_out) | 0.126 | 0.90 / 41 | 0.890 (L5.mlp_out) | 0.406 | 0.212 (L7) | 0.260 / 0.277 / 0.280 | 0.868 / 0.899 / 0.896 |
| `text_jepa_scratch_fidelity_dprime_bert15_k4_2m_4e` | 4 | 62.1 | 84.8 (embedding) | -0.007 | 0.94 / 79 | 0.613 (embedding) | 0.349 | 0.176 (L4.mlp_out) | 0.390 / 0.446 / 0.454 | 0.476 / 0.512 / 0.523 |
| `text_lejepa_from_dense_layerwise_all_2m_2e` | 2 | 55.6 | 84.8 (embedding) | 0.008 | 0.04 / 19 | 0.608 (embedding) | 0.351 | 0.053 (embedding) | 0.293 / 0.292 / 0.120 | 0.207 / 0.207 / 0.398 |
| `text_lejepa_paper_from_dense_ep20_6e` | 1 | 83.5 | 84.9 (embedding) | 0.045 | 0.96 / 19 | 0.676 (L7) | 0.354 | 0.101 (embedding) | 0.619 / 0.617 / 0.599 | 0.342 / 0.358 / 0.386 |
| `text_lejepa_scratch_layerwise_all_2m_4e` | 4 | 55.0 | 84.8 (embedding) | 0.004 | 0.04 / 27 | 0.606 (embedding) | 0.349 | 0.046 (embedding) | 0.052 / 0.053 / 0.010 | 0.658 / 0.656 / 0.702 |
| `text_lejepa_w4_8_from_dense_ep20_2m_6e` | 3 | 58.3 | 84.7 (embedding) | 0.064 | 0.11 / 18 | 0.611 (embedding) | 0.350 | 0.048 (embedding) | 0.389 / 0.392 / 0.417 | 0.107 / 0.109 / 0.110 |
| `text_merged_jepa_trunk_private_diffusion_from_ep5_2m_4e` | 4 | 54.4 | 84.7 (embedding) | -0.015 | 0.97 / 2 | 0.612 (embedding) | 0.351 | 0.075 (embedding) | 0.497 / 0.712 / 0.699 | 0.747 / 0.792 / 0.777 |
| `text_stage2_frozen_from_ep5_d2v_lw_all_2m_2e` | 2 | 90.7 | 94.0 (L6.mlp_out) | 0.223 | 0.91 / 23 | 0.908 (L5.mlp_out) | 0.429 | 0.281 (L4.mlp_out) | 0.254 / 0.266 / 0.266 | 2.003 / 1.985 / 1.877 |
| `text_stage2_frozen_from_ijepa_w12_24_2m_2e` | 2 | 89.9 | 93.0 (L5.mlp_out) | 0.235 | 0.88 / 21 | 0.923 (L5.mlp_out) | 0.412 | 0.298 (L6) | 0.233 / 0.246 / 0.246 | 2.335 / 2.335 / 2.180 |
| `text_stage2_frozen_from_ijepa_w4_8_2m_2e` | 2 | 91.3 | 95.1 (L6.mlp_out) | 0.165 | 0.92 / 23 | 0.945 (L6.mlp_out) | 0.407 | 0.271 (L5.mlp_out) | 0.281 / 0.301 / 0.299 | 1.714 / 1.761 / 1.687 |

## 5. JEPA prediction check (mean over supervised layers)

| model | modality | raw C₊ / C₋ / Δ | centred C₊ / C₋ / Δ |
|---|---|---|---|
| `image_data2vec_from_dense_avg_all_randt_1_2m_2e__e001` | image | 0.956 / 0.830 / 0.126 | 0.952 / 0.812 / **0.139** |
| `image_data2vec_from_dense_avg_l4to7_randt_1_2m_2e__e001` | image | 0.940 / 0.644 / 0.296 | 0.931 / 0.586 / **0.345** |
| `image_data2vec_from_dense_layerwise_all_block2d_1_2m_2e__e001` | image | 0.885 / 0.616 / 0.268 | 0.867 / 0.549 / **0.318** |
| `image_data2vec_from_dense_layerwise_all_block2d_1_2m_6e_continued__e005` | image | 0.904 / 0.638 / 0.266 | 0.879 / 0.530 / **0.349** |
| `image_data2vec_from_dense_layerwise_all_ijepa_2m_2e__e001` | image | 0.876 / 0.623 / 0.253 | 0.858 / 0.560 / **0.298** |
| `image_data2vec_from_dense_layerwise_all_ijepa_2m_2e_seed20260930__e001` | image | 0.877 / 0.625 / 0.252 | 0.859 / 0.562 / **0.297** |
| `image_data2vec_from_dense_layerwise_all_ijepa_2m_6e_continued__e005` | image | 0.896 / 0.640 / 0.255 | 0.871 / 0.542 / **0.329** |
| `image_data2vec_from_dense_layerwise_all_randt_1_2m_2e__e001` | image | 0.915 / 0.651 / 0.263 | 0.904 / 0.599 / **0.305** |
| `image_data2vec_from_dense_layerwise_l4to7_randt_1_2m_2e__e001` | image | 0.922 / 0.541 / 0.381 | 0.909 / 0.458 / **0.450** |
| `image_data2vec_scratch_block2d_30pct_avg_l4to7_1_2m_4e__e003` | image | 0.981 / 0.937 / 0.043 | 0.980 / 0.936 / **0.045** |
| `image_data2vec_scratch_block2d_30pct_layerwise_l4to7_1_2m_4e__e003` | image | 0.978 / 0.926 / 0.052 | 0.977 / 0.922 / **0.055** |
| `image_data2vec_scratch_block2d_65pct_avg_l4to7_1_2m_4e__e003` | image | 0.991 / 0.957 / 0.033 | 0.991 / 0.956 / **0.034** |
| `image_data2vec_scratch_block2d_65pct_layerwise_l4to7_1_2m_4e__e003` | image | 0.990 / 0.948 / 0.042 | 0.990 / 0.946 / **0.044** |
| `image_ijepa_sweep_blk_s05_t60__e003` | image | 0.957 / 0.764 / 0.193 | 0.917 / 0.541 / **0.375** |
| `image_ijepa_sweep_blk_s10_t30__e003` | image | 0.951 / 0.763 / 0.188 | 0.898 / 0.515 / **0.383** |
| `image_ijepa_sweep_blk_s15_t45__e003` | image | 0.944 / 0.760 / 0.184 | 0.888 / 0.523 / **0.365** |
| `image_ijepa_sweep_blk_s15_t60__e003` | image | 0.944 / 0.760 / 0.184 | 0.888 / 0.523 / **0.365** |
| `image_ijepa_sweep_blk_s30_t30__e003` | image | 0.945 / 0.761 / 0.183 | 0.884 / 0.507 / **0.377** |
| `image_ijepa_sweep_blk_s30_t60__e003` | image | 0.932 / 0.764 / 0.168 | 0.864 / 0.533 / **0.331** |
| `image_ijepa_sweep_blk_tiny_t45__e003` | image | 0.973 / 0.766 / 0.207 | 0.951 / 0.558 / **0.392** |
| `image_ijepa_sweep_blk_wide_t45__e003` | image | 0.946 / 0.760 / 0.186 | 0.893 / 0.525 / **0.367** |
| `image_ijepa_sweep_rand_t45__e003` | image | 0.985 / 0.745 / 0.240 | 0.979 / 0.585 / **0.395** |
| `image_ijepa_sweep_rand_t75__e003` | image | 0.967 / 0.757 / 0.210 | 0.944 / 0.569 / **0.375** |
| `image_lejepa_blk_s15_t45_from_dense_ep12__e000` | image | 0.997 / 0.950 / 0.048 | 0.996 / 0.927 / **0.069** |
| `mm_paired_merged_jepa_trunk_private_diff_1_2m_3e__e001` | image | 0.939 / 0.939 / 0.001 | 0.334 / 0.316 / **0.018** |
| `mm_paired_merged_jepa_trunk_private_diff_1_2m_3e__e001` | text | 0.915 / 0.914 / 0.001 | 0.103 / 0.085 / **0.018** |
| `mm_paired_stage1_jepa_lw_all_1_2m_2e__e001` | image | 0.851 / 0.723 / 0.128 | 0.832 / 0.656 / **0.177** |
| `mm_paired_stage1_jepa_lw_all_1_2m_2e__e001` | text | 0.754 / 0.541 / 0.213 | 0.715 / 0.442 / **0.273** |
| `mm_trunk_ijepa_from_ep6_4e__e003` | image | 0.935 / 0.758 / 0.177 | 0.871 / 0.512 / **0.359** |
| `mm_trunk_ijepa_from_ep6_4e__e003` | text | 0.972 / 0.635 / 0.338 | 0.943 / 0.223 / **0.720** |
| `mm_trunk_ijepa_private_diff_from_ep6_4e__e003` | image | 0.934 / 0.758 / 0.176 | 0.870 / 0.512 / **0.358** |
| `mm_trunk_ijepa_private_diff_from_ep6_4e__e003` | text | 0.972 / 0.636 / 0.337 | 0.943 / 0.225 / **0.718** |
| `mm_trunk_ijepa_sigreg_from_ep6_4e__e003` | image | 0.965 / 0.930 / 0.035 | 0.821 / 0.573 / **0.248** |
| `mm_trunk_ijepa_sigreg_from_ep6_4e__e003` | text | 0.983 / 0.895 / 0.088 | 0.902 / 0.248 / **0.654** |
| `mm_unpaired_jepa_adversarial_1_2m_2e__e001` | image | 0.923 / 0.665 / 0.258 | 0.902 / 0.575 / **0.327** |
| `mm_unpaired_jepa_adversarial_1_2m_2e__e001` | text | 0.798 / 0.526 / 0.273 | 0.724 / 0.351 / **0.373** |
| `mm_unpaired_jepa_fixedL6_1_2m_2e__e001` | image | 0.869 / 0.465 / 0.404 | 0.841 / 0.344 / **0.497** |
| `mm_unpaired_jepa_fixedL6_1_2m_2e__e001` | text | 0.712 / 0.351 / 0.362 | 0.615 / 0.145 / **0.470** |
| `mm_unpaired_jepa_fixedL6_moments_1_2m_2e__e001` | image | 0.976 / 0.976 / 0.000 | 0.848 / 0.845 / **0.003** |
| `mm_unpaired_jepa_fixedL6_moments_1_2m_2e__e001` | text | 0.972 / 0.963 / 0.009 | 0.839 / 0.786 / **0.053** |
| `mm_unpaired_jepa_moments_1_2m_2e__e001` | image | 0.964 / 0.963 / 0.001 | 0.838 / 0.834 / **0.004** |
| `mm_unpaired_jepa_moments_1_2m_2e__e001` | text | 0.956 / 0.943 / 0.013 | 0.826 / 0.767 / **0.059** |
| `mm_unpaired_jepa_moments_w10_1_2m_2e__e001` | image | 0.959 / 0.958 / 0.000 | 0.833 / 0.832 / **0.002** |
| `mm_unpaired_jepa_moments_w10_1_2m_2e__e001` | text | 0.949 / 0.943 / 0.007 | 0.819 / 0.792 / **0.027** |
| `mm_unpaired_merged_jepa_trunk_private_diff_1_2m_2e__e001` | image | 0.970 / 0.964 / 0.006 | 0.662 / 0.533 / **0.129** |
| `mm_unpaired_merged_jepa_trunk_private_diff_1_2m_2e__e001` | text | 0.718 / 0.661 / 0.057 | 0.356 / 0.203 / **0.153** |
| `mm_unpaired_merged_moments_1_2m_2e__e001` | image | 0.549 / 0.549 / 0.000 | 0.047 / 0.046 / **0.001** |
| `mm_unpaired_merged_moments_1_2m_2e__e001` | text | 0.495 / 0.494 / 0.001 | 0.064 / 0.063 / **0.001** |
| `multimodal_unpaired_d2v_from_image_dense_1_2m_2e__e001` | image | 0.913 / 0.647 / 0.267 | 0.902 / 0.592 / **0.310** |
| `multimodal_unpaired_d2v_from_image_dense_1_2m_2e__e001` | text | 0.931 / 0.792 / 0.140 | 0.915 / 0.720 / **0.195** |
| `multimodal_unpaired_d2v_from_text_dense_1_2m_2e__e001` | image | 0.921 / 0.876 / 0.044 | 0.890 / 0.822 / **0.067** |
| `multimodal_unpaired_d2v_from_text_dense_1_2m_2e__e001` | text | 0.803 / 0.551 / 0.252 | 0.700 / 0.313 / **0.387** |
| `multimodal_unpaired_d2v_from_unpaired_dense_ema_1_2m_2e__e001` | image | 0.923 / 0.665 / 0.258 | 0.902 / 0.575 / **0.327** |
| `multimodal_unpaired_d2v_from_unpaired_dense_ema_1_2m_2e__e001` | text | 0.798 / 0.526 / 0.273 | 0.724 / 0.351 / **0.373** |
| `multimodal_unpaired_d2v_from_unpaired_dense_sigreg_1_2m_2e__e001` | image | 0.907 / 0.061 / 0.845 | 0.909 / 0.057 / **0.852** |
| `multimodal_unpaired_d2v_from_unpaired_dense_sigreg_1_2m_2e__e001` | text | 0.824 / 0.337 / 0.487 | 0.716 / 0.133 / **0.583** |
| `multimodal_unpaired_lejepa_scratch_1_2m_4e__e003` | image | 0.860 / 0.031 / 0.829 | 0.862 / 0.028 / **0.834** |
| `multimodal_unpaired_lejepa_scratch_1_2m_4e__e003` | text | 0.822 / 0.259 / 0.563 | 0.702 / 0.101 / **0.601** |
| `text_data2vec_from_dense_2m_2e__e001` | text | 0.953 / 0.776 / 0.176 | 0.872 / 0.397 / **0.475** |
| `text_data2vec_from_dense_avg_l4to7_2m_2e__e001` | text | 0.911 / 0.565 / 0.347 | 0.826 / 0.179 / **0.647** |
| `text_data2vec_from_dense_fixedL6_all8_2m_2e__e001` | text | 0.810 / 0.338 / 0.472 | 0.725 / 0.078 / **0.647** |
| `text_data2vec_from_dense_fixedL6_l4to7_2m_2e__e001` | text | 0.843 / 0.366 / 0.477 | 0.760 / 0.079 / **0.680** |
| `text_data2vec_from_dense_layerwise_all_2m_2e__e001` | text | 0.877 / 0.496 / 0.380 | 0.797 / 0.217 / **0.580** |
| `text_data2vec_from_dense_layerwise_all_2m_8e_continued__e007` | text | 0.872 / 0.462 / 0.410 | 0.790 / 0.160 / **0.630** |
| `text_data2vec_from_dense_layerwise_all_w12_24_45pct_2m_2e__e001` | text | 0.813 / 0.472 / 0.341 | 0.707 / 0.181 / **0.526** |
| `text_data2vec_from_dense_layerwise_all_w12_24_45pct_2m_2e_seed20260930__e001` | text | 0.814 / 0.477 / 0.337 | 0.707 / 0.182 / **0.525** |
| `text_data2vec_from_dense_layerwise_all_w12_24_45pct_2m_6e_continued__e005` | text | 0.852 / 0.545 / 0.307 | 0.723 / 0.137 / **0.587** |
| `text_data2vec_from_dense_layerwise_all_w4_8_15pct_2m_2e__e001` | text | 0.945 / 0.426 / 0.519 | 0.914 / 0.151 / **0.763** |
| `text_data2vec_from_dense_layerwise_all_w4_8_15pct_2m_2e_seed20260930__e001` | text | 0.944 / 0.423 / 0.522 | 0.914 / 0.151 / **0.763** |
| `text_data2vec_from_dense_layerwise_all_w4_8_15pct_2m_6e_continued__e005` | text | 0.957 / 0.480 / 0.477 | 0.925 / 0.125 / **0.799** |
| `text_data2vec_from_dense_layerwise_l3to6_2m_2e__e000` | text | 0.262 / 0.219 / 0.044 | 0.089 / 0.018 / **0.071** |
| `text_data2vec_from_dense_layerwise_l4to7_2m_2e__e001` | text | 0.872 / 0.436 / 0.436 | 0.788 / 0.119 / **0.669** |
| `text_data2vec_from_ep5_layerwise_all_2m_2e__e001` | text | 0.894 / 0.582 / 0.312 | 0.790 / 0.223 / **0.567** |
| `text_data2vec_scratch_layerwise_all_joint_2m_4e__e000` | text | 0.152 / 0.152 / 0.000 | 0.049 / 0.049 / **0.000** |
| `text_data2vec_scratch_layerwise_all_randt_2m_4e__e003` | text | 0.895 / 0.451 / 0.445 | 0.859 / 0.282 / **0.577** |
| `text_data2vec_scratch_window12_24_30pct_avg_l4to7_2m_4e__e003` | text | 0.860 / 0.450 / 0.410 | 0.797 / 0.210 / **0.587** |
| `text_data2vec_scratch_window12_24_30pct_layerwise_l4to7_2m_4e__e003` | text | 0.881 / 0.446 / 0.435 | 0.828 / 0.185 / **0.644** |
| `text_data2vec_scratch_window12_24_60pct_avg_l4to7_2m_4e__e003` | text | 0.821 / 0.513 / 0.308 | 0.731 / 0.271 / **0.460** |
| `text_data2vec_scratch_window12_24_60pct_layerwise_l4to7_2m_4e__e003` | text | 0.823 / 0.466 / 0.357 | 0.739 / 0.197 / **0.542** |
| `text_data2vec_scratch_window4_8_15pct_avg_l4to7_2m_4e__e003` | text | 0.969 / 0.446 / 0.523 | 0.954 / 0.208 / **0.745** |
| `text_data2vec_scratch_window4_8_15pct_layerwise_l4to7_2m_4e__e003` | text | 0.974 / 0.417 / 0.558 | 0.965 / 0.193 / **0.772** |
| `text_ijepa_w4_8_from_dense_ep12_2m_6e__e005` | text | 0.968 / 0.661 / 0.307 | 0.912 / 0.129 / **0.783** |
| `text_ijepa_w4_8_from_dense_ep12_2m_full__e028` | text | 0.968 / 0.534 / 0.434 | 0.935 / 0.063 / **0.872** |
| `text_ijepa_w4_8_from_dense_ep16_2m_6e__e005` | text | 0.972 / 0.708 / 0.264 | 0.909 / 0.129 / **0.780** |
| `text_ijepa_w4_8_from_dense_ep16_2m_full__e024` | text | 0.967 / 0.569 / 0.398 | 0.929 / 0.069 / **0.860** |
| `text_ijepa_w4_8_from_dense_ep20_2m_6e__e005` | text | 0.974 / 0.743 / 0.232 | 0.905 / 0.125 / **0.780** |
| `text_ijepa_w4_8_from_dense_ep20_2m_full__e020` | text | 0.971 / 0.636 / 0.335 | 0.924 / 0.077 / **0.847** |
| `text_ijepa_w4_8_from_dense_ep4_2m_6e__e005` | text | 0.957 / 0.481 / 0.475 | 0.925 / 0.125 / **0.799** |
| `text_ijepa_w4_8_from_dense_ep4_2m_full__e036` | text | 0.971 / 0.509 / 0.461 | 0.940 / 0.052 / **0.889** |
| `text_ijepa_w4_8_from_dense_ep8_2m_6e__e005` | text | 0.962 / 0.594 / 0.368 | 0.914 / 0.129 / **0.785** |
| `text_ijepa_w4_8_from_dense_ep8_2m_full__e032` | text | 0.968 / 0.524 / 0.444 | 0.934 / 0.056 / **0.878** |
| `text_jepa_scratch_fidelity_dprime_bert15_k4_2m_4e__e003` | text | 0.975 / 0.052 / 0.923 | 0.975 / 0.052 / **0.923** |
| `text_lejepa_from_dense_layerwise_all_2m_2e__e001` | text | 0.906 / 0.171 / 0.736 | 0.908 / 0.176 / **0.733** |
| `text_lejepa_scratch_layerwise_all_2m_4e__e003` | text | 0.949 / 0.064 / 0.885 | 0.950 / 0.062 / **0.888** |
| `text_lejepa_w4_8_from_dense_ep20_2m_6e__e002` | text | 0.996 / 0.754 / 0.241 | 0.996 / 0.736 / **0.259** |
| `text_merged_jepa_trunk_private_diffusion_from_ep5_2m_4e__e003` | text | 0.919 / 0.766 / 0.153 | 0.815 / 0.449 / **0.366** |
