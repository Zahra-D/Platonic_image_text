# Existing evaluation results (collected, not re-run)

Collected by `scripts/collect_existing_evals.py` from earlier eval folders; newest result per checkpoint and eval. **Mixed script versions**: strict binding, centred/z-scored sensitivity and the JEPA prediction check exist only for recent runs. `view`: trunk = shared trunk only (private LoRA dropped), full = whole model. All layers are in `outputs/existing_evals_long.csv`.

## text (41 checkpoint views)

| run | epoch | view | probe L7 | probe best | RSA L7 | cos / rank L7 | conj mAP best | scene ρ_conj best | S_bind L7 | S_cont L7 |
|---|---|---|---|---|---|---|---|---|---|---|
| `mm_trunk_ijepa_from_ep6_4e` | 0 | trunk | 0.738 | 0.849 (embedding) | 0.099 | 0.95 / 9 | 0.614 (embedding) | 0.242 (L4.mlp_out) | 0.425 | 0.907 |
| `mm_trunk_ijepa_from_ep6_4e` | 1 | trunk | 0.729 | 0.849 (embedding) | 0.099 | 0.95 / 9 | 0.614 (embedding) | 0.220 (L4.mlp_out) | 0.418 | 0.869 |
| `mm_trunk_ijepa_from_ep6_4e` | 2 | trunk | 0.728 | 0.849 (embedding) | 0.101 | 0.95 / 10 | 0.615 (embedding) | — | 0.418 | 0.864 |
| `mm_trunk_ijepa_from_ep6_4e` | 3 | trunk | 0.728 | 0.849 (embedding) | 0.103 | 0.95 / 11 | 0.616 (embedding) | — | 0.416 | 0.824 |
| `mm_trunk_ijepa_private_diff_from_ep6_4e` | 0 | trunk | 0.738 | 0.849 (embedding) | 0.100 | 0.95 / 9 | 0.614 (embedding) | 0.241 (L3.mlp_out) | 0.428 | 0.914 |
| `mm_trunk_ijepa_private_diff_from_ep6_4e` | 1 | trunk | 0.729 | 0.849 (embedding) | 0.101 | 0.95 / 10 | 0.614 (embedding) | 0.239 (L4.mlp_out) | 0.421 | 0.874 |
| `mm_trunk_ijepa_private_diff_from_ep6_4e` | 2 | trunk | 0.729 | 0.849 (embedding) | 0.105 | 0.95 / 10 | 0.615 (embedding) | 0.220 (L1.mlp_out) | 0.418 | 0.870 |
| `mm_trunk_ijepa_sigreg_from_ep6_4e` | 0 | trunk | 0.680 | 0.849 (embedding) | 0.024 | 0.97 / 48 | 0.614 (embedding) | 0.123 (embedding) | 0.595 | 0.624 |
| `mm_trunk_ijepa_sigreg_from_ep6_4e` | 1 | trunk | 0.663 | 0.849 (embedding) | 0.019 | 0.96 / 46 | 0.614 (embedding) | 0.127 (L1.mlp_out) | 0.594 | 0.601 |
| `mm_trunk_ijepa_sigreg_from_ep6_4e` | 2 | trunk | 0.661 | 0.849 (embedding) | 0.020 | 0.96 / 45 | 0.614 (embedding) | 0.117 (embedding) | 0.593 | 0.583 |
| `mm_trunk_ijepa_sigreg_from_ep6_4e` | 3 | trunk | 0.652 | 0.849 (embedding) | 0.028 | 0.95 / 44 | 0.614 (embedding) | 0.115 (embedding) | 0.583 | 0.581 |
| `mm_unpaired_dense_private_lora_r128_2m_40e` | 0 | full | 0.806 | 0.849 (embedding) | 0.108 | 0.85 / 14 | 0.714 (L7) | 0.149 (L1.mlp_out) | 0.328 | 1.159 |
| `mm_unpaired_dense_private_lora_r128_2m_40e` | 0 | trunk | 0.805 | 0.849 (embedding) | 0.118 | 0.96 / 15 | 0.612 (embedding) | 0.137 (embedding) | 0.362 | 0.693 |
| `mm_unpaired_dense_private_lora_r128_2m_40e` | 1 | full | 0.841 | 0.849 (embedding) | 0.181 | 0.93 / 15 | 0.742 (L7) | 0.170 (L5) | 0.357 | 2.333 |
| `mm_unpaired_dense_private_lora_r128_2m_40e` | 1 | trunk | 0.811 | 0.849 (embedding) | 0.140 | 0.99 / 14 | 0.613 (embedding) | 0.134 (embedding) | 0.412 | 1.079 |
| `mm_unpaired_dense_private_lora_r128_2m_40e` | 3 | trunk | 0.809 | 0.849 (embedding) | 0.127 | 1.00 / 10 | 0.613 (embedding) | 0.136 (L1.mlp_out) | 0.372 | 1.083 |
| `mm_unpaired_dense_private_lora_r128_2m_40e` | 5 | trunk | 0.800 | 0.849 (embedding) | 0.107 | 1.00 / 10 | 0.614 (embedding) | 0.126 (L1.mlp_out) | 0.342 | 0.734 |
| `mm_unpaired_dense_private_lora_r128_2m_40e` | 6 | trunk | 0.797 | 0.849 (embedding) | 0.109 | 1.00 / 10 | 0.614 (embedding) | 0.123 (embedding) | 0.344 | 0.679 |
| `mm_unpaired_dense_private_lora_r128_2m_40e` | 9 | trunk | 0.783 | 0.849 (embedding) | 0.071 | 1.00 / 8 | 0.614 (embedding) | — | 0.249 | 0.587 |
| `text_data2vec_from_dense_layerwise_all_2m_2e` | 1 | full | 0.895 | 0.895 (L7) | 0.199 | 0.89 / 20 | — | — | — | — |
| `text_data2vec_from_dense_layerwise_all_w12_24_45pct_2m_2e` | 1 | full | 0.900 | 0.900 (L7) | 0.241 | 0.88 / 21 | — | — | — | — |
| `text_data2vec_from_dense_layerwise_all_w4_8_15pct_2m_2e` | 1 | full | 0.913 | 0.951 (L6.mlp_out) | 0.165 | 0.92 / 23 | — | — | — | — |
| `text_data2vec_scratch_layerwise_all_randt_2m_4e` | 3 | full | — | — | — | — | 0.612 (embedding) | 0.114 (L3.mlp_out) | 0.072 | 0.351 |
| `text_dense_diffusion_2m_12e_continued` | 9 | full | 0.930 | 0.953 (L5.mlp_out) | 0.162 | 0.92 / 25 | 0.966 (L5.mlp_out) | 0.257 (L6) | 0.538 | 1.885 |
| `text_dense_diffusion_2m_20e_continued` | 19 | full | — | — | — | — | 0.962 (L5.mlp_out) | 0.298 (L6) | 0.603 | 1.827 |
| `text_dense_diffusion_2m_40e_continued` | 20 | full | 0.919 | 0.946 (L5.mlp_out) | 0.119 | 0.93 / 24 | — | — | 0.608 | 1.826 |
| `text_dense_diffusion_2m_40e_continued` | 21 | full | 0.928 | 0.948 (L6.mlp_out) | 0.096 | 0.91 / 22 | — | — | 0.497 | 1.766 |
| `text_dense_diffusion_2m_40e_continued` | 25 | full | 0.930 | 0.949 (L5.mlp_out) | 0.116 | 0.95 / 24 | — | — | — | — |
| `text_dense_diffusion_2m_40e_continued` | 39 | full | — | — | — | — | 0.967 (L5.mlp_out) | 0.297 (L6) | 0.707 | 1.705 |
| `text_dense_diffusion_2m_4e_matched` | 0 | full | 0.847 | 0.849 (L6) | 0.115 | 0.82 / 17 | 0.731 (L7) | 0.134 (L5) | 0.276 | 1.302 |
| `text_dense_diffusion_2m_4e_matched` | 1 | full | 0.869 | 0.869 (L7) | 0.169 | 0.85 / 20 | 0.805 (L6.mlp_out) | 0.197 (L6) | 0.294 | 1.551 |
| `text_dense_diffusion_2m_4e_matched` | 3 | full | 0.912 | 0.920 (L5.mlp_out) | 0.186 | 0.91 / 23 | 0.924 (L5.mlp_out) | 0.207 (L6) | 0.323 | 1.828 |
| `text_dense_diffusion_2m_8e_continued` | 6 | full | 0.927 | 0.945 (L5.mlp_out) | 0.161 | 0.94 / 25 | 0.952 (L5.mlp_out) | 0.241 (L6) | 0.506 | 1.897 |
| `text_dense_private_frozen_hsic_2m_2e` | 0 | full | 0.888 | 0.890 (L6) | 0.156 | 0.82 / 19 | — | — | — | — |
| `text_ijepa_w4_8_from_dense_ep20_2m_6e` | 0 | full | 0.926 | 0.967 (L6.mlp_out) | 0.255 | 0.96 / 24 | — | — | 0.302 | 2.429 |
| `text_ijepa_w4_8_from_dense_ep20_2m_6e` | 1 | full | 0.928 | 0.969 (L6.mlp_out) | 0.237 | 0.96 / 25 | — | — | 0.297 | 2.229 |
| `text_ijepa_w4_8_from_dense_ep20_2m_6e` | 5 | full | 0.933 | 0.972 (L7.mlp_out) | 0.209 | 0.96 / 28 | 0.982 (L5.mlp_out) | 0.285 (L4.mlp_out) | 0.302 | 1.617 |
| `text_ijepa_w4_8_from_dense_ep4_2m_6e` | 5 | full | — | — | — | — | 0.950 (L5.mlp_out) | 0.232 (L5.mlp_out) | 0.245 | 1.137 |
| `text_lejepa_paper_from_dense_ep20_6e` | 0 | full | 0.835 | 0.849 (embedding) | 0.045 | 0.96 / 19 | — | — | 0.619 | 0.342 |
| `text_lejepa_w4_8_from_dense_ep20_2m_6e` | 0 | full | 0.585 | 0.848 (embedding) | 0.064 | 0.15 / 15 | — | — | 0.366 | 0.141 |
| `text_lejepa_w4_8_from_dense_ep20_2m_6e` | 1 | full | 0.589 | 0.848 (embedding) | 0.067 | 0.11 / 16 | — | — | 0.352 | 0.123 |

## image (67 checkpoint views)

| run | epoch | view | probe L7 | probe best | RSA L7 | cos / rank L7 | conj mAP best | scene ρ_conj best | binding L7 / best | triples d′ centred L7 | S L7 raw / centred |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `image_data2vec_from_dense_avg_all_randt_1_2m_2e` | 1 | full | — | — | — | — | — | — | 0.708 / 0.729 | 0.254 | — |
| `image_data2vec_from_dense_avg_l4to7_randt_1_2m_2e` | 1 | full | — | — | — | — | — | — | 0.739 / 0.749 | 0.218 | — |
| `image_data2vec_from_dense_layerwise_all_block2d_1_2m_2e` | 1 | full | 0.812 | 0.812 (L7) | 0.186 | 0.75 / 31 | — | — | 0.822 / 0.822 | 0.333 | — |
| `image_data2vec_from_dense_layerwise_all_block2d_1_2m_6e_continued` | 5 | full | — | — | — | — | — | — | 0.824 / 0.897 | 0.387 | — |
| `image_data2vec_from_dense_layerwise_all_ijepa_2m_2e` | 1 | full | 0.809 | 0.822 (L7.mlp_out) | 0.188 | 0.73 / 31 | — | — | 0.819 / 0.888 | 0.370 | — |
| `image_data2vec_from_dense_layerwise_all_ijepa_2m_2e_seed20260930` | 1 | full | — | — | — | — | — | — | 0.814 / 0.893 | 0.338 | — |
| `image_data2vec_from_dense_layerwise_all_ijepa_2m_6e_continued` | 5 | full | — | — | — | — | — | — | 0.817 / 0.885 | 0.293 | — |
| `image_data2vec_from_dense_layerwise_all_randt_1_2m_2e` | 1 | full | — | — | — | — | — | — | 0.815 / 0.815 | 0.220 | — |
| `image_data2vec_from_dense_layerwise_l4to7_randt_1_2m_2e` | 1 | full | — | — | — | — | — | — | 0.815 / 0.815 | 0.325 | — |
| `image_data2vec_scratch_block2d_30pct_avg_l4to7_1_2m_4e` | 3 | full | 0.616 | 0.679 (embedding) | 0.095 | 0.57 / 22 | — | — | 0.554 / 0.626 | 0.101 | — |
| `image_data2vec_scratch_block2d_30pct_layerwise_l4to7_1_2m_4e` | 3 | full | 0.604 | 0.680 (embedding) | 0.096 | 0.45 / 25 | 0.406 (embedding) | 0.133 (L2.mlp_out) | 0.533 / 0.626 | 0.085 | 0.124 / — |
| `image_data2vec_scratch_block2d_65pct_avg_l4to7_1_2m_4e` | 3 | full | — | — | — | — | — | — | 0.533 / 0.626 | 0.091 | — |
| `image_data2vec_scratch_block2d_65pct_layerwise_l4to7_1_2m_4e` | 3 | full | — | — | — | — | — | — | 0.521 / 0.630 | 0.097 | — |
| `image_dense_diffusion_1_2m_4e` | 3 | full | 0.807 | 0.807 (L7) | 0.104 | 0.85 / 19 | — | — | 0.803 / 0.803 | — | — |
| `image_dense_diffusion_1_2m_6e_continued` | 5 | full | — | — | — | — | — | — | 0.804 / 0.804 | 0.446 | — |
| `image_dense_diffusion_2_5m_40e` | 0 | full | 0.789 | 0.789 (L7) | 0.113 | 0.81 / 19 | 0.584 (L7.mlp_out) | 0.159 (L7) | 0.762 / 0.797 | 0.386 | 0.464 / — |
| `image_dense_diffusion_2_5m_40e` | 1 | full | 0.808 | 0.808 (L7) | 0.104 | 0.84 / 16 | 0.653 (L7.mlp_out) | 0.158 (L7) | 0.793 / 0.818 | 0.342 | 0.452 / — |
| `image_dense_diffusion_2_5m_40e` | 3 | full | 0.813 | 0.813 (L7) | 0.109 | 0.91 / 17 | 0.649 (L7.mlp_out) | 0.151 (L7) | 0.804 / 0.835 | 0.215 | 0.277 / — |
| `image_dense_diffusion_2_5m_40e` | 6 | full | 0.809 | 0.809 (L7) | 0.112 | 0.95 / 15 | 0.609 (L6.mlp_out) | 0.148 (L7) | 0.804 / 0.835 | 0.339 | 0.475 / — |
| `image_dense_diffusion_2_5m_40e` | 7 | full | — | — | — | — | — | — | — | 0.277 | — |
| `image_dense_diffusion_2_5m_40e` | 9 | full | 0.811 | 0.811 (L7) | 0.120 | 0.96 / 14 | 0.615 (L6.mlp_out) | 0.135 (L7) | 0.801 / 0.831 | 0.269 | 0.344 / — |
| `image_dense_diffusion_2_5m_40e` | 11 | full | — | — | — | — | 0.610 (L6.mlp_out) | 0.132 (L7) | 0.801 / 0.834 | 0.290 | 0.375 / — |
| `image_dense_diffusion_2_5m_40e` | 12 | full | 0.814 | 0.814 (L7) | 0.111 | 0.97 / 11 | — | — | — | — | 0.331 / — |
| `image_dense_diffusion_2_5m_40e` | 13 | full | — | — | — | — | — | — | 0.805 / 0.836 | — | — |
| `image_dense_diffusion_2_5m_40e` | 15 | full | 0.809 | 0.809 (L7) | 0.110 | 0.97 / 11 | 0.607 (L6.mlp_out) | — | 0.797 / 0.836 | 0.262 | 0.385 / — |
| `image_dense_diffusion_2_5m_40e` | 17 | full | — | — | — | — | — | — | 0.801 / 0.842 | — | — |
| `image_dense_diffusion_2_5m_40e` | 19 | full | — | — | — | — | — | — | 0.801 / 0.839 | 0.290 | — |
| `image_dense_diffusion_2_5m_40e` | 21 | full | — | — | — | — | — | — | 0.808 / 0.838 | — | — |
| `image_dense_diffusion_2_5m_40e` | 22 | full | — | — | — | — | — | — | 0.802 / 0.839 | 0.153 | — |
| `image_dense_diffusion_2_5m_40e` | 23 | full | — | — | — | — | — | — | — | 0.266 | — |
| `image_dense_diffusion_2_5m_40e` | 24 | full | — | — | — | — | 0.609 (L7) | 0.121 (L7) | — | — | 0.359 / — |
| `image_dense_private_frozen_hsic_from_d2v_lw_block2d_1_2m_2e` | 1 | full | — | — | — | — | — | — | 0.803 / 0.803 | 0.247 | — |
| `image_ijepa_sweep_blk_s05_t60` | 3 | full | — | — | — | — | — | — | 0.806 / 0.902 | 0.295 | — |
| `image_ijepa_sweep_blk_s10_t30` | 3 | full | — | — | — | — | — | — | 0.817 / 0.896 | 0.271 | — |
| `image_ijepa_sweep_blk_s15_t45` | 0 | full | 0.817 | 0.826 (L7.mlp_out) | 0.153 | 0.90 / 24 | — | — | — | — | 0.237 / — |
| `image_ijepa_sweep_blk_s15_t45` | 3 | full | 0.817 | 0.837 (L7.mlp_out) | 0.162 | 0.90 / 30 | 0.735 (L7.mlp_out) | 0.193 (L7) | 0.823 / 0.898 | 0.190 | 0.240 / — |
| `image_ijepa_sweep_blk_s15_t60` | 3 | full | — | — | — | — | — | — | 0.820 / 0.897 | 0.190 | — |
| `image_ijepa_sweep_blk_s30_t30` | 3 | full | — | — | — | — | — | — | 0.809 / 0.895 | 0.223 | — |
| `image_ijepa_sweep_blk_s30_t60` | 2 | full | — | — | — | — | — | — | — | 0.273 | — |
| `image_ijepa_sweep_blk_s30_t60` | 3 | full | — | — | — | — | 0.705 (L7.mlp_out) | 0.198 (L7) | 0.802 / 0.890 | — | 0.322 / — |
| `image_ijepa_sweep_blk_tiny_t45` | 3 | full | — | — | — | — | — | — | 0.812 / 0.904 | 0.232 | — |
| `image_ijepa_sweep_blk_wide_t45` | 3 | full | — | — | — | — | — | — | 0.818 / 0.900 | — | — |
| `image_ijepa_sweep_rand_t45` | 3 | full | — | — | — | — | — | — | 0.805 / 0.868 | — | — |
| `image_ijepa_sweep_rand_t75` | 3 | full | — | — | — | — | — | — | 0.820 / 0.885 | — | — |
| `image_lejepa_blk_s15_t45_from_dense_ep12` | 0 | full | 0.554 | 0.705 (embedding) | 0.039 | 0.02 / 22 | — | — | — | — | 0.483 / — |
| `image_lejepa_paper_from_dense_ep12_4e` | 3 | full | — | — | — | — | — | — | 0.644 / 0.779 | — | — |
| `image_stage2_frozen_from_ijepa_1_2m_2e` | 1 | full | — | — | — | — | — | — | — | 0.259 | — |
| `mm_trunk_ijepa_from_ep6_4e` | 0 | trunk | 0.789 | 0.811 (L7.mlp_out) | 0.163 | 0.85 / 23 | 0.651 (L7.mlp_out) | 0.178 (L7) | 0.770 / 0.852 | 0.202 | 0.305 / — |
| `mm_trunk_ijepa_from_ep6_4e` | 1 | trunk | 0.789 | 0.814 (L7.mlp_out) | 0.160 | 0.86 / 23 | 0.670 (L7.mlp_out) | 0.170 (L7) | 0.768 / 0.868 | 0.200 | 0.332 / — |
| `mm_trunk_ijepa_from_ep6_4e` | 2 | trunk | 0.791 | 0.818 (L7.mlp_out) | 0.161 | 0.87 / 25 | 0.681 (L7.mlp_out) | — | 0.767 / 0.873 | 0.173 | 0.257 / — |
| `mm_trunk_ijepa_from_ep6_4e` | 3 | trunk | 0.793 | 0.820 (L7.mlp_out) | 0.157 | 0.89 / 26 | 0.690 (L7.mlp_out) | — | 0.767 / 0.881 | 0.198 | 0.292 / — |
| `mm_trunk_ijepa_private_diff_from_ep6_4e` | 0 | trunk | 0.791 | 0.811 (L7.mlp_out) | 0.158 | 0.85 / 22 | 0.658 (L7.mlp_out) | 0.175 (L7) | 0.766 / 0.851 | 0.179 | 0.240 / — |
| `mm_trunk_ijepa_private_diff_from_ep6_4e` | 1 | trunk | 0.790 | 0.815 (L7.mlp_out) | 0.162 | 0.86 / 24 | 0.674 (L7.mlp_out) | 0.170 (L7) | 0.766 / 0.874 | 0.229 | 0.376 / — |
| `mm_trunk_ijepa_private_diff_from_ep6_4e` | 2 | trunk | 0.791 | 0.819 (L7.mlp_out) | 0.154 | 0.87 / 25 | 0.686 (L7.mlp_out) | 0.165 (L7) | 0.765 / 0.879 | 0.218 | 0.346 / — |
| `mm_trunk_ijepa_sigreg_from_ep6_4e` | 0 | trunk | 0.742 | 0.742 (L7) | 0.113 | 0.97 / 44 | 0.450 (L1) | 0.135 (L6) | 0.693 / 0.722 | 0.254 | 0.324 / — |
| `mm_trunk_ijepa_sigreg_from_ep6_4e` | 1 | trunk | 0.733 | 0.735 (L6) | 0.085 | 0.97 / 46 | 0.439 (L1) | 0.104 (L6) | 0.708 / 0.726 | 0.412 | 0.580 / — |
| `mm_trunk_ijepa_sigreg_from_ep6_4e` | 2 | trunk | 0.716 | 0.730 (L1) | 0.076 | 0.98 / 44 | 0.434 (L1) | 0.106 (L1) | 0.668 / 0.690 | 0.221 | 0.299 / — |
| `mm_trunk_ijepa_sigreg_from_ep6_4e` | 3 | trunk | 0.708 | 0.721 (L1) | 0.069 | 0.97 / 46 | 0.426 (L1) | 0.102 (L1) | 0.671 / 0.682 | 0.201 | 0.317 / — |
| `mm_unpaired_dense_private_lora_r128_2m_40e` | 0 | full | 0.783 | 0.803 (L6.mlp_out) | 0.096 | 0.84 / 14 | 0.611 (L6.mlp_out) | 0.154 (L7) | 0.741 / 0.803 | 0.310 | 0.399 / — |
| `mm_unpaired_dense_private_lora_r128_2m_40e` | 0 | trunk | 0.737 | 0.752 (L6.mlp_out) | 0.061 | 0.90 / 7 | 0.455 (L6.mlp_out) | 0.101 (L7) | 0.656 / 0.716 | 0.176 | 0.230 / — |
| `mm_unpaired_dense_private_lora_r128_2m_40e` | 1 | full | 0.804 | 0.821 (L6.mlp_out) | 0.106 | 0.91 / 14 | 0.685 (L6.mlp_out) | 0.155 (L7) | 0.774 / 0.834 | 0.259 | 0.298 / — |
| `mm_unpaired_dense_private_lora_r128_2m_40e` | 1 | trunk | 0.751 | 0.762 (L6.mlp_out) | 0.075 | 0.95 / 8 | 0.469 (L6.mlp_out) | 0.109 (L7) | 0.682 / 0.761 | 0.109 | 0.134 / — |
| `mm_unpaired_dense_private_lora_r128_2m_40e` | 3 | trunk | 0.761 | 0.770 (L6.mlp_out) | 0.092 | 0.98 / 9 | 0.485 (L6.mlp_out) | 0.117 (L7) | 0.714 / 0.784 | 0.172 | 0.227 / — |
| `mm_unpaired_dense_private_lora_r128_2m_40e` | 5 | trunk | 0.766 | 0.780 (L6.mlp_out) | 0.103 | 0.98 / 10 | 0.503 (L6.mlp_out) | 0.122 (L7) | 0.719 / 0.794 | 0.180 | 0.444 / — |
| `mm_unpaired_dense_private_lora_r128_2m_40e` | 6 | trunk | 0.774 | 0.787 (L6.mlp_out) | 0.111 | 0.99 / 10 | 0.519 (L6.mlp_out) | 0.125 (L7) | 0.723 / 0.808 | 0.155 | 0.366 / — |
| `mm_unpaired_dense_private_lora_r128_2m_40e` | 9 | trunk | 0.775 | 0.792 (L6.mlp_out) | 0.118 | 0.99 / 11 | 0.545 (L6.mlp_out) | — | 0.736 / 0.822 | 0.158 | 0.371 / — |
| `multimodal_paired_dense_1_2m_4e` | 3 | full | — | — | — | — | — | — | 0.791 / 0.791 | — | — |

