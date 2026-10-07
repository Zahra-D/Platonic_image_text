#!/usr/bin/env bash
set -euo pipefail
# Whole-layer cross-pattern retrieval for every final text checkpoint on the
# shared retrieval gallery.  Results: outputs/layer_retrieval/<label>/.
cd /home/zd25e122/clevr_discrete_diffusion
py=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
gallery=outputs/shared_private_semantic_template_retrieval/plain_lora_epoch003/variants.jsonl
while read -r label ckpt; do
  [ -e "outputs/layer_retrieval/$label/results.json" ] && { echo "skip $label"; continue; }
  echo "== $label"
  $py evaluate_layer_retrieval.py --checkpoint "$ckpt" --gallery "$gallery" --output "outputs/layer_retrieval/$label" 2>&1 | grep -v -i warn
done <<'LIST'
dense_epoch003 outputs/text_dense_diffusion_2m_4e_matched/epoch_003.pt
plain_lora_epoch003 outputs/text_lora_diffusion_2m_4e_matched/epoch_003.pt
avg_no_hsic_epoch003 outputs/text_module_jepa_avg_no_hsic_2m_4e/epoch_003.pt
avg_hsic_epoch003 outputs/text_module_jepa_avg_hsic_2m_4e/epoch_003.pt
layerwise_no_hsic_epoch003 outputs/text_module_jepa_layerwise_no_hsic_2m_4e/epoch_003.pt
layerwise_hsic_epoch003 outputs/text_module_jepa_layerwise_hsic_2m_4e/epoch_003.pt
avg_hsic_calibrated_old_epoch003 outputs/text_module_jepa_avg_hsic_calibrated_2m_4e/epoch_003.pt
data2vec_avg_gated_epoch003 outputs/text_data2vec_avg_gated_shared_calibrated_jepa_hsic_2m_4e/epoch_003.pt
data2vec_noavg_gated_epoch003 outputs/text_data2vec_noavg_gated_shared_calibrated_jepa_hsic_2m_4e/epoch_003.pt
LIST
