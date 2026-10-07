#!/usr/bin/env bash
set -euo pipefail
# Binding-swap minimal-pair evaluation on every final text-only checkpoint.
# Two tmux jobs (GPUs 2 and 3) share one output directory; each skips labels
# that already have a result.  Job A also writes the model-free controls.
cd /home/zd25e122/clevr_discrete_diffusion
out=outputs/binding_swap_eval
py=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
mkdir -p "$out"
# Each job rebuilds the captions and items from the same seed; construction is
# deterministic, so both jobs score identical data (checked by hash afterwards).
tmux new-session -d -s clevr_binding_swap_a "cd $PWD && exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${1:-2} PYTHONUNBUFFERED=1 $py evaluate_binding_swap.py --output-dir $out --controls \
  --checkpoint dense_epoch003=outputs/text_dense_diffusion_2m_4e_matched/epoch_003.pt \
  --checkpoint plain_lora_epoch003=outputs/text_lora_diffusion_2m_4e_matched/epoch_003.pt \
  --checkpoint layerwise_no_hsic_epoch003=outputs/text_module_jepa_layerwise_no_hsic_2m_4e/epoch_003.pt \
  --checkpoint layerwise_hsic_epoch003=outputs/text_module_jepa_layerwise_hsic_2m_4e/epoch_003.pt \
  --checkpoint data2vec_avg_gated_epoch003=outputs/text_data2vec_avg_gated_shared_calibrated_jepa_hsic_2m_4e/epoch_003.pt > $out/launcher_a.log 2>&1"
tmux new-session -d -s clevr_binding_swap_b "cd $PWD && exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${2:-3} PYTHONUNBUFFERED=1 $py evaluate_binding_swap.py --output-dir $out \
  --checkpoint avg_no_hsic_epoch003=outputs/text_module_jepa_avg_no_hsic_2m_4e/epoch_003.pt \
  --checkpoint avg_hsic_epoch003=outputs/text_module_jepa_avg_hsic_2m_4e/epoch_003.pt \
  --checkpoint avg_hsic_calibrated_old_epoch003=outputs/text_module_jepa_avg_hsic_calibrated_2m_4e/epoch_003.pt \
  --checkpoint data2vec_noavg_gated_epoch003=outputs/text_data2vec_noavg_gated_shared_calibrated_jepa_hsic_2m_4e/epoch_003.pt > $out/launcher_b.log 2>&1"
echo "started clevr_binding_swap_a and clevr_binding_swap_b"
