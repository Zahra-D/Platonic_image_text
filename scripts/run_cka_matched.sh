#!/bin/bash
# Matched-epoch cross-modal CKA: JEPA pair vs dense pair where each model has
# the same number of training epochs as its counterpart.
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
free_gpu(){ while :; do for g in 0 1 2 3; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$u" -lt 9000 ] && { echo $g; return; }; done; sleep 120; done; }
# wait for the renderers to release the GPUs
while tmux has-session -t triples 2>/dev/null; do sleep 120; done
g=$(free_gpu); log "matched CKA on GPU $g"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g python3 evaluate_cross_modal_structure.py \
  --text-checkpoint  dense26_TEXT=outputs/text_dense_diffusion_2m_40e_continued/epoch_025.pt \
  --text-checkpoint  ijepa26_TEXT=outputs/text_ijepa_w4_8_from_dense_ep20_2m_6e/epoch_005.pt \
  --image-checkpoint dense16_IMAGE=outputs/image_dense_diffusion_2_5m_40e/epoch_015.pt \
  --image-checkpoint ijepa16_IMAGE=outputs/image_ijepa_sweep_blk_s15_t45/epoch_003.pt \
  --image-cache outputs/image_only_2_5m_token_cache/val_tokens.pt \
  --num-scenes 4000 --seed 20260922 --output-dir outputs/cka_matched \
  && log "MATCHED CKA DONE" || log "FAILED"
