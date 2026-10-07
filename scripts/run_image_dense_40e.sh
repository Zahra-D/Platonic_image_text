#!/bin/bash
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
Q=outputs/image40; mkdir -p $Q
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
C=image_dense_diffusion_2_5m_40e
free_gpu(){ while :; do for g in 0 1 2 3; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$u" -lt 2000 ] && { echo $g; return; }; done; sleep 120; done; }
g=$(free_gpu); log "TRAIN $C on GPU $g (40 epochs x 2.53M images)"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g python3 train_multimodal.py \
  --config configs/$C.yaml > $Q/train.log 2>&1 || { log "FAILED $C"; exit 1; }
log "TRAINED $C"
