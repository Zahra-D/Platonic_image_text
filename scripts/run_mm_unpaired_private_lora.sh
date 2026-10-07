#!/bin/bash
# 40-epoch unpaired multimodal diffusion: shared dense trunk (grad from both
# modalities) + modality-private LoRA (r128). Holds the image sweep's GPU lock
# (owner file "mmprivate") so its queued arms do not stack onto this GPU.
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
RUN=mm_unpaired_dense_private_lora_r128_2m_40e; G=${GPU:-3}; L=outputs/image_jepa_sweep/locks
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
[ -e outputs/$RUN ] && { log "REFUSE: outputs/$RUN exists"; exit 1; }
until mkdir $L/$G 2>/dev/null || grep -qx mmprivate $L/$G/owner 2>/dev/null; do sleep 120; done
echo mmprivate > $L/$G/owner
mkdir -p outputs/$RUN; log "START $RUN on GPU $G"
CUDA_VISIBLE_DEVICES=$G python3 train_multimodal.py --config configs/$RUN.yaml > outputs/$RUN/launcher.log 2>&1 \
  && log "TRAINED $RUN" || log "FAILED $RUN"
rm -f $L/$G/owner; rmdir $L/$G 2>/dev/null
