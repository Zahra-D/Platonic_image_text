#!/bin/bash
# Corrected unpaired shared-trunk + private-LoRA base runs (see the config headers):
#   GPU_A: private-branch dropout 0.5 + per-modality trunk gradient balancing
#   GPU_B: gradient balancing only
# Holds the sweep's GPU locks (owner "privfix"); refuses to overwrite outputs.
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
Q=outputs/private_fix; L=outputs/image_jepa_sweep/locks; mkdir -p $Q $L
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
train(){  # run gpu
  local r=$1 g=$2
  [ -e outputs/$r ] && { log "REFUSE $r: outputs/$r exists"; return 1; }
  until mkdir $L/$g 2>/dev/null || grep -qx privfix $L/$g/owner 2>/dev/null; do sleep 120; done
  echo privfix > $L/$g/owner; log "START $r on GPU $g"
  CUDA_VISIBLE_DEVICES=$g python3 train_multimodal.py --config configs/$r.yaml > $Q/$r.train.log 2>&1 \
    && log "TRAINED $r" || log "FAILED $r"
  rm -f $L/$g/owner; rmdir $L/$g 2>/dev/null
}
train mm_unpaired_private_dropout_gradbal_2m_40e ${GPU_A:-0} &
sleep 120; train mm_unpaired_private_gradbal_2m_40e ${GPU_B:-3} &
wait; log "ALL DONE"
