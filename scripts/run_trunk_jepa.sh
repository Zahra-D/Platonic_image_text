#!/bin/bash
# Three JEPA variants on the shared trunk of mm_unpaired_dense_private_lora_r128_2m_40e
# (init epoch_005), one per GPU. Holds the sweep's GPU locks (owner "trunkjepa").
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
Q=outputs/trunk_jepa; L=outputs/image_jepa_sweep/locks; mkdir -p $Q $L
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
train(){  # run gpu
  local r=$1 g=$2
  [ -e outputs/$r ] && { log "REFUSE $r: outputs/$r exists"; return 1; }
  until mkdir $L/$g 2>/dev/null || grep -qx trunkjepa $L/$g/owner 2>/dev/null; do sleep 120; done
  echo trunkjepa > $L/$g/owner; log "START $r on GPU $g"
  CUDA_VISIBLE_DEVICES=$g python3 train_multimodal.py --config configs/$r.yaml > $Q/$r.train.log 2>&1 \
    && log "TRAINED $r" || log "FAILED $r"
  rm -f $L/$g/owner; rmdir $L/$g 2>/dev/null
}
train mm_trunk_ijepa_from_ep6_4e              ${GPU_A:-0} &
sleep 90; train mm_trunk_ijepa_private_diff_from_ep6_4e ${GPU_B:-1} &
sleep 90; train mm_trunk_ijepa_sigreg_from_ep6_4e       ${GPU_C:-2} &
wait; log "ALL TRUNK JEPA RUNS DONE"
