#!/bin/bash
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
Q=outputs/ijepa_init_sweep; L=$Q/locks; mkdir -p $Q $L
MAXPAR=4
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# Claim a GPU: it must be physically idle AND not locked by a sibling that is
# still in its slow startup window (model load takes ~90s before memory shows).
claim_gpu(){ while :; do for g in 0 1 2 3; do
    if mkdir "$L/$g" 2>/dev/null; then
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
      if [ "$u" -lt 800 ]; then echo $g; return; fi
      rmdir "$L/$g" 2>/dev/null
    fi
  done; sleep 60; done; }

run_one(){ # name  kind(train|resume)
  local c=$1
  local g; g=$(claim_gpu)
  log "START $c on GPU $g"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g python3 train_multimodal.py \
      --config configs/$c.yaml > $Q/$c.train.log 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then log "FAILED $c (rc=$rc)"; rmdir "$L/$g" 2>/dev/null; return 1; fi
  log "TRAINED $c"
  local last; last=$(ls outputs/$c/epoch_*.pt 2>/dev/null | grep -v adapter | sort | tail -1)
  if [ -n "$last" ]; then
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g python3 evaluate_semantic_dprime.py \
      --lexicon-matched --num-worlds 2000 --seed 20260922 --batch-size 64 \
      --output-dir outputs/semantic_dprime_lexmatched \
      --checkpoint "${c}=$last" >> $Q/$c.eval.log 2>&1 && log "EVALED $c"
  fi
  rmdir "$L/$g" 2>/dev/null
}

# Longest job first so it is not left stranded at the end.
JOBS=(text_dense_diffusion_2m_40e_continued
      text_ijepa_w4_8_from_dense_ep20_2m_6e
      text_ijepa_w4_8_from_dense_ep16_2m_6e
      text_ijepa_w4_8_from_dense_ep12_2m_6e
      text_ijepa_w4_8_from_dense_ep8_2m_6e
      text_ijepa_w4_8_from_dense_ep4_2m_6e)
for c in "${JOBS[@]}"; do
  [ -f "configs/$c.yaml" ] || { log "SKIP $c (no config)"; continue; }
  while [ "$(jobs -rp | wc -l)" -ge $MAXPAR ]; do sleep 30; done
  run_one "$c" &
  sleep 150   # let this job allocate before the next claim scan
done
wait
log "INIT SWEEP COMPLETE"
