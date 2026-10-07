#!/bin/bash
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
Q=outputs/ijepa_full; L=$Q/locks; mkdir -p $Q $L
MAXPAR=4
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
claim_gpu(){ while :; do for g in 0 1 2 3; do
    if mkdir "$L/$g" 2>/dev/null; then
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
      if [ "$u" -lt 2000 ]; then echo $g; return; fi
      rmdir "$L/$g" 2>/dev/null
    fi
  done; sleep 90; done; }
run_one(){
  local c=$1 g; g=$(claim_gpu)
  log "START $c on GPU $g"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g python3 train_multimodal.py \
      --config configs/$c.yaml > $Q/$c.train.log 2>&1
  if [ $? -ne 0 ]; then log "FAILED $c"; rmdir "$L/$g" 2>/dev/null; return 1; fi
  log "TRAINED $c"
  # score every checkpoint by TOTAL epochs seen (init + ijepa epochs)
  local init=${c#text_ijepa_w4_8_from_dense_ep}; init=${init%%_*}
  local args=()
  for f in $(ls outputs/$c/epoch_*.pt 2>/dev/null | grep -v adapter | sort); do
    local n=$(basename $f .pt); n=${n#epoch_}; n=$((10#$n))
    local tot=$((init + n + 1))
    [ -f "outputs/semantic_dprime_lexmatched/ijfull_from${init}_tot${tot}.json" ] && continue
    args+=(--checkpoint "ijfull_from${init}_tot${tot}=$f")
  done
  if [ ${#args[@]} -gt 0 ]; then
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g python3 evaluate_semantic_dprime.py \
      --lexicon-matched --num-worlds 2000 --seed 20260922 --batch-size 64 \
      --output-dir outputs/semantic_dprime_lexmatched "${args[@]}" >> $Q/$c.eval.log 2>&1 \
      && log "EVALED $c ($((${#args[@]}/2)) checkpoints)"
  fi
  rmdir "$L/$g" 2>/dev/null
}
# longest first
for c in text_ijepa_w4_8_from_dense_ep4_2m_full \
         text_ijepa_w4_8_from_dense_ep8_2m_full \
         text_ijepa_w4_8_from_dense_ep12_2m_full \
         text_ijepa_w4_8_from_dense_ep16_2m_full \
         text_ijepa_w4_8_from_dense_ep20_2m_full; do
  [ -f "configs/$c.yaml" ] || { log "SKIP $c"; continue; }
  while [ "$(jobs -rp | wc -l)" -ge $MAXPAR ]; do sleep 60; done
  run_one "$c" & sleep 180
done
wait
log "IJEPA FULL COMPLETE"
