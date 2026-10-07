#!/bin/bash
# Unattended: waits for the dense image run to reach BRANCH_EP, then sweeps
# I-JEPA masking settings branched from it, evaluates each, and ranks them.
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
Q=outputs/image_jepa_sweep; L=$Q/locks; mkdir -p $Q $L
BRANCH_EP=${BRANCH_EP:-12}          # dense epoch to branch from (1-indexed)
SWEEP_EPOCHS=${SWEEP_EPOCHS:-4}
MAXPAR=${MAXPAR:-2}
DENSE=outputs/image_dense_diffusion_2_5m_40e
CKPT=$(printf '%s/epoch_%03d.pt' "$DENSE" "$((BRANCH_EP-1))")
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "waiting for $CKPT (dense epoch $BRANCH_EP)"
while [ ! -f "$CKPT" ]; do sleep 600; done
sleep 120                            # let the file finish being written
log "branch checkpoint ready: $CKPT"

mapfile -t RUNS < <(python3 scripts/gen_image_jepa_sweep.py "$CKPT" "$SWEEP_EPOCHS")
log "generated ${#RUNS[@]} sweep configs"

claim_gpu(){ while :; do for g in 0 1 2 3; do
    if mkdir "$L/$g" 2>/dev/null; then
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
      if [ "$u" -lt 12000 ]; then echo $g; return; fi
      rmdir "$L/$g" 2>/dev/null
    fi
  done; sleep 180; done; }

run_one(){
  local c=$1 g; g=$(claim_gpu)
  if [ -f "outputs/$c/epoch_$(printf %03d $((SWEEP_EPOCHS-1))).pt" ]; then
    log "SKIP $c (already trained)"; rmdir "$L/$g" 2>/dev/null; return 0; fi
  log "START $c on GPU $g"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g python3 train_multimodal.py \
      --config configs/$c.yaml > $Q/$c.train.log 2>&1
  if [ $? -ne 0 ]; then log "FAILED $c"; rmdir "$L/$g" 2>/dev/null; return 1; fi
  log "TRAINED $c"
  local last; last=$(ls outputs/$c/epoch_*.pt 2>/dev/null | grep -v adapter | sort | tail -1)
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g python3 evaluate_image_binding.py \
      --per-layer --checkpoint "${c}=$last" --seed 20260924 \
      --image-cache outputs/image_only_2_5m_token_cache/val_tokens.pt \
      --output-dir outputs/image_binding_sweep >> $Q/$c.eval.log 2>&1 \
      && log "EVALED $c" || log "EVAL FAILED $c"
  rmdir "$L/$g" 2>/dev/null
}

for c in "${RUNS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge $MAXPAR ]; do sleep 120; done
  run_one "$c" & sleep 240
done
wait
log "ALL SWEEP ARMS DONE - ranking"
python3 scripts/rank_image_jepa_sweep.py > $Q/RESULTS.md 2>&1 || log "ranking failed"
log "phase 1 complete - see $Q/RESULTS.md"

# ---- phase 2: re-run the winning setting from a deeper dense trunk ----
DEEP_EP=${DEEP_EP:-24}
DEEP_EPOCHS=${DEEP_EPOCHS:-6}
BEST=$(grep -oE '^\*\*Best setting: `[^`]+`' $Q/RESULTS.md | sed 's/.*`\(.*\)`/\1/')
if [ -z "$BEST" ]; then log "no winner parsed, stopping"; exit 0; fi
log "winner = $BEST ; waiting for dense epoch $DEEP_EP"
DEEPCK=$(printf '%s/epoch_%03d.pt' "$DENSE" "$((DEEP_EP-1))")
while [ ! -f "$DEEPCK" ]; do sleep 600; done
sleep 120
log "deep branch ready: $DEEPCK"
python3 scripts/gen_image_jepa_sweep.py "$DEEPCK" "$DEEP_EPOCHS" >/dev/null
WIN=image_ijepa_sweep_$BEST
DEEPRUN=image_ijepa_best_${BEST}_from_ep${DEEP_EP}
cp configs/$WIN.yaml configs/$DEEPRUN.yaml
sed -i "s|outputs/$WIN|outputs/$DEEPRUN|g; s|$WIN|$DEEPRUN|g" configs/$DEEPRUN.yaml
g=$(claim_gpu); log "START $DEEPRUN on GPU $g"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g python3 train_multimodal.py \
    --config configs/$DEEPRUN.yaml > $Q/$DEEPRUN.train.log 2>&1 \
  && log "TRAINED $DEEPRUN" || { log "FAILED $DEEPRUN"; rmdir "$L/$g" 2>/dev/null; exit 1; }
LASTD=$(ls outputs/$DEEPRUN/epoch_*.pt | grep -v adapter | sort | tail -1)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g python3 evaluate_image_binding.py \
    --per-layer --checkpoint "${DEEPRUN}=$LASTD" --seed 20260924 \
    --image-cache outputs/image_only_2_5m_token_cache/val_tokens.pt \
    --output-dir outputs/image_binding_sweep >> $Q/$DEEPRUN.eval.log 2>&1 && log "EVALED $DEEPRUN"
rmdir "$L/$g" 2>/dev/null
python3 scripts/rank_image_jepa_sweep.py > $Q/RESULTS.md 2>&1
log "SWEEP COMPLETE (both phases) - see $Q/RESULTS.md"
