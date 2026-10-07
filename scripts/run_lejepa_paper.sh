#!/bin/bash
# Paper-faithful LeJEPA (multi-crop, projector, per-view SIGReg) mirrors of the
# two I-JEPA references, then the cone / probe / CKA eval matched to
# outputs/cka_matched (same scenes and seed).
#
#   ijepa16_IMAGE = image_ijepa_sweep_blk_s15_t45          -> image_lejepa_paper_from_dense_ep12_4e
#   ijepa26_TEXT  = text_ijepa_w4_8_from_dense_ep20_2m_6e  -> text_lejepa_paper_from_dense_ep20_6e
#
# Holds the image sweep's GPU locks (owner file "lejepa") while training so its
# queued arms do not stack onto these GPUs. Refuses to overwrite an output dir.
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
Q=outputs/lejepa_paper; L=outputs/image_jepa_sweep/locks; mkdir -p $Q $L
IMG=image_lejepa_paper_from_dense_ep12_4e
TXT=text_lejepa_paper_from_dense_ep20_6e
GPU_IMG=${GPU_IMG:-2}; GPU_TXT=${GPU_TXT:-3}
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
lock(){ until mkdir $L/$1 2>/dev/null || grep -qx lejepa $L/$1/owner 2>/dev/null; do sleep 120; done
        echo lejepa > $L/$1/owner; }
unlock(){ rm -f $L/$1/owner; rmdir $L/$1 2>/dev/null; }

train(){  # config gpu last-epoch-index
  local c=$1 g=$2 last=$3 rc
  if [ -e outputs/$c ]; then log "REFUSE $c: outputs/$c exists"; return 1; fi
  lock $g; log "START $c on GPU $g"
  CUDA_VISIBLE_DEVICES=$g python3 train_multimodal.py --config configs/$c.yaml > $Q/$c.train.log 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ "$c" = "$IMG" ]; then
    # same binding eval as the sweep, so the run lands in its ranking table
    CUDA_VISIBLE_DEVICES=$g python3 evaluate_image_binding.py \
      --per-layer --checkpoint "${c}=outputs/$c/epoch_$last.pt" --seed 20260924 \
      --image-cache outputs/image_only_2_5m_token_cache/val_tokens.pt \
      --output-dir outputs/image_binding_sweep >> $Q/$c.eval.log 2>&1 \
      && log "EVALED $c" || log "EVAL FAILED $c"
  fi
  [ $rc -eq 0 ] && log "TRAINED $c" || log "FAILED $c (rc=$rc)"
  unlock $g
  return $rc
}

train $IMG $GPU_IMG 003 & train $TXT $GPU_TXT 005 &
wait
if [ -f outputs/$IMG/epoch_003.pt ] && [ -f outputs/$TXT/epoch_005.pt ]; then
  lock $GPU_TXT; log "cone / CKA eval on GPU $GPU_TXT"
  CUDA_VISIBLE_DEVICES=$GPU_TXT python3 evaluate_cross_modal_structure.py \
    --text-checkpoint  lejepaP26_TEXT=outputs/$TXT/epoch_005.pt \
    --image-checkpoint lejepaP16_IMAGE=outputs/$IMG/epoch_003.pt \
    --image-cache outputs/image_only_2_5m_token_cache/val_tokens.pt \
    --num-scenes 4000 --seed 20260922 --output-dir outputs/cka_matched_lejepa_paper \
    > $Q/cone_eval.log 2>&1 && log "CONE EVAL DONE" || log "CONE EVAL FAILED"
else
  log "a run did not finish, skipping the cone eval"
fi
unlock $GPU_TXT
log "ALL DONE"
