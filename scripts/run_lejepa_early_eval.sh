#!/bin/bash
# Early LeJEPA checkpoints vs dense and I-JEPA at the same total epoch count.
#   image: dense ep13 = dense/epoch_012 ; I-JEPA & LeJEPA epoch_000 (from dense ep12)
#   text : dense ep21/22 = 40e_continued/epoch_020/021 ; I-JEPA & LeJEPA epoch_000/001 (from dense ep20)
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-3}
Q=outputs/lejepa_early_eval; mkdir -p $Q
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
ID=outputs/image_dense_diffusion_2_5m_40e/epoch_012.pt
IJ=outputs/image_ijepa_sweep_blk_s15_t45/epoch_000.pt
IL=outputs/image_lejepa_blk_s15_t45_from_dense_ep12/epoch_000.pt
TD21=outputs/text_dense_diffusion_2m_40e_continued/epoch_020.pt
TD22=outputs/text_dense_diffusion_2m_40e_continued/epoch_021.pt
TJ21=outputs/text_ijepa_w4_8_from_dense_ep20_2m_6e/epoch_000.pt
TJ22=outputs/text_ijepa_w4_8_from_dense_ep20_2m_6e/epoch_001.pt
TL21=outputs/text_lejepa_w4_8_from_dense_ep20_2m_6e/epoch_000.pt
TL22=outputs/text_lejepa_w4_8_from_dense_ep20_2m_6e/epoch_001.pt

log "structure (cone / probe / RSA / CKA)"
python3 evaluate_cross_modal_structure.py \
  --text-checkpoint dense22_TEXT=$TD22 --text-checkpoint ijepa22_TEXT=$TJ22 \
  --text-checkpoint lejepa22_TEXT=$TL22 \
  --text-checkpoint dense21_TEXT=$TD21 --text-checkpoint ijepa21_TEXT=$TJ21 \
  --text-checkpoint lejepa21_TEXT=$TL21 \
  --image-checkpoint dense13_IMAGE=$ID --image-checkpoint ijepa13_IMAGE=$IJ \
  --image-checkpoint lejepa13_IMAGE=$IL \
  --image-cache outputs/image_only_2_5m_token_cache/val_tokens.pt \
  --num-scenes 4000 --seed 20260922 --output-dir $Q/structure \
  > $Q/structure.log 2>&1 && log "structure done" || log "structure FAILED"

log "image sensitivity"
python3 evaluate_semantic_sensitivity.py --output-dir $Q/sens_image \
  --checkpoint dense13=$ID --checkpoint ijepa13=$IJ --checkpoint lejepa13=$IL \
  > $Q/sens_image.log 2>&1 && log "image sensitivity done" || log "image sensitivity FAILED"

log "text sensitivity"
python3 evaluate_text_sensitivity.py --num-worlds 2000 --output-dir $Q/sens_text \
  --checkpoint dense22=$TD22 --checkpoint ijepa22=$TJ22 --checkpoint lejepa22=$TL22 \
  > $Q/sens_text.log 2>&1 && log "text sensitivity done" || log "text sensitivity FAILED"

log "cross-modal retrieval"
python3 evaluate_cross_modal_retrieval.py \
  --num-scenes 8000 --test 1000 --val 1000 --output-dir $Q/xret \
  --text-checkpoint dense22_TEXT=$TD22 --text-checkpoint ijepa22_TEXT=$TJ22 \
  --text-checkpoint lejepa22_TEXT=$TL22 \
  --image-checkpoint dense13_IMAGE=$ID --image-checkpoint ijepa13_IMAGE=$IJ \
  --image-checkpoint lejepa13_IMAGE=$IL \
  --pair "dense22_TEXT|dense13_IMAGE" --pair "ijepa22_TEXT|ijepa13_IMAGE" \
  --pair "lejepa22_TEXT|lejepa13_IMAGE" \
  > $Q/xret.log 2>&1 && log "xret done" || log "xret FAILED"
log "EARLY EVAL COMPLETE"
