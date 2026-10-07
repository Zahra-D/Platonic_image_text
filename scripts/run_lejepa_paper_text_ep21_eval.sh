#!/bin/bash
# Paper-LeJEPA text epoch_000 (= 21 epochs) vs dense / I-JEPA / old LeJEPA at 21.
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-3}
Q=outputs/lejepa_early_eval; log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
TP=outputs/text_lejepa_paper_from_dense_ep20_6e/epoch_000.pt
TD=outputs/text_dense_diffusion_2m_40e_continued/epoch_020.pt
TJ=outputs/text_ijepa_w4_8_from_dense_ep20_2m_6e/epoch_000.pt
TL=outputs/text_lejepa_w4_8_from_dense_ep20_2m_6e/epoch_000.pt
ID=outputs/image_dense_diffusion_2_5m_40e/epoch_012.pt
log "structure"
python3 evaluate_cross_modal_structure.py \
  --text-checkpoint lejepaP21_TEXT=$TP --image-checkpoint dense13_IMAGE=$ID \
  --image-cache outputs/image_only_2_5m_token_cache/val_tokens.pt \
  --num-scenes 4000 --seed 20260922 --output-dir $Q/structure \
  > $Q/structure_paper21.log 2>&1 && log "structure done" || log "structure FAILED"
log "text sensitivity"
python3 evaluate_text_sensitivity.py --num-worlds 2000 --output-dir $Q/sens_text \
  --checkpoint dense21=$TD --checkpoint ijepa21=$TJ --checkpoint lejepa21=$TL --checkpoint lejepaP21=$TP \
  > $Q/sens_text_paper21.log 2>&1 && log "text sensitivity done" || log "text sensitivity FAILED"
log "cross-modal retrieval vs a fixed dense image model"
python3 evaluate_cross_modal_retrieval.py --num-scenes 8000 --test 1000 --val 1000 \
  --output-dir $Q/xret_paper21 \
  --text-checkpoint dense21_TEXT=$TD --text-checkpoint ijepa21_TEXT=$TJ \
  --text-checkpoint lejepaP21_TEXT=$TP --image-checkpoint dense13_IMAGE=$ID \
  --pair "dense21_TEXT|dense13_IMAGE" --pair "ijepa21_TEXT|dense13_IMAGE" \
  --pair "lejepaP21_TEXT|dense13_IMAGE" \
  > $Q/xret_paper21.log 2>&1 && log "xret done" || log "xret FAILED"
log "DONE"
