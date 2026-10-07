#!/bin/bash
# Remaining evaluations: experiment 2 for text, the corrected image experiment 2,
# and cross-modal retrieval through a fitted linear map.
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
WHAT=${1:-all}

if [ "$WHAT" = sens ] || [ "$WHAT" = all ]; then
  # image experiment 2: the earlier "random_init" was epoch_000.pt, which has
  # already seen one full epoch; keep it under its true name, add a real one
  S=outputs/semantic_sensitivity
  [ -f $S/random_init.json ] && [ ! -f $S/dense_ep1.json ] && mv $S/random_init.json $S/dense_ep1.json
  log "IMAGE sensitivity (true random baseline)"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU_S:-2} python3 evaluate_semantic_sensitivity.py \
    --output-dir $S \
    --checkpoint "random_init=RANDOM:outputs/image_dense_diffusion_2_5m_40e/epoch_015.pt" \
    > logs/sens_image_random.log 2>&1 && log "IMAGE sensitivity done" || log "IMAGE sensitivity FAILED"
  log "TEXT sensitivity"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU_S:-2} python3 evaluate_text_sensitivity.py \
    --num-worlds 2000 --output-dir outputs/semantic_sensitivity_text \
    --checkpoint "random_init=RANDOM:outputs/text_dense_diffusion_2m_4e_matched/epoch_003.pt" \
    --checkpoint "dense_ep4=outputs/text_dense_diffusion_2m_4e_matched/epoch_003.pt" \
    --checkpoint "dense_ep20=outputs/text_dense_diffusion_2m_20e_continued/epoch_019.pt" \
    --checkpoint "dense_ep41=outputs/text_dense_diffusion_2m_40e_continued/epoch_039.pt" \
    --checkpoint "jepa_from_ep20=outputs/text_ijepa_w4_8_from_dense_ep20_2m_6e/epoch_005.pt" \
    --checkpoint "jepa_from_ep4=outputs/text_ijepa_w4_8_from_dense_ep4_2m_6e/epoch_005.pt" \
    --checkpoint "jepa_scratch=outputs/text_data2vec_scratch_layerwise_all_randt_2m_4e/epoch_003.pt" \
    > logs/sens_text.log 2>&1 && log "TEXT sensitivity done" || log "TEXT sensitivity FAILED"
fi

if [ "$WHAT" = xret ] || [ "$WHAT" = all ]; then
  log "CROSS-MODAL retrieval"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU_X:-3} python3 evaluate_cross_modal_retrieval.py \
    --num-scenes 8000 --test 1000 --val 1000 --output-dir outputs/cross_modal_retrieval \
    --text-checkpoint  "random_TEXT=RANDOM:outputs/text_dense_diffusion_2m_40e_continued/epoch_025.pt" \
    --text-checkpoint  "dense26_TEXT=outputs/text_dense_diffusion_2m_40e_continued/epoch_025.pt" \
    --text-checkpoint  "ijepa26_TEXT=outputs/text_ijepa_w4_8_from_dense_ep20_2m_6e/epoch_005.pt" \
    --text-checkpoint  "paired_TEXT=outputs/multimodal_paired_dense_1_2m_4e/epoch_003.pt" \
    --image-checkpoint "random_IMAGE=RANDOM:outputs/image_dense_diffusion_2_5m_40e/epoch_015.pt" \
    --image-checkpoint "dense16_IMAGE=outputs/image_dense_diffusion_2_5m_40e/epoch_015.pt" \
    --image-checkpoint "ijepa16_IMAGE=outputs/image_ijepa_sweep_blk_s15_t45/epoch_003.pt" \
    --image-checkpoint "paired_IMAGE=outputs/multimodal_paired_dense_1_2m_4e/epoch_003.pt" \
    --pair "random_TEXT|random_IMAGE"   --pair "random_TEXT|dense16_IMAGE"  --pair "random_TEXT|ijepa16_IMAGE" \
    --pair "dense26_TEXT|random_IMAGE"  --pair "dense26_TEXT|dense16_IMAGE" --pair "dense26_TEXT|ijepa16_IMAGE" \
    --pair "ijepa26_TEXT|random_IMAGE"  --pair "ijepa26_TEXT|dense16_IMAGE" --pair "ijepa26_TEXT|ijepa16_IMAGE" \
    --pair "paired_TEXT|paired_IMAGE" \
    > logs/xret_full.log 2>&1 && log "CROSS-MODAL retrieval done" || log "CROSS-MODAL retrieval FAILED"
fi
log "FINAL EVALS ($WHAT) COMPLETE"
