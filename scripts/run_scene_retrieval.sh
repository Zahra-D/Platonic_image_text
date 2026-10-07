#!/bin/bash
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
Q=outputs/scene_retrieval; mkdir -p $Q
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
free_gpu(){ while :; do for g in 0 1 2 3; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$u" -lt 9000 ] && { echo $g; return; }; done; sleep 120; done; }
# (no longer waits for the probe job; runs alongside it)
g=${GPU_R:-2}; log "IMAGE retrieval on GPU $g"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g python3 evaluate_scene_retrieval.py \
  --modality image --num-scenes 2500 --output-dir $Q/image \
  --checkpoint "random_init=RANDOM:outputs/image_dense_diffusion_2_5m_40e/epoch_011.pt" \
  --checkpoint "dense_ep12=outputs/image_dense_diffusion_2_5m_40e/epoch_011.pt" \
  --checkpoint "dense_ep25=outputs/image_dense_diffusion_2_5m_40e/epoch_024.pt" \
  --checkpoint "jepa_s15_t45=outputs/image_ijepa_sweep_blk_s15_t45/epoch_003.pt" \
  --checkpoint "jepa_s30_t60=outputs/image_ijepa_sweep_blk_s30_t60/epoch_003.pt" \
  --checkpoint "jepa_scratch=outputs/image_data2vec_scratch_block2d_30pct_layerwise_l4to7_1_2m_4e/epoch_003.pt" \
  > $Q/image.log 2>&1 && log "IMAGE DONE" || log "IMAGE FAILED"
g=${GPU_R:-2}; log "TEXT retrieval on GPU $g"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g python3 evaluate_scene_retrieval.py \
  --modality text \
  --manifest /home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_1m/val_text_only_human.jsonl \
  --num-scenes 2500 --output-dir $Q/text \
  --checkpoint "random_init=RANDOM:outputs/text_dense_diffusion_2m_4e_matched/epoch_003.pt" \
  --checkpoint "dense_ep4=outputs/text_dense_diffusion_2m_4e_matched/epoch_003.pt" \
  --checkpoint "dense_ep20=outputs/text_dense_diffusion_2m_20e_continued/epoch_019.pt" \
  --checkpoint "dense_ep41=outputs/text_dense_diffusion_2m_40e_continued/epoch_039.pt" \
  --checkpoint "jepa_from_ep20=outputs/text_ijepa_w4_8_from_dense_ep20_2m_6e/epoch_005.pt" \
  --checkpoint "jepa_from_ep4=outputs/text_ijepa_w4_8_from_dense_ep4_2m_6e/epoch_005.pt" \
  --checkpoint "jepa_scratch=outputs/text_data2vec_scratch_layerwise_all_randt_2m_4e/epoch_003.pt" \
  > $Q/text.log 2>&1 && log "TEXT DONE" || log "TEXT FAILED"
log "RETRIEVAL COMPLETE"
