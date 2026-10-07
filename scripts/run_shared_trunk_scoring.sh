#!/bin/bash
# Step 0: score the unpaired shared-trunk models we already have, with the same
# linear-map retrieval + CKA + raw-cosine (no map) pipeline.  Waits for the
# current evaluations so it does not slow them down.
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
while tmux ls 2>/dev/null | grep -qE '^(pimg_f|pimg_r|ptxt_f|ptxt_r|xret):'; do sleep 120; done
echo "[$(date '+%H:%M')] scoring shared-trunk models"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python3 evaluate_cross_modal_retrieval.py \
  --num-scenes 8000 --test 1000 --val 1000 --output-dir outputs/cross_modal_retrieval_shared \
  --text-checkpoint  "mm_dense4_TEXT=outputs/multimodal_unpaired_dense_1_2m_4e/epoch_003.pt" \
  --text-checkpoint  "mm_dense8_TEXT=outputs/mm_unpaired_dense_8e_continued/epoch_007.pt" \
  --text-checkpoint  "mm_merged_TEXT=outputs/mm_unpaired_merged_jepa_trunk_private_diff_1_2m_2e/epoch_001.pt" \
  --image-checkpoint "mm_dense4_IMAGE=outputs/multimodal_unpaired_dense_1_2m_4e/epoch_003.pt" \
  --image-checkpoint "mm_dense8_IMAGE=outputs/mm_unpaired_dense_8e_continued/epoch_007.pt" \
  --image-checkpoint "mm_merged_IMAGE=outputs/mm_unpaired_merged_jepa_trunk_private_diff_1_2m_2e/epoch_001.pt" \
  --pair "mm_dense4_TEXT|mm_dense4_IMAGE" --pair "mm_dense8_TEXT|mm_dense8_IMAGE" \
  --pair "mm_merged_TEXT|mm_merged_IMAGE"
echo "[$(date '+%H:%M')] SHARED-TRUNK SCORING DONE"
