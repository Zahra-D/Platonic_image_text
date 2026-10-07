#!/usr/bin/env bash
# Launch the three unpaired multimodal runs as GPUs free up, in priority order.
set -uo pipefail
cd /home/zd25e122/clevr_discrete_diffusion
wait_for() { while tmux has-session -t "$1" 2>/dev/null; do sleep 120; done; }

wait_for clevr_image_data2vec_from_dense_layerwise_all_randt_1_2m_2e
echo "$(date '+%F %T') GPU 0 free; launching dense unpaired"
scripts/launch_data2vec_stage1_variant_vnode10.sh multimodal_unpaired_dense_1_2m_4e 0

wait_for clevr_image_data2vec_from_dense_layerwise_all_block2d_1_2m_2e
echo "$(date '+%F %T') GPU 2 free; launching data2vec from text dense"
scripts/launch_data2vec_stage1_variant_vnode10.sh multimodal_unpaired_d2v_from_text_dense_1_2m_2e 2

wait_for clevr_image_lora_diffusion_1_2m_12e_continued
echo "$(date '+%F %T') GPU 1 free; launching data2vec from image dense"
scripts/launch_data2vec_stage1_variant_vnode10.sh multimodal_unpaired_d2v_from_image_dense_1_2m_2e 1
