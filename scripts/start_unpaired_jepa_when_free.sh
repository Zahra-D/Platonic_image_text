#!/usr/bin/env bash
# Launch the three unpaired JEPA runs as GPUs free up.
set -uo pipefail
cd /home/zd25e122/clevr_discrete_diffusion
wait_for() { while tmux has-session -t "$1" 2>/dev/null; do sleep 120; done; }

wait_for clevr_text_lejepa_from_dense_layerwise_all_2m_2e
echo "$(date '+%F %T') GPU 0 free; EMA version from the unpaired dense trunk"
scripts/launch_data2vec_stage1_variant_vnode10.sh multimodal_unpaired_d2v_from_unpaired_dense_ema_1_2m_2e 0

wait_for clevr_text_dense_private_trainable_hsic_from_d2v_lw_all_2m_2e
echo "$(date '+%F %T') GPU 3 free; SIGReg version from the unpaired dense trunk"
scripts/launch_data2vec_stage1_variant_vnode10.sh multimodal_unpaired_d2v_from_unpaired_dense_sigreg_1_2m_2e 3

wait_for clevr_text_lejepa_scratch_layerwise_all_2m_4e
echo "$(date '+%F %T') GPU 1 free; LeJEPA from scratch on both modalities"
scripts/launch_data2vec_stage1_variant_vnode10.sh multimodal_unpaired_lejepa_scratch_1_2m_4e 1
