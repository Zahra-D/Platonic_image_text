#!/usr/bin/env bash
# Starts the two remaining from-scratch window runs (4-8 token windows, 15%)
# when GPU 2 and GPU 3 free up.
set -uo pipefail
cd /home/zd25e122/clevr_discrete_diffusion
wait_for() { while tmux has-session -t "$1" 2>/dev/null; do sleep 120; done; }

wait_for clevr_text_data2vec_from_dense_layerwise_all_2m_2e
echo "$(date '+%F %T') GPU 2 free"
scripts/launch_data2vec_stage1_variant_vnode10.sh text_data2vec_scratch_window4_8_15pct_avg_l4to7_2m_4e 2

wait_for clevr_text_all_layers
echo "$(date '+%F %T') GPU 3 free"
scripts/launch_data2vec_stage1_variant_vnode10.sh text_data2vec_scratch_window4_8_15pct_layerwise_l4to7_2m_4e 3
