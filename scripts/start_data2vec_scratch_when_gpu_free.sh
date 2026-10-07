#!/usr/bin/env bash
# Starts the from-scratch data2vec run on GPU 3 once the all-layer LoRA run ends.
set -uo pipefail
cd /home/zd25e122/clevr_discrete_diffusion
while tmux has-session -t clevr_text_all_layers 2>/dev/null; do sleep 120; done
echo "$(date '+%F %T') GPU 3 free; launching from-scratch data2vec"
scripts/launch_data2vec_stage1_variant_vnode10.sh text_data2vec_scratch_avg_all_2m_4e 3
