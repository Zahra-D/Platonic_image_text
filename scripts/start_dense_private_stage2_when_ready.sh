#!/usr/bin/env bash
# Waits for stage 1 to write its final checkpoint, then starts condition A on
# the GPU stage 1 vacates; condition B follows when the image dense run ends.
set -uo pipefail

repo_dir=/home/zd25e122/clevr_discrete_diffusion
cd "$repo_dir"
stage1=outputs/text_data2vec_from_dense_2m_2e/epoch_001.pt

while tmux has-session -t clevr_text_d2v_stage1 2>/dev/null; do sleep 120; done
if [ ! -f "$stage1" ]; then
  echo "$(date '+%F %T') stage 1 ended without $stage1; not launching" >&2
  exit 1
fi
echo "$(date '+%F %T') stage 1 finished; launching frozen condition on GPU 2"
scripts/launch_dense_private_stage2_vnode10.sh frozen 2

while tmux has-session -t clevr_image_dense 2>/dev/null; do sleep 120; done
echo "$(date '+%F %T') image dense run finished; launching trainable condition on GPU 0"
scripts/launch_dense_private_stage2_vnode10.sh trainable 0
