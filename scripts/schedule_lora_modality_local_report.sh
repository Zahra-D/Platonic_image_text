#!/usr/bin/env bash
# Wait without competing with active training, then start text/image reports.
set -euo pipefail
cd /home/zd25e122/clevr_discrete_diffusion

while true; do
  mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  free=()
  for index in "${!used[@]}"; do
    # A model process uses ~12 GiB.  This threshold avoids sharing its GPU.
    if [ "${used[$index]// /}" -lt 500 ]; then
      free+=("$index")
    fi
  done
  if [ "${#free[@]}" -ge 2 ]; then
    tmux new-session -d -s lora_modality_report_text \
      ./scripts/run_lora_modality_local_report.sh text "${free[0]}"
    tmux new-session -d -s lora_modality_report_image \
      ./scripts/run_lora_modality_local_report.sh image "${free[1]}"
    exit 0
  fi
  sleep 60
done
