#!/usr/bin/env bash
# Stop the image Tri-LoRA continuation at the first epoch that reaches the
# dense model's validation loss, then report which checkpoint to evaluate.
set -uo pipefail
cd /home/zd25e122/clevr_discrete_diffusion
target=3.6340
log=outputs/image_lora_diffusion_1_2m_12e_continued/train.log
session=clevr_image_lora_diffusion_1_2m_12e_continued
while tmux has-session -t "$session" 2>/dev/null; do
  if [ -f "$log" ]; then
    while read -r epoch loss; do
      if awk "BEGIN{exit !($loss <= $target)}"; then
        echo "$(date '+%F %T') epoch $epoch reached $loss <= $target; stopping"
        tmux kill-session -t "$session" 2>/dev/null
        echo "evaluate: outputs/image_lora_diffusion_1_2m_12e_continued/epoch_$(printf '%03d' "$epoch").pt"
        exit 0
      fi
    done < <(grep "validation epoch" "$log" | sed 's/.*epoch=\([0-9]*\).*loss=\([0-9.]*\)/\1 \2/')
  fi
  sleep 300
done
echo "$(date '+%F %T') run ended without reaching $target"
