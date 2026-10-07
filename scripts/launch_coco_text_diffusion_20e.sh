#!/usr/bin/env bash
set -euo pipefail

repo=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
session=coco_text_diffusion_20e
gpu=${1:-1}
output=outputs/text_diffusion_coco_one_caption_scratch_20e

cd "$repo"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Refusing to reuse active tmux session: $session" >&2
  exit 1
fi
if test -e "$output/last.pt"; then
  echo "Refusing to overwrite completed run: $output/last.pt" >&2
  exit 1
fi
mkdir -p "$output"
tmux new-session -d -s "$session" \
  "cd $repo && exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 \
   $python_bin train_multimodal.py --config configs/text_diffusion_coco_one_caption_scratch_20e.yaml \
   > $output/launcher.log 2>&1"

echo "Started $session on physical GPU $gpu"
echo "Log: $repo/$output/launcher.log"
