#!/usr/bin/env bash
set -euo pipefail

repo_dir=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
output=outputs/pretraining_unpaired_dense_absolute_70e
session=clevr_dense_unpaired_70e
cd "$repo_dir"

if [[ -e "$output" ]]; then
  echo "Refusing to overwrite existing output: $output" >&2
  exit 1
fi
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Refusing to reuse active tmux session: $session" >&2
  exit 1
fi
mkdir -p "$output"
tmux new-session -d -s "$session" \
  "cd $repo_dir && exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 $python_bin train_multimodal.py --config configs/pretraining_unpaired_dense_absolute_70e.yaml > $output/launcher.log 2>&1"

tmux list-sessions
