#!/usr/bin/env bash
set -euo pipefail

# Strict no-base Tri-LoRA pretraining with both objectives active from update 1:
# masked-to-clean shared-token JEPA at every Transformer block, and per-layer
# SIGReg to prevent the zero/collapsed shared representation from satisfying
# the prediction objective trivially.
repo_dir=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
config=configs/pretraining_unpaired_lora_shared_jepa_sigreg_per_layer_absolute_70e.yaml
output=outputs/pretraining_unpaired_lora_shared_jepa_sigreg_per_layer_absolute_70e
session=clevr_shared_jepa_sigreg_per_layer_70e
gpu=${1:-0}

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
  "cd $repo_dir && exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $python_bin train_multimodal.py --config $config > $output/launcher.log 2>&1"

echo "Started $session on physical GPU $gpu"
echo "Log: $repo_dir/$output/launcher.log"
