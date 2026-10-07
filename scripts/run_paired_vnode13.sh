#!/usr/bin/env bash
set -euo pipefail

repo_dir=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
cd "$repo_dir"

launch() {
  session=$1
  gpu=$2
  config=$3
  output=$4
  mkdir -p "$output"
  tmux new-session -d -s "$session" \
    "cd $repo_dir && exec env CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $python_bin train_multimodal.py --config $config > $output/launcher.log 2>&1"
}

launch clevr_paired_dense_70e 2 \
  configs/pretraining_paired_dense_absolute_70e.yaml \
  outputs/pretraining_paired_dense_absolute_70e

launch clevr_paired_lora_70e 3 \
  configs/pretraining_paired_lora_absolute_70e.yaml \
  outputs/pretraining_paired_lora_absolute_70e

tmux list-sessions
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
