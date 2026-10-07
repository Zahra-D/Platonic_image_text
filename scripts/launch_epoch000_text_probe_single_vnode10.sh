#!/usr/bin/env bash
set -euo pipefail

# Run one independently mergeable first-epoch text probe on an otherwise free GPU.
# Usage: script NAME CHECKPOINT GPU
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 NAME CHECKPOINT GPU" >&2
  exit 2
fi
name=$1
checkpoint=$2
gpu=$3
repo_dir=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
output_dir="outputs/modulewise_jepa_hsic_epoch000_modality_probe/chunks/$name"
session="clevr_epoch000_probe_$name"

cd "$repo_dir"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Refusing to reuse active tmux session: $session" >&2
  exit 1
fi
if test -f "$output_dir/results.json"; then
  echo "Refusing to overwrite completed evaluation: $output_dir/results.json" >&2
  exit 1
fi
mkdir -p "$output_dir"
tmux new-session -d -s "$session" \
  "cd $repo_dir && exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $python_bin evaluate_jepa_modality.py \\
    --checkpoint $name=$checkpoint \\
    --train-dir /home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_2m \\
    --val-dir /home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_1m \\
    --train-manifest /home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_2m/train_text_only_human.jsonl \\
    --val-manifest /home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_1m/val_text_only_human.jsonl \\
    --token-cache outputs/unused_text_only_cache \\
    --caption-field caption_human --data-mode text_only --modality text --layers 2 3 4 \\
    --mask-ratios 0.8 --train-samples 2048 --val-samples 2048 --batch-size 32 \\
    --token-probe-train 4000 --token-probe-val 2000 --probe-epochs 15 --seed 20260916 \\
    --output $output_dir/results.json > $output_dir/launcher.log 2>&1"
echo "Started $session on physical GPU $gpu"
echo "Log: $repo_dir/$output_dir/launcher.log"
