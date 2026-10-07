#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/launch_exact_template_conflict_vnode10.sh LABEL CHECKPOINT [GPU]
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 LABEL CHECKPOINT [GPU]" >&2
  exit 2
fi

repo_dir=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
label=$1
checkpoint=$2
gpu=${3:-2}
output_dir="outputs/exact_template_conflict/${label}"
session="clevr_template_conflict_${label}"

cd "$repo_dir"
if [ ! -f "$checkpoint" ]; then
  echo "Checkpoint does not exist: $checkpoint" >&2
  exit 1
fi
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Active tmux session already exists: $session" >&2
  exit 1
fi
if [ -e "$output_dir/results.json" ]; then
  echo "Completed result already exists: $output_dir/results.json" >&2
  exit 1
fi
mkdir -p "$output_dir"
tmux new-session -d -s "$session" \
  "cd $repo_dir && exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $python_bin evaluate_exact_template_conflict.py \\
    --checkpoint $checkpoint \\
    --manifest /home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_1m/val_text_only_human.jsonl \\
    --num-worlds 256 --batch-size 64 --bootstrap 1000 --seed 20260917 --overwrite \\
    --output $output_dir > $output_dir/launcher.log 2>&1"
echo "Started $session on physical GPU $gpu"
