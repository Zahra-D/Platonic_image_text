#!/usr/bin/env bash
set -euo pipefail

# Read-only held-out nearest-neighbor evaluation.  Each invocation evaluates
# one checkpoint and writes an independent, non-overwriting result directory.
# Usage: ./scripts/launch_shared_private_retrieval_vnode10.sh LABEL CHECKPOINT [GPU]

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 LABEL CHECKPOINT [GPU]" >&2
  exit 2
fi

repo_dir=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
label=$1
checkpoint=$2
gpu=${3:-2}
output_dir="outputs/shared_private_semantic_template_retrieval/${label}"
session="clevr_retrieval_${label}"

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
  "cd $repo_dir && exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $python_bin evaluate_shared_private_retrieval.py \\
    --checkpoint $checkpoint \\
    --manifest /home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_1m/val_text_only_human.jsonl \\
    --num-worlds 256 --variants-per-pattern 2 --batch-size 64 --bootstrap 1000 --seed 20260916 --overwrite \\
    --output $output_dir > $output_dir/launcher.log 2>&1"
echo "Started $session on physical GPU $gpu"
echo "Log: $repo_dir/$output_dir/launcher.log"
