#!/usr/bin/env bash
# Evaluate every epoch checkpoint of the given runs as soon as it appears.
#
#   scripts/watch_and_evaluate_epoch_checkpoints.sh <gpu> <run_dir> [run_dir ...]
#
# For each outputs/<run>/epoch_00N.pt it runs the hard-retrieval and
# binding-swap probes under the label <run>_eN, skipping any label that already
# has a result file. Both probes are a couple of minutes per checkpoint and use
# a few GB, so this is safe to run alongside training on a shared GPU.
set -uo pipefail
cd /home/zd25e122/clevr_discrete_diffusion
py=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
gpu=${1:?usage: $0 <gpu> <run_dir> [run_dir ...]}; shift
hard_dir=outputs/hard_retrieval_eval_all_sublayers
bind_dir=outputs/binding_swap_eval
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1

while true; do
  pending=0
  for run_dir in "$@"; do
    run=$(basename "$run_dir")
    for checkpoint in "$run_dir"/epoch_*.pt; do
      [ -f "$checkpoint" ] || continue
      epoch=$(basename "$checkpoint" .pt | sed 's/epoch_0*//')
      label="${run}_e${epoch:-0}"
      if [ ! -f "$hard_dir/$label.json" ]; then
        echo "$(date '+%F %T') hard retrieval: $label"
        $py evaluate_hard_retrieval.py --num-worlds 2000 --sublayer-layers 0 1 2 3 4 5 6 7 \
          --output-dir "$hard_dir" --checkpoint "$label=$checkpoint" \
          >> "$hard_dir/watcher.log" 2>&1
      fi
      if [ ! -f "$bind_dir/$label.json" ]; then
        echo "$(date '+%F %T') binding swap: $label"
        $py evaluate_binding_swap.py --output-dir "$bind_dir" \
          --checkpoint "$label=$checkpoint" >> "$bind_dir/watcher.log" 2>&1
      fi
    done
    # Keep watching while the run's tmux session is alive.
    tmux has-session -t "clevr_$run" 2>/dev/null && pending=1
  done
  [ "$pending" -eq 1 ] || { echo "$(date '+%F %T') all runs finished and evaluated"; break; }
  sleep 300
done
