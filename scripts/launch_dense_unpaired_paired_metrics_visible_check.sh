#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python_bin="${PYTHON_BIN:-/home/zd25e122/miniconda3/envs/unix/bin/python3.10}"
tmux_session="${TMUX_SESSION:-clevr_dense_unpaired_paired_metrics_visible_check}"
config="configs/multimodal_unpaired_dense_paired_metrics_correct_dataset_visible_check.yaml"
output_dir="outputs/multimodal_unpaired_dense_paired_metrics_correct_dataset_visible_check"
gpu="${CUDA_DEVICE:-2}"

if tmux has-session -t "$tmux_session" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$tmux_session" >&2
  exit 1
fi
if [[ -e "$output_dir/train.log" || -e "$output_dir/best.pt" || -e "$output_dir/last.pt" ]]; then
  printf 'refusing to overwrite existing dense paired-metrics run: %s\n' "$output_dir" >&2
  exit 1
fi

mkdir -p "$output_dir"
printf -v launch_command \
  'cd %q && exec env CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 %q train_multimodal.py --config %q >>%q 2>&1' \
  "$PWD" "$gpu" "$python_bin" "$config" "$output_dir/launcher.log"
tmux new-session -d -s "$tmux_session" -n train "$launch_command"
pid="$(tmux display-message -p -t "$tmux_session:train" '#{pane_pid}')"
printf '%s\n' "$pid" >"$output_dir/launcher.pid"
printf 'GPU %s PID %s window=%s:train config=%s log=%s\n' \
  "$gpu" "$pid" "$tmux_session" "$config" "$output_dir/launcher.log"
