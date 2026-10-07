#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python_bin="${PYTHON_BIN:-/home/zd25e122/miniconda3/envs/unix/bin/python3.10}"
gpu_id="${GPU_ID:-0}"
tmux_session="${TMUX_SESSION:-clevr_instruction_lora_token_routed}"
config="configs/instruction_tune_lora_token_routed_correct_dataset_visible_check.yaml"
output_dir="outputs/instruction_tune_lora_token_routed_correct_dataset_visible_check"

if tmux has-session -t "$tmux_session" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$tmux_session" >&2
  exit 1
fi
if [[ -e "$output_dir/train.log" || -e "$output_dir/best.pt" || -e "$output_dir/last.pt" ]]; then
  printf 'refusing to overwrite existing corrected instruction run: %s\n' "$output_dir" >&2
  exit 1
fi

mkdir -p "$output_dir"
printf -v launch_command \
  'cd %q && exec env CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 %q train_multimodal.py --config %q >>%q 2>&1' \
  "$PWD" "$gpu_id" "$python_bin" "$config" "$output_dir/launcher.log"
tmux new-session -d -s "$tmux_session" -n token_routed "$launch_command"
pid="$(tmux display-message -p -t "$tmux_session:token_routed" '#{pane_pid}')"
printf '%s\n' "$pid" >"$output_dir/launcher.pid"
printf 'GPU %s PID %s window=%s:token_routed config=%s log=%s\n' \
  "$gpu_id" "$pid" "$tmux_session" "$config" "$output_dir/launcher.log"
