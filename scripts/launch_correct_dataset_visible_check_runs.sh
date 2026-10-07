#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python_bin="${PYTHON_BIN:-/home/zd25e122/miniconda3/envs/unix/bin/python3.10}"
tmux_session="${TMUX_SESSION:-clevr_correct_dataset_visible_check}"

configs=(
  configs/multimodal_unpaired_lora_correct_dataset_visible_check.yaml
  configs/multimodal_unpaired_lora_adversarial_correct_dataset_visible_check.yaml
  configs/multimodal_dense_correct_dataset_visible_check.yaml
  configs/multimodal_unpaired_dense_correct_dataset_visible_check.yaml
)

output_dirs=(
  outputs/multimodal_unpaired_lora_correct_dataset_visible_check
  outputs/multimodal_unpaired_lora_adversarial_correct_dataset_visible_check
  outputs/multimodal_dense_correct_dataset_visible_check
  outputs/multimodal_unpaired_dense_correct_dataset_visible_check
)

window_names=(lora_no_adv lora_adv dense_paired dense_unpaired)

if tmux has-session -t "$tmux_session" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$tmux_session" >&2
  exit 1
fi

for gpu in 0 1 2 3; do
  output_dir="${output_dirs[$gpu]}"
  mkdir -p "$output_dir"
  printf -v launch_command \
    'cd %q && exec env CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 %q train_multimodal.py --config %q >>%q 2>&1' \
    "$PWD" "$gpu" "$python_bin" "${configs[$gpu]}" "$output_dir/launcher.log"
  if (( gpu == 0 )); then
    tmux new-session -d -s "$tmux_session" -n "${window_names[$gpu]}" "$launch_command"
  else
    tmux new-window -d -t "$tmux_session" -n "${window_names[$gpu]}" "$launch_command"
  fi
  pid="$(tmux display-message -p -t "$tmux_session:${window_names[$gpu]}" '#{pane_pid}')"
  printf '%s\n' "$pid" >"$output_dir/launcher.pid"
  printf 'GPU %s PID %s window=%s:%s config=%s log=%s\n' \
    "$gpu" "$pid" "$tmux_session" "${window_names[$gpu]}" \
    "${configs[$gpu]}" "$output_dir/launcher.log"
done
