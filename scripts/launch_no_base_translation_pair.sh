#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python_bin="${PYTHON_BIN:-/home/zd25e122/miniconda3/envs/unix/bin/python3.10}"
tmux_session="${TMUX_SESSION:-clevr_no_base_translation_pair}"

configs=(
  configs/multimodal_unpaired_lora_adversarial_backtranslation_contrastive_no_base_no_stage0_correct_dataset_visible_check.yaml
  configs/multimodal_unpaired_lora_backtranslation_contrastive_from_shared_stage0_10e_no_base_correct_dataset_visible_check.yaml
)
output_dirs=(
  outputs/multimodal_unpaired_lora_adversarial_backtranslation_contrastive_no_base_no_stage0_correct_dataset_visible_check
  outputs/multimodal_unpaired_lora_backtranslation_contrastive_from_shared_stage0_10e_no_base_correct_dataset_visible_check
)
window_names=(no_base_dann no_base_stage0_no_dann)
gpu_ids=(0 1)

if tmux has-session -t "$tmux_session" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$tmux_session" >&2
  exit 1
fi

for output_dir in "${output_dirs[@]}"; do
  if [[ -e "$output_dir/train.log" || -e "$output_dir/best.pt" || -e "$output_dir/last.pt" ]]; then
    printf 'refusing to overwrite existing run: %s\n' "$output_dir" >&2
    exit 1
  fi
done

for slot in "${!configs[@]}"; do
  gpu="${gpu_ids[$slot]}"
  output_dir="${output_dirs[$slot]}"
  mkdir -p "$output_dir"
  printf -v launch_command \
    'cd %q && exec env CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 %q train_multimodal.py --config %q >>%q 2>&1' \
    "$PWD" "$gpu" "$python_bin" "${configs[$slot]}" "$output_dir/launcher.log"
  if (( slot == 0 )); then
    tmux new-session -d -s "$tmux_session" -n "${window_names[$slot]}" "$launch_command"
  else
    tmux new-window -d -t "$tmux_session" -n "${window_names[$slot]}" "$launch_command"
  fi
  pid="$(tmux display-message -p -t "$tmux_session:${window_names[$slot]}" '#{pane_pid}')"
  printf '%s\n' "$pid" >"$output_dir/launcher.pid"
  printf 'GPU %s PID %s window=%s:%s config=%s\n' \
    "$gpu" "$pid" "$tmux_session" "${window_names[$slot]}" "${configs[$slot]}"
done
