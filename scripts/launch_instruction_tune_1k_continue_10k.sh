#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python_bin="${PYTHON_BIN:-/home/zd25e122/miniconda3/envs/unix/bin/python3.10}"
tmux_session="${TMUX_SESSION:-clevr_instruction_tuning_1k_pure_paired_10k}"

# Each run starts at optimizer step 0 from its designated pretraining
# checkpoint.  DANN, back-translation, and gradient balancing are disabled;
# training is only the paired text->image and image->text denoising losses.
configs=(
  configs/instruction_tune_1k_dense_from_unpaired_dense.yaml
  configs/instruction_tune_1k_lora_from_unpaired_no_adversary_frozen_base.yaml
  configs/instruction_tune_1k_lora_from_correct_dann_frozen_base.yaml
  configs/instruction_tune_1k_lora_from_translation_correct_dann_frozen_base.yaml
)
output_dirs=(
  outputs/instruction_tune_1k_pairs_dense_from_unpaired_dense_pure_paired_10k_steps
  outputs/instruction_tune_1k_pairs_tri_lora_from_unpaired_no_adversary_frozen_base_pure_paired_10k_steps
  outputs/instruction_tune_1k_pairs_tri_lora_from_correct_dann_frozen_base_pure_paired_10k_steps
  outputs/instruction_tune_1k_pairs_tri_lora_from_translation_correct_dann_frozen_base_pure_paired_10k_steps
)
window_names=(dense unpaired dann translation)
gpu_ids=(0 1 2 3)

if tmux has-session -t "$tmux_session" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$tmux_session" >&2
  exit 1
fi

for output_dir in "${output_dirs[@]}"; do
  if [[ -e "$output_dir/train.log" || -e "$output_dir/best.pt" || -e "$output_dir/last.pt" ]]; then
    printf 'refusing to overwrite existing fresh run: %s\n' "$output_dir" >&2
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
