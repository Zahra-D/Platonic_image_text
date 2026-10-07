#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python_bin="${PYTHON_BIN:-/home/zd25e122/miniconda3/envs/unix/bin/python3.10}"
session="${TMUX_SESSION:-clevr_normalized_dann_no_base_pretraining}"
configs=(
  configs/multimodal_unpaired_lora_adversarial_l2_normalized_no_base_no_stage0_correct_dataset_visible_check.yaml
  configs/multimodal_unpaired_lora_adversarial_l2_normalized_after_shared_stage0_10e_no_base_correct_dataset_visible_check.yaml
)
output_dirs=(
  outputs/multimodal_unpaired_lora_adversarial_l2_normalized_no_base_no_stage0_correct_dataset_visible_check
  outputs/multimodal_unpaired_lora_adversarial_l2_normalized_after_shared_stage0_10e_no_base_correct_dataset_visible_check
)
gpu_ids=(2 3)

if tmux has-session -t "$session" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$session" >&2
  exit 1
fi
for index in "${!configs[@]}"; do
  test -f "${configs[$index]}"
  if [[ -e "${output_dirs[$index]}/train.log" || -e "${output_dirs[$index]}/best.pt" || -e "${output_dirs[$index]}/last.pt" ]]; then
    printf 'refusing to overwrite existing run: %s\n' "${output_dirs[$index]}" >&2
    exit 1
  fi
  mkdir -p "${output_dirs[$index]}"
done

for index in "${!configs[@]}"; do
  gpu="${gpu_ids[$index]}"
  command="cd $(printf '%q' "$PWD") && env CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $(printf '%q' "$python_bin") train_multimodal.py --config $(printf '%q' "${configs[$index]}") >>$(printf '%q' "${output_dirs[$index]}/launcher.log") 2>&1"
  if (( index == 0 )); then
    tmux new-session -d -s "$session" -n "gpu${gpu}" "bash -lc $(printf '%q' "$command")"
  else
    tmux new-window -d -t "$session" -n "gpu${gpu}" "bash -lc $(printf '%q' "$command")"
  fi
done

printf '%s\n' "from-start config=${configs[0]} gpu=${gpu_ids[0]}"
printf '%s\n' "stage0-then-DANN config=${configs[1]} gpu=${gpu_ids[1]}"
printf '%s\n' "monitor: tmux attach -t $session"
