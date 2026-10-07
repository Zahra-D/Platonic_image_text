#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python_bin="${PYTHON_BIN:-/home/zd25e122/miniconda3/envs/unix/bin/python3.10}"
session="${TMUX_SESSION:-clevr_instruction_tune_2k_10e_finalset}"

configs=(
  configs/instruction_tune_2k_10e_dense_finalset.yaml
  configs/instruction_tune_2k_10e_lora_no_stage_finalset.yaml
  configs/instruction_tune_2k_10e_lora_dann_finalset.yaml
  configs/instruction_tune_2k_10e_lora_stage0_finalset.yaml
  configs/instruction_tune_2k_10e_lora_stage0_then_dann_finalset.yaml
)
output_dirs=(
  outputs/instruction_tune_2k_10e_dense_finalset
  outputs/instruction_tune_2k_10e_lora_no_stage_finalset
  outputs/instruction_tune_2k_10e_lora_dann_finalset
  outputs/instruction_tune_2k_10e_lora_stage0_finalset
  outputs/instruction_tune_2k_10e_lora_stage0_then_dann_finalset
)

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

# CUDA IDs 2 and 3 are the currently idle physical devices on this host.
# Each worker executes its queue serially; no existing training run is touched.
worker_2='set -euo pipefail; '
for index in 0 3; do
  printf -v worker_2 '%s cd %q && env CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 %q train_multimodal.py --config %q >>%q 2>&1; ' \
    "$worker_2" "$PWD" "$python_bin" "${configs[$index]}" "${output_dirs[$index]}/launcher.log"
done
worker_3='set -euo pipefail; '
for index in 1 2 4; do
  printf -v worker_3 '%s cd %q && env CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 %q train_multimodal.py --config %q >>%q 2>&1; ' \
    "$worker_3" "$PWD" "$python_bin" "${configs[$index]}" "${output_dirs[$index]}/launcher.log"
done

tmux new-session -d -s "$session" -n gpu2 "bash -lc $(printf '%q' "$worker_2")"
tmux new-window -d -t "$session" -n gpu3 "bash -lc $(printf '%q' "$worker_3")"

for index in "${!configs[@]}"; do
  printf '%s\n' "config=${configs[$index]} output=${output_dirs[$index]} session=$session"
done
printf '%s\n' "monitor: tmux attach -t $session"
