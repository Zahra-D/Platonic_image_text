#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python_bin="${PYTHON_BIN:-/home/zd25e122/miniconda3/envs/unix/bin/python3.10}"
session="${TMUX_SESSION:-clevr_instruction_tune_2k_10e_role_routed}"

configs=(
  configs/instruction_tune_2k_10e_role_routed_lora_no_stage.yaml
  configs/instruction_tune_2k_10e_role_routed_lora_dann.yaml
  configs/instruction_tune_2k_10e_role_routed_lora_stage0.yaml
  configs/instruction_tune_2k_10e_role_routed_lora_stage0_then_dann.yaml
)
output_dirs=(
  outputs/instruction_tune_2k_10e_role_routed_lora_no_stage
  outputs/instruction_tune_2k_10e_role_routed_lora_dann
  outputs/instruction_tune_2k_10e_role_routed_lora_stage0
  outputs/instruction_tune_2k_10e_role_routed_lora_stage0_then_dann
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

# GPUs 0 and 2 were free at launch; each runs two experiments serially.
worker_0='set -euo pipefail; '
for index in 0 2; do
  printf -v worker_0 '%s cd %q && env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 %q train_multimodal.py --config %q >>%q 2>&1; ' \
    "$worker_0" "$PWD" "$python_bin" "${configs[$index]}" "${output_dirs[$index]}/launcher.log"
done
worker_2='set -euo pipefail; '
for index in 1 3; do
  printf -v worker_2 '%s cd %q && env CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 %q train_multimodal.py --config %q >>%q 2>&1; ' \
    "$worker_2" "$PWD" "$python_bin" "${configs[$index]}" "${output_dirs[$index]}/launcher.log"
done

tmux new-session -d -s "$session" -n gpu0 "bash -lc $(printf '%q' "$worker_0")"
tmux new-window -d -t "$session" -n gpu2 "bash -lc $(printf '%q' "$worker_2")"

for index in "${!configs[@]}"; do
  printf '%s\n' "config=${configs[$index]} output=${output_dirs[$index]} session=$session"
done
printf '%s\n' "monitor: tmux attach -t $session"
