#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python_bin="${PYTHON_BIN:-/home/zd25e122/miniconda3/envs/unix/bin/python3.10}"
session="${TMUX_SESSION:-clevr_instruction_tune_2k_continue_10e_all}"

configs=(
  configs/instruction_tune_2k_10e_dense_finalset.yaml
  configs/instruction_tune_2k_10e_lora_no_stage_finalset.yaml
  configs/instruction_tune_2k_10e_lora_dann_finalset.yaml
  configs/instruction_tune_2k_10e_lora_stage0_finalset.yaml
  configs/instruction_tune_2k_10e_lora_stage0_then_dann_finalset.yaml
  configs/instruction_tune_2k_10e_lora_dann_translation_finalset.yaml
  configs/instruction_tune_2k_10e_role_routed_lora_no_stage.yaml
  configs/instruction_tune_2k_10e_role_routed_lora_dann.yaml
  configs/instruction_tune_2k_10e_role_routed_lora_stage0.yaml
  configs/instruction_tune_2k_10e_role_routed_lora_stage0_then_dann.yaml
)
output_dirs=(
  outputs/instruction_tune_2k_10e_dense_finalset
  outputs/instruction_tune_2k_10e_lora_no_stage_finalset
  outputs/instruction_tune_2k_10e_lora_dann_finalset
  outputs/instruction_tune_2k_10e_lora_stage0_finalset
  outputs/instruction_tune_2k_10e_lora_stage0_then_dann_finalset
  outputs/instruction_tune_2k_10e_lora_dann_translation_finalset
  outputs/instruction_tune_2k_10e_role_routed_lora_no_stage
  outputs/instruction_tune_2k_10e_role_routed_lora_dann
  outputs/instruction_tune_2k_10e_role_routed_lora_stage0
  outputs/instruction_tune_2k_10e_role_routed_lora_stage0_then_dann
)
run_names=(
  2k_20e__dense__finalset
  2k_20e__lora_no_stage__finalset
  2k_20e__lora_dann__finalset
  2k_20e__lora_stage0__finalset
  2k_20e__lora_stage0_then_dann__finalset
  2k_20e__lora_dann_translation__finalset
  2k_20e__role_routed__lora_no_stage
  2k_20e__role_routed__lora_dann
  2k_20e__role_routed__lora_stage0
  2k_20e__role_routed__lora_stage0_then_dann
)
gpu_ids=(0 1 2 3)

if tmux has-session -t "$session" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$session" >&2
  exit 1
fi

for index in "${!configs[@]}"; do
  test -f "${configs[$index]}"
  test -f "${output_dirs[$index]}/last.pt"
done

for worker in "${!gpu_ids[@]}"; do
  gpu="${gpu_ids[$worker]}"
  command='set -euo pipefail; '
  for index in "${!configs[@]}"; do
    if (( index % ${#gpu_ids[@]} != worker )); then
      continue
    fi
    printf -v command '%s cd %q && env CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 %q train_multimodal.py --config %q --resume %q --init-checkpoint %q --epochs 20 --max-steps 640 --wandb-group %q --wandb-run-name %q >>%q 2>&1; ' \
      "$command" "$PWD" "$gpu" "$python_bin" "${configs[$index]}" \
      "${output_dirs[$index]}/last.pt" "" "instruction_tuning_2k_20e_continuations" "${run_names[$index]}" \
      "${output_dirs[$index]}/continuation_launcher.log"
  done
  tmux new-session -d -s "$session" -n "gpu${gpu}" "bash -lc $(printf '%q' "$command")" 2>/dev/null || \
    tmux new-window -d -t "$session" -n "gpu${gpu}" "bash -lc $(printf '%q' "$command")"
done

for index in "${!configs[@]}"; do
  printf '%s\n' "config=${configs[$index]} resume=${output_dirs[$index]}/last.pt run=${run_names[$index]}"
done
printf '%s\n' "monitor: tmux attach -t $session"
