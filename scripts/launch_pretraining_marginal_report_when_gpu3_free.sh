#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python_bin="${PYTHON_BIN:-/home/zd25e122/miniconda3/envs/unix/bin/python3.10}"
session="${TMUX_SESSION:-clevr_pretraining_marginal_report}"
output_dir="outputs/pretraining_marginal_allmask_384steps"
wait_logs=(
  outputs/instruction_tune_2k_10e_lora_stage0_finalset/continuation_launcher.log
  outputs/instruction_tune_2k_10e_role_routed_lora_dann/continuation_launcher.log
)

if tmux has-session -t "$session" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$session" >&2
  exit 1
fi
if [[ -e "$output_dir/index.html" || -e "$output_dir/results.json" ]]; then
  printf 'refusing to overwrite existing report: %s\n' "$output_dir" >&2
  exit 1
fi
mkdir -p "$output_dir"

wait_command='set -euo pipefail; '
for log in "${wait_logs[@]}"; do
  printf -v wait_command '%s until grep -q %q %q; do sleep 30; done; ' \
    "$wait_command" 'training complete' "$log"
done
printf -v wait_command '%s cd %q && env CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 %q generate_pretraining_marginal_report.py --output-dir %q --num-samples 4 --num-steps 384 --temperature 1.0 --seed 20260827 --device cuda --checkpoint %q --checkpoint %q --checkpoint %q --checkpoint %q --checkpoint %q --checkpoint %q --checkpoint %q >>%q 2>&1; ' \
  "$wait_command" "$PWD" "$python_bin" "$output_dir" \
  'Dense unpaired=outputs/multimodal_unpaired_dense_correct_dataset_visible_check/best.pt' \
  'LoRA unpaired=outputs/multimodal_unpaired_lora_no_base_no_stage0_correct_dataset_visible_check/best.pt' \
  'LoRA DANN=outputs/multimodal_unpaired_lora_adversarial_no_base_no_stage0_correct_dataset_visible_check/best.pt' \
  'LoRA stage 0=outputs/multimodal_unpaired_lora_shared_stage0_10e_no_base_correct_dataset_visible_check/best.pt' \
  'LoRA stage 0 then DANN=outputs/multimodal_unpaired_lora_adversarial_after_shared_stage0_10e_no_base_correct_dataset_visible_check/best.pt' \
  'LoRA DANN plus translation=outputs/multimodal_unpaired_lora_adversarial_backtranslation_contrastive_no_base_no_stage0_correct_dataset_visible_check/best.pt' \
  'Dense paired reference=outputs/multimodal_dense_paired_exposure_matched_correct_dataset_visible_check/best.pt' \
  "$output_dir/launcher.log"

tmux new-session -d -s "$session" -n marginal "bash -lc $(printf '%q' "$wait_command")"
printf '%s\n' "scheduled report=$output_dir/index.html after the GPU 3 continuation queue completes"
