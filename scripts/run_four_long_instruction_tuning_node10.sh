#!/usr/bin/env bash
set -euo pipefail

repo_dir=/home/zd25e122/clevr_discrete_diffusion
continue_worker=$repo_dir/scripts/continue_instruction_tune_2k_one.sh
fresh_worker=$repo_dir/scripts/run_instruction_tune_2k_long_one.sh
cd "$repo_dir"

for gpu in 0 1 2 3; do
  used=$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{sum += $1} END {print sum + 0}')
  if (( used > 1000 )); then
    echo "GPU $gpu has ${used} MiB of active compute memory; refusing to overlap jobs." >&2
    exit 1
  fi
done

plain_output=outputs/instruction_tune_2k_10e_from_pretraining_unpaired_lora_absolute_no_stage0_70e
stage0_output=outputs/instruction_tune_2k_10e_from_pretraining_unpaired_lora_absolute_stage0_70e
dann_output=outputs/instruction_tune_2k_up_to100e_from_pretraining_unpaired_lora_absolute_dann_70e
stage0_dann_output=outputs/instruction_tune_2k_up_to100e_from_pretraining_unpaired_lora_absolute_stage0_then_dann_70e
mkdir -p "$dann_output" "$stage0_dann_output"

tmux -L clevr_it2k_long_node10 new-session -d -s plain_lora_gpu0 \
  "cd $repo_dir && env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 $continue_worker configs/pretraining_unpaired_lora_absolute_no_stage0.yaml $plain_output btclg7cs instruction_tune_2k_from_unpaired_lora_no_stage lora tri-lora no-base from-unpaired no-stage0 >> $plain_output/launcher.log 2>&1"

tmux -L clevr_it2k_long_node10 new-session -d -s stage0_lora_gpu1 \
  "cd $repo_dir && env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 $continue_worker configs/pretraining_unpaired_lora_absolute_stage0_70e.yaml $stage0_output wgfke2ho instruction_tune_2k_from_unpaired_lora_stage0 lora tri-lora no-base from-unpaired from-stage0 >> $stage0_output/launcher.log 2>&1"

tmux -L clevr_it2k_long_node10 new-session -d -s dann_lora_gpu2 \
  "cd $repo_dir && env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 $fresh_worker configs/pretraining_unpaired_lora_absolute_dann_70e.yaml outputs/pretraining_unpaired_lora_absolute_dann_70e/best.pt $dann_output instruction_tune_2k_from_unpaired_lora_dann lora tri-lora no-base from-unpaired dann-pretrained > $dann_output/launcher.log 2>&1"

tmux -L clevr_it2k_long_node10 new-session -d -s stage0_dann_lora_gpu3 \
  "cd $repo_dir && env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 $fresh_worker configs/pretraining_unpaired_lora_absolute_stage0_then_dann_70e.yaml outputs/pretraining_unpaired_lora_absolute_stage0_then_dann_70e/best.pt $stage0_dann_output instruction_tune_2k_from_unpaired_lora_stage0_then_dann lora tri-lora no-base from-unpaired from-stage0 dann-pretrained > $stage0_dann_output/launcher.log 2>&1"

tmux -L clevr_it2k_long_node10 list-sessions
