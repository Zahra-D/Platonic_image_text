#!/usr/bin/env bash
set -euo pipefail

repo_dir=/home/zd25e122/clevr_discrete_diffusion
worker=$repo_dir/scripts/run_instruction_tune_2k_one.sh
cd "$repo_dir"

assert_gpu_idle() {
  gpu=$1
  used=$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{sum += $1} END {print sum + 0}')
  if (( used > 1000 )); then
    echo "GPU $gpu has ${used} MiB of active compute memory; refusing to overlap jobs." >&2
    exit 1
  fi
}

assert_gpu_idle 1
assert_gpu_idle 2

plain_output=outputs/instruction_tune_2k_10e_from_pretraining_unpaired_lora_absolute_no_stage0_70e
stage0_output=outputs/instruction_tune_2k_10e_from_pretraining_unpaired_lora_absolute_stage0_70e
sigreg_stage0_output=outputs/instruction_tune_2k_10e_from_pretraining_unpaired_lora_absolute_stage0_then_sigreg_70e

mkdir -p "$plain_output" "$stage0_output" "$sigreg_stage0_output"

tmux new-session -d -s clevr_it2k_lora_queue \
  "cd $repo_dir && env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 $worker configs/pretraining_unpaired_lora_absolute_no_stage0.yaml outputs/pretraining_unpaired_lora_absolute_no_stage0_70e/best.pt $plain_output instruction_tune_2k_10e_from_unpaired_lora_no_stage lora tri-lora no-base from-unpaired no-stage0 > $plain_output/launcher.log 2>&1 && env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 $worker configs/pretraining_unpaired_lora_absolute_stage0_70e.yaml outputs/pretraining_unpaired_lora_absolute_stage0_70e/best.pt $stage0_output instruction_tune_2k_10e_from_unpaired_lora_stage0 lora tri-lora no-base from-unpaired from-stage0 > $stage0_output/launcher.log 2>&1"

tmux new-session -d -s clevr_it2k_baselines_queue \
  "cd $repo_dir && env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 $worker configs/pretraining_unpaired_lora_absolute_stage0_70e.yaml outputs/pretraining_unpaired_lora_absolute_stage0_then_sigreg_70e/best.pt $sigreg_stage0_output instruction_tune_2k_10e_from_unpaired_lora_stage0_then_sigreg lora tri-lora no-base from-unpaired from-stage0 sigreg-pretrained > $sigreg_stage0_output/launcher.log 2>&1"

tmux list-sessions
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
