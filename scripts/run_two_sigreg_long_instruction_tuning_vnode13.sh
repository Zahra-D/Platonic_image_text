#!/usr/bin/env bash
set -euo pipefail

repo_dir=/home/zd25e122/clevr_discrete_diffusion
continue_worker=$repo_dir/scripts/continue_instruction_tune_2k_one.sh
fresh_worker=$repo_dir/scripts/run_instruction_tune_2k_long_one.sh
cd "$repo_dir"

sigreg_output=outputs/instruction_tune_2k_up_to100e_from_pretraining_unpaired_lora_sigreg_absolute_no_stage0_70e
stage0_sigreg_output=outputs/instruction_tune_2k_10e_from_pretraining_unpaired_lora_absolute_stage0_then_sigreg_70e
mkdir -p "$sigreg_output"

tmux new-session -d -s sigreg_lora_long_gpu0 \
  "cd $repo_dir && env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 $fresh_worker configs/pretraining_unpaired_lora_absolute_no_stage0.yaml outputs/pretraining_unpaired_lora_sigreg_absolute_no_stage0_70e/best.pt $sigreg_output instruction_tune_2k_from_unpaired_lora_sigreg lora tri-lora no-base from-unpaired sigreg-pretrained > $sigreg_output/launcher.log 2>&1"

tmux new-session -d -s stage0_sigreg_lora_long_gpu1 \
  "cd $repo_dir && env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 $continue_worker configs/pretraining_unpaired_lora_absolute_stage0_70e.yaml $stage0_sigreg_output 9p1wygpv instruction_tune_2k_from_unpaired_lora_stage0_then_sigreg lora tri-lora no-base from-unpaired from-stage0 sigreg-pretrained >> $stage0_sigreg_output/launcher.log 2>&1"

tmux list-sessions
