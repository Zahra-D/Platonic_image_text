#!/usr/bin/env bash
set -euo pipefail

repo_dir=/home/zd25e122/clevr_discrete_diffusion
worker=$repo_dir/scripts/continue_instruction_tune_2k_one.sh
cd "$repo_dir"

plain_output=outputs/instruction_tune_2k_10e_from_pretraining_unpaired_lora_absolute_no_stage0_70e
stage0_output=outputs/instruction_tune_2k_10e_from_pretraining_unpaired_lora_absolute_stage0_70e
sigreg_output=outputs/instruction_tune_2k_10e_from_pretraining_unpaired_lora_absolute_stage0_then_sigreg_70e

# GPU 0: extend the two already-completed ordinary LoRA runs sequentially.
tmux -L clevr_it2k_extend_node10 new-session -d -s lora_extensions_gpu0 \
  "cd $repo_dir && env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 $worker configs/pretraining_unpaired_lora_absolute_no_stage0.yaml $plain_output btclg7cs instruction_tune_2k_10e_from_unpaired_lora_no_stage lora tri-lora no-base from-unpaired no-stage0 >> $plain_output/launcher.log 2>&1 && env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 $worker configs/pretraining_unpaired_lora_absolute_stage0_70e.yaml $stage0_output wgfke2ho instruction_tune_2k_10e_from_unpaired_lora_stage0 lora tri-lora no-base from-unpaired from-stage0 >> $stage0_output/launcher.log 2>&1"

# GPU 2: wait for the currently running 10-epoch SIGReg-pretrained phase, then
# continue the same checkpoint/optimizer and append to the same W&B run.
tmux -L clevr_it2k_extend_node10 new-session -d -s sigreg_extension_gpu2 \
  "cd $repo_dir && while ! grep -q 'training complete' $sigreg_output/launcher.log; do sleep 20; done; env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 $worker configs/pretraining_unpaired_lora_absolute_stage0_70e.yaml $sigreg_output 9p1wygpv instruction_tune_2k_10e_from_unpaired_lora_stage0_then_sigreg lora tri-lora no-base from-unpaired from-stage0 sigreg-pretrained >> $sigreg_output/launcher.log 2>&1"

tmux -L clevr_it2k_extend_node10 list-sessions
