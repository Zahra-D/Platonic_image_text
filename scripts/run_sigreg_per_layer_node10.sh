#!/usr/bin/env bash
set -euo pipefail

# Fresh strict no-base Tri-LoRA pretraining runs.  SIGReg is applied to each
# Transformer's own pooled shared-adapter vector; the eight layer statistics
# are averaged before the configured 0.01 loss weight is applied.
repo_dir=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
cd "$repo_dir"

launch() {
  session=$1
  gpu=$2
  config=$3
  output=$4
  run_name=$5
  sigreg_start_epoch=$6
  shift 6
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing output: $output" >&2
    exit 1
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Refusing to reuse active tmux session: $session" >&2
    exit 1
  fi
  mkdir -p "$output"
  tmux new-session -d -s "$session" \
    "cd $repo_dir && exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $python_bin train_multimodal.py --config $config --sigreg --sigreg-per-layer --sigreg-start-epoch $sigreg_start_epoch --sigreg-weight 0.01 --sigreg-warmup-steps 1000 --sigreg-num-slices 256 --sigreg-num-points 17 --sigreg-t-max 3.0 --output-dir $output --wandb-run-name $run_name --wandb-tags pretraining unpaired lora tri-lora no-base trainable-shared-bias no-modality-embedding learned-absolute-position sigreg sigreg-per-layer matched-shuffled-null-validation $* > $output/launcher.log 2>&1"
}

launch clevr_sigreg_per_layer_plain_70e 0 \
  configs/pretraining_unpaired_lora_absolute_no_stage0.yaml \
  outputs/pretraining_unpaired_lora_sigreg_per_layer_absolute_no_stage0_70e \
  pretraining_unpaired_lora_sigreg_per_layer_absolute_no_stage0_70e \
  0 \
  no-stage0 sigreg-from-start no-dann

launch clevr_sigreg_per_layer_stage0_70e 1 \
  configs/pretraining_unpaired_lora_absolute_stage0_70e.yaml \
  outputs/pretraining_unpaired_lora_absolute_stage0_then_sigreg_per_layer_70e \
  pretraining_unpaired_lora_absolute_stage0_then_sigreg_per_layer_70e \
  10 \
  stage0 stage0-10e sigreg-after-stage0 no-dann

tmux list-sessions
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
