#!/usr/bin/env bash
set -euo pipefail

repo_dir=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
output_dir=outputs/presentation_current_bidirectional_384
cd "$repo_dir"

if [[ -e "$output_dir/results.json" ]]; then
  echo "refusing to overwrite existing sample report: $output_dir" >&2
  exit 1
fi
mkdir -p "$output_dir"

exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$python_bin" generate_bidirectional_samples.py \
  --checkpoints \
    "Dense_paired=outputs/pretraining_paired_dense_absolute_70e/best.pt" \
    "LoRA=outputs/instruction_tune_2k_10e_from_pretraining_unpaired_lora_absolute_no_stage0_70e/best.pt" \
    "Stage0=outputs/instruction_tune_2k_10e_from_pretraining_unpaired_lora_absolute_stage0_70e/best.pt" \
    "DANN=outputs/instruction_tune_2k_up_to100e_from_pretraining_unpaired_lora_absolute_dann_70e/best.pt" \
    "Stage0_DANN=outputs/instruction_tune_2k_up_to100e_from_pretraining_unpaired_lora_absolute_stage0_then_dann_70e/best.pt" \
    "SIGReg=outputs/instruction_tune_2k_up_to100e_from_pretraining_unpaired_lora_sigreg_absolute_no_stage0_70e/best.pt" \
    "Stage0_SIGReg=outputs/instruction_tune_2k_10e_from_pretraining_unpaired_lora_absolute_stage0_then_sigreg_70e/best.pt" \
  --manifest /home/zd25e122/clevr-dataset-gen_clone/output/platonic_clevr_v1_100k_gpu_visible_s20260825/val_pairs_diverse_human.jsonl \
  --token-cache outputs/platonic_token_cache_correct_dataset_visible_check/val_tokens.pt \
  --dataset-root /home/zd25e122/clevr-dataset-gen_clone/output/platonic_clevr_v1_100k_gpu_visible_s20260825 \
  --indices 4491 \
  --output-dir "$output_dir" \
  --num-steps 384 \
  --reveal-orders confidence random \
  --image-temperature 1.0 \
  --text-temperature 1.0 \
  --seed 20260902 \
  --device cuda
