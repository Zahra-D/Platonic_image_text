#!/usr/bin/env bash
set -euo pipefail

repo_dir=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
cd "$repo_dir"

image_out=outputs/presentation_pretraining_image_marginals_384
text_out=outputs/presentation_pretraining_text_marginals_24

if [[ -e "$image_out" || -e "$text_out" ]]; then
  echo "Refusing to overwrite an existing presentation marginal archive." >&2
  exit 1
fi

common=(
  --checkpoint dense_unpaired=outputs/multimodal_unpaired_dense_correct_dataset_visible_check/best.pt
  --checkpoint dense_paired=outputs/pretraining_paired_dense_absolute_70e/best.pt
  --checkpoint lora=outputs/pretraining_unpaired_lora_absolute_no_stage0_70e/best.pt
  --checkpoint stage0=outputs/pretraining_unpaired_lora_absolute_stage0_70e/best.pt
  --checkpoint dann=outputs/pretraining_unpaired_lora_absolute_dann_70e/best.pt
  --checkpoint stage0_dann=outputs/pretraining_unpaired_lora_absolute_stage0_then_dann_70e/best.pt
  --checkpoint sigreg=outputs/pretraining_unpaired_lora_sigreg_absolute_no_stage0_70e/best.pt
  --checkpoint stage0_sigreg=outputs/pretraining_unpaired_lora_absolute_stage0_then_sigreg_70e/best.pt
)

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$python_bin" generate_pretraining_marginal_report.py \
  "${common[@]}" --output-dir "$image_out" --num-samples 1 --num-steps 384 --temperature 1.0 --seed 20260902 --device cuda

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$python_bin" generate_pretraining_text_marginal_report.py \
  "${common[@]}" --output-dir "$text_out" --slot-counts 16 --num-steps 24 --temperature 1.0 --seed 20260902 --device cuda
