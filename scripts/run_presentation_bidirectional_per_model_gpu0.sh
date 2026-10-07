#!/usr/bin/env bash
set -euo pipefail

repo_dir=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
root=outputs/presentation_bidirectional_per_model_384
cd "$repo_dir"
mkdir -p "$root"

run_one() {
  name=$1
  checkpoint=$2
  output="$root/$name"
  if [[ -f "$output/results.json" ]]; then
    echo "$name already complete"
    return
  fi
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite incomplete output: $output" >&2
    echo "Move it aside after inspection, then rerun this script." >&2
    exit 1
  fi
  mkdir -p "$output"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$python_bin" generate_bidirectional_samples.py \
    --checkpoints "$name=$checkpoint" \
    --manifest /home/zd25e122/clevr-dataset-gen_clone/output/platonic_clevr_v1_100k_gpu_visible_s20260825/val_pairs_diverse_human.jsonl \
    --token-cache outputs/platonic_token_cache_correct_dataset_visible_check/val_tokens.pt \
    --dataset-root /home/zd25e122/clevr-dataset-gen_clone/output/platonic_clevr_v1_100k_gpu_visible_s20260825 \
    --indices 4491 \
    --output-dir "$output" \
    --num-steps 384 \
    --reveal-orders confidence random \
    --image-temperature 1.0 \
    --text-temperature 1.0 \
    --seed 20260902
}

run_one dense_paired outputs/pretraining_paired_dense_absolute_70e/best.pt
run_one lora outputs/instruction_tune_2k_10e_from_pretraining_unpaired_lora_absolute_no_stage0_70e/best.pt
run_one stage0 outputs/instruction_tune_2k_10e_from_pretraining_unpaired_lora_absolute_stage0_70e/best.pt
run_one dann outputs/instruction_tune_2k_up_to100e_from_pretraining_unpaired_lora_absolute_dann_70e/best.pt
run_one stage0_dann outputs/instruction_tune_2k_up_to100e_from_pretraining_unpaired_lora_absolute_stage0_then_dann_70e/best.pt
run_one sigreg outputs/instruction_tune_2k_up_to100e_from_pretraining_unpaired_lora_sigreg_absolute_no_stage0_70e/best.pt
run_one stage0_sigreg outputs/instruction_tune_2k_10e_from_pretraining_unpaired_lora_absolute_stage0_then_sigreg_70e/best.pt
