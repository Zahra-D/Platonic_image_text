#!/usr/bin/env bash
# Evaluate every completed LoRA pretraining family for one modality.
# Usage: ./scripts/run_lora_modality_local_report.sh {text|image} GPU
set -euo pipefail

modality=${1:?modality required: text or image}
gpu=${2:?GPU index required}
case "$modality" in text|image) ;; *) echo "modality must be text or image" >&2; exit 2;; esac

cd /home/zd25e122/clevr_discrete_diffusion
root=/home/zd25e122/clevr-dataset-gen_clone/output/platonic_clevr_v1_100k_gpu_visible_s20260825
output="outputs/jepa_modality_eval_lora_family_${modality}/results.json"
test ! -e "$(dirname "$output")" || { echo "Output directory already exists: $(dirname "$output")" >&2; exit 1; }

exec env CUDA_VISIBLE_DEVICES="$gpu" /home/zd25e122/miniconda3/envs/unix/bin/python3.10 \
  evaluate_jepa_modality.py \
  --train-dir "$root" --val-dir "$root" \
  --train-manifest "$root/train_pairs_diverse_human.jsonl" \
  --val-manifest "$root/val_pairs_diverse_human.jsonl" \
  --token-cache outputs/platonic_token_cache_correct_dataset_visible_check \
  --modality "$modality" --layers 2 3 4 \
  --mask-ratios 0.1 0.2 0.4 0.6 0.8 \
  --train-samples 512 --val-samples 256 --batch-size 16 \
  --token-probe-train 2000 --token-probe-val 1000 --probe-epochs 8 \
  --checkpoint plain_lora_no_stage=outputs/pretraining_unpaired_lora_absolute_no_stage0_70e/last.pt \
  --checkpoint lora_stage0=outputs/pretraining_unpaired_lora_absolute_stage0_70e/last.pt \
  --checkpoint lora_dann=outputs/pretraining_unpaired_lora_absolute_dann_70e/last.pt \
  --checkpoint lora_stage0_then_dann=outputs/pretraining_unpaired_lora_absolute_stage0_then_dann_70e/last.pt \
  --checkpoint jepa_per_layer=outputs/pretraining_unpaired_lora_shared_jepa_sigreg_per_layer_absolute_70e/last.pt \
  --checkpoint jepa_normalized=outputs/pretraining_unpaired_lora_middle_l2_l4_normalized_jepa_sigreg_absolute_70e/last.pt \
  --checkpoint jepa_strong_normalized=outputs/pretraining_unpaired_lora_middle_l2_l4_strong_balanced_normalized_jepa_sigreg_absolute_70e/last.pt \
  --checkpoint jepa_strong_raw_l2=outputs/pretraining_unpaired_lora_middle_l2_l4_strong_balanced_raw_l2_jepa_sigreg_absolute_70e/last.pt \
  --output "$output"
