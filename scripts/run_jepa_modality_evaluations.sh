#!/usr/bin/env bash
# Persistent launcher for the modality-local JEPA report.  Run one named job:
#   ./scripts/run_jepa_modality_evaluations.sh text_ema 0
#   ./scripts/run_jepa_modality_evaluations.sh legacy_text 1
#   ./scripts/run_jepa_modality_evaluations.sh legacy_image 2
#   ./scripts/run_jepa_modality_evaluations.sh image_ema 0
set -euo pipefail

job=${1:?job required}
gpu=${2:?GPU index required}
cd /home/zd25e122/clevr_discrete_diffusion
root=/home/zd25e122/clevr-dataset-gen_clone/output/platonic_clevr_v1_100k_gpu_visible_s20260825
common=(
  --train-dir "$root" --val-dir "$root"
  --train-manifest "$root/train_pairs_diverse_human.jsonl"
  --val-manifest "$root/val_pairs_diverse_human.jsonl"
  --token-cache outputs/platonic_token_cache_correct_dataset_visible_check
  --layers 2 3 4 --mask-ratios 0.1 0.2 0.4 0.6 0.8
  --train-samples 512 --val-samples 256 --batch-size 16
  --token-probe-train 2000 --token-probe-val 1000 --probe-epochs 8
)
python=/home/zd25e122/miniconda3/envs/unix/bin/python3.10

case "$job" in
  text_ema)
    args=(--modality text
      --checkpoint text_diffusion_only=outputs/text_diffusion_only_ema_jepa_pilot/last.pt
      --checkpoint text_ema_fixed=outputs/text_ema_jepa_fixed_lambda050/last.pt
      --checkpoint text_ema_dynamic_r010=outputs/text_ema_jepa_dynamic_r010/last.pt
      --checkpoint text_ema_dynamic_r025=outputs/text_ema_jepa_dynamic_r025/last.pt
      --checkpoint text_ema_dynamic_r050=outputs/text_ema_jepa_dynamic_r050/last.pt
      --output outputs/jepa_modality_eval_text_ema/results.json)
    ;;
  image_ema)
    args=(--modality image
      --checkpoint image_diffusion_only=outputs/image_diffusion_only_ema_jepa_pilot/last.pt
      --checkpoint image_ema_fixed=outputs/image_ema_jepa_fixed_lambda050_retry/last.pt
      --checkpoint image_ema_dynamic_r010=outputs/image_ema_jepa_dynamic_r010/last.pt
      --checkpoint image_ema_dynamic_r025=outputs/image_ema_jepa_dynamic_r025/last.pt
      --checkpoint image_ema_dynamic_r050=outputs/image_ema_jepa_dynamic_r050/last.pt
      --output outputs/jepa_modality_eval_image_ema/results.json)
    ;;
  legacy_text|legacy_image)
    modality=${job#legacy_}
    args=(--modality "$modality"
      --checkpoint plain_lora=outputs/pretraining_unpaired_lora_absolute_no_stage0_70e/last.pt
      --checkpoint jepa_per_layer=outputs/pretraining_unpaired_lora_shared_jepa_sigreg_per_layer_absolute_70e/last.pt
      --checkpoint jepa_normalized=outputs/pretraining_unpaired_lora_middle_l2_l4_normalized_jepa_sigreg_absolute_70e/last.pt
      --checkpoint jepa_strong_normalized=outputs/pretraining_unpaired_lora_middle_l2_l4_strong_balanced_normalized_jepa_sigreg_absolute_70e/last.pt
      --checkpoint jepa_strong_raw_l2=outputs/pretraining_unpaired_lora_middle_l2_l4_strong_balanced_raw_l2_jepa_sigreg_absolute_70e/last.pt
      --output "outputs/jepa_modality_eval_${job}/results.json")
    ;;
  *) echo "unknown job: $job" >&2; exit 2;;
esac

exec env CUDA_VISIBLE_DEVICES="$gpu" "$python" evaluate_jepa_modality.py "${common[@]}" "${args[@]}"
