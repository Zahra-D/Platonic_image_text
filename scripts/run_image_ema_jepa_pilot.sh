#!/usr/bin/env bash
# Run one image-only EMA-JEPA ablation in a tmux session or directly.
# Usage: ./scripts/run_image_ema_jepa_pilot.sh {fixed|r010|r025|r050} GPU
set -euo pipefail

variant=${1:?variant required: fixed, r010, r025, or r050}
gpu=${2:?GPU index required}
cd /home/zd25e122/clevr_discrete_diffusion

common=(
  --config configs/image_ema_jepa_pilot.yaml
  --wandb-project Platonic_CLEVR_single_modality_ema_jepa
  --wandb-group image_ema_jepa_pilot
)

case "$variant" in
  fixed)
    args=(
      --output-dir outputs/image_ema_jepa_fixed_lambda050_retry
      --wandb-run-name image_ema_jepa_fixed_lambda050_retry
      --wandb-tags pretraining image-only lora tri-lora no-base no-modality-embedding
        learned-absolute-position ema-teacher shared-jepa normalized-jepa middle-layers
        l2-l4 no-sigreg fixed-jepa-weight ema-jepa-pilot retry-after-oom
    )
    ;;
  r010|r025|r050)
    ratio=${variant#r}
    ratio="${ratio:0:1}.${ratio:1}"
    args=(
      --shared-jepa-dynamic-weight --shared-jepa-gradient-ratio "$ratio"
      --output-dir "outputs/image_ema_jepa_dynamic_${variant}"
      --wandb-run-name "image_ema_jepa_dynamic_${variant}"
      --wandb-tags pretraining image-only lora tri-lora no-base no-modality-embedding
        learned-absolute-position ema-teacher shared-jepa normalized-jepa middle-layers
        l2-l4 no-sigreg dynamic-jepa-weight "jepa-gradient-ratio-${variant}" ema-jepa-pilot
    )
    ;;
  *) echo "Unknown variant: $variant" >&2; exit 2;;
esac

exec env CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/zd25e122/miniconda3/envs/unix/bin/python3.10 train_multimodal.py \
  "${common[@]}" "${args[@]}"
