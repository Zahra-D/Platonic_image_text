#!/usr/bin/env bash
# I-JEPA-style frozen linear probes for every image checkpoint + baselines.
# Usage: scripts/run_linear_probes_image.sh GPU SHARD NSHARDS
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${1:-2}; SHARD=${2:-0}; NSHARDS=${3:-1}
OUT=outputs/linear_probes_image

specs=()
# 1.2M-image runs: every epoch checkpoint, so models can be compared at equal images seen.
for ckpt in outputs/image_*/epoch_0[0-9][0-9].pt; do
  run=$(basename "$(dirname "$ckpt")"); ep=$(basename "$ckpt" .pt)
  specs+=("--checkpoint" "${run#image_}__${ep}=${ckpt}")
done
# 100k-image EMA-JEPA runs (different training set, only last.pt saved).
for run in image_diffusion_only_ema_jepa_pilot image_ema_jepa_dynamic_r010 image_ema_jepa_dynamic_r025 \
           image_ema_jepa_dynamic_r050 image_ema_jepa_fixed_lambda050_retry; do
  specs+=("--checkpoint" "${run#image_}__last=outputs/${run}/last.pt")
done

# Shard the checkpoint list; shard 0 also runs the baselines.
mine=(); n=0
for ((i = 0; i < ${#specs[@]}; i += 2)); do
  if (( n % NSHARDS == SHARD )); then mine+=("${specs[i]}" "${specs[i+1]}"); fi
  n=$((n + 1))
done
extra=()
if (( SHARD == 0 )); then
  extra=(--easy-baselines
         --random-init untrained_dense=outputs/image_dense_diffusion_1_2m_4e/epoch_000.pt
         --random-init untrained_lora=outputs/image_lora_diffusion_1_2m_4e/epoch_000.pt)
fi
CUDA_VISIBLE_DEVICES=$GPU exec python -u evaluate_linear_probes.py --output-dir "$OUT" "${extra[@]}" "${mine[@]}"
