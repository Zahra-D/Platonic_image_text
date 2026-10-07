#!/usr/bin/env bash
set -euo pipefail

# Image-only 1.2M-image study: dense control, plain Tri-LoRA control, and the
# best text recipe (gated-predictor layerwise EMA-JEPA + HSIC) at two HSIC
# strengths.  One run per GPU.  Requires the VQ token cache to exist already
# (scripts/pretokenize_image_only_sharded.py).
repo=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
cache=$repo/outputs/image_only_1_2m_token_cache
cd "$repo"

for split in train val; do
  [ -s "$cache/${split}_tokens.pt" ] || { echo "missing $cache/${split}_tokens.pt; tokenize first" >&2; exit 1; }
done

configs=(
  configs/image_dense_diffusion_1_2m_4e.yaml
  configs/image_lora_diffusion_1_2m_4e.yaml
  configs/image_module_jepa_layerwise_gated_hsic_matched_1_2m_4e.yaml
  configs/image_module_jepa_layerwise_gated_hsic_strong_1_2m_4e.yaml
)
outputs=(
  outputs/image_dense_diffusion_1_2m_4e
  outputs/image_lora_diffusion_1_2m_4e
  outputs/image_module_jepa_layerwise_gated_hsic_matched_1_2m_4e
  outputs/image_module_jepa_layerwise_gated_hsic_strong_1_2m_4e
)
sessions=(clevr_image_dense clevr_image_lora clevr_image_gated_hsic_matched clevr_image_gated_hsic_strong)
# Optional: RUNS="0 1 2" selects a subset of the four runs (same order as the
# arrays above), so a GPU can be left free for other work.
read -r -a picked <<< "${RUNS:-0 1 2 3}"
if [[ $# -eq 0 ]]; then gpus=(0 1 2 3); else gpus=("$@"); fi
[[ ${#gpus[@]} -ge ${#picked[@]} ]] || { echo "need at least ${#picked[@]} GPUs for RUNS='${picked[*]}'" >&2; exit 2; }
configs=($(for i in "${picked[@]}"; do echo "${configs[$i]}"; done))
outputs=($(for i in "${picked[@]}"; do echo "${outputs[$i]}"; done))
sessions=($(for i in "${picked[@]}"; do echo "${sessions[$i]}"; done))

for i in "${!configs[@]}"; do
  session=${sessions[$i]}; output=${outputs[$i]}
  tmux has-session -t "$session" 2>/dev/null && { echo "session exists: $session" >&2; exit 1; }
  if find "$output" -maxdepth 1 -type f \( -name 'best.pt' -o -name 'last.pt' -o -name 'epoch_*.pt' \) -print -quit 2>/dev/null | grep -q .; then
    echo "refusing to overwrite checkpointed output: $output" >&2; exit 1
  fi
done

for i in "${!configs[@]}"; do
  mkdir -p "${outputs[$i]}"
  tmux new-session -d -s "${sessions[$i]}" \
    "cd $repo && exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${gpus[$i]} PYTHONUNBUFFERED=1 $python_bin train_multimodal.py --config ${configs[$i]} > ${outputs[$i]}/launcher.log 2>&1"
  echo "started ${sessions[$i]} on GPU ${gpus[$i]}: ${outputs[$i]}/launcher.log"
done
