#!/usr/bin/env bash
set -euo pipefail

# Matched pair for the gated-predictor correction: same-layer teacher target and
# the JEPA predictor retained, but the predictor gradient now routed end-to-end
# into the shared LoRA qkv/out_proj/mlp.0/mlp.3 of layers 2-4 rather than being
# confined to each supervised layer.  The only change from
# text_module_jepa_layerwise_{no_,}hsic_2m_4e is that routing, so the pair is a
# direct single-variable comparison against those completed cells.

repo_dir=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10

configs=(
  configs/text_module_jepa_layerwise_gated_no_hsic_2m_4e.yaml
  configs/text_module_jepa_layerwise_gated_hsic_2m_4e.yaml
)
outputs=(
  outputs/text_module_jepa_layerwise_gated_no_hsic_2m_4e
  outputs/text_module_jepa_layerwise_gated_hsic_2m_4e
)
sessions=(
  clevr_module_jepa_layerwise_gated_no_hsic
  clevr_module_jepa_layerwise_gated_hsic
)

if [[ $# -eq 0 ]]; then
  gpus=(0 1)
else
  gpus=("$@")
fi
if [[ ${#gpus[@]} -ne 2 ]]; then
  echo "Usage: $0 [GPU_GATED_NO_HSIC GPU_GATED_HSIC]" >&2
  exit 2
fi

cd "$repo_dir"

for index in "${!configs[@]}"; do
  output=${outputs[$index]}
  session=${sessions[$index]}
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Refusing to reuse active tmux session: $session" >&2
    exit 1
  fi
  if find "$output" -maxdepth 1 -type f \( -name 'best.pt' -o -name 'last.pt' -o -name 'epoch_*.pt' \) -print -quit 2>/dev/null | grep -q .; then
    echo "Refusing to overwrite checkpointed output: $output" >&2
    exit 1
  fi
done

for index in "${!configs[@]}"; do
  config=${configs[$index]}
  output=${outputs[$index]}
  session=${sessions[$index]}
  gpu=${gpus[$index]}
  mkdir -p "$output"
  tmux new-session -d -s "$session" \
    "cd $repo_dir && exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $python_bin train_multimodal.py --config $config > $output/launcher.log 2>&1"
  echo "Started $session on physical GPU $gpu; log: $repo_dir/$output/launcher.log"
done
