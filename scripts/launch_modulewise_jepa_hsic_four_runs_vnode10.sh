#!/usr/bin/env bash
set -euo pipefail

# Launch the four-cell modulewise EMA-JEPA × HSIC sweep as independent,
# persistent tmux jobs.  Invoke this on vnode10, where GPUs 0–3 are visible.
# The previously attempted jobs stopped at update 10 before saving a checkpoint;
# this launcher treats any real checkpoint as a safety stop rather than silently
# replacing scientific output.

repo_dir=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10

configs=(
  configs/text_module_jepa_avg_no_hsic_2m_4e.yaml
  configs/text_module_jepa_avg_hsic_2m_4e.yaml
  configs/text_module_jepa_layerwise_no_hsic_2m_4e.yaml
  configs/text_module_jepa_layerwise_hsic_2m_4e.yaml
)
outputs=(
  outputs/text_module_jepa_avg_no_hsic_2m_4e
  outputs/text_module_jepa_avg_hsic_2m_4e
  outputs/text_module_jepa_layerwise_no_hsic_2m_4e
  outputs/text_module_jepa_layerwise_hsic_2m_4e
)
sessions=(
  clevr_module_jepa_avg_no_hsic
  clevr_module_jepa_avg_hsic
  clevr_module_jepa_layerwise_no_hsic
  clevr_module_jepa_layerwise_hsic
)
if [[ $# -eq 0 ]]; then
  gpus=(0 1 2 3)
else
  gpus=("$@")
fi

if [[ ${#gpus[@]} -ne 4 ]]; then
  echo "Usage: $0 [GPU_AVG_NO_HSIC GPU_AVG_HSIC GPU_LAYERWISE_NO_HSIC GPU_LAYERWISE_HSIC]" >&2
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
  if find "$output" -maxdepth 1 -type f \( -name 'best.pt' -o -name 'last.pt' -o -name 'epoch_*.pt' \) -print -quit | grep -q .; then
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
    "cd $repo_dir && exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $python_bin train_multimodal.py --config $config > $output/launcher_rerun.log 2>&1"
  echo "Started $session on physical GPU $gpu; log: $repo_dir/$output/launcher_rerun.log"
done
