#!/usr/bin/env bash
set -euo pipefail

# Full-shared-LoRA EMA-JEPA + calibrated-HSIC: qkv, out_proj, mlp.0 and mlp.3
# at each selected Transformer layer.  This is distinct from the earlier
# out_proj/mlp.3-only ablation and never reuses its checkpoint directory.
repo_dir=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
config=configs/text_module_jepa_avg_all_shared_hsic_calibrated_2m_4e.yaml
output=outputs/text_module_jepa_avg_all_shared_hsic_calibrated_2m_4e
session=clevr_text_module_jepa_avg_all_shared_hsic_calibrated_2m_4e
gpu=${1:-0}

cd "$repo_dir"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Refusing to reuse active tmux session: $session" >&2
  exit 1
fi
if find "$output" -maxdepth 1 -type f \( -name 'best.pt' -o -name 'last.pt' -o -name 'epoch_*.pt' \) -print -quit 2>/dev/null | grep -q .; then
  echo "Refusing to overwrite checkpointed output: $output" >&2
  exit 1
fi

mkdir -p "$output"
tmux new-session -d -s "$session" \
  "cd $repo_dir && exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $python_bin train_multimodal.py --config $config > $output/launcher.log 2>&1"
echo "Started $session on physical GPU $gpu"
echo "Log: $repo_dir/$output/launcher.log"
