#!/usr/bin/env bash
set -euo pipefail
repo_dir=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
config=configs/text_data2vec_avg_gated_shared_calibrated_jepa_hsic_2m_4e.yaml
output=outputs/text_data2vec_avg_gated_shared_calibrated_jepa_hsic_2m_4e
session=clevr_text_data2vec_avg_gated_shared_calibrated_jepa_hsic_2m_4e
gpu=${1:-0}
cd "$repo_dir"
if tmux has-session -t "$session" 2>/dev/null; then echo "Active session exists" >&2; exit 1; fi
if find "$output" -maxdepth 1 -type f \( -name 'best.pt' -o -name 'last.pt' -o -name 'epoch_*.pt' \) -print -quit 2>/dev/null | grep -q .; then echo "Checkpointed output exists" >&2; exit 1; fi
mkdir -p "$output"
tmux new-session -d -s "$session" "cd $repo_dir && exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $python_bin train_multimodal.py --config $config > $output/launcher.log 2>&1"
echo "Started $session on physical GPU $gpu"
