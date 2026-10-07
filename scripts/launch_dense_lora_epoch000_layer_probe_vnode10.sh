#!/usr/bin/env bash
set -euo pipefail
repo_dir=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
output=outputs/dense_lora_epoch000_layer_probe/results.json
session=clevr_dense_lora_epoch000_layer_probe
gpu=${1:-3}
cd "$repo_dir"
if tmux has-session -t "$session" 2>/dev/null || test -f "$output"; then
  echo "Refusing to reuse active/completed evaluation" >&2; exit 1
fi
mkdir -p "$(dirname "$output")"
tmux new-session -d -s "$session" \
  "cd $repo_dir && exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $python_bin evaluate_text_layer_representations.py \\
    --checkpoint dense_epoch000=outputs/text_dense_diffusion_2m_4e_matched/epoch_000.pt \\
    --checkpoint plain_lora_epoch000=outputs/text_lora_diffusion_2m_4e_matched/epoch_000.pt \\
    --train-dir /home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_2m \\
    --val-dir /home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_1m \\
    --train-manifest /home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_2m/train_text_only_human.jsonl \\
    --val-manifest /home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_1m/val_text_only_human.jsonl \\
    --token-cache outputs/unused_text_only_cache --caption-field caption_human \\
    --layers 2 3 4 --train-samples 2048 --val-samples 2048 --batch-size 32 --mask-ratio 0.8 --seed 20260916 \\
    --output $output > $(dirname "$output")/launcher.log 2>&1"
echo "Started $session on GPU $gpu"
