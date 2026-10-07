#!/usr/bin/env bash
set -euo pipefail

# Standard frozen shared/private probe (clean-trained, 80%-masked-text) on the
# final checkpoints, placing the two collapsed gated-data2vec runs beside the
# plain Tri-LoRA control and the two layerwise JEPA cells.  Identical protocol
# to launch_modulewise_jepa_hsic_epoch000_probe_vnode10.sh, at epoch 3.
repo_dir=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
output_dir=outputs/collapsed_shared_final_modality_probe
session=clevr_collapsed_shared_final_probe
gpu=${1:-2}

cd "$repo_dir"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Refusing to reuse active tmux session: $session" >&2
  exit 1
fi
if [ -e "$output_dir/results.json" ]; then
  echo "Refusing to overwrite completed evaluation: $output_dir/results.json" >&2
  exit 1
fi

mkdir -p "$output_dir"
tmux new-session -d -s "$session" \
  "cd $repo_dir && exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $python_bin evaluate_jepa_modality.py \\
    --checkpoint plain_lora_epoch003=outputs/text_lora_diffusion_2m_4e_matched/epoch_003.pt \\
    --checkpoint layerwise_no_hsic_epoch003=outputs/text_module_jepa_layerwise_no_hsic_2m_4e/epoch_003.pt \\
    --checkpoint layerwise_hsic_epoch003=outputs/text_module_jepa_layerwise_hsic_2m_4e/epoch_003.pt \\
    --checkpoint data2vec_avg_gated_epoch003=outputs/text_data2vec_avg_gated_shared_calibrated_jepa_hsic_2m_4e/epoch_003.pt \\
    --checkpoint data2vec_noavg_gated_epoch003=outputs/text_data2vec_noavg_gated_shared_calibrated_jepa_hsic_2m_4e/epoch_003.pt \\
    --train-dir /home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_2m \\
    --val-dir /home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_1m \\
    --train-manifest /home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_2m/train_text_only_human.jsonl \\
    --val-manifest /home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_1m/val_text_only_human.jsonl \\
    --token-cache outputs/unused_text_only_cache \\
    --caption-field caption_human --data-mode text_only --modality text --layers 2 3 4 \\
    --mask-ratios 0.8 --train-samples 2048 --val-samples 2048 --batch-size 32 \\
    --token-probe-train 4000 --token-probe-val 2000 --probe-epochs 15 --seed 20260916 \\
    --output $output_dir/results.json > $output_dir/launcher.log 2>&1"
echo "Started $session on physical GPU $gpu"
echo "Log: $repo_dir/$output_dir/launcher.log"
