#!/usr/bin/env bash
set -euo pipefail

# First-epoch frozen shared/private probe: a clean-trained, 80%-masked-text
# semantic and local-token evaluation for the matched Tri-LoRA control and the
# four completed modulewise JEPA cells.  The calibrated-HSIC run is deliberately
# absent because it has not yet written epoch_000.pt.
repo_dir=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
output_dir=outputs/modulewise_jepa_hsic_epoch000_modality_probe
session=clevr_modulewise_jepa_hsic_epoch000_probe
gpu=${1:-3}

cd "$repo_dir"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Refusing to reuse active tmux session: $session" >&2
  exit 1
fi
if find "$output_dir" -maxdepth 1 -type f -name 'results.json' -print -quit 2>/dev/null | grep -q .; then
  echo "Refusing to overwrite completed evaluation: $output_dir/results.json" >&2
  exit 1
fi

mkdir -p "$output_dir"
tmux new-session -d -s "$session" \
  "cd $repo_dir && exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $python_bin evaluate_jepa_modality.py \\
    --checkpoint plain_lora_epoch000=outputs/text_lora_diffusion_2m_4e_matched/epoch_000.pt \\
    --checkpoint avg_no_hsic_epoch000=outputs/text_module_jepa_avg_no_hsic_2m_4e/epoch_000.pt \\
    --checkpoint avg_hsic_epoch000=outputs/text_module_jepa_avg_hsic_2m_4e/epoch_000.pt \\
    --checkpoint layerwise_no_hsic_epoch000=outputs/text_module_jepa_layerwise_no_hsic_2m_4e/epoch_000.pt \\
    --checkpoint layerwise_hsic_epoch000=outputs/text_module_jepa_layerwise_hsic_2m_4e/epoch_000.pt \\
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
