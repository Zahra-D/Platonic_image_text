#!/usr/bin/env bash
set -euo pipefail

repo=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
data_root=/home/zd25e122/clevr-dataset-gen_clone/output/platonic_clevr_v1_100k_gpu_visible_s20260825
manifest="$data_root/val_pairs_diverse_human.jsonl"
cache="$repo/outputs/platonic_token_cache_correct_dataset_visible_check/val_tokens.pt"
report_dir="$repo/outputs/paired_representation_analysis"

checkpoints=(
  "lora_unpaired_no_base=$repo/outputs/multimodal_unpaired_lora_no_base_no_stage0_correct_dataset_visible_check/best.pt"
  "lora_unpaired_stage0_no_base=$repo/outputs/multimodal_unpaired_lora_shared_stage0_10e_no_base_correct_dataset_visible_check/best.pt"
  "lora_unpaired_adv_no_base=$repo/outputs/multimodal_unpaired_lora_adversarial_no_base_no_stage0_correct_dataset_visible_check/best.pt"
  "lora_unpaired_stage0_then_adv_no_base=$repo/outputs/multimodal_unpaired_lora_adversarial_after_shared_stage0_10e_no_base_correct_dataset_visible_check/best.pt"
  "dense_unpaired=$repo/outputs/multimodal_unpaired_dense_correct_dataset_visible_check/best.pt"
)

mkdir -p "$report_dir"
args=()
for checkpoint in "${checkpoints[@]}"; do
  args+=(--checkpoint "$checkpoint")
done

CUDA_VISIBLE_DEVICES=2 "$python_bin" "$repo/analyze_branch_ablation_multiratio.py" \
  "${args[@]}" \
  --split-dir "$data_root" \
  --pair-manifest "$manifest" \
  --token-cache "$cache" \
  --num-samples 128 \
  --batch-size 32 \
  --mask-ratio 0.75 \
  --mask-ratio 1.0 \
  --device cuda \
  --output "$report_dir/no_base_selected_branch_ablation_masks_075_100_128.json" \
  > "$report_dir/no_base_selected_branch_ablation_masks_075_100_128.log" 2>&1 &
ablation_pid=$!

CUDA_VISIBLE_DEVICES=3 "$python_bin" "$repo/analyze_branch_layerwise_recall.py" \
  "${args[@]}" \
  --split-dir "$data_root" \
  --pair-manifest "$manifest" \
  --token-cache "$cache" \
  --num-samples 128 \
  --batch-size 32 \
  --device cuda \
  --output "$report_dir/no_base_selected_layerwise_branch_recall_128.json" \
  > "$report_dir/no_base_selected_layerwise_branch_recall_128.log" 2>&1 &
recall_pid=$!

wait "$ablation_pid"
wait "$recall_pid"
