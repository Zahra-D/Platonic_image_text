#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python_bin="${PYTHON_BIN:-/home/zd25e122/miniconda3/envs/unix/bin/python3.10}"
cuda_id="${CUDA_ID:-2}"
wait_pid="${WAIT_PID:-1245033}"
output_dir="outputs/multimodal_dense_paired_exposure_matched_correct_dataset_visible_check"
config="configs/multimodal_dense_paired_exposure_matched_correct_dataset_visible_check.yaml"

if [[ -e "$output_dir/train.log" || -e "$output_dir/best.pt" || -e "$output_dir/last.pt" ]]; then
  printf 'refusing to overwrite existing matched paired-dense run: %s\n' "$output_dir" >&2
  exit 1
fi

printf 'waiting for paired-dense PID %s to release CUDA device %s\n' "$wait_pid" "$cuda_id"
while kill -0 "$wait_pid" 2>/dev/null; do
  sleep 30
done

mkdir -p "$output_dir"
printf 'GPU released; launching %s\n' "$config"
exec env CUDA_VISIBLE_DEVICES="$cuda_id" PYTHONUNBUFFERED=1 \
  "$python_bin" train_multimodal.py --config "$config"
