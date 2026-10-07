#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 CONFIG CHECKPOINT OUTPUT_DIR RUN_NAME TAG [TAG ...]" >&2
  exit 2
fi

repo_dir=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
config=$1
checkpoint=$2
output=$3
run_name=$4
shift 4

cd "$repo_dir"
if [[ ! -f "$checkpoint" ]]; then
  echo "missing checkpoint: $checkpoint" >&2
  exit 1
fi
if [[ -e "$output/last.pt" || -e "$output/train.log" ]]; then
  echo "refusing to overwrite an existing run: $output" >&2
  exit 1
fi

mkdir -p "$output"
exec env PYTHONUNBUFFERED=1 "$python_bin" train_multimodal.py \
  --config "$config" \
  --data-mode paired \
  --objective both \
  --full-mask-probability 0.15 \
  --unweighting \
  --no-use-modality-embeddings \
  --no-modality-adversarial \
  --no-sigreg \
  --no-backtranslation \
  --shared-only-epochs 0 \
  --batch-size 64 \
  --gradient-accumulation-steps 1 \
  --epochs 100 \
  --early-stopping-patience 15 \
  --early-stopping-min-delta 0.0 \
  --lr 1.0e-4 \
  --shared-lora-lr 7.5e-5 \
  --private-lora-lr 1.0e-4 \
  --embedding-lr 1.0e-4 \
  --max-train-samples 2000 \
  --train-subset-seed 20260827 \
  --val-max-samples 2048 \
  --log-every 4 \
  --eval-every 1 \
  --save-every 1 \
  --paired-val \
  --paired-val-max-samples 512 \
  --paired-val-mask-ratios 0.5 0.75 1.0 \
  --paired-val-control-ratios 0.75 1.0 \
  --step-eval-every 16 \
  --step-eval-max-samples 128 \
  --step-eval-mask-ratios 1.0 \
  --step-eval-control-ratios 1.0 \
  --validation-selection paired_t1_matched \
  --init-checkpoint "$checkpoint" \
  --output-dir "$output" \
  --wandb \
  --wandb-project Platonic_CLEVR_shared_lora_alignment \
  --wandb-entity zahra-delbari-university-of-bern \
  --wandb-group instruction_tuning_2k_100e_earlystop \
  --wandb-run-name "$run_name" \
  --wandb-tags instruction-tuning paired 2k-pairs up-to-100-epochs early-stopping no-modality-embedding bidirectional matched-shuffled-null-validation "$@"
