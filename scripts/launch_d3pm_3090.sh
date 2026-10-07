#!/usr/bin/env bash
# Single-GPU Stage-2 (masked-diffusion Transformer) training launch.
# All hyperparameters live in configs/d3pm.yaml — edit that file to change
# them, or pass CLI flags here to override just this run (e.g. --epochs 50).
#
# Wrap this with whatever your cluster's scheduler needs (sbatch, qsub, etc.) —
# this script itself just runs the training process on one GPU.
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python train_d3pm.py "$@"
