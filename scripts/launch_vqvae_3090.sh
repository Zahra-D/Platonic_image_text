#!/usr/bin/env bash
# Single-GPU VQ-VAE training launch for the RTX 3090 cluster.
# No DDP: this model is small enough that one GPU is plenty (see project README).
# All hyperparameters live in configs/vqvae.yaml — edit that file to change
# them, or pass CLI flags here to override just this run (e.g. --epochs 50).
#
# Wrap this with whatever your cluster's scheduler needs (sbatch, qsub, etc.) —
# this script itself just runs the training process on one GPU.
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python train_vqvae.py "$@"
