#!/usr/bin/env bash
# Cached-token Stage-2 experiment recommended after the initial diagnosis:
# learned absolute positions + unweighted masked-token CE.
#
# The VQ token cache is built once with:
#   python pretokenize_clevr.py
# This run writes checkpoints and samples to its own directory, so it will not
# overwrite the five-epoch speed-check run.
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cache_dir="outputs/token_cache"
if [[ ! -f "$cache_dir/train_tokens.pt" || ! -f "$cache_dir/val_tokens.pt" ]]; then
    echo "Missing VQ token cache in $cache_dir. Build it first with:"
    echo "  python pretokenize_clevr.py"
    exit 1
fi

exec python train_d3pm.py \
    --token-cache-dir "$cache_dir" \
    --pos-embed-type absolute \
    --no-weight-by-t \
    --epochs 50 \
    --output-dir outputs/d3pm_absolute_noweight_cached_50ep \
    --wandb-run-name cache-absolute-noweight-50ep \
    "$@"
