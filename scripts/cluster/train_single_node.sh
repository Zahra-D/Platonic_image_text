#!/bin/bash
# Single-node multi-GPU DDP training (no SLURM needed), e.g. inside an interactive job.
#
#   export CLEVR_DATA=/path/to/clevr_data
#   NGPU=4 bash scripts/cluster/train_single_node.sh configs/cluster/mm_unpaired_dense_2m_40e_ddp.yaml [extra flags]
set -euo pipefail
cd "$(dirname "$0")/../.."
CONFIG=${1:?usage: train_single_node.sh CONFIG [extra trainer flags]}; shift
: "${CLEVR_DATA:?export CLEVR_DATA=/path/to/clevr_data first}"
export CLEVR_DATA OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
torchrun --standalone --nproc_per_node="$NGPU" train_multimodal.py --config "$CONFIG" "$@"
