#!/bin/bash
# Multi-node DDP training with SLURM + torchrun (one torchrun launcher per node).
#
#   export CLEVR_DATA=/path/to/clevr_data          # see CLUSTER.md for the layout
#   CONFIG=configs/cluster/mm_unpaired_dense_2m_40e_ddp.yaml sbatch scripts/cluster/train_ddp.slurm
#   # extra trainer flags go after the script, e.g.  sbatch scripts/cluster/train_ddp.slurm --epochs 10
#
# Edit the #SBATCH lines for your cluster (account / partition / GPUs per node / time).
#SBATCH --job-name=clevr-ddp
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=YOUR_PARTITION
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs

# --- environment: adapt (module load / conda activate / venv) -------------------
# module load cuda/12.x
# source ~/venvs/clevr/bin/activate
: "${CLEVR_DATA:?export CLEVR_DATA=/path/to/clevr_data before sbatch}"
export CLEVR_DATA
# -----------------------------------------------------------------------------

CONFIG=${CONFIG:-configs/cluster/mm_unpaired_dense_2m_40e_ddp.yaml}
GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
# export NCCL_DEBUG=INFO          # uncomment to debug inter-node communication

echo "nodes=$SLURM_NNODES gpus/node=$GPUS_PER_NODE master=$MASTER_ADDR:$MASTER_PORT config=$CONFIG"
srun --kill-on-bad-exit=1 torchrun \
  --nnodes="$SLURM_NNODES" --nproc_per_node="$GPUS_PER_NODE" \
  --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  train_multimodal.py --config "$CONFIG" "$@"
