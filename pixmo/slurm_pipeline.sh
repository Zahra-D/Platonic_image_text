#!/bin/bash
#SBATCH --job-name=pixmo_pipeline
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_pixmo_%j.out
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=YOUR_PARTITION
# One time-limited slot of the three-stage PixMo pipeline; resubmits itself until every stage has converged.
# Submit from the repository root (branch `pixmo`):
#   PIXMO_DATA=/scratch/$USER/platonic_pixmo_data PIXMO_RUNS=/scratch/$USER/platonic_pixmo_runs \
#   PIXMO_VENV=~/venvs/clevr sbatch --account=<acct> pixmo/slurm_pipeline.sh
# Optional: PIXMO_BUDGET_MIN (default 80, keep ~10 min below --time), PIXMO_FLAGS (extra pipeline.py flags, e.g.
# "--wandb"), PIXMO_RESUBMIT=0 to run a single slot.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
if [ -n "${PIXMO_VENV:-}" ]; then source "$PIXMO_VENV/bin/activate"; fi
: "${PIXMO_DATA:?set PIXMO_DATA}" "${PIXMO_RUNS:?set PIXMO_RUNS}"
export PIXMO_DATA PIXMO_RUNS
python pixmo/pipeline.py --ngpu "${SLURM_GPUS_ON_NODE:-4}" --budget-min "${PIXMO_BUDGET_MIN:-80}" ${PIXMO_FLAGS:-}
if [ "$(python pixmo/pipeline.py --status)" != "done" ] && [ "${PIXMO_RESUBMIT:-1}" = "1" ]; then
  sbatch --export=ALL pixmo/slurm_pipeline.sh
fi
