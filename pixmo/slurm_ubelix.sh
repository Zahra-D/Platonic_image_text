#!/bin/bash
#SBATCH --job-name=pixmo_pipeline
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=6G             # UBELIX rejects --mem; 32 x 6 GB = 192 GB
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=slurm_pixmo_%j.out
# UBELIX (Uni Bern) slot of the three-stage PixMo pipeline; resubmits itself until every stage has converged.
# Defaults: the GPU request your accounts allow (gratis, 4x H100, preemptable). Override on the command line, e.g.
#   sbatch -A <account> -p <partition> --qos=<qos> --gres=gpu:h200:4 --time=01:30:00 pixmo/slurm_ubelix.sh
# Data in ~/platonic_pixmo_data, runs in ~/platonic_pixmo_runs, env ~/venvs/pixmo (setup_env_ubelix.sh).
# Preemption or the time limit only loses the current epoch: the next slot resumes from last.pt.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/platonic_pixmo}"
module load Anaconda3/2024.02-1 >/dev/null 2>&1 || true
source "${PIXMO_VENV:-$HOME/venvs/pixmo}/bin/activate"
export PIXMO_DATA=${PIXMO_DATA:-$HOME/platonic_pixmo_data} PIXMO_RUNS=${PIXMO_RUNS:-$HOME/platonic_pixmo_runs}
LIMIT_MIN=$(( $(squeue -h -j "$SLURM_JOB_ID" -o %l | awk -F: '{ if (NF==3) print $1*60+$2; else print $1 }') ))
BUDGET=${PIXMO_BUDGET_MIN:-$(( LIMIT_MIN > 15 ? LIMIT_MIN - 10 : LIMIT_MIN ))}
python pixmo/pipeline.py --budget-min "$BUDGET" ${PIXMO_FLAGS:-}
if [ "$(python pixmo/pipeline.py --status)" != "done" ] && [ "${PIXMO_RESUBMIT:-1}" = "1" ]; then
  sbatch --export=ALL ${PIXMO_SBATCH_ARGS:-} pixmo/slurm_ubelix.sh
fi
