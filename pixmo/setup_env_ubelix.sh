#!/bin/bash
# Python env of the PixMo pipeline on UBELIX, inside the project folder (run on a login node; needs internet):
#   bash ~/platonic_pixmo/code/pixmo/setup_env_ubelix.sh > ~/platonic_pixmo/logs/setup_env.log 2>&1
set -e
PROJECT=${PIXMO_PROJECT:-$HOME/platonic_pixmo}
module load Anaconda3/2024.02-1
[ -d "$PROJECT/venv" ] || python3 -m venv "$PROJECT/venv"
source "$PROJECT/venv/bin/activate"
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu128   # 2.12 is not built for cu128; gets 2.11
pip install -q -r "$PROJECT/code/requirements.txt"
python -c "import torch, yaml, numpy; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
echo ENV READY
