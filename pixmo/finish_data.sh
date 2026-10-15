#!/bin/bash
# Dev machine only: keep tokenizing newly downloaded chunks until the PixMo download has finished, then merge and pack.
#   PIXMO_DATA=~/platonic_pixmo_data nohup bash pixmo/finish_data.sh > ~/platonic_align/logs/pixmo_finish_data.log 2>&1 &
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/platonic_align/.venv/bin/python}
export OMP_NUM_THREADS=3 CUDA_DEVICE_ORDER=PCI_BUS_ID
downloading() { ps -eo args | grep -q "^[^ ]*python scripts/download_pixmo.py"; }
tokenizing() { ps -eo args | grep -q "^[^ ]*python pixmo/prepare_pixmo.py --tokenize"; }
while tokenizing; do sleep 60; done
while downloading; do
  $PY pixmo/prepare_pixmo.py --tokenize --gpu ${GPU:-0} || exit 1
  sleep 300
done
$PY pixmo/prepare_pixmo.py --tokenize --gpu ${GPU:-0} || exit 1
$PY pixmo/prepare_pixmo.py --merge || exit 1
bash pixmo/pack_data.sh "${PACK:-$HOME/platonic_pixmo_data.tar}" && echo "DATA READY $(date)"
