#!/usr/bin/env bash
# Stage-1 data2vec variants: $1 = config basename (without .yaml), $2 = GPU.
set -euo pipefail
repo_dir=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
name=${1:?usage: $0 <config-basename> <gpu>}
gpu=${2:?usage: $0 <config-basename> <gpu>}
config=configs/${name}.yaml
output=outputs/${name}
session=clevr_${name}
cd "$repo_dir"
[ -f "$config" ] || { echo "No such config: $config" >&2; exit 1; }
# `tmux has-session -t` accepts unique prefixes.  Several ablations deliberately
# share a name prefix, so require an exact session-name match here.
if tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq "$session"; then
  echo "Active session exists" >&2
  exit 1
fi
if find "$output" -maxdepth 1 -type f \( -name 'best.pt' -o -name 'last.pt' -o -name 'epoch_*.pt' \) -print -quit 2>/dev/null | grep -q .; then echo "Checkpointed output exists" >&2; exit 1; fi
mkdir -p "$output"
tmux new-session -d -s "$session" "cd $repo_dir && exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $python_bin train_multimodal.py --config $config > $output/launcher.log 2>&1"
echo "Started $session on physical GPU $gpu"
