#!/usr/bin/env bash
# Stage 2 of the dense-shared plan: dense trunk as the shared route plus
# rank-128 modality-private adapters, diffusion loss and HSIC.
#   $1  condition: frozen | trainable
#   $2  physical GPU index
set -euo pipefail

repo_dir=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
condition=${1:?usage: $0 <frozen|trainable> <gpu>}
gpu=${2:?usage: $0 <frozen|trainable> <gpu>}
case "$condition" in
  frozen|trainable) ;;
  *) echo "condition must be frozen or trainable" >&2; exit 1 ;;
esac
name=text_dense_private_${condition}_hsic_2m_2e
config=configs/${name}.yaml
output=outputs/${name}
session=clevr_${name}
cd "$repo_dir"
if tmux has-session -t "$session" 2>/dev/null; then echo "Active session exists" >&2; exit 1; fi
if find "$output" -maxdepth 1 -type f \( -name 'best.pt' -o -name 'last.pt' -o -name 'epoch_*.pt' \) -print -quit 2>/dev/null | grep -q .; then echo "Checkpointed output exists" >&2; exit 1; fi
init=$(sed -n 's/^  init_checkpoint: //p' "$config")
if [ ! -f "$init" ]; then echo "Stage-1 checkpoint $init is missing" >&2; exit 1; fi
mkdir -p "$output"
tmux new-session -d -s "$session" "cd $repo_dir && exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $python_bin train_multimodal.py --config $config > $output/launcher.log 2>&1"
echo "Started $session on physical GPU $gpu (init from $init)"
