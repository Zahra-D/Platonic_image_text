#!/usr/bin/env bash
set -euo pipefail

repo=/home/zd25e122/clevr_discrete_diffusion
python_bin=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
gpu=${1:-1}
checkpoint=${2:-outputs/text_jepa_coco_one_caption_scratch/last.pt}
destination=${3:-outputs/text_jepa_coco_one_caption_scratch/retrieval}

cd "$repo"
mkdir -p "$destination"
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
  "$python_bin" evaluate_coco_caption_retrieval.py \
  --checkpoint "$checkpoint" \
  --retrieval-manifest data/ms_coco_2017/manifests/val2017_retrieval.jsonl \
  --max-images 5000 --batch-size 64 --query-chunk-size 128 \
  --output "$destination/trained.json"

env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
  "$python_bin" evaluate_coco_caption_retrieval.py \
  --checkpoint "$checkpoint" --ema-teacher \
  --retrieval-manifest data/ms_coco_2017/manifests/val2017_retrieval.jsonl \
  --max-images 5000 --batch-size 64 --query-chunk-size 128 \
  --output "$destination/ema_teacher.json"

env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
  "$python_bin" evaluate_coco_caption_retrieval.py \
  --checkpoint "$checkpoint" --random-init \
  --retrieval-manifest data/ms_coco_2017/manifests/val2017_retrieval.jsonl \
  --max-images 5000 --batch-size 64 --query-chunk-size 128 \
  --output "$destination/random_init.json"
