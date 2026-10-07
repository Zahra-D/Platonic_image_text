#!/bin/bash
# Paired stage 1 -> stage 2, run as a chain: stage 2 needs stage 1's checkpoint.
# Each stage is scored on the CORRECT 1.2M validation data.
set -u
cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
G=${1:-2}
O=outputs/paired_sequential; mkdir -p $O
score(){
  last=$(ls outputs/$1/epoch_*.pt 2>/dev/null | grep -v adapter | sort | tail -1)
  [ -z "$last" ] && { echo "[$(date '+%T')] no checkpoint for $1"; return; }
  CUDA_VISIBLE_DEVICES=$G python3 evaluate_paired_modality_alignment.py --num-scenes 2000     --manifest /home/zd25e122/clevr-dataset-gen_clone/output/platonic_clevr_v1_5M_train_gpu_visible/val_pairs_human.jsonl --image-cache outputs/image_only_1_2m_token_cache/val_tokens.pt --batch-size 16 --output-dir $O/alignment     --model "$1=$last" >> $O/$1.eval.log 2>&1
  echo "[$(date '+%T')] SCORED $1: $(grep -ahoE 'cka [0-9.]+' $O/$1.eval.log | tail -1)"
}
run(){
  have=$(ls outputs/$1/epoch_*.pt 2>/dev/null | grep -vc adapter)
  if [ "$have" -ge 2 ]; then echo "[$(date '+%T')] SKIP $1 ($have epochs)"; else
    echo "[$(date '+%T')] TRAIN $1"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G python3 train_multimodal.py       --config configs/$1.yaml > $O/$1.train.log 2>&1 || { echo "FAILED $1"; return 1; }
  fi
  score $1
}
run mm_paired_stage1_jepa_lw_all_1_2m_2e || exit 1
run mm_paired_stage2_frozen_1_2m_2e
run mm_paired_stage2_trainable_1_2m_2e
echo "[$(date '+%T')] PAIRED SEQUENTIAL COMPLETE"
