#!/bin/bash
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p outputs/dense_20e_queue
free_gpu(){ while :; do for g in 0 1 2 3; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$u" -lt 800 ] && { echo $g; return; }; done; sleep 120; done; }
C=text_dense_diffusion_2m_20e_continued
have=$(ls outputs/$C/epoch_*.pt 2>/dev/null | grep -vc adapter)
if [ "${have:-0}" -lt 8 ]; then
  g=$(free_gpu); echo "[$(date '+%T')] TRAIN $C on GPU $g"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g python3 train_multimodal.py \
    --config configs/$C.yaml > outputs/dense_20e_queue/train.log 2>&1 \
    || { echo "[$(date '+%T')] FAILED $C"; exit 1; }
fi
# epoch_011.pt is the 12th epoch, so epoch_0NN.pt is epoch NN+1.
args=()
for n in 12 13 14 15 16 17 18 19; do
  f=$(printf 'outputs/%s/epoch_%03d.pt' "$C" "$n")
  [ -f "$f" ] && args+=(--checkpoint "dense_$((n+1))ep=$f")
done
echo "[$(date '+%T')] EVAL ${#args[@]} args: ${args[*]}"
if [ "${#args[@]}" -gt 0 ]; then
  g=$(free_gpu)
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g python3 evaluate_semantic_dprime.py \
    --lexicon-matched --num-worlds 2000 --seed 20260922 --batch-size 64 \
    --output-dir outputs/semantic_dprime_lexmatched "${args[@]}" \
    >> outputs/dense_20e_queue/eval.log 2>&1
fi
echo "[$(date '+%T')] DONE $C"
