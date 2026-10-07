#!/bin/bash
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p outputs/ijepa_continued
free_gpu(){ while :; do for g in 0 1 2 3; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$u" -lt 800 ] && { echo $g; return; }; done; sleep 120; done; }
for c in text_data2vec_from_dense_layerwise_all_w12_24_45pct_2m_6e_continued \
         text_data2vec_from_dense_layerwise_all_w4_8_15pct_2m_6e_continued \
         image_data2vec_from_dense_layerwise_all_ijepa_2m_6e_continued \
         image_data2vec_from_dense_layerwise_all_block2d_1_2m_6e_continued; do
  [ -f "configs/$c.yaml" ] || { echo "no config $c"; continue; }
  have=$(ls outputs/$c/epoch_*.pt 2>/dev/null | grep -vc adapter)
  [ "${have:-0}" -ge 6 ] && { echo "[$(date '+%T')] SKIP $c"; continue; }
  g=$(free_gpu); echo "[$(date '+%T')] TRAIN $c on GPU $g"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g python3 train_multimodal.py --config configs/$c.yaml \
    > outputs/ijepa_continued/$c.train.log 2>&1 || { echo "FAILED $c"; continue; }
  last=$(ls outputs/$c/epoch_*.pt | grep -v adapter | sort | tail -1)
  if [[ "$c" == text_* ]]; then
    CUDA_VISIBLE_DEVICES=$g python3 evaluate_semantic_dprime.py --lexicon-matched --num-worlds 2000 \
      --batch-size 24 --output-dir outputs/semantic_dprime_lexmatched --checkpoint "${c}=$last" \
      >> outputs/ijepa_continued/$c.eval.log 2>&1
  else
    CUDA_VISIBLE_DEVICES=$g python3 evaluate_image_binding.py --per-layer --batch-size 24 --bootstrap 400 \
      --output-dir outputs/image_binding_perlayer --checkpoint "${c}=$last" >> outputs/ijepa_continued/$c.eval.log 2>&1
  fi
  echo "[$(date '+%T')] DONE $c"
done
echo "[$(date '+%T')] CONTINUATIONS COMPLETE"
