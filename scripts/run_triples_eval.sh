#!/bin/bash
# After the renders land: tokenize the triples, then score every image model.
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
R=outputs/image_eval_triples
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
free_gpu(){ while :; do for g in 0 1 2 3; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$u" -lt 9000 ] && { echo $g; return; }; done; sleep 120; done; }

while tmux has-session -t triples 2>/dev/null; do sleep 120; done
for s in anchor swap para; do
  n=$(ls $R/$s/*.png 2>/dev/null | wc -l)
  log "$s: $n images"
  [ "$n" -lt 100 ] && { log "ABORT: $s has too few renders"; exit 1; }
done

g=$(free_gpu); log "tokenizing triples on GPU $g"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g python3 scripts/tokenize_image_triples.py \
    --root $R > $R/tokenize.log 2>&1 || { log "TOKENIZE FAILED"; exit 1; }
log "$(tail -1 $R/tokenize.log)"

args=()
while read -r line; do [ -n "$line" ] && args+=(--checkpoint "$line"); done < $R/model_list.txt
log "scoring ${#args[@]} checkpoints"
g=$(free_gpu)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g python3 evaluate_image_triples.py \
    --triples $R/triples_tokens.pt --output-dir outputs/image_triples_eval \
    "${args[@]}" > $R/evaluate.log 2>&1 || log "SOME EVALS FAILED"
log "TRIPLES EVAL COMPLETE: $(ls outputs/image_triples_eval/*.json 2>/dev/null | wc -l) models scored"
