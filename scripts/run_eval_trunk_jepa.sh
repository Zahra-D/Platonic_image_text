#!/bin/bash
# Trunk-only eval of the trunk-JEPA variants (TRUNK: = shared trunk alone, private
# LoRA dropped) vs their starting trunk and dense single-modality models at the
# same total epoch count (7). Runs ONE eval at a time and aborts an eval if host
# MemAvailable < MIN_MB, so the training jobs are never OOM-killed.
#   usage: run_eval_trunk_jepa.sh LABEL=CHECKPOINT ...   (all loaded trunk-only)
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-0}; MIN_MB=${MIN_MB:-4000}; START_MB=${START_MB:-9000}
Q=${OUT:-outputs/eval_trunk_jepa}; DE=${DE:-7}; mkdir -p $Q
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
CACHE=outputs/image_only_2_5m_token_cache/val_tokens.pt
TXTVAL=/home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_1m/val_text_only_human.jsonl
DT=${DT:-outputs/text_dense_diffusion_2m_8e_continued/epoch_006.pt}
DI=${DI:-outputs/image_dense_diffusion_2_5m_40e/epoch_006.pt}
SPECS=("$@")
mm_ck(){ for s in "${SPECS[@]}"; do printf -- "--checkpoint %s=TRUNK:%s " "${s%%=*}" "${s#*=}"; done; }
mm_tx(){ for s in "${SPECS[@]}"; do printf -- "--text-checkpoint %s_TEXT=TRUNK:%s " "${s%%=*}" "${s#*=}"; done; }
mm_im(){ for s in "${SPECS[@]}"; do printf -- "--image-checkpoint %s_IMAGE=TRUNK:%s " "${s%%=*}" "${s#*=}"; done; }
mm_pairs(){ for s in "${SPECS[@]}"; do printf -- "--pair %s_TEXT|%s_IMAGE " "${s%%=*}" "${s%%=*}"; done; }
avail(){ awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
attempt(){ local name=$1; shift
  # wait (up to 30 min) for headroom before starting
  local waited=0; while [ "$(avail)" -lt $START_MB ] && [ $waited -lt 1800 ]; do sleep 30; waited=$((waited+30)); done
  "$@" > $Q/$name.log 2>&1 & local pid=$! low=0
  while kill -0 $pid 2>/dev/null; do
    if [ "$(avail)" -lt $MIN_MB ]; then low=$((low+1)); else low=0; fi
    # abort only if RAM stays low for 3 checks (15 s): a training job would be killed instead
    if [ $low -ge 3 ]; then kill $pid; wait $pid 2>/dev/null; return 2; fi
    sleep 5; done
  wait $pid; }
run(){ local name=$1; shift; log "START $name"
  attempt $name "$@"; local rc=$?
  if [ $rc -eq 2 ]; then log "ABORTED $name (low RAM), retrying once"; attempt $name "$@"; rc=$?; fi
  case $rc in 0) log "DONE $name";; 2) log "ABORTED $name again (low RAM)";; *) log "FAILED $name";; esac; }
run structure python3 evaluate_cross_modal_structure.py --text-checkpoint denseT${DE}_TEXT=$DT \
  --image-checkpoint denseI${DE}_IMAGE=$DI $(mm_tx) $(mm_im) \
  --image-cache $CACHE --num-scenes 4000 --seed 20260922 --output-dir $Q/structure
run xret python3 evaluate_cross_modal_retrieval.py --num-scenes 8000 --test 1000 --val 1000 --output-dir $Q/xret \
  --text-checkpoint denseT${DE}_TEXT=$DT --image-checkpoint denseI${DE}_IMAGE=$DI $(mm_tx) $(mm_im) \
  --pair "denseT${DE}_TEXT|denseI${DE}_IMAGE" $(mm_pairs)
run probes_image python3 evaluate_probe_suite.py --modality image --num-scenes 6000 --noise-levels 0.0 0.75 \
  --output-dir $Q/probe_suite/image --checkpoint denseI${DE}=$DI $(mm_ck)
run probes_text python3 evaluate_probe_suite.py --modality text --manifest $TXTVAL --num-scenes 6000 --noise-levels 0.0 0.75 \
  --output-dir $Q/probe_suite/text --checkpoint denseT${DE}=$DT $(mm_ck)
run binding python3 evaluate_image_binding.py --per-layer --seed 20260924 --image-cache $CACHE \
  --output-dir $Q/binding --checkpoint denseI${DE}=$DI $(mm_ck)
run sens_text python3 evaluate_text_sensitivity.py --num-worlds 2000 --output-dir $Q/sens_text --checkpoint denseT${DE}=$DT $(mm_ck)
run sens_image python3 evaluate_semantic_sensitivity.py --output-dir $Q/sens_image --checkpoint denseI${DE}=$DI $(mm_ck)
run triples python3 evaluate_image_triples.py --triples outputs/image_eval_triples/triples_tokens.pt \
  --output-dir $Q/triples --checkpoint denseI${DE}=$DI $(mm_ck)
run scene_image python3 evaluate_scene_retrieval.py --modality image --num-scenes 2500 \
  --output-dir $Q/scene_retrieval/image --checkpoint denseI${DE}=$DI $(mm_ck)
run scene_text python3 evaluate_scene_retrieval.py --modality text --manifest $TXTVAL --num-scenes 2500 \
  --output-dir $Q/scene_retrieval/text --checkpoint denseT${DE}=$DT $(mm_ck)
log "EVAL PASS COMPLETE ($*)"
