#!/bin/bash
# Shared-trunk-only eval (private LoRA ignored: TRUNK: loads the dense trunk alone)
# of mm_unpaired_dense_private_lora_r128_2m_40e at epochs 1, 2, 4, plus dense
# single-modality baselines at epoch 4. Random / dense ep1 / ep2 results already
# in outputs/eval_mm_private are reused. Three chains on GPU_A/B/C.
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
Q=outputs/eval_mm_private; mkdir -p $Q
GPU_A=${GPU_A:-0}; GPU_B=${GPU_B:-1}; GPU_C=${GPU_C:-2}
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
MM=outputs/mm_unpaired_dense_private_lora_r128_2m_40e
T1=TRUNK:$MM/epoch_000.pt; T2=TRUNK:$MM/epoch_001.pt; T4=TRUNK:$MM/epoch_003.pt
DT=outputs/text_dense_diffusion_2m_4e_matched; DI=outputs/image_dense_diffusion_2_5m_40e
DT1=$DT/epoch_000.pt; DT2=$DT/epoch_001.pt; DT4=$DT/epoch_003.pt
DI1=$DI/epoch_000.pt; DI2=$DI/epoch_001.pt; DI4=$DI/epoch_003.pt
CACHE=outputs/image_only_2_5m_token_cache/val_tokens.pt
TXTVAL=/home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_1m/val_text_only_human.jsonl
MMC="--checkpoint mmT1=$T1 --checkpoint mmT2=$T2 --checkpoint mmT4=$T4"
run(){ local name=$1 gpu=$2; shift 2; log "START $name (GPU $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu "$@" > $Q/trunk_$name.log 2>&1 && log "DONE $name" || log "FAILED $name"; }
chain_a(){
  run structure $GPU_A python3 evaluate_cross_modal_structure.py \
    --text-checkpoint denseT1_TEXT=$DT1 --text-checkpoint denseT2_TEXT=$DT2 --text-checkpoint denseT4_TEXT=$DT4 \
    --text-checkpoint mmT1_TEXT=$T1 --text-checkpoint mmT2_TEXT=$T2 --text-checkpoint mmT4_TEXT=$T4 \
    --image-checkpoint denseI1_IMAGE=$DI1 --image-checkpoint denseI2_IMAGE=$DI2 --image-checkpoint denseI4_IMAGE=$DI4 \
    --image-checkpoint mmT1_IMAGE=$T1 --image-checkpoint mmT2_IMAGE=$T2 --image-checkpoint mmT4_IMAGE=$T4 \
    --image-cache $CACHE --num-scenes 4000 --seed 20260922 --output-dir $Q/structure
  run xret $GPU_A python3 evaluate_cross_modal_retrieval.py --num-scenes 8000 --test 1000 --val 1000 \
    --output-dir $Q/xret \
    --text-checkpoint denseT4_TEXT=$DT4 --text-checkpoint mmT1_TEXT=$T1 --text-checkpoint mmT2_TEXT=$T2 --text-checkpoint mmT4_TEXT=$T4 \
    --image-checkpoint denseI4_IMAGE=$DI4 --image-checkpoint mmT1_IMAGE=$T1 --image-checkpoint mmT2_IMAGE=$T2 --image-checkpoint mmT4_IMAGE=$T4 \
    --pair "denseT4_TEXT|denseI4_IMAGE" --pair "mmT1_TEXT|mmT1_IMAGE" \
    --pair "mmT2_TEXT|mmT2_IMAGE" --pair "mmT4_TEXT|mmT4_IMAGE"
}
chain_b(){
  run probes_image $GPU_B python3 evaluate_probe_suite.py --modality image --num-scenes 6000 \
    --noise-levels 0.0 0.25 0.5 0.75 --output-dir $Q/probe_suite/image --checkpoint denseI4=$DI4 $MMC
  run probes_text $GPU_B python3 evaluate_probe_suite.py --modality text --manifest $TXTVAL --num-scenes 6000 \
    --noise-levels 0.0 0.25 0.5 0.75 --output-dir $Q/probe_suite/text --checkpoint denseT4=$DT4 $MMC
}
chain_c(){
  run sens_image $GPU_C python3 evaluate_semantic_sensitivity.py --output-dir $Q/sens_image --checkpoint denseI4=$DI4 $MMC
  run sens_text $GPU_C python3 evaluate_text_sensitivity.py --num-worlds 2000 --output-dir $Q/sens_text --checkpoint denseT4=$DT4 $MMC
  run triples $GPU_C python3 evaluate_image_triples.py --triples outputs/image_eval_triples/triples_tokens.pt \
    --output-dir $Q/triples --checkpoint denseI4=$DI4 $MMC
  run scene_image $GPU_C python3 evaluate_scene_retrieval.py --modality image --num-scenes 2500 \
    --output-dir $Q/scene_retrieval/image --checkpoint denseI4=$DI4 $MMC
  run scene_text $GPU_C python3 evaluate_scene_retrieval.py --modality text --manifest $TXTVAL --num-scenes 2500 \
    --output-dir $Q/scene_retrieval/text --checkpoint denseT4=$DT4 $MMC
  run binding $GPU_C python3 evaluate_image_binding.py --per-layer --seed 20260924 --image-cache $CACHE \
    --output-dir $Q/binding --checkpoint denseI4=$DI4 $MMC
}
chain_a & chain_b & chain_c & wait
log "ALL TRUNK EVALS COMPLETE"
