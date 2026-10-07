#!/bin/bash
# Full eval suite on the unpaired shared-trunk + private-LoRA model
# (mm_unpaired_dense_private_lora_r128_2m_40e, epochs 1 and 2) against
# single-modality dense models after the same number of epochs on the same
# corpora (text 2M, image 2.53M) and random init.  One mm checkpoint serves as
# both the TEXT and the IMAGE model.  Chain A on GPU_A, chain B on GPU_B.
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
Q=outputs/eval_mm_private; mkdir -p $Q
GPU_A=${GPU_A:-0}; GPU_B=${GPU_B:-1}
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
MM=outputs/mm_unpaired_dense_private_lora_r128_2m_40e
MP1=$MM/epoch_000.pt; MP2=$MM/epoch_001.pt
DT1=outputs/text_dense_diffusion_2m_4e_matched/epoch_000.pt
DT2=outputs/text_dense_diffusion_2m_4e_matched/epoch_001.pt
DI1=outputs/image_dense_diffusion_2_5m_40e/epoch_000.pt
DI2=outputs/image_dense_diffusion_2_5m_40e/epoch_001.pt
RT="RANDOM:$DT1"; RI="RANDOM:$DI1"
CACHE=outputs/image_only_2_5m_token_cache/val_tokens.pt
TXTVAL=/home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_1m/val_text_only_human.jsonl
run(){ local name=$1 gpu=$2; shift 2; log "START $name (GPU $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu "$@" > $Q/$name.log 2>&1 && log "DONE $name" || log "FAILED $name"; }

chain_a(){
  run structure $GPU_A python3 evaluate_cross_modal_structure.py \
    --text-checkpoint denseT1_TEXT=$DT1 --text-checkpoint denseT2_TEXT=$DT2 \
    --text-checkpoint mmP1_TEXT=$MP1 --text-checkpoint mmP2_TEXT=$MP2 \
    --image-checkpoint denseI1_IMAGE=$DI1 --image-checkpoint denseI2_IMAGE=$DI2 \
    --image-checkpoint mmP1_IMAGE=$MP1 --image-checkpoint mmP2_IMAGE=$MP2 \
    --image-cache $CACHE --num-scenes 4000 --seed 20260922 --output-dir $Q/structure
  run xret $GPU_A python3 evaluate_cross_modal_retrieval.py --num-scenes 8000 --test 1000 --val 1000 \
    --output-dir $Q/xret \
    --text-checkpoint random_TEXT=$RT --text-checkpoint denseT2_TEXT=$DT2 \
    --text-checkpoint mmP1_TEXT=$MP1 --text-checkpoint mmP2_TEXT=$MP2 \
    --image-checkpoint random_IMAGE=$RI --image-checkpoint denseI2_IMAGE=$DI2 \
    --image-checkpoint mmP1_IMAGE=$MP1 --image-checkpoint mmP2_IMAGE=$MP2 \
    --pair "random_TEXT|random_IMAGE" --pair "denseT2_TEXT|denseI2_IMAGE" \
    --pair "mmP1_TEXT|mmP1_IMAGE" --pair "mmP2_TEXT|mmP2_IMAGE"
  run sens_image $GPU_A python3 evaluate_semantic_sensitivity.py --output-dir $Q/sens_image \
    --checkpoint random=$RI --checkpoint denseI1=$DI1 --checkpoint denseI2=$DI2 \
    --checkpoint mmP1=$MP1 --checkpoint mmP2=$MP2
  run sens_text $GPU_A python3 evaluate_text_sensitivity.py --num-worlds 2000 --output-dir $Q/sens_text \
    --checkpoint random=$RT --checkpoint denseT1=$DT1 --checkpoint denseT2=$DT2 \
    --checkpoint mmP1=$MP1 --checkpoint mmP2=$MP2
  run triples $GPU_A python3 evaluate_image_triples.py --triples outputs/image_eval_triples/triples_tokens.pt \
    --output-dir $Q/triples --checkpoint denseI1=$DI1 --checkpoint denseI2=$DI2 \
    --checkpoint mmP1=$MP1 --checkpoint mmP2=$MP2
}
chain_b(){
  run probes_image $GPU_B python3 evaluate_probe_suite.py --modality image --num-scenes 6000 \
    --noise-levels 0.0 0.25 0.5 0.75 --output-dir $Q/probe_suite/image \
    --checkpoint random=$RI --checkpoint denseI1=$DI1 --checkpoint denseI2=$DI2 \
    --checkpoint mmP1=$MP1 --checkpoint mmP2=$MP2
  run probes_text $GPU_B python3 evaluate_probe_suite.py --modality text --manifest $TXTVAL --num-scenes 6000 \
    --noise-levels 0.0 0.25 0.5 0.75 --output-dir $Q/probe_suite/text \
    --checkpoint random=$RT --checkpoint denseT1=$DT1 --checkpoint denseT2=$DT2 \
    --checkpoint mmP1=$MP1 --checkpoint mmP2=$MP2
  run scene_image $GPU_B python3 evaluate_scene_retrieval.py --modality image --num-scenes 2500 \
    --output-dir $Q/scene_retrieval/image \
    --checkpoint random=$RI --checkpoint denseI1=$DI1 --checkpoint denseI2=$DI2 \
    --checkpoint mmP1=$MP1 --checkpoint mmP2=$MP2
  run scene_text $GPU_B python3 evaluate_scene_retrieval.py --modality text --manifest $TXTVAL --num-scenes 2500 \
    --output-dir $Q/scene_retrieval/text \
    --checkpoint random=$RT --checkpoint denseT1=$DT1 --checkpoint denseT2=$DT2 \
    --checkpoint mmP1=$MP1 --checkpoint mmP2=$MP2
  run binding $GPU_B python3 evaluate_image_binding.py --per-layer --seed 20260924 --image-cache $CACHE \
    --output-dir $Q/binding --checkpoint denseI1=$DI1 --checkpoint denseI2=$DI2 \
    --checkpoint mmP1=$MP1 --checkpoint mmP2=$MP2
}
chain_a & chain_b & wait
log "ALL EVALS COMPLETE"
