#!/bin/bash
# Unpaired without private LoRA (U, 1.2M/epoch) vs with private LoRA (P = original r128 run, D = dropout+gradbal
# run; both TRUNK only) vs separately trained dense (T text / I image), at epochs 2, 4, 8.
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
export CLEVR_DATA=${CLEVR_DATA:-/home/zd25e122/clevr-dataset-gen_clone/output}
Q=outputs/eval_compare_private; mkdir -p $Q
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
declare -A U=([2]=outputs/multimodal_unpaired_dense_1_2m_4e/epoch_001.pt [4]=outputs/multimodal_unpaired_dense_1_2m_4e/epoch_003.pt [8]=outputs/mm_unpaired_dense_8e_continued/epoch_007.pt)
P=outputs/mm_unpaired_dense_private_lora_r128_2m_40e; D=outputs/mm_unpaired_private_dropout_gradbal_2m_40e
declare -A T=([2]=outputs/text_dense_diffusion_2m_4e_matched/epoch_001.pt [4]=outputs/text_dense_diffusion_2m_4e_matched/epoch_003.pt [8]=outputs/text_dense_diffusion_2m_8e_continued/epoch_007.pt)
img(){ printf 'outputs/image_dense_diffusion_2_5m_40e/epoch_%03d.pt' $(( $1 - 1 )); }
TX=(); IM=(); PAIRS=(); BIND=()
for e in 2 4 8; do f=$(printf '%03d' $((e-1)))
  for m in "U_e$e=${U[$e]}" "P_e$e=TRUNK:$P/epoch_$f.pt" "D_e$e=TRUNK:$D/epoch_$f.pt"; do
    l=${m%%=*}; p=${m#*=}; TX+=(--text-checkpoint "${l}_TEXT=$p"); IM+=(--image-checkpoint "${l}_IMAGE=$p"); PAIRS+=(--pair "${l}_TEXT|${l}_IMAGE"); BIND+=(--checkpoint "$l=$p"); done
  TX+=(--text-checkpoint "dense_e${e}_TEXT=${T[$e]}"); IM+=(--image-checkpoint "dense_e${e}_IMAGE=$(img $e)")
  PAIRS+=(--pair "dense_e${e}_TEXT|dense_e${e}_IMAGE"); BIND+=(--checkpoint "dense_e$e=$(img $e)")
done
TXS=(); for ((i=0;i<${#TX[@]};i+=2)); do TXS+=(--text-checkpoint "${TX[i+1]}"); done
IMS=(); for ((i=0;i<${#IM[@]};i+=2)); do IMS+=(--image-checkpoint "${IM[i+1]}"); done
( CUDA_VISIBLE_DEVICES=1 python3 evaluate_image_binding.py --per-layer --seed 20260924 --image-cache outputs/image_only_2_5m_token_cache/val_tokens.pt --output-dir $Q/binding "${BIND[@]}" > $Q/binding.log 2>&1 && log "DONE binding" || log "FAILED binding"
  CUDA_VISIBLE_DEVICES=1 python3 evaluate_cross_modal_structure.py "${IMS[@]}" --image-cache outputs/image_only_2_5m_token_cache/val_tokens.pt --num-scenes 4000 --seed 20260922 --output-dir $Q/structure > $Q/structure_image.log 2>&1 && log "DONE structure_image" || log "FAILED structure_image" ) &
( CUDA_VISIBLE_DEVICES=2 python3 evaluate_cross_modal_retrieval.py --num-scenes 8000 --test 1000 --val 1000 --output-dir $Q/xret "${TX[@]}" "${IM[@]}" "${PAIRS[@]}" > $Q/xret.log 2>&1 && log "DONE xret" || log "FAILED xret" ) &
( CUDA_VISIBLE_DEVICES=3 python3 evaluate_cross_modal_structure.py "${TXS[@]}" --image-cache outputs/image_only_2_5m_token_cache/val_tokens.pt --num-scenes 4000 --seed 20260922 --output-dir $Q/structure > $Q/structure_text.log 2>&1 && log "DONE structure_text" || log "FAILED structure_text" ) &
wait; log "ALL DONE"
