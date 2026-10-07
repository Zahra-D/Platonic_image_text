#!/bin/bash
# usage: run_probes_tuned.sh <image|text> <fwd|rev> <gpu>
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
MOD=$1; DIR=$2; GPU=$3
if [ "$MOD" = image ]; then
  M=("random_init=RANDOM:outputs/image_dense_diffusion_2_5m_40e/epoch_011.pt"
     "dense_ep1=outputs/image_dense_diffusion_2_5m_40e/epoch_000.pt"
     "dense_ep12=outputs/image_dense_diffusion_2_5m_40e/epoch_011.pt"
     "dense_ep16=outputs/image_dense_diffusion_2_5m_40e/epoch_015.pt"
     "dense_ep25=outputs/image_dense_diffusion_2_5m_40e/epoch_024.pt"
     "jepa_s15_t45=outputs/image_ijepa_sweep_blk_s15_t45/epoch_003.pt"
     "jepa_s30_t60=outputs/image_ijepa_sweep_blk_s30_t60/epoch_003.pt"
     "jepa_scratch=outputs/image_data2vec_scratch_block2d_30pct_layerwise_l4to7_1_2m_4e/epoch_003.pt")
  EXTRA=(--modality image)
else
  M=("random_init=RANDOM:outputs/text_dense_diffusion_2m_4e_matched/epoch_003.pt"
     "dense_ep4=outputs/text_dense_diffusion_2m_4e_matched/epoch_003.pt"
     "dense_ep20=outputs/text_dense_diffusion_2m_20e_continued/epoch_019.pt"
     "dense_ep41=outputs/text_dense_diffusion_2m_40e_continued/epoch_039.pt"
     "jepa_from_ep4=outputs/text_ijepa_w4_8_from_dense_ep4_2m_6e/epoch_005.pt"
     "jepa_from_ep20=outputs/text_ijepa_w4_8_from_dense_ep20_2m_6e/epoch_005.pt"
     "jepa_scratch=outputs/text_data2vec_scratch_layerwise_all_randt_2m_4e/epoch_003.pt")
  EXTRA=(--modality text --manifest /home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_1m/val_text_only_human.jsonl)
fi
[ "$DIR" = rev ] && M=($(printf '%s\n' "${M[@]}" | tac))
ARGS=(); for m in "${M[@]}"; do ARGS+=(--checkpoint "$m"); done
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU python3 evaluate_probe_suite.py "${EXTRA[@]}" \
  --num-scenes 6000 --noise-levels 0.0 0.25 0.5 0.75 --output-dir outputs/probe_suite/$MOD "${ARGS[@]}"
echo "[$(date '+%H:%M')] $MOD $DIR DONE"
