#!/usr/bin/env bash
# Priority pass: dense vs JEPA/data2vec at matched budget, then everything else.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/linear_probes_image; GPU=${1:-2}
spec() { for c in "$@"; do r=$(basename "$(dirname "$c")"); echo "--checkpoint ${r#image_}__$(basename "$c" .pt)=$c"; done; }
A=$(spec outputs/image_dense_diffusion_1_2m_4e/epoch_00[0-3].pt outputs/image_dense_diffusion_1_2m_6e_continued/epoch_00[45].pt \
         outputs/image_data2vec_from_dense_*/epoch_00[01].pt)
B=$(spec outputs/image_data2vec_scratch_*/epoch_00[0-3].pt)
CUDA_VISIBLE_DEVICES=$GPU python -u evaluate_linear_probes.py --output-dir $OUT $A > $OUT/priorityA.log 2>&1 &
CUDA_VISIBLE_DEVICES=$GPU python -u evaluate_linear_probes.py --output-dir $OUT $B > $OUT/priorityB.log 2>&1 &
wait
scripts/run_linear_probes_image.sh $GPU 0 2 > $OUT/rest0.log 2>&1 &
scripts/run_linear_probes_image.sh $GPU 1 2 > $OUT/rest1.log 2>&1 &
wait
echo ALL_DONE >> $OUT/priorityA.log
