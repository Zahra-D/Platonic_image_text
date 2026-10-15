#!/bin/bash
# Per-checkpoint evaluation of the shared-trunk cross-view JEPA runs (trunk only, private LoRA off).
# Runs on GPU 1; scores every R2 / R3 checkpoint as it lands, plus the reference points once.
cd /home/zd25e122/clevr_discrete_diffusion
export CLEVR_DATA=/home/zd25e122/clevr-dataset-gen_clone/output CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-1}
export PYTHONUNBUFFERED=1 PYTHONPATH=.:scripts PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
O=outputs/eval_shared_xview; mkdir -p $O/logs $O/done
evaluate() {   # label checkpoint
  local lab=$1 ck=$2; [ -f $O/done/$lab ] && return
  echo "[$(date +%H:%M)] eval $lab <- $ck"
  python3 - "$lab" "$ck" > $O/logs/${lab}_encode.log 2>&1 <<'PY'
import argparse, sys, evaluate_cross_modal_structure as X
from evaluate_cross_modal_retrieval import encode_all
from pathlib import Path
from clevr_paths import CLEVR_DATA_ROOT, CLEVR_GEN_ROOT
lab, ck = sys.argv[1], sys.argv[2]
a = argparse.Namespace(text_checkpoint=[f"{lab}_TEXT=TRUNK:{ck}"], image_checkpoint=[f"{lab}_IMAGE=TRUNK:{ck}"], device="cuda", batch_size=64,
    num_scenes=8000, seed=20260922, caption_generator=CLEVR_GEN_ROOT + "/generate_human_captions.py",
    image_manifest=CLEVR_DATA_ROOT + "/platonic_clevr_v1_5M_train_gpu_visible/val_image_only.jsonl",
    image_cache="outputs/image_only_2_5m_token_cache/val_tokens.pt")
_, t, c = X.load_scenes(a); encode_all(a, Path("outputs/eval_all/xret"), t, c)
PY
  python3 evaluate_cross_modal_retrieval.py --num-scenes 8000 --test 1000 --val 1000 --output-dir outputs/eval_all/xret --pair "${lab}_TEXT|${lab}_IMAGE" > $O/logs/${lab}_oracle.log 2>&1
  python3 evaluate_hard_xret_variants.py --pair "${lab}_TEXT=TRUNK:$ck|${lab}_IMAGE" --output-dir outputs/eval_hard_xret_variants > $O/logs/${lab}_hard.log 2>&1
  python3 scripts/view_invariance.py --checkpoint "$lab=TRUNK:$ck" > $O/logs/${lab}_invariance.log 2>&1
  python3 evaluate_text_sensitivity.py --num-worlds 2000 --output-dir outputs/eval_all/sens_text --checkpoint ref_text_dense_4e=outputs/text_dense_diffusion_2m_4e_matched/epoch_003.pt --checkpoint "$lab=TRUNK:$ck" > $O/logs/${lab}_sens.log 2>&1
  touch $O/done/$lab; echo "[$(date +%H:%M)] done $lab"
}
B=outputs/mm_unpaired_private_dropout_gradbal_2m_40e; A=outputs/mm_dropout_trunk_ijepa_from_ep9_4e
RUNS="mm_xview_xpara_trunk_jepa_from_ep9:xpr mm_xview_trunk_jepa_from_ep9:xvo"
while :; do
  # R2 / R3 checkpoints first (including continuation stages: <run>_c2, _c3, ...)
  for spec in $RUNS; do run=${spec%%:*}; tag=${spec#*:}
    for d in outputs/${run} outputs/${run}_c*; do [ -d $d ] || continue
      for ck in $(ls $d/epoch_[0-9][0-9][0-9].pt 2>/dev/null | sort); do
        e=$(basename $ck .pt | cut -d_ -f2); [ $(( $(date +%s) - $(stat -c %Y $ck) )) -gt 120 ] && evaluate ${tag}_e$e $ck
      done
    done
  done
  # reference points: base at epoch 9 (= the starting point), JEPA A after 2M / 4M / 8M samples
  evaluate base_e09 $B/epoch_008.pt
  evaluate jepaA_e000 $A/epoch_000.pt
  evaluate jepaA_e001 $A/epoch_001.pt
  evaluate jepaA_e003 $A/epoch_003.pt
  [ -f $O/STOP ] && break
  sleep 120
done
