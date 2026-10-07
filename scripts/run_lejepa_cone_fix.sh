#!/bin/bash
# LeJEPA (SIGReg, no EMA teacher, no stop-gradient) mirrors of the two I-JEPA
# reference runs, then the cone / CKA eval matched to outputs/cka_matched.
#
#   ijepa16_IMAGE = image_ijepa_sweep_blk_s15_t45          -> image_lejepa_blk_s15_t45_from_dense_ep12
#   ijepa26_TEXT  = text_ijepa_w4_8_from_dense_ep20_2m_6e  -> text_lejepa_w4_8_from_dense_ep20_2m_6e
#
# Holds the image sweep's GPU locks while training so its queued arms do not
# stack onto the same GPU. Refuses to overwrite an existing output directory.
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
Q=outputs/lejepa_cone_fix; L=outputs/image_jepa_sweep/locks; mkdir -p $Q $L
IMG=image_lejepa_blk_s15_t45_from_dense_ep12
TXT=text_lejepa_w4_8_from_dense_ep20_2m_6e
GPU_IMG=${GPU_IMG:-2}; GPU_TXT=${GPU_TXT:-3}
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
lock(){ until mkdir $L/$1 2>/dev/null; do sleep 120; done; }

train(){  # config gpu
  local c=$1 g=$2 rc
  if [ -e outputs/$c ]; then log "REFUSE $c: outputs/$c exists"; return 1; fi
  lock $g; log "START $c on GPU $g"
  CUDA_VISIBLE_DEVICES=$g python3 train_multimodal.py --config configs/$c.yaml > $Q/$c.train.log 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ "$c" = "$IMG" ]; then
    # same binding eval as the sweep, so the run lands in its ranking table
    CUDA_VISIBLE_DEVICES=$g python3 evaluate_image_binding.py \
      --per-layer --checkpoint "${c}=outputs/$c/epoch_003.pt" --seed 20260924 \
      --image-cache outputs/image_only_2_5m_token_cache/val_tokens.pt \
      --output-dir outputs/image_binding_sweep >> $Q/$c.eval.log 2>&1 \
      && log "EVALED $c" || log "EVAL FAILED $c"
  fi
  rmdir $L/$g 2>/dev/null
  [ $rc -eq 0 ] && log "TRAINED $c" || log "FAILED $c (rc=$rc)"
  return $rc
}

train $IMG $GPU_IMG & train $TXT $GPU_TXT &
wait
[ -f outputs/$IMG/epoch_003.pt ] && [ -f outputs/$TXT/epoch_005.pt ] \
  || { log "a run did not finish, skipping the cone eval"; exit 1; }

lock $GPU_TXT; log "cone / CKA eval on GPU $GPU_TXT"
CUDA_VISIBLE_DEVICES=$GPU_TXT python3 evaluate_cross_modal_structure.py \
  --text-checkpoint  lejepa26_TEXT=outputs/$TXT/epoch_005.pt \
  --image-checkpoint lejepa16_IMAGE=outputs/$IMG/epoch_003.pt \
  --image-cache outputs/image_only_2_5m_token_cache/val_tokens.pt \
  --num-scenes 4000 --seed 20260922 --output-dir outputs/cka_matched_lejepa \
  > $Q/cone_eval.log 2>&1 && log "CONE EVAL DONE" || log "CONE EVAL FAILED"
rmdir $L/$GPU_TXT 2>/dev/null

python3 - > $Q/RESULTS.md 2>&1 <<'EOF'
import json
rows = [("cka_matched", "dense16_IMAGE"), ("cka_matched", "ijepa16_IMAGE"),
        ("cka_matched_lejepa", "lejepa16_IMAGE"), ("cka_matched", "dense26_TEXT"),
        ("cka_matched", "ijepa26_TEXT"), ("cka_matched_lejepa", "lejepa26_TEXT")]
print("mean pairwise cosine / effective rank / probe %, pooled clean features\n")
for d, label in rows:
    m = json.load(open(f"outputs/{d}/{label}.json"))
    m = m.get("metrics", m)
    print(f"{label:15s} " + "  ".join(
        f"{k}: {m[k]['mean_pairwise_cosine']:.2f}/{m[k]['effective_rank']:.0f}/{100 * m[k]['probe_accuracy']:.0f}"
        for k in ("L1", "L3", "L5", "L7")))
for d in ("cka_matched", "cka_matched_lejepa"):
    print(f"\n{d}/cross_modal.json:\n" + open(f"outputs/{d}/cross_modal.json").read()[:1500])
EOF
log "ALL DONE - see $Q/RESULTS.md"
