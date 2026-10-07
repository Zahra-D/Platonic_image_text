#!/bin/bash
# Sequential experiment queue: train each config, score its cross-modal CKA,
# append one summary line. Resumable -- a run whose final checkpoint already
# exists is skipped, so re-launching after a disconnect continues where it
# stopped rather than starting over.
set -u
cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
GPU="${1:-0}"
QUEUE="${2:-scripts/cka_queue.txt}"
RESULTS=outputs/cka_queue/results.tsv
mkdir -p outputs/cka_queue
[ -f "$RESULTS" ] || printf "run\tepochs\tval\tCKA_L7\tCKA_best\tat\tgap\tprobe_text\tprobe_image\n" > "$RESULTS"

log(){ echo "[$(date '+%F %T')] $*"; }

while read -r name; do
  case "$name" in ''|\#*) continue;; esac
  out="outputs/$name"
  want=$(python3 - "$name" <<'PY'
import sys, yaml
raw=open(f"configs/{sys.argv[1]}.yaml").read()
print(yaml.safe_load('tokenizer:'+raw.split('tokenizer:',1)[1])['train']['epochs'])
PY
)
  have=$(ls "$out"/epoch_*.pt 2>/dev/null | grep -vc adapter || true)
  if [ "${have:-0}" -ge "$want" ]; then
    log "SKIP $name (already has $have/$want epochs)"
  else
    log "TRAIN $name  ($have/$want epochs present)"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
      python3 train_multimodal.py --config "configs/$name.yaml" \
      > "outputs/cka_queue/$name.train.log" 2>&1
    if [ $? -ne 0 ]; then
      log "FAILED $name -- see outputs/cka_queue/$name.train.log"
      printf "%s\tTRAIN FAILED\n" "$name" >> "$RESULTS"
      continue
    fi
  fi

  last=$(ls "$out"/epoch_*.pt 2>/dev/null | grep -v adapter | sort | tail -1)
  [ -z "$last" ] && { log "no checkpoint for $name"; continue; }

  log "EVAL $name  ($last)"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
    python3 evaluate_paired_modality_alignment.py --num-scenes 2000 \
      --manifest /home/zd25e122/clevr-dataset-gen_clone/output/platonic_clevr_v1_5M_train_gpu_visible/val_pairs_human.jsonl \
      --image-cache outputs/image_only_1_2m_token_cache/val_tokens.pt \
      --output-dir outputs/cka_queue/alignment \
      --model "$name=$last" > "outputs/cka_queue/$name.eval.log" 2>&1

  python3 - "$name" "$out" >> "$RESULTS" <<'PY'
import json, os, re, sys
name, out = sys.argv[1], sys.argv[2]
row = {"CKA_L7":"-","CKA_best":"-","at":"-","gap":"-"}
path = "outputs/cka_queue/alignment/alignment.json"
if os.path.exists(path):
    blob = json.load(open(path)).get(name, {}).get("metrics", {})
    if blob:
        order = ["h0"] + [f"L{i}" for i in range(8)]
        have = [k for k in order if k in blob]
        best = max(have, key=lambda k: blob[k]["cka"])
        row = {"CKA_L7": f"{blob['L7']['cka']:.3f}" if "L7" in blob else "-",
               "CKA_best": f"{blob[best]['cka']:.3f}", "at": best,
               "gap": f"{blob['L7'].get('modality_gap', float('nan')):.3f}" if "L7" in blob else "-"}
log = os.path.join(out, "train.log")
val = "-"
if os.path.exists(log):
    found = re.findall(r"validation epoch=\d+ .*?loss=([0-9.]+)", open(log, errors="ignore").read())
    val = found[-1] if found else "-"
epochs = len([p for p in os.listdir(out) if re.match(r"epoch_\d+\.pt$", p)])
print(f"{name}\t{epochs}\t{val}\t{row['CKA_L7']}\t{row['CKA_best']}\t{row['at']}\t{row['gap']}\t-\t-")
PY
  log "DONE $name -> $(tail -1 "$RESULTS")"
done < "$QUEUE"
log "QUEUE COMPLETE"
