#!/bin/bash
set -u; cd /home/zd25e122/clevr_discrete_diffusion
export PYTHONUNBUFFERED=1
R=/home/zd25e122/clevr-dataset-gen_clone/output/platonic_clevr_v1_5M_train_gpu_visible
EXTRA=$R/train_image_only_extra.jsonl
OUT=outputs/image_only_extra_token_cache
MERGED=outputs/image_only_2_5m_token_cache
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# 1. wait for the manifest builder to finish
log "manifest rows: $(wc -l < $EXTRA)"

# 2. tokenize the extra images into their own cache
if [ ! -f "$OUT/train_tokens.pt" ]; then
  log "tokenizing extra images -> $OUT"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 pretokenize_clevr.py \
    --tokenizer-checkpoint outputs/vqvae_training_bs128/best.pt \
    --train-dir "$R" --train-manifest "$EXTRA" \
    --val-dir "$R"   --val-manifest "$R/val_image_only.jsonl" \
    --cache-dir "$OUT" --batch-size 512 --num-workers 16 || { log "TOKENIZE FAILED"; exit 1; }
  log "tokenize done"
else
  log "extra cache already present, skipping"
fi

# 3. merge the 1.2M and the extra into a single 2.53M cache
log "merging caches -> $MERGED"
python3 - <<'PY'
import torch, json
from pathlib import Path
A=torch.load("outputs/image_only_1_2m_token_cache/train_tokens.pt",map_location="cpu",weights_only=False)
B=torch.load("outputs/image_only_extra_token_cache/train_tokens.pt",map_location="cpu",weights_only=False)
ta,tb=A["tokens"],B["tokens"]
assert ta.shape[1:]==tb.shape[1:], (ta.shape,tb.shape)
out=Path("outputs/image_only_2_5m_token_cache"); out.mkdir(parents=True,exist_ok=True)
md=dict(A["metadata"]); md["num_images"]=int(ta.shape[0]+tb.shape[0])
md["merged_from"]=["image_only_1_2m_token_cache","image_only_extra_token_cache"]
md.pop("image_manifest_sha256",None)
torch.save({"tokens":torch.cat([ta,tb]),"metadata":md}, out/"train_tokens.pt")
import shutil; shutil.copy("outputs/image_only_1_2m_token_cache/val_tokens.pt", out/"val_tokens.pt")
print("merged train tokens:",tuple(torch.cat([ta,tb]).shape))
PY
# 4. merged manifest in the SAME order as the merged tokens
cat $R/train_image_only.jsonl $EXTRA > $R/train_image_only_2_5m.jsonl
log "merged manifest rows: $(wc -l < $R/train_image_only_2_5m.jsonl)"
log "TOKENIZE+MERGE COMPLETE"
