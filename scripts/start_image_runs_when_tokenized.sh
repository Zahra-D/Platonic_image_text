#!/usr/bin/env bash
set -euo pipefail
# Wait for the image token cache, verify it against the manifests, then launch
# the four image-only runs.  Verification failures stop before any training.
repo=/home/zd25e122/clevr_discrete_diffusion
py=/home/zd25e122/miniconda3/envs/unix/bin/python3.10
cache=$repo/outputs/image_only_1_2m_token_cache
data=/home/zd25e122/clevr-dataset-gen_clone/output/platonic_clevr_v1_5M_train_gpu_visible
cd "$repo"
until [ -s "$cache/train_tokens.pt" ]; do
  pgrep -f pretokenize_image_only_sharded >/dev/null || { echo "tokenizer exited without writing train_tokens.pt" >&2; exit 1; }
  sleep 120
done
$py - <<'PY'
import torch, sys
sys.path.insert(0, "/home/zd25e122/clevr_discrete_diffusion")
from data.multimodal_dataset import image_manifest_fingerprint, read_jsonl
cache = "/home/zd25e122/clevr_discrete_diffusion/outputs/image_only_1_2m_token_cache"
data = "/home/zd25e122/clevr-dataset-gen_clone/output/platonic_clevr_v1_5M_train_gpu_visible"
for split, manifest in (("train", f"{data}/train_image_only.jsonl"), ("val", f"{data}/val_image_only.jsonl")):
    payload = torch.load(f"{cache}/{split}_tokens.pt", map_location="cpu", weights_only=False)
    tokens, meta = payload["tokens"], payload["metadata"]
    rows = read_jsonl(manifest)
    assert len(tokens) == len(rows), f"{split}: {len(tokens)} tokens vs {len(rows)} manifest rows"
    assert tuple(tokens.shape[1:]) == (16, 24), f"{split}: grid {tuple(tokens.shape[1:])}"
    assert meta["image_manifest_sha256"] == image_manifest_fingerprint(rows), f"{split}: manifest fingerprint mismatch"
    top = int(tokens.reshape(-1)[:1_000_000].to(torch.int32).max())
    assert top < 512, f"{split}: code {top} outside the 512-code book"
    print(f"{split}: {tuple(tokens.shape)} verified against {manifest}")
PY
# GPU 3 is reserved for the all-layer text experiment; the fourth image
# variant (HSIC at 5%) is left for later.
exec env RUNS="0 1 2" "$repo/scripts/launch_image_only_1_2m_vnode10.sh" 0 1 2
