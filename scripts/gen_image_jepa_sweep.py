"""Generate the image I-JEPA masking sweep from a dense branch checkpoint."""
import re, sys
from pathlib import Path

BRANCH = sys.argv[1]                       # dense image checkpoint to init from
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
BASE = Path("configs/image_data2vec_from_dense_layerwise_all_ijepa_2m_2e.yaml").read_text()

# name: (mask_block_2d, train_fixed_t, scale, aspect)
# Varies the three things that define an I-JEPA mask: how much is hidden,
# how big each hidden region is, and how elongated it is.
SWEEP = {
    "blk_s15_t45":   (True,  0.45, [0.15, 0.20], [0.75, 1.50]),   # current ijepa, reference
    "blk_s10_t30":   (True,  0.30, [0.10, 0.25], [0.75, 1.50]),   # current block2d, reference
    "blk_tiny_t45":  (True,  0.45, [0.02, 0.05], [0.75, 1.50]),   # many tiny blocks
    "blk_s05_t60":   (True,  0.60, [0.05, 0.10], [0.75, 1.50]),   # small blocks, most hidden
    "blk_s15_t60":   (True,  0.60, [0.15, 0.20], [0.75, 1.50]),   # ijepa blocks, most hidden
    "blk_s30_t30":   (True,  0.30, [0.25, 0.40], [0.75, 1.50]),   # few large blocks
    "blk_s30_t60":   (True,  0.60, [0.25, 0.40], [0.75, 1.50]),   # large blocks, most hidden
    "blk_wide_t45":  (True,  0.45, [0.15, 0.20], [0.30, 3.00]),   # elongated strips
    "rand_t45":      (False, 0.45, None, None),                   # scattered tokens
    "rand_t75":      (False, 0.75, None, None),                   # scattered, most hidden
}

def emit(name, block2d, t, scale, aspect):
    run = f"image_ijepa_sweep_{name}"
    s = BASE
    s = s.replace("outputs/image_data2vec_from_dense_layerwise_all_ijepa_2m_2e", f"outputs/{run}")
    s = s.replace("image_data2vec_from_dense_layerwise_all_ijepa_2m_2e", run)
    s = re.sub(r"(?m)^(  epochs: ).*$", rf"\g<1>{EPOCHS}", s)
    s = re.sub(r"(?m)^  init_checkpoint: .*$", f"  init_checkpoint: {BRANCH}", s)
    s = re.sub(r"(?m)^(  train_fixed_t: ).*$", rf"\g<1>{t}", s)
    s = re.sub(r"(?m)^(  mask_block_2d: ).*$", rf"\g<1>{'true' if block2d else 'false'}", s)
    # rewrite the two-line scale / aspect lists in place
    def pair(key, vals):
        nonlocal s
        if vals is None: return
        s = re.sub(rf"(?m)^  {key}:\n  - .*\n  - .*$",
                   f"  {key}:\n  - {vals[0]}\n  - {vals[1]}", s)
    pair("mask_block_scale", scale)
    pair("mask_block_aspect", aspect)
    # train on the full 2.53M corpus
    s = s.replace("outputs/image_only_1_2m_token_cache", "outputs/image_only_2_5m_token_cache")
    s = s.replace("/train_image_only.jsonl", "/train_image_only_2_5m.jsonl")
    Path(f"configs/{run}.yaml").write_text(s)
    return run

made = [emit(n, *v) for n, v in SWEEP.items()]
print("\n".join(made))
