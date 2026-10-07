# CLEVR Discrete Diffusion — Stage 1: VQ-VAE Tokenizer

Small, from-scratch VQ-VAE tokenizer for CLEVR, first stage of a
CLEVR → discrete tokens → absorbing-state D3PM Transformer pipeline.

For the detailed multimodal research code (paired/unpaired data, Tri-LoRA,
stage 0, DANN normalization, matched/shuffled/null validation, generation,
and optional back-translation), read [MULTIMODAL_README.md](MULTIMODAL_README.md).

- Input: 64×96 RGB (aspect-preserving resize of CLEVR's native 480×320)
- Encoder: 4× spatial downsample → 16×24 latent grid
- Codebook: K=512, embedding dim=64, EMA updates, commitment β=0.25
- Loss: L1 reconstruction + commitment loss
- Output: 384 discrete tokens/image (this is what Stage 2's D3PM Transformer will model)

## Data

Reuses the CLEVR PNGs already staged at `~/Omni/data/clevr/{train,val,test}`
(70,000 / 15,000 / 15,000 images) — nothing to download.

## Config

Every default (data paths, model architecture, optimization, logging, W&B)
lives in one place: `configs/vqvae.yaml`. Both `train_vqvae.py` and
`eval_reconstructions.py` load it automatically — edit the file to change a
default for good, or pass the equivalent CLI flag (e.g. `--batch-size 64`)
to override it for a single run. Point `--config other.yaml` at a different
file to switch configs entirely.

## Usage

Sanity-check the model shapes:

```bash
python -c "
import torch
from models.vqvae import VQVAE
m = VQVAE()
x = torch.randn(2, 3, 64, 96)
out = m(x)
print('recon', out['recon'].shape, 'indices', out['indices'].shape, 'ppl', out['perplexity'].item())
"
```

Local smoke test (few hundred steps, small data subset — just checks nothing crashes):

```bash
python train_vqvae.py --max-train-samples 512 --max-steps 200 \
    --batch-size 32 --eval-every 1 --output-dir outputs/smoke
```

Full training run (intended for the RTX 3090 cluster — wrap with your scheduler):

```bash
scripts/launch_vqvae_3090.sh
```

Resume from a checkpoint:

```bash
python train_vqvae.py --resume outputs/run1/last.pt --output-dir outputs/run1
```

## Stage-1 gate before touching the diffusion model

Once training is done, run the reconstruction sanity check and **look at the
contact sheet** — object count, shape, color, material, size, and position
should all survive tokenization before Stage 2 (D3PM Transformer) starts:

```bash
python eval_reconstructions.py --checkpoint outputs/run1/best.pt
```

This prints mean L1 / PSNR, codebook perplexity, and dead-code count, and
saves `outputs/eval/contact_sheet.png` (top row = originals, bottom row =
reconstructions).

## Stage-2 token cache

Before Stage-2 training, encode each image once with the frozen tokenizer:

```bash
python pretokenize_clevr.py
```

This writes `outputs/token_cache/{train,val}_tokens.pt`. `train_d3pm.py` uses
that cache by default, avoiding repeated PNG decoding and VQ-VAE encoding on
every diffusion epoch. Pass `--no-token-cache` only to benchmark the older
on-the-fly path.

## Multimodal text + image training

`train_multimodal.py` is the joint from-scratch training path. It uses the same
frozen VQ-VAE token cache, learns a small CLEVR word vocabulary from
`train/text.jsonl`, and combines text and image-code tokens in one Transformer
vocabulary. It does not download a pretrained language model or tokenizer.

The expected split layout is:

```text
train/
  images.jsonl   # {"modality":"image", "image_path":"images/...png"}
  text.jsonl     # {"modality":"text", "text":"..."}
  images/
val/
  images.jsonl
  text.jsonl
  images/
```

In `data.mode: paired`, image row `i` and text row `i` are treated as a real
pair. Their manifest lengths must match. With `diffusion.objective: both`, each
microbatch performs image denoising with clean paired text as context and text
denoising with clean paired image tokens as context. Set the objective to
`image` when only text-to-image conditional training is wanted.

In `data.mode: unpaired`, training follows Omni's balanced-unpaired design. Two
independently seeded permutations are constructed and the image permutation is
a derangement of the text permutation, so a caption is never carried beside its
source image. Every microstep contains exactly half text and half image examples.
The carrier batch is immediately split into a text sub-batch and image sub-batch,
and the model performs two independent forwards and losses. There is no joint
attention or cross-modal loss between the unrelated examples. This learns the
two marginal distributions without leaking false correspondences; it does **not**
by itself teach text-to-image alignment.

Modality is represented by markers, separate image-code IDs, and Tri-LoRA
routing; the learned modality embedding is optional:

- sequences start with a non-maskable `<text>` or `<image>` token;
- tokens receive a learned text/image modality embedding only when
  `model.use_modality_embeddings: true`;
- image codes occupy a separate, offset vocabulary range;
- each forward carries a batch-level route ID for Tri-LoRA.

Tri-LoRA gives every target matrix a shared branch, a text-private branch, and
an image-private branch. In the current multimodal LoRA path there is no frozen
dense base matrix or bias: an unpaired text forward evaluates `shared +
text-private`; an image forward evaluates `shared + image-private`. The
inactive private branch receives no gradient. In paired caption-to-image mode,
the complete sequence routes as an image-generation task while its text and
image tokens retain their own marker tokens and modality embeddings. During
bidirectional paired pretraining, the image loss routes through image-private
LoRA and the text loss routes through text-private LoRA.

Dense from-scratch training:

```bash
scripts/launch_multimodal_dense_3090.sh
```

Dense balanced-unpaired pretraining:

```bash
scripts/launch_multimodal_unpaired_dense_3090.sh
```

LoRA from-scratch training:

```bash
scripts/launch_multimodal_lora_3090.sh
```

Omni-style balanced unpaired Tri-LoRA training:

```bash
scripts/launch_multimodal_unpaired_lora_3090.sh
```

The same run with DANN-style adversarial modality invariance:

```bash
scripts/launch_multimodal_unpaired_lora_adversarial_3090.sh
```

The optional adversary reads the final `d_model`-wide shared Tri-LoRA delta,
pools it over non-padding tokens, and predicts text versus image. A gradient
reversal layer lets the discriminator minimize modality classification loss
while sending the opposite gradient only into the shared LoRA branch. Its input
is detached before the shared delta is recomputed, preventing adversarial
gradients from leaking into earlier private branches. The masked text/image task
losses remain active to discourage trivial feature collapse. Configure it under
`alignment.modality_adversarial_*`; the launch script enables the otherwise-off
option and writes to a separate output directory.

The LoRA runs remove selected dense attention/MLP matrices and biases, then
train Tri-LoRA plus token/position embeddings (and optionally modality
embeddings), LayerNorms, and the vocabulary head. Shared, private, and embedding
parameter groups can use different learning rates. The unpaired LoRA config also uses gradient accumulation and Omni-style
EMA norm balancing: only shared-branch gradients are rescaled, while private
gradients retain their original modality loss scale. Both a complete resumable
checkpoint and an `*_adapter.pt` diagnostic adapter-only state are written. The
complete checkpoint is required for inference because it contains all learned
non-adapter parameters as well.

Dense and Tri-LoRA unpaired configs are available as
`configs/multimodal_unpaired.yaml` and
`configs/multimodal_unpaired_lora.yaml`, respectively.
To initialize a paired run from an unpaired checkpoint, keep the architecture
and tokenizer settings identical and pass `--resume`; note that resume also
restores the optimizer and epoch counter.

Generate from one or more prompts:

```bash
python generate_multimodal.py \
  --checkpoint outputs/multimodal_dense/best.pt \
  --prompts "A large red rubber cube is left of a small blue sphere." \
  --output outputs/prompted.png
```

Run the dependency-light unit tests with:

```bash
python -m unittest tests/test_multimodal.py
```
