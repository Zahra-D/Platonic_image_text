# Cross-modal retrieval between separately trained dense models

How the claim "independently trained text and image models converge (linearly)" was
measured: the two models, their training, the evaluation data, the linear map, the
metrics, every number, and the caveats.

**Data-overlap caveat.** The 8000 evaluation images are in the training set of the
2.53M-image corpus (§3.3). The image model was trained on these exact images, without
their captions. Treat the numbers below as preliminary until they are re-run on held-out
images.

## 1. Question

Take one text model trained only on captions and one image model trained only on images,
never trained together. Is there a linear map between their representations that
retrieves an image's own caption among 1000 candidates (and back)? If yes, and if it gets
easier as both models train longer, the two models are converging on shared scene
structure.

The models are trained **without pairing**. The *evaluation* uses pairs: 6000 matched
caption–image pairs fit the linear map, and different scenes are used for scoring.

## 2. The two models

Both are the same architecture: a masked discrete diffusion transformer, 8 blocks,
d_model 384, 6 heads, MLP ratio 4, dropout 0.1, learned absolute positions, no modality
embedding, `train_mode: dense`.

| | Text model | Image model |
|---|---|---|
| Runs (one continuous training, resumed) | `text_dense_diffusion_2m_4e_matched` (ep 1–4) → `_8e_continued` (5–8) → `_12e_continued` (9–12) → `_20e_continued` (13–20) → `_40e_continued` (21–40) | `image_dense_diffusion_2_5m_40e` (ep 1–40, one run) |
| Training data | `platonic_text_only_v1_2m/train_text_only_human.jsonl`, **2,000,000 captions** (field `caption_human`), text-only scenes | `train_image_only_2_5m.jsonl`, **2,529,500 images**, as VQ tokens (16×24 grid, 512 codes, VQ-VAE `outputs/vqvae_training_bs128/best.pt`) |
| Tokens per sample | `[TEXT] BOS` + word tokens (vocabulary 170, max 192) + `EOS` | `[IMAGE] BOS` + 384 VQ codes + `EOS` |
| Objective | masked diffusion: t ~ U(ε, 1), ε = 1e-3, each content token masked with probability t, cross-entropy on masked tokens weighted by 1/t | same |
| Batch | 64 × 4 gradient accumulation = **256** | same |
| Optimiser | AdamW, lr 3e-4 constant, weight decay 0, grad clip 50, bf16 | same |
| Steps per epoch | 7,812 | 9,881 |
| Seed | 20260915 | 20260921 |

"Epoch e" in the results means each model's checkpoint after e full epochs:

| e | Text checkpoint | Image checkpoint |
|---|---|---|
| 1 | `text_dense_diffusion_2m_4e_matched/epoch_000.pt` | `image_dense_diffusion_2_5m_40e/epoch_000.pt` |
| 2 | `…_4e_matched/epoch_001.pt` | `…/epoch_001.pt` |
| 4 | `…_4e_matched/epoch_003.pt` | `…/epoch_003.pt` |
| 7 | `…_8e_continued/epoch_006.pt` | `…/epoch_006.pt` |
| 10 | `…_12e_continued/epoch_009.pt` | `…/epoch_009.pt` |
| 20 | `…_20e_continued/epoch_019.pt` | `…/epoch_019.pt` |
| 40 | `…_40e_continued/epoch_039.pt` | `…/epoch_039.pt` |

## 3. Evaluation data

### 3.1 Scenes and images
- The first **8000 rows** of `val_image_only.jsonl`, with image indices 1,210,000–1,217,999.
- Their VQ tokens come from `outputs/image_only_2_5m_token_cache/val_tokens.pt`, matched by row.

### 3.2 Captions
- For each scene, one caption is **generated fresh** from its scene graph (`world`) with the dataset's
  caption generator (`generate_human_captions.py`, through `binding_swap_captions`).
- The generator is seeded per scene: `Random(20260922 · 1000003 + position)`.
- The captions follow the text corpus's "human" style.
- Example: "You can see five objects here: a cube that is sizable, red, and rubber; a tube that is
  smallish, grey, and metal; …"
- **No spatial relations:** the code reads `world["relations"]`, but the manifests store
  `world["relationships"]`, so captions contain only object descriptions (EVALUATIONS.md §4.1).
  The text corpus captions do contain relations.

### 3.3 Overlap with training data (checked 2026-10-08)
- **Image side:** all 8000 eval images are in `train_image_only_2_5m.jsonl`, with the same
  `image_path` and the same scene. That manifest = the original 1.2M (indices 0–1,199,999) plus an
  "extra" 1.33M block built on Oct 2, which includes the validation block (1,210,000–1,229,999).
  The image model **trained on these images**, unsupervised and without captions.
  - This affects every model trained on the 2.53M corpus: the dense image run, the I-JEPA sweep, the
    image LeJEPA runs, and all 2M unpaired runs including the dropout run and trunk JEPA.
  - Models on the 1.2M corpus (`train_image_only.jsonl`, `train_pairs_human.jsonl`) are not affected.
- **Text side:** the text model never saw these captions. Scenes are sets of object attributes, so by
  chance 1022 of the 7968 distinct eval scenes (13%, nearly all with 3–4 objects) also occur in the
  text corpus, with different captions.

## 4. Features

Each model encodes its own modality on **clean** input (no masking), in eval mode. For each
caption and each image:
1. **Hooks** record 17 readouts: the input embedding (input to block 0), each block's residual
   output L0–L7 (L7 taken before the final LayerNorm), and each block's MLP write L0–L7.mlp_out.
2. **Pooling:** each readout is **mean-pooled** over content tokens only, excluding `[TEXT]`/`[IMAGE]`,
   BOS, EOS and padding.
3. **Normalisation:** L2-normalised and cached as float32 (`outputs/eval_all/xret/features/*.npz`,
   8000 × 384 per readout).

## 5. Linear map ("with linear probe")

**Steps** (`evaluate_cross_modal_retrieval.py`):
1. **Split** the 8000 scenes with a seeded permutation (seed 20260922) into **6000 fit, 1000 val and
   1000 test**. Test scenes are never used to fit or select anything.
2. **Cells:** every pair (text readout, image readout) is a cell, 17 × 17 = 289 cells.
3. **Centre** each modality with the mean of its fit rows; the same mean is applied to val and test.
4. **Fit a ridge map from image to text features:** W = (XᵀX + α·s̄·I)⁻¹XᵀY, where X are image and
   Y text features of the fit pairs, and s̄ is the mean eigenvalue of XᵀX. The grid is
   α ∈ {1e-3, 1e-2, 0.1, 1, 10}, all solved from one eigendecomposition.
5. **Score:** s_ij = cos(x_i·W, y_j).
   - **Image→text** ranks the 1000 test captions for each image (rows of s).
   - **Text→image** ranks the 1000 test images for each caption (columns of the same matrix).
6. **Select:** for each cell, pick α by the mean of the two directions' MRR on val, then the best cell
   on val.
7. **Refit** the chosen cell on fit + val (7000 pairs) and report on test.

**Metrics:**
- R@k = fraction of queries whose true partner ranks in the top k.
- MRR = mean of 1/rank.
- Median rank.
- 95% bootstrap CI (1000 resamples of test queries).
- Chance with 1000 candidates: R@1 0.001, MRR 0.0075.

**Controls:**
- **Procrustes:** an orthogonal map (rotation only, W = UVᵀ from the SVD of XᵀY), selected separately.
  Asks whether a rotation suffices.
- **Shuffled:** the fit pairs randomly permuted, at the chosen cell. Must be at chance.
- **Random encoders:** the same architectures with untrained weights, through the same protocol.
- **No map ("without linear probe"):** text and image vectors compared directly by cosine, raw and with
  each modality's mean removed. For separately trained models this must be at chance, since their
  coordinates are unrelated.
- **Sample efficiency:** the same procedure with only 100 to 6000 fit pairs.

## 6. Results (`outputs/eval_all/xret/results/denseT_eXXX_TEXT__denseI_eXXX_IMAGE.json`)

| Epochs | Chosen cell (text \| image) | α | Text→image R@1 [95% CI] | R@5 | R@10 | MRR | Median rank | Image→text R@1 / MRR | Procrustes MRR (i→t / t→i) | Shuffled MRR | CKA (test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | embedding \| L7 | 0.1 | 0.109 [0.091, 0.129] | 0.294 | 0.427 | 0.207 | 15 | 0.070 / 0.144 | 0.066 / 0.062 | 0.006 | 0.228 |
| 2 | L6 \| L7 | 0.01 | 0.175 [0.152, 0.199] | 0.401 | 0.528 | 0.287 | 9 | 0.129 / 0.219 | 0.073 / 0.092 | 0.008 | 0.260 |
| 4 | L6 \| L7 | 0.01 | 0.436 [0.408, 0.467] | 0.726 | 0.799 | 0.566 | 2 | 0.293 / 0.416 | 0.126 / 0.131 | 0.008 | 0.280 |
| 7 | L6 \| L7 | 0.01 | 0.634 [0.605, 0.663] | 0.858 | 0.910 | 0.734 | 1 | 0.468 / 0.590 | 0.143 / 0.197 | 0.006 | 0.359 |
| 10 | L5 \| L7 | 0.01 | 0.621 [0.591, 0.652] | 0.844 | 0.909 | 0.723 | 1 | 0.470 / 0.592 | 0.167 / 0.167 | 0.009 | 0.319 |
| 20 | L5 \| L7 | 0.001 | 0.692 [0.665, 0.719] | 0.884 | 0.928 | 0.776 | 1 | 0.550 / 0.662 | 0.181 / 0.174 | 0.009 | 0.310 |
| 40 | L5 \| L7 | 0.001 | **0.695** [0.668, 0.722] | 0.897 | 0.929 | **0.783** | 1 | **0.548 / 0.666** | 0.215 / 0.198 | 0.009 | 0.279 |
| random init | embedding \| L0 | — | 0.024 | — | — | 0.059 | — | 0.033 / 0.067 | 0.026 | 0.007 | — |

- **No map, 40 epochs:** text→image MRR 0.011 at best (L6, centred). That is chance, as expected.
- **Sample efficiency, 40 epochs** (text→image R@1 / MRR by number of fit pairs):

  | Fit pairs | 100 | 300 | 1000 | 3000 | 6000 |
  |---|---|---|---|---|---|
  | R@1 / MRR | 0.146 / 0.253 | 0.341 / 0.474 | 0.499 / 0.617 | 0.607 / 0.716 | 0.681 / 0.773 |

## 7. What this does and does not show

**What it supports:**
- **Alignment grows with training.** A linear map from image features to caption features gets far
  better as both models train: text→image R@1 0.11 → 0.18 → 0.44 → 0.63 → 0.69 → 0.70. The shuffled
  control stays at chance and random encoders stay at 0.02–0.03, so it's not a property of the
  architecture or of a 384×384 map by itself.
- **It needs an affine map, not a rotation.** Procrustes stays at MRR ≈ 0.2: the spaces are linearly
  related, not isometric. With no map at all, the models share no coordinates (chance).
- **Few pairs suffice.** With 100 pairs the map already reaches R@1 0.15; with 1000, 0.50.
- **The deep layers carry it.** The chosen cells sit there (text L5–L6, image L7), not at the input
  embedding (except at epoch 1).

**Limits:**
1. **Overlap:** the eval images are training images of the image model (§3.3). The text model never
   saw the captions. Re-run on held-out images.
2. **No relations:** captions and scenes contain no spatial relations, so matching can rely on object
   inventories (which colours, shapes, sizes and materials are present).
3. **One seed** per model, and **one 1000-scene test split**. The CIs cover query sampling only.
4. **Best cell selected on val:** test numbers are honest, but the chosen layers change over epochs.
5. **The map uses paired data.** "Unpaired" refers to training only; this measures linear
   *alignability* given 6000 pairs, not unsupervised alignment.

## 8. Reproduce

```bash
python3 evaluate_cross_modal_retrieval.py --num-scenes 8000 --test 1000 --val 1000 \
  --output-dir outputs/eval_all/xret \
  --text-checkpoint  denseT_e040_TEXT=outputs/text_dense_diffusion_2m_40e_continued/epoch_039.pt \
  --image-checkpoint denseI_e040_IMAGE=outputs/image_dense_diffusion_2_5m_40e/epoch_039.pt \
  --pair "denseT_e040_TEXT|denseI_e040_IMAGE"
```
