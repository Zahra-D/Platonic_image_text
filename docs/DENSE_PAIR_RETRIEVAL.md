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

## 9. Hard retrieval: the true caption vs binding swaps (`evaluate_cross_modal_hard_retrieval.py`)

Same models, same 8000 scenes, same split, same centring and ridge map (image → text) as above.
Only the candidate set changes.

**Candidates:** for every test image, its own caption plus **7 captions of swapped scenes**. Each swapped
scene exchanges one attribute (colour, shape, material or size) between two of the scene's objects.
- All 8 captions share one phrasing plan, each with its own random object order.
- They contain exactly the **same word multiset**; this is verified for every task.
- So an inventory of attributes cannot separate them; only which attribute belongs to which object can.
- Chance R@1 = 1/8 = 0.125. Ties count at random, so the input-embedding readout scores exactly 0.125.

**Scenes kept:** scenes with at least 3 objects, no identical twins and at least 7 distinct swaps, giving
**695 test** and 667 val tasks.

**Direction:** image → text only, since swapped *images* would need rendering.

**Cells reported:**
- **easy cell:** the cell and α chosen for the ordinary retrieval above.
- **hard-selected:** the cell and α chosen on the 667 val hard tasks, refit on fit + val.
- **shuffled:** a map fitted on shuffled pairs, at the hard-selected cell.

| Epochs | Ordinary i→t R@1 (1000 candidates) | Hard R@1, easy cell [95% CI] | Hard R@1, hard-selected cell [95% CI] | Hard MRR (hard-selected) | Shuffled |
|---|---|---|---|---|---|
| 1 | 0.070 | 0.125 (embedding: exact tie) | 0.118 [0.095, 0.142] (L3.mlp_out \| L6.mlp_out) | 0.338 | 0.082 |
| 2 | 0.129 | 0.121 [0.096, 0.145] | 0.128 [0.104, 0.153] (L5.mlp_out \| L5.mlp_out) | 0.361 | 0.127 |
| 4 | 0.293 | 0.112 [0.089, 0.135] | 0.194 [0.165, 0.223] (L6.mlp_out \| L7) | 0.412 | 0.106 |
| 7 | 0.468 | 0.160 [0.134, 0.187] | 0.140 [0.115, 0.165] (L6.mlp_out \| L4.mlp_out) | 0.369 | 0.099 |
| 10 | 0.470 | 0.122 [0.098, 0.150] | 0.168 [0.141, 0.196] (L5.mlp_out \| L5.mlp_out) | 0.404 | 0.118 |
| 20 | 0.550 | 0.158 [0.132, 0.188] | 0.197 [0.168, 0.227] (L5.mlp_out \| L7) | 0.436 | 0.078 |
| 40 | 0.548 | 0.150 [0.122, 0.176] | 0.183 [0.154, 0.210] (L5.mlp_out \| L6) | 0.415 | 0.098 |
| random init | 0.033 | 0.125 | 0.115 [0.091, 0.137] | 0.334 | 0.150 |

**Reading:**
- **The map carries almost no binding.** Ordinary retrieval climbs to 0.55. Hard retrieval stays near
  chance: 0.15 at the easy cell and at most 0.20 at the best cell, against 0.125 chance. The shuffled
  control varies between 0.08 and 0.15, which shows how noisy 695 tasks are.
- **So the ordinary R@1 comes almost entirely from the attribute inventory.** The map learns which
  colours, shapes, materials and sizes are present, not which object has which.
- **Binding is in both models, but in directions the map and cosine don't use.**
  - Unsupervised cosine ranking in text alone (caption vs reworded swaps, `evaluate_hard_retrieval.py`)
    also sits near chance: R@1 0.22–0.23 at L7 with 8 candidates.
  - A trained linear probe reads binding from the same pooled features at ~90% for text and ~80–84%
    for images (§10).
  - So binding lives in low-variance directions. Cosine scoring and a ridge map fitted to predict the
    whole caption vector are dominated by attribute inventory and word order (§10.3).
- **Same caveats as §7:** image overlap, no relations, one seed.

Results: `outputs/eval_hard_xret/<text>__<image>.json`; tasks and captions: `outputs/eval_hard_xret/tasks.json`.

## 10. Binding inside each model separately: linear-probe tests

These test binding **within one model**, with a trained probe, not across modalities. They are the
reference for §9: is binding information present in the pooled features at all?

### 10.1 Image model (`evaluate_image_binding.py`)

**Pairs:**
- From 20,000 val scenes (`val_image_only.jsonl`, tokens from `image_only_2_5m_token_cache/val_tokens.pt`),
  keep **content-matched pairs**: two real rendered scenes with identical per-attribute multisets
  (same colours, shapes, materials, sizes) but a different attribute-to-object assignment.
  - Example: red cube + blue sphere vs blue cube + red sphere.
- Only scenes after the first 12,000 are used, giving **397 pairs**.

**Facts:**
- Each scene has a set of **conjunction facts**, one per object and per pair of attributes, e.g.
  "some object is red AND a cube".
- 72 facts occur at least 50 times as positives and as negatives.
- In a matched pair some facts flip: true in one scene, false in the other.

**Steps:**
1. **Features:** image model, clean input, content tokens mean-pooled and L2-normalised, at every layer.
2. **Probe:** on the first 12,000 scenes, one class-balanced logistic regression per fact, on
   standardised features (Adam, 300 steps, lr 0.05, weight decay 1e-4).
3. **Score:** on each held-out pair, for every flipped fact:
   - **pairwise:** the probe's logit is higher in the scene where the fact is true. Chance 50%.
   - **strict:** both scenes are classified correctly, logit > 0 where true and < 0 where false.
4. **CI:** bootstrap over pairs. **Probe** = ordinary balanced accuracy of the probe on held-out scenes.

**Results** (`outputs/eval_compare_private/binding/dense_e{2,4,8}.json`,
`outputs/eval_all/binding/image_dense_diffusion_2_5m_40e__e039.json`):

| Image dense, epoch | Pairwise, L7 [95% CI] | Strict, L7 | Probe, L7 | Best layer, pairwise | Input embedding, pairwise |
|---|---|---|---|---|---|
| 2 | 79.3 [77.8, 80.8] | 52.3 | 82.3 | 81.8 (L6.mlp_out) | 64.1 |
| 4 | 80.2 [78.8, 81.5] | 54.5 | 83.1 | 83.5 (L6.mlp_out) | 64.1 |
| 8 | 80.3 [78.8, 81.7] | 55.5 | 82.8 | 84.0 (L6.mlp_out) | 64.7 |
| 40 | 80.6 [79.1, 82.1] | 54.3 | 82.7 | 83.7 (L6.mlp_out) | 64.5 |

- **The floor is ~64%, not 50%.** A red cube and a red sphere give different VQ codes, so even the bag
  of image tokens carries some binding information.
- **Flat after epoch 2.**
- **Overlap caveat (§3.3):** these scenes are also in the image model's training set.

### 10.2 Text model (`evaluate_binding_swap.py`)

**Items:**
- 512 held-out scenes from `val_text_only_human.jsonl`.
- Each query caption has a **swapped caption with exactly the same words**: one attribute exchanged
  between two objects, or a relation's subject and anchor swapped.
- Item counts: colour 493, shape 496, material 383, size 401, relation 413.

**Steps:**
1. **Probe:** the same kind, trained on 20,000 training captions
   (`platonic_text_only_v1_2m/train_text_only_human.jsonl`).
2. **Score:** for each flipped fact, whether the probe ranks the true caption above its swapped twin.
   Pairwise only; the text eval has no strict variant yet.
3. **Floor:** a bag of words or the input embedding gives identical inputs and scores exactly 50%.

**Results** (`outputs/binding_swap_eval_epoch000/dense_epoch000.json`,
`outputs/binding_swap_eval/dense_epoch003.json`). Only epochs 1 and 4 were run:

| Text dense, epoch | Readout | Probe | Colour | Shape | Material | Size | Relation |
|---|---|---|---|---|---|---|---|
| 1 | L7 | 84.3 | 76.6 | 86.1 | 78.3 | 74.7 | 96.6 |
| 1 | L5.mlp_out | 82.9 | 79.2 | 82.1 | 80.1 | 74.9 | 95.9 |
| 4 | L7 | 89.7 | 90.8 | 87.7 | 93.1 | 93.5 | 48.2 (unexplained; 99% at L5–L6) |
| 4 | L5.mlp_out (best) | 90.5 | 95.8 | 94.5 | 97.9 | 96.1 | 99.5 |
| any | embedding | 84.5 | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 |

Text binding grows with training (epoch 1 → 4) and is higher than image binding at epoch 4: 88–98%
vs 80–84%. The text floor is cleaner (50%), so the gap above the floor is much larger for text.

### 10.3 Why the probe sees binding and cosine does not

The same text eval also asks, by cosine (L7, epoch 4), whether a query caption is closer to:

| Comparison | True caption preferred |
|---|---|
| the same scene **reworded** vs the swap in the **same words** | 0% |
| the same scene **reordered** vs the swap in the same order | 20% |
| a **different scene** in the same words | 99.8% |

**What this means:**
- Raw cosine is dominated by word choice and word order, not binding.
- The binding information that the probe finds is a small, low-variance part of the vector.
- §9 combines both problems:
  - its 8 candidates differ in object order;
  - the ridge map is fitted to predict the **whole** caption vector, which is mostly inventory and
    wording.
  
  So the binding directions barely enter the image→text map.

**Fair comparisons:**
- probe vs probe (§10.1 vs §10.2);
- a cross-modal test that scores only binding directions, e.g. text and image fact probes, or a map
  fitted to predict fact logits.

### 10.4 Missing

- Text binding probe at epochs 7, 10, 20, 40, to match the image table.
- A strict metric on the text side.

## 11. Hard retrieval with better scorers (`evaluate_hard_xret_variants.py`)

§10 suggested hard retrieval fails because ridge and cosine follow the high-variance directions
(inventory, wording), while binding sits in weak directions. This section tests that directly.

**Setup:**
- Same scenes, split and 695 test hard tasks as §9: image → text, true caption vs 7 same-word swaps,
  chance R@1 0.125.
- Readouts searched: text {L5, L6, L7, L5.mlp_out, L6.mlp_out} × image {L5, L6, L7, L6.mlp_out}.
- Each scorer's readout pair and hyper-parameters are chosen on the 667 val hard tasks, refitted on
  fit + val, and scored once on test.

**Scorers:**
- **ridge:** the §9 map: ridge image → text, cosine in text space.
- **whitened:** both modalities PCA-whitened on the fit rows (eigenvalue floor ε·mean, ε ∈ {1e-3, 1e-2,
  0.1}), then ridge and cosine in whitened text space. Every direction gets equal weight.
- **CCA:** regularised CCA (reg ∈ {0.01, 0.1}) between image and text features of the fit pairs;
  image and candidates are compared by cosine in the top k canonical components, k ∈ {8, …, 128}.
- **map → text fact probe:** a class-balanced logistic probe for the 72 conjunction facts, trained on
  the text features of the fit scenes. The ridge-mapped image and each candidate are compared by
  cosine of their centred probe logits.
  - Fitting ridge from image features straight to the probe logits gives identical numbers, because
    the probe is linear. So these are one test.
- **fact probe vs fact probe:** an image fact probe on the image vs the text fact probe on each
  candidate. It uses fact labels on both sides, so it is a supervised reference, not an alignment.

**Results** (hard R@1 on test):

| Epoch | Ridge | Whitened | **CCA** | Map → text fact probe | Fact probe vs fact probe |
|---|---|---|---|---|---|
| 1 | 0.135 | 0.142 | **0.210** | 0.145 | 0.167 |
| 2 | 0.145 | 0.242 | **0.260** | 0.262 | 0.240 |
| 4 | 0.194 | 0.338 | **0.412** | 0.355 | 0.355 |
| 7 | 0.170 | 0.338 | **0.424** | 0.374 | 0.345 |
| 10 | 0.151 | 0.360 | **0.485** | 0.328 | 0.387 |
| 20 | 0.197 | 0.342 | **0.469** | 0.376 | 0.367 |
| 40 | 0.183 [0.154, 0.210] | 0.358 [0.325, 0.394] | **0.452 [0.414, 0.489]** | 0.341 [0.305, 0.378] | 0.351 [0.315, 0.387] |
| random init | 0.121 | 0.115 | 0.121 | 0.118 | 0.117 |

**Choices at epoch 40** (text | image):

| Scorer | Cells | Settings |
|---|---|---|
| ridge | L5.mlp_out \| L6 | α 0.001 |
| whitened | L5.mlp_out \| L6.mlp_out | ε 0.1, α 10 |
| CCA | L6.mlp_out \| L6.mlp_out | reg 0.01, k 32 |
| map → probe | L6.mlp_out \| L6.mlp_out | — |
| fact vs fact | L5.mlp_out \| L6.mlp_out | — |

**Shuffled-pairs control, epoch 40, at the chosen settings:** CCA 0.109, whitened 0.085. Both are
chance, so the gains need true pairs.

**Reading:**
- **Ridge was the problem.** Giving weak directions equal weight doubles hard R@1 (whitened 0.36);
  CCA reaches 0.45–0.49, close to 4× chance. Binding information in the two separately trained models
  is partly linearly aligned. Plain ridge could not show it, because it fits the strong inventory and
  wording directions.
- **CCA improves with training up to epoch 10, then plateaus.** That matches the image binding probe,
  which is also flat (§10.1).
- **CCA beats the label-based fact-vs-fact scorer** (0.45 vs 0.35). The shared structure goes beyond
  the 72 hand-defined attribute-pair facts.
- **Mid-depth MLP writes are where the alignment is.** All label-free scorers end up on L5/L6 MLP
  outputs, not the residual stream at L7.
- **Caveats as in §7:** image overlap with the image training set, no relations, one seed per model,
  and settings chosen on val from a modest grid.

**Next:**
- Rerun the ordinary 1000-candidate retrieval with CCA and whitening, to see whether they also help
  there or trade easy retrieval for binding.
- The same tests for the paired dense model and the unpaired dropout trunk.

Results: `outputs/eval_hard_xret_variants/<text>__<image>.json`. Cached candidate features:
`outputs/eval_hard_xret_variants/candidates_<text>.npz`.

## 12. Does text I-JEPA improve cross-modal binding alignment? (JEPA branches vs diffusion-only text)

**Question:** I-JEPA makes text binding easier to read out. Does it also make it more alignable with
the image model's binding, measured by hard R@1 (§9) with the better scorers of §11?

**Text models compared at the same total epochs:**
- **Diffusion-only:** the dense text model of §2 (checkpoints at 4, 10, 14, 18, 22, 26, 40).
- **JEPA branches** `text_ijepa_w4_8_from_dense_ep{k}_2m_6e`, for k = 4, 8, 12, 16, 20:
  - start from that dense text model at epoch k;
  - train text-only I-JEPA on the same 2M captions: EMA teacher, layerwise targets on all blocks,
    span masking 4–8 tokens, no diffusion loss;
  - the `_6e` checkpoint (epoch_005) is at **k + 6** total epochs;
  - the `_full` continuation is at **41** total epochs, compared with diffusion-only at 40.

**Protocol:**
- **Image model fixed:** every text model is paired with the **same image model**,
  `image_dense_diffusion_2_5m_40e/epoch_039`. Differences therefore come from the text side.
- **Eval:** the same 8000 scenes, split, hard tasks (695 test, chance 0.125) and scorers as §11, with
  readouts and settings chosen on val.

**Results** (hard R@1 on test; 95% CI about ±0.035):

| Total epochs | Text model | Ridge | Whitened | **CCA** | Map → text probe | Fact vs fact |
|---|---|---|---|---|---|---|
| 10 | diffusion | 0.165 | 0.365 | **0.452** | 0.335 | 0.377 |
| 10 | JEPA from 4, +6 | 0.292 | 0.334 | **0.458** | 0.407 | 0.407 |
| 14 | diffusion | 0.190 | 0.340 | **0.463** | 0.328 | 0.351 |
| 14 | JEPA from 8, +6 | 0.286 | 0.364 | **0.475** | 0.430 | 0.404 |
| 18 | diffusion | 0.209 | 0.329 | **0.446** | 0.341 | 0.354 |
| 18 | JEPA from 12, +6 | 0.296 | 0.364 | **0.439** | 0.416 | 0.387 |
| 22 | diffusion | 0.176 | 0.347 | **0.479** | 0.368 | 0.378 |
| 22 | JEPA from 16, +6 | 0.288 | 0.354 | **0.452** | 0.459 | 0.424 |
| 26 | diffusion | 0.176 | 0.358 | **0.412** | 0.341 | 0.331 |
| 26 | JEPA from 20, +6 | 0.305 | 0.338 | **0.478** | 0.432 | 0.391 |
| 40 | diffusion | 0.183 | 0.358 | **0.452** | 0.341 | 0.351 |
| 41 | JEPA from 4, full | 0.260 | 0.271 | **0.383** | 0.368 | 0.344 |
| 41 | JEPA from 8, full | 0.220 | 0.266 | **0.393** | 0.386 | 0.370 |
| 41 | JEPA from 12, full | 0.272 | 0.288 | **0.397** | 0.367 | 0.335 |
| 41 | JEPA from 16, full | 0.268 | 0.292 | **0.417** | 0.380 | 0.348 |
| 41 | JEPA from 20, full | 0.259 | 0.367 | **0.447** | 0.414 | 0.390 |

**Readouts chosen by CCA:**
- **Usually L6.mlp_out or L5.mlp_out**, with k = 16–32 components and reg 0.01.
- **Exceptions:** L7 for diffusion at 26 and for JEPA from 12 at 18.

**Reading:**
1. **With CCA, JEPA gives no gain in cross-modal binding alignment.**
   - Matched differences (JEPA − diffusion) are +0.006, +0.012, −0.007, −0.027 and +0.066, about +0.01 on
     average. All but the last are within noise, and the +0.066 at 26 comes from a weak diffusion point.
   - **Long JEPA lowers the ceiling:** all 41-epoch branches are at or below diffusion-only at 40
     (0.38–0.45 vs 0.45). The earlier the branch, the worse.
2. **With plain ridge, JEPA helps consistently** (+0.1: 0.17–0.21 → 0.29–0.31). A short JEPA phase moves
   text binding into higher-variance directions, so even a naive map picks it up.
3. **Probe-based scorers gain +0.04 to +0.09 with short JEPA.** These are map → text probe and fact vs
   fact. They track JEPA's faster within-text binding, but their best (0.459) does not exceed CCA on
   diffusion-only text.
4. **Whitened ridge:** no gain from short JEPA; four of the five long branches drop to 0.27–0.29.

**Conclusion:** short text I-JEPA makes binding more *prominent*, easier for simple readouts and probes,
but it does not raise the amount of binding structure that is linearly shared with the image model.
Longer JEPA reduces it.

**Caveats:**
- Only one image model (40 epochs). Pairs with the image model at each text model's own epoch are still
  running.
- One seed per model, and image overlap as in §3.3.

Results: `outputs/eval_hard_xret_variants/<text label>__denseI_e040_IMAGE.json`, where text labels are
`denseT_eNNN_TEXT` and `ijepa_fromKK_totNNN_TEXT`.

## 13. JEPA text × JEPA image vs dense text × dense image (same total epochs)

**Question:** with JEPA on **both** sides, is cross-modal alignment better than between two
diffusion-only models at the same total epochs?

**Models:** both JEPA models start from their dense model at **epoch 12**.
- **Text:** `text_ijepa_w4_8_from_dense_ep12_2m_6e`, text I-JEPA as in §12.
- **Image:** `image_ijepa_sweep_blk_s15_t45`, image I-JEPA from `image_dense_diffusion_2_5m_40e/epoch_011`.
  - EMA teacher, layerwise targets, 2-D block masking; context 15% scale, targets 45%.
  - Same 2.53M images, no diffusion loss.
  - Chosen from the 10-config sweep for its L7 binding and probe accuracy.
  - The sweep config with the best best-layer binding, `blk_tiny_t45`, is added at 16 as a check.
- **Matched total epochs:**
  - 14 = 12 dense + 2 JEPA (`epoch_001` on both sides);
  - 16 = 12 + 4 (text `epoch_003`, image `epoch_003`).
  
  Dense pairs at the same totals: text `text_dense_diffusion_2m_20e_continued/epoch_{013,015}` × image
  `image_dense_diffusion_2_5m_40e/epoch_{013,015}`.
- **Only 14 and 16 can be matched,** because the image JEPA exists only from dense epoch 12.
- **Mixed pairs** (JEPA × dense) were not run.

**Protocol:** the same 8000 scenes and split as everywhere above.
- Ordinary retrieval and CKA as in §5–6: 1000 test candidates, ridge cell chosen on val; CKA on test
  features at the chosen cell and at the L7 diagonal.
- Hard retrieval as in §11: 695 tasks, chance 0.125.

### 13.1 Ordinary retrieval and CKA

| Total epochs | Pair | Cell (text \| image) | t→i R@1 [95% CI] | t→i MRR | i→t R@1 / MRR | Procrustes i→t MRR | CKA, chosen cell | CKA, L7 \| L7 |
|---|---|---|---|---|---|---|---|---|
| 14 | dense × dense | L5 \| L7 | 0.626 [0.596, 0.658] | 0.727 | 0.448 / 0.574 | 0.179 | 0.358 | 0.227 |
| 14 | JEPA × JEPA (s15_t45) | L6 \| L7 | **0.657** [0.630, 0.687] | 0.757 | 0.457 / 0.574 | **0.242** | 0.390 | **0.357** |
| 16 | dense × dense | L5 \| L7 | **0.649** [0.620, 0.678] | 0.748 | **0.534 / 0.640** | 0.192 | 0.312 | 0.246 |
| 16 | JEPA × JEPA (s15_t45) | L5.mlp_out \| L7 | 0.575 [0.544, 0.606] | 0.685 | 0.475 / 0.587 | **0.314** | 0.281 | **0.353** |
| 16 | JEPA × JEPA (tiny_t45) | L5.mlp_out \| L7 | 0.593 [0.564, 0.623] | 0.702 | 0.470 / 0.591 | 0.287 | 0.277 | 0.352 |

### 13.2 Hard retrieval (true caption vs 7 same-word swaps)

| Total epochs | Pair | Ridge | Whitened | **CCA** [95% CI] | Map → text probe | Fact vs fact |
|---|---|---|---|---|---|---|
| 14 | dense × dense | 0.178 | 0.319 | **0.435** [0.400, 0.471] | 0.344 | 0.374 |
| 14 | JEPA × JEPA (s15_t45) | 0.255 | 0.400 | **0.515** [0.481, 0.553] | 0.468 | 0.460 |
| 16 | dense × dense | 0.197 | 0.348 | **0.502** [0.463, 0.541] | 0.378 | 0.383 |
| 16 | JEPA × JEPA (s15_t45) | 0.318 | 0.414 | **0.511** [0.473, 0.548] | 0.420 | 0.452 |
| 16 | JEPA × JEPA (tiny_t45) | 0.245 | 0.396 | **0.538** [0.502, 0.573] | 0.453 | 0.475 |

CCA chose L5/L6.mlp_out or L7 (text) × L6.mlp_out (image), k = 16–32, reg 0.01.

**Reading:**
1. **Hard retrieval improves with JEPA on both sides.**
   - JEPA×JEPA beats dense×dense on every hard scorer at both epochs:
     - ridge +0.05 to +0.12;
     - whitened +0.05 to +0.08;
     - probe-based scorers +0.04 to +0.12.
   - With CCA the gain is +0.08 at 14 (CIs do not overlap) and +0.01 / +0.04 at 16 (within noise).
   - 0.538 is the best hard R@1 measured so far.
2. **Plain R@1 does not improve.**
   - At 14 it is about level: 0.657 vs 0.626, CIs overlap.
   - At 16 JEPA×JEPA is lower: 0.575–0.593 vs 0.649 text→image, 0.47 vs 0.53 image→text.
   - JEPA trades some inventory-driven retrieval for binding alignment.
3. **The two JEPA spaces are geometrically closer.**
   - Layer-matched CKA at L7 rises from 0.23–0.25 to 0.35.
   - Procrustes (rotation only) rises from 0.18–0.19 to 0.24–0.31.
   - CKA at the ridge-chosen cell does not follow (0.28–0.39 vs 0.31–0.36), because that cell is chosen for
     plain retrieval, not similarity.
4. **The image side is the likely source.** In §12, JEPA on the text side alone (with the 40-epoch dense
   image model) gave no CCA gain. Mixed pairs at 14/16 would confirm this.

**Caveats:**
- Two matched epochs only.
- One seed per model.
- The image config was chosen by binding.
- Image overlap as in §3.3.

Results:
- ordinary: `outputs/eval_all/xret/results/<text>__<image>.json`;
- hard: `outputs/eval_hard_xret_variants/<text>__<image>.json`.
- Labels: `denseT_e01{4,6}_TEXT`, `denseI_e01{4,6}_IMAGE`, `ijepa_from12_tot01{4,6}_TEXT`,
  `ijepaI_{s15t45,tinyt45}_tot01{4,6}_IMAGE`.
