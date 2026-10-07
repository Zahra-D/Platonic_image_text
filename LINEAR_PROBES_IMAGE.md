# Frozen linear probes for the CLEVR image models

Do the image models learn scene structure that a *linear* read-out can recover,
and did the JEPA/data2vec objective help compared with dense diffusion?

**Short answer:** dense diffusion learns the most linearly readable
representation per image seen. Replacing the diffusion loss with data2vec,
starting from a trained dense model, gains at most about 1–2 points on count or
depth. It loses significantly on shape and material, and generation gets worse.
Data2vec from scratch collapses. None of the image runs added JEPA on top of
diffusion, so that question is still open (see [Open question](#open-question)).

---

## 1. Protocol

It follows I-JEPA (Assran et al., 2023), which evaluates low-level scene
understanding with VTAB CLEVR/Count and CLEVR/Dist:

- **Frozen encoder.** Only a linear layer is trained.
- **Clean input.** The full 16×24 grid of VQ tokens, with no masking.
- **Scene features.** The residual stream averaged over the 384 image tokens,
  read after the embedding and after every block (L0–L7). The encoder has no
  [CLS] token, so this matches I-JEPA's average pooling.
- **Headline feature (as in I-JEPA).** The better of the last block and the
  concatenation of the last four blocks, chosen on probe-val. "Best layer" is
  also reported: the best single layer, chosen on probe-val.
- **Probes.**
  - Classification: multinomial logistic regression on standardized features,
    fit with L-BFGS; weight decay from {1e-5 … 1e-1}, chosen on probe-val.
  - Regression: ridge; alpha from {1e-3 … 100}, chosen on probe-val.

### Data and splits (identical for every model)

- **Scenes.** 20,000 held-out scenes: the val split of
  `platonic_clevr_v1_5M_train_gpu_visible`, which no 1.2M run trained on.
  Labels come from the Blender scene JSONs.
- **One fixed split** (seed 20260929): 12,000 probe-train, 3,000 probe-val,
  5,000 test. Test is scored once.
- **Objects** (108,907 in total) inherit their scene's split, so no scene
  appears in two splits.
- **Uncertainty.** Each score has a 95% bootstrap interval over test items.
  Model-vs-model differences use a **paired** bootstrap on the same test items.

### Tasks

| Task | Level | Target | Metric |
|---|---|---|---|
| `count` | scene | number of objects, 3–8 (6 classes); VTAB CLEVR/Count | accuracy |
| `dist` | scene | camera depth of the closest object, binned at [8, 8.5, 9, 9.5, 10]; VTAB CLEVR/Dist | accuracy |
| `color` / `shape` / `material` / `size` | scene | number of objects with each value (e.g. #red, #cube) | mean R² over values |
| `position` | scene | 3-D (x, y) of the closest object | R² |
| `obj_color` / `obj_shape` / `obj_material` / `obj_size` | object | attribute of the object, read from the token under its projected centre (8 / 3 / 2 / 2 classes) | accuracy |

---

## 2. Training budget per model

One image is 384 VQ tokens. "Images seen" is cumulative and includes the
checkpoint a run was initialized from.

| Run | Objective | Init | Images seen (per saved epoch) | Image tokens |
|---|---|---|---|---|
| `dense_diffusion_1_2m_4e` | diffusion, dense | scratch | 1.2 / 2.4 / 3.6 / 4.8M | 0.46 → 1.84B |
| `dense_diffusion_1_2m_6e_continued` | diffusion, dense | resumes dense epoch 3 | 6.0 / 7.2M | 2.30 / 2.77B |
| `lora_diffusion_1_2m_4e` | diffusion, Tri-LoRA, no base | scratch | 1.2 → 4.8M | 0.46 → 1.84B |
| `lora_diffusion_1_2m_12e_continued` | diffusion, Tri-LoRA | resumes LoRA epoch 3 | 6.0 → 9.6M (stopped at epoch 7) | 2.30 → 3.69B |
| `data2vec_scratch_*` (4 variants) | **data2vec only** (`diffusion.weight: 0`) | scratch | 1.2 → 4.8M | 0.46 → 1.84B |
| `data2vec_from_dense_*` (5 variants) | **data2vec only** (`diffusion.weight: 0`) | dense epoch 3 (4.8M) | 6.0 / 7.2M | 2.30 / 2.77B |
| `dense_private_frozen_hsic_from_d2v_lw_block2d` | diffusion + HSIC, dense_private | data2vec from dense, block-mask variant, epoch 1 (7.2M) | 8.4 / 9.6M | 3.23 / 3.69B |
| `ema_jepa_*` (4 runs) | diffusion + shared EMA-JEPA, LoRA | scratch | 6.31M (**70 epochs of a 90k-image set**) | 2.42B |
| `diffusion_only_ema_jepa_pilot` | diffusion, LoRA | scratch | 0.09M (90k set) | 0.03B |

The EMA-JEPA runs trained on a different, smaller dataset
(`platonic_clevr_v1_100k_gpu_visible_s20260825`), so they aren't budget-matched
to the 1.2M runs.

---

## 3. Baselines

| Baseline | count | dist | color R² | shape R² | material R² | size R² | position R² | obj color | obj shape | obj material | obj size |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Chance (always guess the most common class / predict the train mean) | 24.4 | 21.7 | 0 | 0 | 0 | 0 | 0 | 23.3 | 34.1 | 51.7 | 50.7 |
| Chance (uniform guess) | 16.7 | 16.7 | – | – | – | – | – | 12.5 | 33.3 | 50.0 | 50.0 |
| Easy: bag of VQ codes (512-d histogram) | 37.9 | 34.8 | 0.559 | 0.309 | 0.528 | 0.549 | 0.228 | – | – | – | – |
| Easy: mean VQ codebook vector (64-d) | 36.8 | 31.8 | 0.461 | 0.274 | 0.482 | 0.465 | 0.142 | – | – | – | – |
| Easy: raw pixels, 32×48 RGB | 32.5 | 61.2 | 0.233 | 0.210 | 0.286 | 0.552 | 0.377 | – | – | – | – |
| Easy: VQ code under the object (one-hot) | – | – | – | – | – | – | – | 80.2 | 63.5 | 67.6 | 67.6 |
| Easy: 12×12 pixel patch around the object | – | – | – | – | – | – | – | 97.3 | 79.5 | 89.9 | 98.0 |
| Untrained dense (random weights, same architecture) | 35.9 | 35.9 | 0.484 | 0.277 | 0.458 | 0.500 | 0.299 | 79.8 | 63.4 | 66.9 | 74.2 |
| Untrained LoRA (random weights, same architecture) | 36.7 | 35.5 | 0.507 | 0.290 | 0.470 | 0.496 | 0.226 | 79.8 | 63.5 | 67.3 | 73.0 |

How to read the baselines:

- A randomly initialized network is no better than its input VQ codes, so
  anything clearly above about 37% on count is learned.
- `dist` is partly low-level: raw pixels already reach 61%.
- `obj_color` and `obj_size` are nearly solved by a local pixel patch.
  `count`, `shape` and `obj_shape` separate models best.

---

## 4. Results (headline feature: last block or last four blocks, chosen on probe-val)

Accuracy is in %, and the R² columns are R². Only selected epochs are shown;
every epoch is in `outputs/linear_probes_image/summary.csv`.

| Model | Images | count | dist | color | shape | material | size | position | obj color | obj shape | obj material | obj size |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense, epoch 0 | 1.2M | 65.0 | 70.7 | 0.740 | 0.513 | 0.796 | 0.910 | 0.723 | 95.9 | 82.5 | 95.7 | 99.4 |
| dense, epoch 3 | 4.8M | 78.7 | 77.5 | 0.773 | 0.782 | 0.905 | 0.958 | 0.725 | 96.7 | 94.2 | 98.0 | 99.8 |
| dense continued, epoch 4 | 6.0M | 79.7 | 77.6 | 0.772 | 0.814 | 0.910 | 0.960 | 0.728 | 96.8 | 94.8 | 97.9 | 99.8 |
| dense continued, epoch 5 | 7.2M | 79.9 | 77.4 | 0.771 | **0.835** | 0.914 | 0.963 | 0.730 | 96.9 | **95.5** | 98.0 | 99.8 |
| data2vec from dense, all-layer average target | 7.2M | 78.2 | 77.2 | 0.749 | 0.731 | 0.877 | 0.955 | 0.715 | 96.6 | 93.2 | 97.7 | 99.8 |
| data2vec from dense, layers 4–7 average target | 7.2M | 80.7 | **79.2** | 0.748 | 0.745 | 0.886 | 0.959 | 0.736 | 96.7 | 93.4 | 97.7 | 99.7 |
| data2vec from dense, layer-wise all layers, block mask | 7.2M | 80.5 | 78.8 | 0.765 | 0.752 | 0.900 | 0.958 | 0.747 | 96.4 | 93.2 | 97.8 | 99.7 |
| data2vec from dense, layer-wise all layers | 7.2M | 81.0 | 78.0 | 0.779 | 0.749 | 0.893 | 0.959 | 0.741 | 96.6 | 93.2 | 97.7 | 99.8 |
| data2vec from dense, layer-wise layers 4–7 | 7.2M | 80.7 | 78.4 | **0.783** | 0.754 | 0.894 | 0.959 | 0.742 | 96.6 | 93.4 | 97.7 | 99.7 |
| data2vec scratch, 30% block mask, layers 4–7 average | 4.8M | 40.6 | 56.9 | 0.305 | 0.263 | 0.333 | 0.637 | 0.619 | 82.3 | 64.6 | 70.1 | 93.3 |
| data2vec scratch, 30% block mask, layers 4–7 layer-wise | 4.8M | 38.2 | 48.0 | 0.269 | 0.241 | 0.303 | 0.558 | 0.556 | 80.2 | 63.7 | 67.3 | 85.8 |
| data2vec scratch, 65% block mask, layers 4–7 average | 4.8M | 39.0 | 53.8 | 0.184 | 0.207 | 0.249 | 0.569 | 0.593 | 69.2 | 58.4 | 63.5 | 84.6 |
| data2vec scratch, 65% block mask, layers 4–7 layer-wise | 4.8M | 36.6 | 46.8 | 0.169 | 0.186 | 0.225 | 0.515 | 0.536 | 68.1 | 58.2 | 62.0 | 80.2 |
| dense_private + HSIC (from data2vec), epoch 0 | 8.4M | 81.9 | **79.7** | 0.774 | 0.826 | 0.912 | 0.965 | 0.729 | 96.9 | 95.1 | 98.0 | 99.8 |
| dense_private + HSIC (from data2vec), epoch 1 | 9.6M | **82.4** | 79.1 | 0.774 | **0.838** | **0.916** | **0.967** | 0.736 | 96.9 | 95.5 | **98.1** | 99.8 |
| LoRA, epoch 0 | 1.2M | 45.8 | 58.8 | 0.560 | 0.356 | 0.576 | 0.719 | 0.597 | 91.9 | 69.1 | 83.2 | 97.0 |
| LoRA, epoch 3 | 4.8M | 66.8 | 70.4 | 0.663 | 0.573 | 0.807 | 0.916 | 0.673 | 95.9 | 87.2 | 96.7 | 99.5 |
| LoRA continued, epoch 4 | 6.0M | 70.0 | 72.5 | 0.692 | 0.635 | 0.832 | 0.929 | 0.688 | 96.2 | 89.7 | 97.1 | 99.6 |
| LoRA continued, epoch 7 | 9.6M | 75.7 | 75.1 | 0.714 | 0.741 | 0.864 | 0.944 | 0.695 | 96.5 | 93.4 | 97.6 | 99.7 |
| EMA-JEPA dynamic, ratio 0.10 (90k set) | 6.3M | 71.7 | 74.8 | 0.689 | 0.655 | 0.845 | 0.934 | 0.689 | 96.2 | 90.2 | 97.2 | 99.6 |
| EMA-JEPA dynamic, ratio 0.25 (90k set) | 6.3M | 71.1 | 74.6 | 0.657 | 0.658 | 0.836 | 0.930 | 0.689 | 96.2 | 90.7 | 97.3 | 99.6 |
| EMA-JEPA dynamic, ratio 0.50 (90k set) | 6.3M | 70.9 | 72.8 | 0.696 | 0.650 | 0.830 | 0.932 | 0.689 | 96.6 | 90.6 | 97.1 | 99.7 |
| EMA-JEPA fixed λ=0.5 (90k set) | 6.3M | 73.4 | 74.9 | 0.689 | 0.683 | 0.855 | 0.939 | 0.686 | 96.3 | 92.1 | 97.3 | 99.8 |
| EMA-JEPA pilot, diffusion only (90k set) | 0.09M | 36.4 | 36.0 | 0.494 | 0.286 | 0.477 | 0.510 | 0.256 | 79.6 | 63.4 | 67.1 | 74.9 |

---

## 5. Did JEPA/data2vec help? Paired, budget-matched differences

Each data2vec checkpoint is compared with the dense checkpoint that has **the
same initialization and the same number of images seen**. Values are data2vec
minus dense. Accuracy is in points, R² tasks are in R² units, and `*` marks a
paired 95% CI that excludes zero.

### 5a. Starting from dense (the clean comparison)

Both sides start from dense epoch 3 (4.8M images) and train the same extra
1.2M or 2.4M images. The only difference is the objective: data2vec only, or
diffusion only. Headline feature:

| Images | data2vec target | count | dist | color | shape | material | size | position | obj color | obj shape | obj material | obj size |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6.0M | all-layer average | −2.16* | −1.52* | −0.025* | −0.087* | −0.035* | −0.007* | −0.013* | −0.20* | −1.87* | −0.24* | −0.06* |
| 6.0M | layers 4–7 average | +0.64 | +0.96 | −0.025* | −0.073* | −0.024* | −0.003* | +0.007* | −0.14* | −1.55* | −0.26* | −0.04 |
| 6.0M | layer-wise all, block mask | −0.06 | +0.50 | −0.012* | −0.073* | −0.014* | −0.004* | +0.013* | −0.34* | −1.95* | −0.25* | −0.03 |
| 6.0M | layer-wise all | +1.36* | +0.30 | +0.002 | −0.066* | −0.020* | −0.002* | +0.012* | −0.15* | −1.70* | −0.23* | −0.07* |
| 6.0M | layer-wise layers 4–7 | +1.10* | +0.88 | +0.006* | −0.061* | −0.016* | −0.002* | +0.009* | −0.10 | −1.54* | −0.23* | −0.07* |
| 7.2M | all-layer average | −1.66* | −0.18 | −0.022* | −0.103* | −0.037* | −0.008* | −0.016* | −0.28* | −2.24* | −0.32* | −0.04 |
| 7.2M | layers 4–7 average | +0.84 | +1.82* | −0.023* | −0.089* | −0.028* | −0.004* | +0.006 | −0.24* | −2.05* | −0.26* | −0.05* |
| 7.2M | layer-wise all, block mask | +0.66 | +1.48* | −0.006* | −0.083* | −0.014* | −0.005* | +0.017* | −0.48* | −2.21* | −0.25* | −0.04 |
| 7.2M | layer-wise all | +1.08 | +0.68 | +0.008* | −0.085* | −0.021* | −0.005* | +0.010* | −0.27* | −2.22* | −0.33* | −0.04 |
| 7.2M | layer-wise layers 4–7 | +0.86 | +1.06 | +0.012* | −0.080* | −0.020* | −0.004* | +0.011* | −0.26* | −2.04* | −0.27* | −0.06* |

What the table shows:

- **Count and dist.** Four of the five targets gain up to about 1–2 points on
  count or dist; 4 of the 20 gains are significant. With the best-single-layer
  probe, every 7.2M count difference is negative (−0.6 to −3.6), so the gain
  isn't robust to feature choice.
- **Shape.** Every variant is significantly worse: about −2 points per object,
  about −0.08 R² on shape counts. The gap widens from 6.0M to 7.2M, because dense
  keeps improving on shape (R² 0.78 → 0.81 → 0.84 over epochs 3–5) while data2vec
  does not.
- **Material.** Every variant is significantly worse on material counts and
  per-object material.
- **Averaging across all layers** is the worst target and is below dense almost
  everywhere.
- **Generation.** From-dense data2vec drops masked-token accuracy from about
  0.25 to 0.10–0.14, and validation diffusion loss (t = 0.75) rises from 3.63 to
  about 4.8–4.9.

### 5b. From scratch

Data2vec from scratch loses to dense from scratch on **every task at every
budget** (1.2 / 2.4 / 3.6 / 4.8M), by 20–42 points on count and dist. The count
gap widens from about −28 to about −40 as dense keeps improving. All
differences are significant.

This looks like representation collapse rather than a slow learner:

- **Layer profile.** Count accuracy by layer rises only from 37% (embedding)
  to 42% (best layer), against 37.5% → 79.4% for dense.
- **Training logs.** Masked-to-target cosine is 0.98–0.99, the logged target
  spread is about 0.05, and validation loss is flat across 4 epochs
  (8.542 → 8.535).
- **Probe scores.** At 4.8M images these models are at the easy-baseline level:
  count 36–41%, and dist below raw pixels.

### 5c. Other comparisons (not budget-matched)

- **LoRA vs dense.** LoRA is well below dense at the same budget: count 66.8% vs
  78.7% at 4.8M, and shape R² 0.57 vs 0.78. Continued to 9.6M it reaches 75.7%,
  still below dense at 4.8M.
- **EMA-JEPA LoRA.** These are the only image runs that actually combine
  diffusion with JEPA. They reach count 71–73% after 6.3M images of a 90k set,
  above 1.2M-data LoRA at 4.8M (66.8%), but the dataset differs and they saw
  more images. The same-data diffusion-only control is the 0.09M-image pilot,
  which is at untrained level, so this is not a controlled test.
- **dense_private + HSIC** has the best count of any run (82.4%), but it's the
  only model at 9.6M images. Dense was trained only to 7.2M, and its count had
  plateaued at about 80% by epoch 4. A dense run continued to 9.6M is needed
  before crediting the HSIC stage or its data2vec starting point.

---

## 6. Conclusions

1. **Dense diffusion gives the best representation per image seen.** Count
   and dist plateau at about 80% and 77% by 6M images; shape keeps improving.
2. **Replacing diffusion with data2vec** (from dense) trades about 1–2 points
   of count or dist, and that gain isn't robust, for significant losses on shape
   and material, plus worse generation.
3. **Data2vec from scratch** collapses in all four variants tried.
4. **LoRA without a base** is substantially less linearly readable than dense
   at a matched budget.

## Open question

All 9 data2vec image runs set `diffusion.weight: 0`: they *replace* diffusion.
"Does adding JEPA to dense diffusion help?" has not been tested. The missing
run is dense continued from `image_dense_diffusion_1_2m_4e/epoch_003.pt` with
diffusion weight 1 plus the data2vec loss (the layer-wise layers 4–7 target was
the best here), probed at 6.0M and 7.2M against
`dense_diffusion_1_2m_6e_continued` epochs 4 and 5. A dense run to 9.6M would
also settle the dense_private + HSIC result.

## Caveats

- These probes read the full residual stream. They don't separate the shared
  and private LoRA branches.
- `obj_size`, `obj_color` and `size` are near their ceiling for all trained
  models and discriminate little.
- The headline feature can differ by model (last block vs last four blocks).
  Both choices are made on probe-val only.
- In `summary.csv`, the images-seen column for the two untrained rows shows
  1.2M; the correct value is 0.

## Reproduce

```bash
# budget tables
python scripts/image_training_budget.py outputs/image_*/last.pt > outputs/image_training_budget.json
python scripts/image_training_budget.py $(ls outputs/image_*/epoch_0??.pt | grep -v vocab) > outputs/image_training_budget_epochs.json
# probes: dense and matched data2vec first, then everything else (skips finished labels)
scripts/run_linear_probes_dense_vs_jepa.sh 2          # GPU id
# tables
python scripts/summarize_linear_probes_image.py         # -> outputs/linear_probes_image/summary.csv
python scripts/compare_dense_vs_jepa_probes.py                                     # headline feature
python scripts/compare_dense_vs_jepa_probes.py outputs/linear_probes_image best_layer
```

Code: `evaluate_linear_probes.py`. Per-model results with every layer, the
chosen hyper-parameters, CIs and per-item scores are in
`outputs/linear_probes_image/*.json`; splits, label histograms and chance levels
are in `construction.json`.
