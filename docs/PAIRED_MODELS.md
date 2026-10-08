# Paired multimodal models: training, validation and evaluation

This file covers every model trained on **paired** data, where each training
sequence holds one scene's caption *and* its VQ image tokens, so the model sees
the correspondence. All five use the **1.2M paired set** (`train_pairs_human.jsonl`).
No paired model exists on the 2M / 2.53M corpora.

Evaluation follows [EVALUATIONS.md](../EVALUATIONS.md) and the full eval in
[FULL_EVAL.md](FULL_EVAL.md), using the last checkpoint of each run.
`dense_private` models are scored **trunk only** (private LoRA dropped). Every
layer is in `outputs/eval_all/all_metrics_long.csv`.

## 1. Runs and lineage

```
multimodal_paired_dense_1_2m_4e            dense paired diffusion, scratch           epochs 1-4
 ├─ mm_paired_stage1_jepa_lw_all_1_2m_2e   + I-JEPA (EMA, layerwise, all 8 blocks)   +2  (total 6)
 │   ├─ mm_paired_stage2_frozen_1_2m_2e    + private LoRA r128, trunk FROZEN         +2  (total 8)
 │   └─ mm_paired_stage2_trainable_1_2m_2e + private LoRA r128, trunk trainable      +2  (total 8)
 └─ mm_paired_merged_jepa_trunk_private_diff_1_2m_3e
                                           JEPA on the trunk + diffusion to private  +2 of 3 planned (total 6)
```

| Run | Train mode | Objective | Init | Saved | Batch / lr |
|---|---|---|---|---|---|
| `multimodal_paired_dense_1_2m_4e` | dense | diffusion | scratch | epochs 0–3 | 32 / 3e-4 |
| `mm_paired_stage1_jepa_lw_all_1_2m_2e` | dense | I-JEPA / data2vec, EMA, layerwise | paired dense, epoch 4 | epochs 0–1 | 32 / 3e-4 |
| `mm_paired_stage2_frozen_1_2m_2e` | dense_private (r128), trunk frozen | diffusion (+HSIC) | stage 1, epoch 2 | epochs 0–1 | 32 / 3e-4 |
| `mm_paired_stage2_trainable_1_2m_2e` | dense_private (r128) | diffusion | stage 1, epoch 2 | epochs 0–1 | 32 / 3e-4 |
| `mm_paired_merged_jepa_trunk_private_diff_1_2m_3e` | dense_private (r128) | diffusion (private only) + I-JEPA on trunk | paired dense, epoch 4 | epochs 0–1 | 32 / 3e-4 |

**Missing for a fair comparison.** Every post-base stage has only 2 epochs. Neither of these exists:
- a **4-epoch** private-LoRA stage, matching the 4 JEPA epochs used in the unpaired runs;
- a **paired dense control trained longer** (6 / 8 epochs of plain diffusion), to separate JEPA from
  simply training more.

See §6.

## 2. Validation loss (logged during training)

1/t-weighted masked cross-entropy on **paired** validation sequences, 75% of tokens masked:

| Run | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 |
|---|---|---|---|---|
| paired dense | 2.774 | 2.490 | 2.361 | **2.260** |
| stage 1 JEPA | 4.489 | 4.555 | — | — |
| stage 2 frozen | 2.149 | **2.101** | — | — |
| stage 2 trainable | 2.113 | **2.033** | — | — |
| merged | 4.682 | 4.089 | — | — |

JEPA runs train no diffusion loss, so their diffusion head drifts (4.5). That number says nothing about
representation quality. Stage 2 retrains the head through the private LoRA and gets back to 2.0–2.1.

## 3. Validation per modality (final checkpoints)

Plain cross-entropy and masked-token accuracy at 75% masking on 2048 **single-modality** inputs
(caption alone or image alone); `scripts/diagnose_private_reliance.py`, output
`outputs/eval_all/paired_val_per_modality.json`. "Trunk" = private LoRA switched off.

| Run | Text CE / acc, full | Text, trunk | Image CE / acc, full | Image, trunk |
|---|---|---|---|---|
| paired dense (4 ep) | 7.02 / 0.300 | — | 3.34 / 0.228 | — |
| stage 1 JEPA (+2) | 5.54 / 0.306 | — | 6.19 / 0.002 | — |
| stage 2 frozen (+2) | 5.78 / **0.340** | 5.54 / 0.306 | 3.74 / 0.217 | 6.19 / 0.002 |
| stage 2 trainable (+2) | 6.36 / **0.348** | 6.23 / 0.335 | **3.45 / 0.230** | 3.81 / 0.163 |
| merged | 5.50 / 0.086 | 9.02 / 0.000 | 5.61 / 0.027 | 8.75 / 0.008 |

- Text alone is out of distribution for paired models, which always saw the caption next to its image.
  Their text-only accuracy is about 0.30–0.35, against about 0.69 for text-only models. Image alone works
  almost normally (0.23, vs 0.26 for image-only dense).
- Stage 2 trainable is the best generator. Its trunk alone still generates images reasonably (0.163).
  The frozen JEPA trunk cannot (0.002).
- Merged is broken on both modalities.

## 4. Cross-modal retrieval and binding

Retrieval: the text and image sides of the same model, a ridge map chosen on val, 1000 test scenes;
chance R@1 0.001, MRR 0.0075. Binding: 397 content-matched image pairs; pairwise chance 50%;
strict = both scenes classified correctly.

| Model | Text→image R@1 / MRR | Image→text R@1 / MRR | Procrustes i→t MRR | Binding L7 / best | Strict L7 / best |
|---|---|---|---|---|---|
| stage 2 trainable | **0.596 / 0.705** | **0.374 / 0.514** | 0.146 | 76.6 / 81.7 | 48.3 / 56.1 |
| stage 1 JEPA | 0.421 / 0.556 | 0.205 / 0.326 | **0.280** | **89.8 / 91.5** | **68.1 / 70.8** |
| stage 2 frozen | 0.421 / 0.556 | 0.205 / 0.326 | 0.280 | 89.8 / 91.5 | 68.1 / 70.8 |
| paired dense | 0.316 / 0.451 | 0.168 / 0.276 | 0.207 | 79.2 / 84.7 | 50.2 / 59.2 |
| merged | 0.024 / 0.068 | 0.028 / 0.065 | 0.040 | 50.6 / 63.4 | 24.0 / 33.9 |
| *reference: 2 separate dense models, 8 ep* | *0.603 / 0.710* | *0.449 / 0.569* | — | *80.3 / 84.0* | *55.5 / 60.6* |
| *reference: unpaired trunk + private LoRA + dropout, 8 ep* | *0.510 / 0.628* | *0.376 / 0.502* | — | *79.7 / 84.7* | *53.9 / 61.2* |

- Stage 2 frozen equals stage 1 exactly: its trunk is frozen, and the trunk is what is scored.
- Paired I-JEPA (stage 1) has the **best binding of any model** in the full eval.
- The best paired retrieval (0.596) only matches two separately trained dense models (0.603), with less
  training (4–8 epochs on 1.2M vs 8 epochs on 2M).

## 5. Representation summary (layer 7; best layer in parentheses)

| Model | Modality | Probe | RSA | Cos / rank | Conj mAP best | Scene ρ_conj best | Sensitivity |
|---|---|---|---|---|---|---|---|
| paired dense | text | 86.0 (86.0) | 0.191 | 0.93 / 26 | 0.603 (emb) | 0.125 | S_bind 0.517, S_cont 2.161 (centred 0.533 / 2.018) |
| paired dense | image | 82.4 (84.1) | 0.241 | 0.74 / 22 | 0.693 | 0.229 | S 0.371 / 0.382 / 0.395 (raw / centred / z); triples d′ 0.297 |
| stage 1 JEPA | text | 85.7 (88.0) | 0.093 | 0.72 / 30 | 0.694 | 0.235 | S_bind 0.262, S_cont 1.186 (centred 0.313 / 1.338) |
| stage 1 JEPA | image | **86.3** (86.3) | **0.337** | 0.83 / 22 | **0.761** | **0.264** | S 0.251 / 0.276 / 0.302; triples d′ 0.173 |
| stage 2 trainable | text | **89.3 (91.7)** | 0.126 | 0.81 / 33 | **0.793** | 0.181 | S_bind 0.417, S_cont 1.517 (centred 0.468 / 1.596) |
| stage 2 trainable | image | 81.4 (81.4) | 0.106 | 0.88 / 15 | 0.633 | 0.144 | S 0.359 / 0.372 / 0.376; triples d′ 0.235 |
| merged | text | 56.3 (84.8, emb) | −0.006 | **1.00 / 2** | 0.603 (emb) | 0.129 | collapsed |
| merged | image | 51.5 (69.5, emb) | 0.014 | **1.00 / 2** | 0.413 (emb) | 0.050 | collapsed |

Stage 2 frozen = stage 1, since the trunk is frozen.

**JEPA prediction check** (mean over supervised layers; Δ = C₊ − C₋, raw / centred):
- stage 1 JEPA: text 0.213 / **0.273**, image 0.128 / **0.177**. Instance-specific, but weaker than the
  unpaired-trunk and image-only I-JEPA runs (centred Δ 0.3–0.7).
- merged: text and image 0.001 / **0.018**. Collapsed: the predictor matches any scene equally well
  (rank 2, cosine 1.00).

**Takeaways:**
- Paired JEPA gives the strongest image binding and scene structure.
- Paired stage 2 (trainable) gives the best paired retrieval and text probes, but loses most of the
  image-side gains.
- The merged recipe (JEPA on the trunk + diffusion only to the private LoRA, from the paired dense
  model) collapses.

## 6. Proposed runs to complete the picture (not started)

1. **Paired dense, continued to 8 epochs** (resume `multimodal_paired_dense_1_2m_4e` from epoch 4). Its
   6- and 8-epoch checkpoints are the matched controls for stage 1 (+2) and stage 2 (+4 total).
2. **Stage 2 trainable for 4 epochs** from stage 1 epoch 2, matching the 4 JEPA epochs of the unpaired runs.
3. Optionally, a **2M paired set**: generate paired captions for the 2.53M image scenes, to match the
   unpaired corpora.

About 1 h per paired epoch on a 3090.
