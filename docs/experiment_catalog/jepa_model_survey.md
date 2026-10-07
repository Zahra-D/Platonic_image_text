# JEPA model survey: every run, its training budget, and its measured semantics

> **Type:** results survey · **Status:** 26 text JEPA-family runs, 11 image runs, 7 multimodal runs · **Updated:** 2026-09-24  
> **Menu:** [experiment catalog](README.md) · **Metrics:** [semantic effect sizes](evaluations/semantic_dprime.md), [cross-modal structure](evaluations/cross_modal_structure.md)

Everything here is measured on the large-data phase. The point of the survey is
to put the training budget next to the result, so a weak number can be read as
"undertrained" or "actually worse" rather than guessed at.

## Training budgets

One epoch is:

| Corpus | Samples per epoch | Tokens per sample | **Tokens per epoch** |
|---|---:|---:|---:|
| Text (2M captions) | 2,000,000 | 91.7 mean content tokens | **183M** |
| Images (1.2M renders) | 1,200,000 | 384 (16 × 24 VQ grid) | **461M** |
| Old 100k paired set | 90,000 | 91.7 text + 384 image | 42.8M |

So, per run:

| Budget | Total tokens |
|---|---:|
| Text, 2 epochs | 0.37B |
| Text, 4 epochs | 0.73B |
| Text, 8 epochs | 1.47B |
| Image, 4 epochs | 1.84B |
| Image, 8 epochs | 3.69B |
| Old paired models, 70 epochs | 3.00B (text + image combined) |

**Two consequences worth holding onto.** Every image model has seen 2.5× the
tokens of a 4-epoch text model, so image results are not weaker for lack of
tokens. And the old 70-epoch models saw 3.0B tokens across both modalities —
more than any single new-phase run — on a dataset of only 90k scenes, so they
are heavily *repeated* rather than undertrained.

## Image models on the 1.2M-image set — the original four

All: 8 blocks, width 384, 6 heads, 16 × 24 VQ grid, batch 64 × 4, 4 epochs,
**1.84B tokens each**.

| Run | Parameterization | Objective | Val loss | Scene probe (L7) | best `rsa_scene` | CKA with text dense |
|---|---|---|---:|---:|---|---:|
| `image_dense_diffusion_1_2m_4e` | dense | diffusion | **3.6340** | **80.7%** | L6 **+0.178** | **0.322** |
| `image_lora_diffusion_1_2m_4e` | Tri-LoRA (shared 256 / private 128) | diffusion | 3.9252 | 75.1% | L5 +0.159 | 0.274 |
| `image_data2vec_scratch_block2d_30pct_avg_l4to7_1_2m_4e` | dense | data2vec averaged 4-7, 2D block masking, from scratch | 8.5352 † | 61.6% | L2 +0.110 | 0.186 |
| `image_data2vec_scratch_block2d_30pct_layerwise_l4to7_1_2m_4e` | dense | data2vec layerwise 4-7, 2D block masking, from scratch | 8.5394 † | 60.4% | embedding +0.109 | 0.139 |

† `diffusion.weight: 0`, so the output head never receives gradient; 8.70 is the
uniform-guess level over the 682-token joint vocabulary. Not a quality measure.

Two more image configs exist and were never trained
(`image_module_jepa_layerwise_gated_hsic_{matched,strong}_1_2m_4e`).

**Budget is not the explanation for the JEPA image models' weakness**: all four
runs saw the same 1.84B tokens. The dense/LoRA gap (3.6340 vs. 3.9252) also
reproduces the text result (1.0758 vs. 1.1638) at equal budget.

A loss-matched control is in progress: `image_lora_diffusion_1_2m_12e_continued`
resumes Tri-LoRA and trains until its validation loss reaches dense's 3.6340,
so the two can be probed at equal reconstruction rather than equal epochs.

## Master table: every text run, its token budget, and its semantics

Grouped by **token budget**, with the dense baseline included at each budget so
every comparison is like-for-like. `d_bind shared` / `private` are the best
shared-write and private-write features; dense models have no such split.

| Budget | Model | Design | Init | Epochs | **Tokens** | Val loss | d_sem (L7) | best d_bind | shared | private |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.73B | dense baseline | diffusion only | scratch | 4 | **0.73B** | 1.0758 | +0.38 | +0.244 | — | — |
| 0.73B | plain Tri-LoRA | diffusion only | scratch | 4 | **0.73B** | 1.1638 | +0.47 | +0.293 | +0.293 | +0.250 |
| 0.73B | sweep: avg, no HSIC | JEPA on shared writes, blocks 2-4 | scratch | 4 | **0.73B** | 1.1435 | -0.07 | +0.147 | +0.089 | +0.147 |
| 0.73B | sweep: avg + HSIC | JEPA on shared writes, blocks 2-4 | scratch | 4 | **0.73B** | 1.1613 | +0.14 | +0.108 | +0.082 | +0.108 |
| 0.73B | sweep: avg + HSIC (calibrated) | JEPA on shared writes, blocks 2-4 | scratch | 4 | **0.73B** | 1.1489 | +0.09 | +0.102 | +0.102 | +0.087 |
| 0.73B | sweep: layerwise, no HSIC | JEPA on shared writes, blocks 2-4 | scratch | 4 | **0.73B** | 1.1595 | +0.03 | +0.117 | +0.106 | +0.112 |
| 0.73B | sweep: layerwise + HSIC | JEPA on shared writes, blocks 2-4 | scratch | 4 | **0.73B** | 1.1677 | +0.05 | +0.224 | +0.174 | +0.222 |
| 0.73B | gated layerwise, no HSIC | gated gradient into shared A/B | scratch | 4 | **0.73B** | 1.1705 | -0.49 | +0.074 | +0.074 | +0.052 |
| 0.73B | gated layerwise + HSIC | gated gradient into shared A/B | scratch | 4 | **0.73B** | 1.1778 | +0.18 | +0.148 | +0.113 | +0.148 |
| 0.73B | gated layerwise + HSIC, all layers | gated, blocks 0-7 | scratch | 4 | **0.73B** | 1.1679 | -0.23 | +0.143 | +0.041 | +0.143 |
| 0.73B | gated data2vec, averaged | no predictor, blocks 2-4 | scratch | 4 | **0.73B** | 1.1571 | -0.02 | +0.111 | +0.084 | +0.102 |
| 0.73B | gated data2vec, per-layer | no predictor, blocks 2-4 | scratch | 4 | **0.73B** | 1.1932 | +0.39 | +0.179 | +0.137 | +0.162 |
| 0.73B | scratch d2v: avg, windows 12-24 | hidden-state data2vec, no diffusion | scratch | 4 | **0.73B** | 8.7964 | +0.42 | +0.198 | — | — |
| 0.73B | scratch d2v: avg, windows 4-8 | hidden-state data2vec, no diffusion | scratch | 4 | **0.73B** | 8.8497 | -0.53 | +0.248 | — | — |
| 0.73B | scratch d2v: layerwise, windows 12-24 | hidden-state data2vec, no diffusion | scratch | 4 | **0.73B** | 8.9335 | -1.72 | +0.085 | — | — |
| 0.73B | scratch d2v: layerwise, windows 4-8 | hidden-state data2vec, no diffusion | scratch | 4 | **0.73B** | 8.9406 | -1.27 | +0.081 | — | — |
| 0.73B | scratch d2v: layerwise all-8, random-t | hidden-state data2vec, no diffusion | scratch | 4 | **0.73B** | 8.9516 | -1.64 | +0.022 | — | — |
| 1.10B | dense baseline (6 epochs) | diffusion only | scratch | 6 | **1.10B** | 1.0520 | +0.44 | +0.443 | — | — |
| 1.10B | d2v hidden: averaged, all 8 | data2vec on hidden states | from dense | 4+2 | **1.10B** | 1.4488 | -0.76 | +0.293 | — | — |
| 1.10B | d2v hidden: averaged, 4-7 | data2vec on hidden states | from dense | 4+2 | **1.10B** | 1.5243 | +0.11 | +0.395 | — | — |
| 1.10B | d2v hidden: layerwise, 4-7 | data2vec on hidden states | from dense | 4+2 | **1.10B** | 1.9885 | +0.66 | +0.445 | — | — |
| 1.10B | d2v hidden: layerwise, all 8 | data2vec on hidden states | from dense | 4+2 | **1.10B** | 1.9656 | +0.75 | +0.527 | — | — |
| 1.47B | dense baseline (8 epochs) | diffusion only | scratch | 8 | **1.47B** | 1.0475 | +0.34 | +0.438 | — | — |
| 1.47B | stage 2: frozen trunk (avg) | diffusion + HSIC, private LoRA only | from d2v avg | 4+2+2 | **1.47B** | 1.0677 | -0.33 | +0.356 | +0.356 | +0.275 |
| 1.47B | stage 2: trainable trunk (avg) | diffusion + HSIC, whole model | from d2v avg | 4+2+2 | **1.47B** | 1.0648 | +0.20 | +0.426 | +0.426 | +0.296 |
| 1.47B | stage 2: frozen trunk (layerwise) | diffusion + HSIC, private LoRA only | from d2v layerwise | 4+2+2 | **1.47B** | 1.0617 | +0.41 | +0.508 | +0.508 | +0.273 |

† The `8.79-8.95` validation losses belong to runs with `diffusion.weight: 0`:
their output head never receives gradient, so 8.70 is uniform guessing over the
682-token joint vocabulary. The `1.45-1.99` ones measure how far a data2vec
phase drifted the trunk away from a frozen head. Neither is a quality measure.

### What the budget-matched comparison changes

Earlier write-ups in this catalog compared JEPA models against the **4-epoch**
dense baseline, whose best binding effect size is +0.244. That baseline is
simply undertrained as a reference: dense reaches **+0.443 at 6 epochs** and
plateaus at +0.438 by 8. Claims of the form "JEPA more than doubles dense's
binding" came from that mismatch and are withdrawn.

At matched budget:

* **0.73B** — every JEPA variant is *below* both dense (+0.244) and plain
  Tri-LoRA (+0.293). The best of them, `sweep: layerwise + HSIC`, reaches +0.224.
  At this budget the JEPA objective costs more than it buys, and plain Tri-LoRA
  is the strongest model.
* **1.10B** — only `d2v hidden: layerwise, all 8` beats dense on both axes
  (+0.75 vs. +0.44 semantic, +0.527 vs. +0.443 binding). `layerwise 4-7` merely
  ties dense (+0.445 vs. +0.443); both averaged variants lose.
* **1.47B** — only `stage 2: frozen trunk (layerwise)` beats dense (+0.41 vs.
  +0.34, +0.508 vs. +0.438). The other two stage-2 runs fall below dense.

**The surviving claim is narrow and specific**: layerwise data2vec over all
eight blocks, applied to an already-trained dense trunk, beats a budget-matched
dense model by roughly 15-20% on binding and by a wider margin on the semantic
axis — and carries that advantage into stage 2. Every other JEPA configuration
in this study is at or below dense at its own budget.

## The compute-matched text dense control — yes, it exists

`text_dense_diffusion_2m_8e_continued` resumes the dense baseline (model,
optimizer and step counter) and trains to **8 epochs = 1.47B tokens**, exactly
matching stage 2's budget.

| Run | Epochs | Tokens | Val loss |
|---|---:|---:|---:|
| `text_dense_diffusion_2m_4e_matched` | 4 | 0.73B | 1.0758 |
| **`text_dense_diffusion_2m_8e_continued`** | **8** | **1.47B** | 1.0713 → 1.0520 → 1.0555 → **1.0475** |
| `stage2_frozen` (private LoRA only) | 4+2+2 | 1.47B | **1.0433** (at 7 epochs) / 1.0677 |
| `stage2_lwall` | 4+2+2 | 1.47B | 1.0440 / 1.0617 |

At matched tokens the gap is **0.004**, not the 0.032 the four-epoch comparison
suggested. The claim that a rank-128 private branch beats dense does not
survive; what survives is that it **matches** dense while updating 12.58M of
27.45M parameters.

One caveat on "matched": tokens are matched, gradient signal is not. Of stage 2's
eight epochs, two ran with `diffusion.weight: 0` (no reconstruction signal at
all) and two updated only the private adapters, while the control trained all
14.87M dense parameters for all eight epochs. If anything that favours the
control.

## Cross-modal agreement, JEPA against JEPA

CKA and RSA between a text model's caption representations and an image model's
VQ-token representations of the same 4,000 scenes. Dense excluded on both sides
in the first block.

| Text side | Image side | CKA (L7) | RSA (L7) | best CKA |
|---|---|---:|---:|---|
| `d2v_lw_all` (from dense) | `image_jepa_avg` | **0.200** | 0.176 | L1 0.266 |
| `d2v_lw_all` | `image_jepa_lw` | 0.149 | 0.111 | L1 0.252 |
| `scratch_w12_24_avg` | `image_jepa_avg` | 0.148 | 0.154 | L1 0.291 |
| `scratch_w12_24_avg` | `image_jepa_lw` | 0.114 | 0.094 | L1 0.276 |
| `scratch_lw_randt` | `image_jepa_avg` | 0.061 | 0.091 | emb 0.149 |
| `scratch_lw_randt` | `image_jepa_lw` | 0.045 | 0.043 | emb 0.147 |

With dense on one side or the other:

| Text side | Image side | CKA (L7) | RSA (L7) |
|---|---|---:|---:|
| `text_dense` | `image_dense` | 0.322 | 0.371 |
| `text_d2v_lw_all` | `image_dense` | 0.349 | 0.381 |
| **`text_stage2_frozen`** | `image_dense` | **0.388** | **0.406** |
| `text_stage2_frozen` | `image_lora` | 0.328 | 0.380 |
| `text_dense` | `image_jepa_avg` | 0.186 | 0.170 |
| `text_dense` | `image_jepa_lw` | 0.139 | 0.106 |

1. **Two JEPA models agree *less* with each other, not more.** Every JEPA↔JEPA
   pair (0.045–0.200) is below the dense↔dense baseline of 0.322. The JEPA
   objective does not impose a common cross-modal structure.
2. **The effect is asymmetric.** On the text side JEPA *helps*: `d2v_lw_all` with
   image dense reaches 0.349 and stage-2 text reaches **0.388/0.406**, both above
   dense↔dense. On the image side JEPA hurts: substituting an image JEPA model
   drops any pairing to 0.15–0.20. The bottleneck is the image JEPA models, whose
   scene probes are 61.6% and 60.4% against image dense's 80.7% — at the *same*
   1.84B-token budget.
3. **JEPA↔JEPA agreement lives in the early layers.** Every such pair peaks at L1
   or the embedding, while dense-involving pairs peak at L7. Whatever
   from-scratch JEPA builds, it stops building it after the first blocks — the
   same shape as those models' own `rsa_scene` peaks.

## Per-model scene content, both modalities

| Model | modality | L7 probe | best `rsa_scene` |
|---|---|---:|---|
| `text_dense` | text | **91.2%** | L6 +0.275 |
| `text_d2v_lw_all` | text | 89.4% | L6 +0.224 |
| `text_stage2_frozen` | text | 88.8% | L5 **+0.295** |
| `text_jepa_scratch_w12_24_avg` | text | 85.8% | L5 **+0.299** |
| `text_jepa_scratch_lw_randt` | text | 62.8% | embedding +0.135 |
| `image_dense` | image | 80.7% | L6 +0.178 |
| `image_lora` | image | 75.1% | L5 +0.159 |
| `image_jepa_avg` | image | 61.6% | L2 +0.110 |
| `image_jepa_lw` | image | 60.4% | embedding +0.109 |

The from-scratch layerwise text model (62.8%, best at the embedding) and the two
image JEPA models (61.6%, 60.4%, best at L2 and the embedding) share one
signature: layerwise-from-scratch fails the same way in both modalities.

## SIGReg / LeJEPA

[LeJEPA](https://arxiv.org/abs/2511.08544) argues that a JEPA needs no EMA
teacher, no stop-gradient and no target normalization: collapse is prevented
instead by pushing the embedding distribution towards an isotropic Gaussian,
tested with a **sliced Epps–Pulley** statistic. We implemented it and ran it in
place of the EMA teacher, keeping everything else — layerwise targets, 8 MLP
predictor heads, the same data and budget.

**Objective.** With `z` the student embeddings and `ẑ` the targets,

```
L = (1 − λ) · L_pred(pred(z_masked), ẑ)  +  λ · SIGReg(z)

SIGReg(z) = mean over 1024 random unit directions u of
            EppsPulley( ⟨z, u⟩ ,  N(0,1) ),  evaluated at 17 points
```

`λ = 0.05` in all four runs. The teacher, the stop-gradient and the target
LayerNorm are all switched off (`data2vec_teacherless`), so the targets are the
model's own live activations.

### Results — all four collapse

| Run | Mod | From | Budget | `d_sem` L7 | best `d_bind` | L7 CKA | text probe | image probe |
|---|---|---|---:|---:|---|---:|---:|---:|
| `text_lejepa_from_dense_layerwise_all_2m_2e` | text | text dense | 1.10B | **−2.68** | +0.013 | — | — | — |
| `text_lejepa_scratch_layerwise_all_2m_4e` | text | scratch | 0.73B | −0.82 | +0.016 | — | — | — |
| `multimodal_unpaired_d2v_from_unpaired_dense_sigreg_1_2m_2e` | both | unpaired dense | 3.43B | −2.14 | +0.022 | 0.023 | 55.7% | 57.2% |
| `multimodal_unpaired_lejepa_scratch_1_2m_4e` | both | scratch | 2.28B | — | — | 0.018 | 54.7% | 53.5% |

Reference points: untrained model +0.004 `d_bind`, bag of words −2.53
`d_semantic`, EMA-teacher layerwise from dense **+0.527** `d_bind`.

**What happened, precisely.** The per-layer CKA profile
([modality alignment](evaluations/modality_alignment.md#per-layer-cka-the-depth-profile))
shows the mechanism: the embedding layer is intact (CKA 0.157 / 0.188, normal
for this study) and then **the first transformer block drops CKA to ~0.02 and
every later block holds it there**. This is not gradual degradation; the
representation is switched off at the first opportunity and never recovers.

**The one thing SIGReg did do**, and it is worth recording: it is the only
mechanism in the study that closed the modality gap. `lejepa_scratch_both`
reaches a gap of 0.050 with AUC 0.576 — genuinely overlapping caption and image
distributions, which no other model came close to. It achieves that by emptying
both representations rather than by finding shared content (CKA 0.018, RSA
−0.008), which is a clean demonstration that **distributional overlap and shared
structure are different goals**.

**What is not yet known.** λ was fixed at 0.05 and never swept, and the
isotropy penalty was applied to the pooled embedding rather than per-token. A λ
sweep and an image-only SIGReg run are the obvious follow-ups; on the present
evidence SIGReg is not a drop-in replacement for the EMA teacher in this setup.

## The multimodal family: paired vs unpaired

Seven runs on the 1.2M paired corpus, one model encoding both modalities.
`unpaired` uses a derangement so a caption is never carried with its own image.

| Run | From | Budget | val | text probe | image probe | image bind | L7 CKA | best CKA |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **`multimodal_paired_dense_1_2m_4e`** | scratch | 2.28B | **2.2596** | **86.0%** | **82.4%** | 0.792 | **0.554** | **L6 0.622** |
| `multimodal_unpaired_dense_1_2m_4e` | scratch | 2.28B | 2.1990 | 73.2% | 79.8% | — | 0.187 | L7 0.187 |
| `multimodal_unpaired_d2v_from_text_dense_1_2m_2e` | text dense | 1.88B | 9.48 | **89.9%** | 59.3% | — | 0.061 | L1 0.207 |
| `multimodal_unpaired_d2v_from_image_dense_1_2m_2e` | image dense | 3.45B | 7.17 | 58.6% | 80.4% | — | 0.182 | L7 0.182 |
| `multimodal_unpaired_d2v_from_unpaired_dense_ema_1_2m_2e` | unpaired dense | 3.43B | 3.2485 | 68.1% | 78.7% | — | 0.116 | L1 0.161 |
| `multimodal_unpaired_d2v_from_unpaired_dense_sigreg_1_2m_2e` | unpaired dense | 3.43B | 8.4424 | 55.7% | 57.2% | — | 0.023 | emb 0.157 |
| `multimodal_unpaired_lejepa_scratch_1_2m_4e` | scratch | 2.28B | 8.7888 | 54.7% | 53.5% | — | 0.018 | emb 0.188 |

`val` is only comparable between the two dense runs; the rest are `no diffusion`.

1. **Paired dense is the reference and nothing is close.** CKA 0.554 at L7 and
   **0.622 at L6**, against 0.187 for the best unpaired model — a 3× gap. It is
   also the only model in the entire study whose CKA *climbs* with depth. Its
   `val` is slightly worse than the unpaired model's (2.2596 vs 2.1990) because
   paired sequences are harder to denoise, so the generative loss and the
   representation quality point in opposite directions here.
2. **The unpaired dense trunk has no text binding at all** (`d_bind` +0.007,
   the untrained floor) despite a 73.2% text probe. Interleaved unpaired batches
   give it attribute presence and no conjunctions. Text-only dense on a
   *smaller* budget gets +0.443.
3. **JEPA inherits its parent's modality.** From the text trunk it reaches the
   best text probe in the family (89.9%) and the best text `d_bind` (+0.381)
   while its image probe falls to 59.3% and its CKA declines monotonically to
   0.061. From the image trunk the mirror happens. Neither produces a model that
   is good at both.
4. **The decisive experiment has not been run.** *Unpaired stage 2* — the dense
   unpaired trunk as the shared route plus a per-modality private LoRA, exactly
   the design that produced the best text model — would test the shared/private
   hypothesis on the multimodal data it was formulated for. See
   [the plan](plans/dense_shared_private_lora_plan.md).
