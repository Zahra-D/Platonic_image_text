# Multimodal study (1.2M paired corpus): run registry

> **Type:** model registry · **Status:** 7 finished · **Updated:** 2026-09-24
> **Menu:** [experiment catalog](../README.md)

Seven runs in which **one model encodes both modalities**. Same architecture as
the text and image studies — 8 blocks, width 384, 6 heads, no modality
embeddings — on the 1.2M-scene paired corpus (`train_pairs_human.jsonl`), where
each scene exists as a caption (91.7 mean content tokens) and as a 16×24 VQ grid
(384 codes). One epoch is therefore **571M tokens**; batch 32 × 8 accumulation.

The two data modes:

- **paired** — a caption and *its own* image occupy the same training sequence,
  so the model can use one to predict the other.
- **unpaired** — a derangement over the batch guarantees a caption is never
  carried with its own image. The model sees both modalities and never sees a
  correspondence.

That single difference is the study's most important comparison, because it
isolates what correspondence supervision buys.

## Runs

| Run | Mode | Objective | From | Ep | Tokens | Val |
|---|---|---|---|---:|---:|---:|
| `multimodal_paired_dense_1_2m_4e` | paired | diffusion, dense | scratch | 4 | 2.28B | **2.2596** |
| `multimodal_unpaired_dense_1_2m_4e` | unpaired | diffusion, dense | scratch | 4 | 2.28B | 2.1990 |
| `multimodal_unpaired_d2v_from_text_dense_1_2m_2e` | unpaired | data2vec layerwise ×8, EMA | text dense | 2 | 1.88B | 9.4818 † |
| `multimodal_unpaired_d2v_from_image_dense_1_2m_2e` | unpaired | data2vec layerwise ×8, EMA | image dense | 2 | 3.45B | 7.1726 † |
| `multimodal_unpaired_d2v_from_unpaired_dense_ema_1_2m_2e` | unpaired | data2vec layerwise ×8, EMA | unpaired dense | 2 | 3.43B | 3.2485 † |
| `multimodal_unpaired_d2v_from_unpaired_dense_sigreg_1_2m_2e` | unpaired | data2vec layerwise ×8, **SIGReg** λ=0.05 | unpaired dense | 2 | 3.43B | 8.4424 † |
| `multimodal_unpaired_lejepa_scratch_1_2m_4e` | unpaired | data2vec layerwise ×8, **SIGReg** λ=0.05 | scratch | 4 | 2.28B | 8.7888 † |

† `diffusion.weight: 0` — the output head gets no gradient, so this number is
drift from a frozen head, not representation quality. Uniform guessing over the
joint vocabulary is ≈ 8.8.

All JEPA runs use the configuration that worked on text: **layerwise targets,
one MLP predictor head per block (8 heads, 4.72M parameters), EMA teacher,
SmoothL1 β = 2.0, target LayerNorm, stop-gradient** — except the two SIGReg runs,
which drop the teacher, the stop-gradient and the target LayerNorm in favour of
the sliced Epps–Pulley isotropy penalty.

## Results

| Run | text probe | image probe | text `d_bind` | L7 CKA | best CKA | gap | image bind |
|---|---:|---:|---:|---:|---|---:|---:|
| **`multimodal_paired_dense_1_2m_4e`** | **86.0%** | **82.4%** | — | **0.554** | **L6 0.622** | 1.275 | 0.792 |
| `multimodal_unpaired_dense_1_2m_4e` | 73.2% | 79.8% | +0.007 | 0.187 | L7 0.187 | 1.561 | — |
| `..._d2v_from_text_dense` | **89.9%** | 59.3% | **+0.381** | 0.061 | L1 0.207 | 1.331 | — |
| `..._d2v_from_image_dense` | 58.6% | 80.3% | +0.010 | 0.182 | L7 0.182 | 0.540 | — |
| `..._d2v_from_unpaired_dense_ema` | 68.1% | 78.7% | +0.021 | 0.116 | L1 0.161 | 1.096 | — |
| `..._d2v_from_unpaired_dense_sigreg` | 55.7% | 57.2% | +0.022 | 0.023 | emb 0.157 | 0.053 | — |
| `..._lejepa_scratch` | 54.7% | 53.5% | +0.016 | 0.018 | emb 0.188 | 0.050 | — |

`text d_bind` is the best feature's binding effect size
([semantic d′](../evaluations/semantic_dprime.md)); the untrained floor is
+0.004 and text-only dense at a comparable budget reaches +0.443.
Per-layer CKA for every row is in
[modality alignment](../evaluations/modality_alignment.md#per-layer-cka-the-depth-profile).

## What this family established

**1. Correspondence supervision is the only thing that produced cross-modal
structure.** Paired dense reaches CKA 0.554 at L7 and 0.622 at L6 — three times
the best unpaired model, and it is the *only* model in the whole study whose CKA
climbs with depth (0.13 → 0.62). No self-supervised objective on unpaired data
came within 3× of it.

**2. The generative loss points the other way.** The unpaired dense model has the
*better* validation loss (2.1990 vs 2.2596) and by far the worse representation.
Paired sequences are harder to denoise; being good at the training objective and
being good at the representation are different things here, and this pair is the
cleanest demonstration of it in the study.

**3. Unpaired dense has essentially no text binding.** `d_bind` +0.007, against
the untrained floor of +0.004, despite a 73.2% text probe. It learns which
attributes are present and not which object has them. Text-only dense on a
*smaller* budget gets +0.443. Interleaving unpaired image batches appears to cost
the text side its conjunction structure.

**4. JEPA inherits its parent's modality and specializes further.** From the text
trunk: best text probe in the family (89.9%), best text binding (+0.381),
image probe down to 59.3%, and CKA declining monotonically with depth to 0.061.
From the image trunk: the mirror image (80.3% / 58.6%, CKA held at 0.182). From
the unpaired trunk, the EMA variant lands between them and the SIGReg variant
collapses. **No configuration produced a model that is good at both.**

**5. SIGReg closes the modality gap by emptying the representation.** Gaps of
0.053 and 0.050 — the only genuinely overlapping modality distributions measured
anywhere — with CKA 0.023 and 0.018 and probes at ~55%. See
[the survey](../jepa_model_survey.md#sigreg--lejepa).

## The experiment this family is missing

**Unpaired stage 2 has never been run.** Every finding above says the shared/
private hypothesis should be tested here: on text, the best model in the study is
a dense trunk used as the shared route with a rank-128 private LoRA
([stage 2](../plans/dense_shared_private_lora_plan.md), `d_bind` +0.555
trunk-only). The multimodal analogue — the unpaired dense trunk as the shared
route, one private LoRA per modality — is the design the hypothesis was actually
formulated for, and it is the one cell of the grid that is empty.

The other open follow-ups: a SIGReg λ sweep (λ was fixed at 0.05 and never
varied), and a paired JEPA run, since every JEPA run here is unpaired.

## Related

- [Modality alignment](../evaluations/modality_alignment.md) — gap, AUC, paired R@1, per-layer CKA
- [Cross-modal structure](../evaluations/cross_modal_structure.md) — probes and RSA
- [Image binding](../evaluations/image_binding.md) — the image-side binding metric
- [Text study registry](text_study_run_registry.md) · [Image study registry](image_study_run_registry.md)
- The earlier 90k models: [dense_paired](dense_paired.md), [dense_unpaired](dense_unpaired.md)
