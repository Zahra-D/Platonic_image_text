# Cross-modal structure: do independently trained text and image models agree?

> **Type:** evaluation · **Status:** 26 models, 20 cross-modal pairs · **Updated:** 2026-09-24  
> **Menu:** [experiment catalog](../README.md) · **Code:** `evaluate_cross_modal_structure.py`

## Why this exists

Two gaps. First, the image runs had **no** semantic evaluation at all — hard
retrieval and binding swap both construct caption minimal pairs, so neither can
score an image checkpoint. Second, the project's premise is that scene
semantics are shared across modalities, and nothing measured that directly.

Both are answered on one set of scenes: 4,000 held-out CLEVR worlds, each
contributing its cached VQ image tokens **and** a caption rendered from the same
`world` JSON, so a text model and an image model are measured on identical
content.

## What is measured

**Per model — how much of the scene is in the representation.**

* `probe_accuracy`: a linear probe (the [binding-swap](binding_swap_evaluation.md)
  fitter) trained on 3,000 scenes to read the scene's conjunction facts out of
  the pooled representation, balanced accuracy on 1,000 held-out scenes.
* `rsa_scene`: Spearman correlation between representation cosine distance and
  scene-graph distance (symmetric difference of the two scenes' fact sets),
  over 200,000 sampled pairs. Graded and without a ceiling: it asks whether the
  geometry is *organized* by scene content.
* `effective_rank` and `mean_pairwise_cosine`, so a high score cannot come from
  a collapsed or degenerate spread.

**Across modalities — do the two models agree?** The two representations live in
unrelated 384-dimensional spaces, so only rotation-invariant comparisons are
meaningful:

* `cka`: linear centered kernel alignment between the two models'
  representations of the same scenes;
* `rsa_cross`: Spearman between their pairwise-distance structures;
* each side's effective rank, since "the same distribution" also implies a
  comparable shape.

A direct distance, MMD or cosine between the two spaces would be meaningless:
any rotation of one space changes it without changing the model.

## Results

### Per-model scene content

| Model | modality | L7 probe | L7 rsa | best rsa_scene |
|---|---|---:|---:|---|
| `paired_dense_IMAGE` | image | 82.4% | +0.241 | `L6` +0.251 |
| `image_d2v_fromdense_block2d` | image | 81.2% | +0.190 | `L7` +0.190 |
| `image_d2v_fromdense_randt` | image | 81.1% | +0.185 | `L7` +0.185 |
| `image_d2v_lw_l4to7` | image | 81.0% | +0.181 | `L7` +0.181 |
| `image_dense` | image | 80.7% | +0.104 | `L6` +0.178 |
| `from_image_IMAGE` | image | 80.4% | +0.186 | `L7` +0.186 |
| `unpaired_dense_IMAGE` | image | 79.8% | +0.092 | `L6` +0.184 |
| `jepa_ema_IMAGE` | image | 78.7% | +0.150 | `L7` +0.150 |
| `image_d2v_avg_l4to7` | image | 78.5% | +0.158 | `L7` +0.158 |
| `image_d2v_avg_all` | image | 78.2% | +0.139 | `L6` +0.150 |
| `image_lora_lossmatched` | image | 77.8% | +0.084 | `L5` +0.152 |
| `image_lora` | image | 75.1% | +0.076 | `L5` +0.159 |
| `image_jepa_avg` | image | 61.6% | +0.095 | `L2` +0.110 |
| `image_jepa_lw` | image | 60.4% | +0.096 | `embedding` +0.109 |
| `from_text_IMAGE` | image | 59.3% | +0.020 | `embedding` +0.098 |
| `jepa_sigreg_IMAGE` | image | 57.2% | +0.059 | `embedding` +0.116 |
| `lejepa_scratch_IMAGE` | image | 53.5% | +0.020 | `embedding` +0.091 |
| `text_dense` | text | 91.2% | +0.186 | `L6` +0.275 |
| `from_text_TEXT` | text | 89.9% | +0.221 | `L6` +0.245 |
| `text_d2v_lw_all` | text | 89.4% | +0.194 | `L6` +0.224 |
| `text_jepa_d2v_lw_all` | text | 89.4% | +0.194 | `L6` +0.224 |
| `text_stage2_frozen` | text | 88.8% | +0.156 | `L5` +0.295 |
| `paired_dense_TEXT` | text | 86.0% | +0.191 | `L5` +0.205 |
| `text_jepa_scratch_w12_24_avg` | text | 85.8% | +0.224 | `L5` +0.299 |
| `unpaired_dense_TEXT` | text | 73.1% | +0.070 | `embedding` +0.152 |
| `jepa_ema_TEXT` | text | 68.1% | +0.052 | `embedding` +0.141 |
| `text_jepa_scratch_lw_randt` | text | 62.8% | +0.014 | `embedding` +0.135 |
| `from_image_TEXT` | text | 58.6% | +0.000 | `embedding` +0.134 |
| `jepa_sigreg_TEXT` | text | 55.7% | +0.014 | `embedding` +0.145 |
| `lejepa_scratch_TEXT` | text | 54.7% | +0.008 | `embedding` +0.137 |

> **The scene probe is no longer the headline image measure.** It spreads the
> image models over 3 points (79.2 – 82.4%) because attribute *presence* is easy
> and every model gets it right. [Image binding](image_binding.md), built from
> content-matched scene pairs, spreads the same models over 29 points with a
> principled 50% floor, a bootstrap CI and per-pair scores. Read the probe as a
> sanity check — "is there anything in there at all" — and the binding number as
> the result. The two disagree in one informative place: `paired_dense_IMAGE` has
> the *best* probe here (82.4%) and slightly *worse* binding than image-only
> dense (0.792 vs 0.802).
>
> Per-layer CKA for every model, rather than L7 alone, is tabulated in
> [modality alignment](modality_alignment.md#per-layer-cka-the-depth-profile).

### Cross-modal agreement at L7

| Text side | Image side | CKA | RSA |
|---|---|---:|---:|
| `text_stage2_frozen` | `image_dense` | 0.388 | +0.406 |
| `text_d2v_lw_all` | `image_dense` | 0.349 | +0.381 |
| `text_stage2_frozen` | `image_lora` | 0.328 | +0.380 |
| `text_dense` | `image_dense` | 0.322 | +0.371 |
| `text_d2v_lw_all` | `image_d2v_fromdense_block2d` | 0.318 | +0.353 |
| `text_dense` | `image_lora_lossmatched` | 0.309 | +0.370 |
| `text_d2v_lw_all` | `image_d2v_fromdense_randt` | 0.309 | +0.311 |
| `text_dense` | `image_d2v_fromdense_block2d` | 0.297 | +0.350 |
| `text_d2v_lw_all` | `image_lora` | 0.296 | +0.359 |
| `text_dense` | `image_d2v_fromdense_randt` | 0.286 | +0.306 |
| `text_dense` | `image_d2v_lw_l4to7` | 0.281 | +0.302 |
| `text_dense` | `image_d2v_avg_all` | 0.279 | +0.231 |
| `text_dense` | `image_d2v_avg_l4to7` | 0.276 | +0.211 |
| `text_dense` | `image_lora` | 0.274 | +0.351 |
| `text_stage2_frozen` | `image_jepa_avg` | 0.215 | +0.194 |
| `text_d2v_lw_all` | `image_jepa_avg` | 0.200 | +0.176 |
| `text_jepa_d2v_lw_all` | `image_jepa_avg` | 0.200 | +0.176 |
| `text_dense` | `image_jepa_avg` | 0.186 | +0.170 |
| `text_stage2_frozen` | `image_jepa_lw` | 0.160 | +0.124 |
| `text_d2v_lw_all` | `image_jepa_lw` | 0.149 | +0.111 |
| `text_jepa_d2v_lw_all` | `image_jepa_lw` | 0.149 | +0.111 |
| `text_jepa_scratch_w12_24_avg` | `image_jepa_avg` | 0.148 | +0.154 |
| `text_dense` | `image_jepa_lw` | 0.139 | +0.106 |
| `text_jepa_scratch_w12_24_avg` | `image_jepa_lw` | 0.114 | +0.094 |
| `text_jepa_scratch_lw_randt` | `image_jepa_avg` | 0.061 | +0.091 |
| `text_jepa_scratch_lw_randt` | `image_jepa_lw` | 0.045 | +0.043 |

## What the numbers say

0. **A paired model is the reference.** `paired_dense`, the only model trained
   with caption and image in one sequence, reaches the highest scene content on
   both sides at once -- 86.0% text and **82.4%** image, with image `rsa_scene`
   **+0.251**, above every image-only model including image dense (+0.178). It is
   also the target for cross-modal agreement; see
   [modality alignment](modality_alignment.md), where its CKA of 0.554 is the
   highest measured.

1. **Two models that never shared a parameter partly agree.** Dense text and
   dense image, trained on different modalities of the same scenes with no
   alignment objective, reach CKA 0.322 and RSA +0.371 at the last block, against
   ~0 for unrelated representations. This is the platonic claim, measured. It is
   also only partial: two models of the *same* modality typically reach 0.8–0.9,
   so these share roughly a third of their structure.
2. **Agreement grows with depth and jumps at the end.** CKA climbs 0.145 →
   0.322 from the embedding to L7, and RSA nearly doubles at the last block
   (0.204 → 0.371). The embedding row is the lowest, so this is not an artifact
   of the input tokens.
3. **Both sides compress to a similar degree.** Effective rank at L7 is 23.2
   (text) and 19.1 (image) out of 384. They differ sharply only at the embedding
   (34.2 vs. 83.8), which follows from 682 image codes against 170 word types.
4. **The plain diffusion models agree most.** Every JEPA or LoRA variant is lower
   at every layer — image LoRA 0.274, image JEPA averaged 0.186, image JEPA
   layerwise 0.139 at L7. On current evidence the plain objective produces the
   more cross-modally aligned representation.
5. **From a pretrained trunk, image JEPA works; from scratch it does not.** The
   from-dense image runs reach 81.0-81.2% with their best structure at the
   deepest block, slightly above the dense trunk they started from (80.7%,
   +0.178). Layerwise beats averaged there exactly as on text (81.0-81.2 against
   78.2-78.5), so the target-design result holds in both modalities. The
   from-scratch runs are the ones that fail.

6. **Every SIGReg run lands at the bottom of the image table** (53.5-57.2%) with
   its best structure at the embedding -- the same signature as the other
   collapses. See [modality alignment](modality_alignment.md) for why their
   near-zero modality gap is not the good news it appears to be.

7. **The image JEPA runs from scratch did not build scene structure.** Their probe accuracy is
   61.6% and 60.4% against image dense's 80.7%, and their best `rsa_scene` sits
   at L2 or even the embedding rather than deep in the network. This mirrors the
   from-scratch text result: data2vec from a random initialization does not
   organize a representation around scene content, in either modality.

## Limits

* The image models trained on 1.2M images while the text models saw 2M captions,
  and the image JEPA runs are from scratch while the text models had a full
  diffusion objective — so "dense agrees most" is partly "the two best-trained
  models agree most".
* One caption per scene, so RSA measures scene-level structure, not sensitivity
  to wording; that is what [`d_semantic`](semantic_dprime.md) is for.
* CKA and RSA establish that the *relational structure* corresponds. Whether the
  two point clouds occupy the same region has no rotation-invariant answer for
  independently trained models; it needs a model that shares parameters across
  modalities, which is what the shared-route design is for.

## Reproduce

```bash
python evaluate_cross_modal_structure.py --num-scenes 4000 \
  --output-dir outputs/cross_modal_structure \
  --text-checkpoint text_dense=outputs/text_dense_diffusion_2m_4e_matched/epoch_003.pt \
  --image-checkpoint image_dense=outputs/image_dense_diffusion_1_2m_4e/epoch_003.pt
```

Files: one JSON per model with the per-layer semantic metrics, and
`cross_modal.json` with every text/image pair's CKA, RSA and effective ranks.

## Does JEPA make the two modalities more similar? Depth decides

The sharpest version of the question: take the best text JEPA model and the
matching image JEPA model (both layerwise data2vec over all 8 blocks, both
started from their modality's dense checkpoint, both random-t masking) and
compare them against the two dense models they descend from.

| Layer | CKA dense | CKA JEPA | Δ | RSA dense | RSA JEPA | Δ |
|---|---:|---:|---:|---:|---:|---:|
| embedding | 0.145 | 0.159 | +0.014 | 0.057 | 0.071 | +0.014 |
| L0 | 0.182 | **0.242** | **+0.060** | 0.149 | 0.181 | +0.032 |
| L1 | 0.190 | **0.253** | **+0.063** | 0.132 | 0.162 | +0.030 |
| L2 | 0.216 | 0.252 | +0.036 | 0.156 | 0.191 | +0.035 |
| L3 | 0.196 | 0.232 | +0.036 | 0.133 | 0.163 | +0.030 |
| L4 | 0.200 | 0.237 | +0.037 | 0.134 | **0.183** | **+0.049** |
| L5 | 0.251 | 0.253 | +0.001 | 0.173 | 0.193 | +0.019 |
| L6 | 0.278 | 0.267 | −0.011 | 0.204 | 0.210 | +0.006 |
| **L7** | **0.322** | 0.309 | −0.013 | **0.371** | 0.311 | **−0.061** |

**JEPA raises cross-modal similarity through most of the network and lowers it
at the top.** Blocks 0-1 gain about a third in CKA (+0.060, +0.063) and RSA
rises at every layer from the embedding through L6 -- but L7, where both pairs
peak, goes the other way, and on RSA by a clear margin. The best cross-modal
readout available therefore still belongs to the dense pair.

The effect is a flattened depth profile: dense climbs 0.145 -> 0.322 from the
embedding to L7, while JEPA starts higher and ends slightly lower. One
structural hint at the cause: effective rank at L7 moves from 23.2 text / 19.1
image in the dense pair to 20.1 text / **30.3** image in the JEPA pair, so the
JEPA image model spreads over a much wider subspace at exactly the layer where
agreement drops.
