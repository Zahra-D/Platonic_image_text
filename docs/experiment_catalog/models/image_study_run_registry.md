# Image study: run registry

> **Type:** model registry · **Status:** 10 finished, 2 running · **Updated:** 2026-09-24  
> **Menu:** [experiment catalog](../README.md)

Same architecture as the text study — 8 blocks, width 384, 6 heads, no modality
embeddings — trained on 1.2M rendered CLEVR images tokenized to a 16×24 VQ grid
(`outputs/image_only_1_2m_token_cache`), microbatch 64 × 4 accumulation, seed
20260921. Validation is masked-token loss at fixed t = 0.75 on 2,048 held-out
images.

## Runs

| Run | Objective | Epochs | Val loss (final) | Config |
|---|---|---:|---:|---|
| `image_dense_diffusion_1_2m_4e` | diffusion, dense | 4 | **3.6340** | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/image_dense_diffusion_1_2m_4e.yaml) |
| `image_lora_diffusion_1_2m_4e` | diffusion, Tri-LoRA | 4 | 3.9252 | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/image_lora_diffusion_1_2m_4e.yaml) |
| `image_data2vec_scratch_block2d_30pct_avg_l4to7_1_2m_4e` | data2vec averaged 4-7, from scratch | 4 | 8.5352 † | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/image_data2vec_scratch_block2d_30pct_avg_l4to7_1_2m_4e.yaml) |
| `image_data2vec_scratch_block2d_30pct_layerwise_l4to7_1_2m_4e` | data2vec layerwise 4-7, from scratch | 4 | 8.5394 † | [yaml](/home/zd25e122/clevr_discrete_diffusion/configs/image_data2vec_scratch_block2d_30pct_layerwise_l4to7_1_2m_4e.yaml) |

† `diffusion.weight: 0`, so the output head never receives gradient and this
number is the uniform-guess level over the 682-token joint vocabulary
(ln(682)/0.75 = 8.70). It says nothing about the representation.

**The dense-versus-LoRA gap reproduces the text result exactly**: 3.6340 vs.
3.9252 on images, 1.0758 vs. 1.1638 on text. Two adapters whose ranks sum to the
dense rank optimize worse than the dense weight in both modalities.

## data2vec from the dense image model

The from-scratch image JEPA runs were weak (scene probe 61%), matching what
from-scratch data2vec did on text. These repeat the configuration that *did*
work on text -- layerwise, all 8 blocks, each predicting its own EMA target --
starting from the trained dense image checkpoint (validation 3.6340).

| Run | Masking | L7 scene probe | best `rsa_scene` | CKA with text dense |
|---|---|---:|---|---:|
| `image_dense_diffusion_1_2m_4e` (the trunk they start from) | — | 80.7% | L6 +0.178 | **0.322** |
| `image_data2vec_from_dense_layerwise_all_randt_1_2m_2e` | random-t | **81.1%** | **L7 +0.185** | 0.286 |
| `image_data2vec_from_dense_layerwise_all_block2d_1_2m_2e` | 2D blocks | **81.2%** | **L7 +0.190** | 0.297 |
| `image_data2vec_scratch_block2d_..._avg_l4to7` | 2D blocks | 61.6% | L2 +0.110 | 0.186 |
| `image_data2vec_scratch_block2d_..._layerwise_l4to7` | 2D blocks | 60.4% | embedding +0.109 | 0.139 |

1. **The pretrained-trunk requirement holds in images.** 81% from dense against
   61% from scratch, and the most scene-organized layer moves back to the
   deepest block (L7) instead of sitting at L2 or the embedding.
2. **The gain over the trunk itself is small**: +0.4 points of probe accuracy
   and +0.007 `rsa_scene`, with the peak moving L6 -> L7.
3. **Masking style barely matters** (2D blocks +0.190 vs. random-t +0.185),
   replicating the text result where masking was not the variable.
4. **Cross-modal agreement drops**: 0.322 for the dense trunk against
   0.286/0.297 for its JEPA descendants. The gain is within-modality only.

### All four target designs, scored on image binding

The three further target designs finished. With
[image binding](../evaluations/image_binding.md) available, the four from-dense
variants can be ranked on a metric with a principled 50% floor rather than on a
3-point spread of probe accuracy. **The text ordering reproduces exactly.**

| Target design | binding | 95% CI | clean probe | Δ vs dense trunk |
|---|---:|---|---:|---:|
| layerwise all-8, 2-D blocks | **0.822** | 0.808 – 0.835 | 0.827 | **+0.020** |
| layerwise all-8, random-t | 0.816 | 0.802 – 0.831 | 0.824 | +0.014 |
| layerwise L4–7, random-t | 0.814 | 0.800 – 0.829 | 0.824 | +0.012 |
| *dense trunk (reference)* | *0.802* | *0.787 – 0.816* | *0.824* | — |
| **averaged** L4–7, random-t | 0.744 | 0.727 – 0.760 | 0.798 | −0.058 |
| **averaged** all-8, random-t | 0.710 | 0.691 – 0.727 | 0.792 | −0.092 |

Layerwise beats dense; averaged loses to dense; the *same layer window* flips
from +0.012 to −0.058 when the target is averaged instead of per-block. This is
the image-side confirmation of the text result, on an independent metric, and it
refutes the "shallow blocks poison the average" explanation in both modalities:
restricting the average to L4–7 recovers only a third of the gap.

### The LoRA loss-matched continuation

`image_lora_diffusion_1_2m_4e` was 4 points of binding below dense partly
because it was undertrained (val 3.9252 vs 3.6340). Continued to 12 epochs
(`image_lora_diffusion_1_2m_12e_continued`, 3.69B tokens, val 3.6214) it finally
*beats* dense on validation loss — and still sits **4 points below it on
binding** (0.761 vs 0.802). Matching the generative loss does not match the
representation, which is the cleanest single demonstration in the image study
that val loss is not a proxy for what these evaluations measure.

### Harder masking, from scratch — in flight

The from-scratch collapse was diagnosed as the objective being *too easy*: the
image JEPA cosine similarity to the target reaches 0.978 after one epoch and
only 0.981 after four, so more epochs do not help. Two runs at ~65% realized
masking (`image_data2vec_scratch_block2d_65pct_{avg,layerwise}_l4to7_1_2m_4e`)
are training to test whether a harder task changes that. Not yet evaluated.

## 2D block masking

The two JEPA runs mask **rectangles** on the token grid rather than scattered
codes: per-block area 10–25% of the 16×24 grid, aspect ratio in [0.75, 1.5],
enough rectangles to hide ~30% (realized 30.4%). A VQ grid is locally redundant,
so scattered codes are recoverable from their neighbours and a contiguous region
is not. Implementation: `block_mask_2d` in `multimodal_diffusion.py`, test
`test_block_masking_hides_rectangles_of_the_image_grid`.

## Evaluation

Hard retrieval and binding swap are text-only — both construct caption minimal
pairs — so the image models are scored by
[cross-modal structure](../evaluations/cross_modal_structure.md), which measures
scene content from image representations and agreement with text models:

| Run | L7 probe accuracy | best `rsa_scene` | CKA with text dense (L7) |
|---|---:|---|---:|
| `image_dense` | **80.7%** | `L6` **+0.178** | **0.322** |
| `image_lora` | 75.1% | `L5` +0.159 | 0.274 |
| `image_jepa_avg` | 61.6% | `L2` +0.110 | 0.186 |
| `image_jepa_lw` | 60.4% | `embedding` +0.109 | 0.139 |

Both JEPA runs are well below the diffusion baselines on scene content, and their
best structure sits at L2 or the embedding rather than deep in the network —
the same failure the from-scratch text runs show.

**Since 2026-09-24 the image models also have their own binding metric.**
[Image binding](../evaluations/image_binding.md) builds text-style minimal pairs
out of *content-matched scenes* — two rendered scenes whose per-attribute
multisets are identical but whose bindings differ, of which 397 occur naturally
in the 20k validation split. It gives images the same 50% floor the text binding
swap has, and it separates the image models over **29 points** where the scene
probe separates them over **3**. It is now the headline image measure; the probe
is reported alongside it as a sanity check.

For reference, the paired multimodal model
(`multimodal_paired_dense_1_2m_4e`) scores 0.792 binding with the best clean
probe of any image model (0.838) — it buys cross-modal structure, not
within-image binding. See
[modality alignment](../evaluations/modality_alignment.md).

## Not run

`image_module_jepa_layerwise_gated_hsic_{matched,strong}_1_2m_4e` exist as
configs but were never trained; the matched variant was stopped before its first
checkpoint.
