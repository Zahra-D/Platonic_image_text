# Image binding: do image models know *which* object has which attribute?

> **Type:** evaluation · **Status:** 11 models · **Updated:** 2026-09-24
> **Menu:** [experiment catalog](../README.md) · **Code:** `evaluate_image_binding.py`

The text side has had a binding test from the start ([binding
swap](binding_swap_evaluation.md), [semantic d′](semantic_dprime.md)): two
captions containing *exactly the same words* that differ only in which object
carries which attribute. The image side had nothing comparable, and the
standing assumption was that building one meant re-rendering counterfactual
scenes with a modified generator.

It did not. **The rendered corpus already contains the minimal pairs.**

## Construction: content-matched scene pairs

Two scenes form a *content-matched pair* when their per-attribute multisets are
identical — the same colours, the same shapes, the same materials, the same
sizes, in the same quantities — but those attributes are attached to different
objects. Formally, writing `M_a(S)` for the multiset of attribute `a` over the
objects of scene `S`:

```
pair(S, S′)  ⟺  M_a(S) = M_a(S′) for every a ∈ {color, shape, material, size}
             ∧  fact-set(S) ≠ fact-set(S′)
```

Concretely, a pair looks like this:

| | scene A | scene B |
|---|---|---|
| object 1 | **large red** metal **cube** | **large blue** metal **cube** |
| object 2 | **small blue** rubber **sphere** | **small red** rubber **sphere** |
| colour multiset | {red, blue} | {red, blue} — identical |
| shape multiset | {cube, sphere} | {cube, sphere} — identical |
| size multiset | {large, small} | {large, small} — identical |
| conjunction facts | red∧cube, blue∧sphere | blue∧cube, red∧sphere — **different** |

A representation that encodes only *which attributes are present in the image*
assigns both scenes the same features and therefore scores **exactly 50%**.
Anything above 50% requires knowing which attributes co-occur **on the same
object**. That is the same floor and the same question as the text binding
swap, which is what makes the two modalities comparable.

In the 20 000-scene image validation split, **397 such pairs** occur naturally.
No re-rendering, no synthetic counterfactuals, no generator changes.

## Protocol

Mirrors `evaluate_binding_swap.py` so the numbers mean the same thing:

1. Fit a linear probe on **12 000 held-out scenes** to read the
   binding-dependent conjunction facts (`color∧shape`, `material∧shape`, …;
   72 labels surviving the `min_label_count = 50` filter) out of the pooled
   image representation.
2. For each content-matched pair, take the facts that **differ** between the
   two scenes and ask whether the probe assigns each fact to the right scene.
3. Score the fraction ranked correctly; bootstrap 1000× over pairs for the CI.

Two numbers come back:

- **`binding_accuracy`** — the score above, chance **0.500**, with a bootstrap
  95% interval. This is the headline.
- **`clean_probe_balanced_accuracy`** — the probe's ordinary balanced accuracy
  on *unmatched* scenes, i.e. "is the probe any good at all". A model can score
  well here and still be at chance on binding; that dissociation is the point.

### Why this replaced the scene probe as the headline image metric

The per-modality scene probe in
[cross-modal structure](cross_modal_structure.md) spreads the image models over
**3 points** (79.2 – 82.4%) because it is dominated by attribute presence,
which every model gets right. Binding accuracy spreads the same models over
**29 points** (53.5 – 82.2%) and has a principled floor. It also ships a
bootstrap CI and per-pair scores (`per_pair` in the JSON), so two models can be
compared with a paired test rather than by eyeballing overlapping intervals.

## Results

Chance = 0.500. Sorted by binding accuracy.

| Model | design / objective | binding | 95% CI | clean probe | Δ vs image dense |
|---|---|---:|---|---:|---:|
| `image_data2vec_from_dense_layerwise_all_block2d_1_2m_2e` | JEPA layerwise ×8, from dense, 2-D block mask | **0.822** | 0.808 – 0.835 | 0.827 | **+0.020** |
| `image_data2vec_from_dense_layerwise_all_randt_1_2m_2e` | JEPA layerwise ×8, from dense, random-t | 0.816 | 0.802 – 0.831 | 0.824 | +0.014 |
| `image_data2vec_from_dense_layerwise_l4to7_randt_1_2m_2e` | JEPA layerwise L4–7, from dense | 0.814 | 0.800 – 0.829 | 0.824 | +0.012 |
| `image_dense_diffusion_1_2m_4e` | dense diffusion (reference) | 0.802 | 0.787 – 0.816 | 0.824 | — |
| `multimodal_paired_dense_1_2m_4e` | paired dense, both modalities | 0.792 | 0.776 – 0.806 | **0.838** | −0.010 |
| `image_lora_diffusion_1_2m_12e_continued` | Tri-LoRA, trained to dense's val loss | 0.761 | 0.746 – 0.776 | 0.794 | −0.041 |
| `image_data2vec_from_dense_avg_l4to7_randt_1_2m_2e` | JEPA **averaged** L4–7, from dense | 0.744 | 0.727 – 0.760 | 0.798 | −0.058 |
| `image_data2vec_from_dense_avg_all_randt_1_2m_2e` | JEPA **averaged** all 8, from dense | 0.710 | 0.691 – 0.727 | 0.792 | −0.092 |
| `image_lora_diffusion_1_2m_4e` | Tri-LoRA, 4 epochs (undertrained) | 0.703 | 0.687 – 0.720 | 0.765 | −0.099 |
| `image_data2vec_scratch_block2d_30pct_avg_l4to7_1_2m_4e` | JEPA-only **from scratch**, averaged | 0.552 | 0.537 – 0.567 | 0.629 | −0.250 |
| `image_data2vec_scratch_block2d_30pct_layerwise_l4to7_1_2m_4e` | JEPA-only **from scratch**, layerwise | 0.535 | 0.521 – 0.550 | 0.612 | −0.267 |

## What the numbers say

**1. Layerwise JEPA on a pretrained image trunk beats dense — the same result
as text, in the same direction, at a smaller margin.** All three layerwise
variants clear the dense reference (+0.012 to +0.020); the best, 2-D block
masking, is +2.0 points. On text the corresponding gain over *budget-matched*
dense is +19% relative on `d_bind`. The image gain is real but modest, and the
intervals of the three layerwise variants overlap each other — they are one
result, not three.

**2. Averaging the target destroys the gain, in images as in text.** Averaged
L4–7 is 0.744 and averaged all-8 is 0.710, both well *below* dense, while the
layerwise version of the *same layer window* is 0.814. The damage tracks
averaging, not which layers are averaged — the same conclusion the text side
reached, and it rules out the "shallow blocks are the problem" hypothesis in
both modalities.

**3. From-scratch JEPA is near the 50% floor.** 0.552 and 0.535, against a
chance of 0.500. The clean probe (0.61 – 0.63) shows these models are not
entirely empty — they know *something* about attribute presence — but they have
essentially no binding. This is the image-side confirmation of the text-side
collapse, on a metric with a principled floor, and it holds for both target
designs.

**4. The paired dense model trades binding for probe accuracy.** It has the
*best* clean probe of any image model (0.838) and slightly *worse* binding than
the image-only dense model (0.792 vs 0.802, intervals overlapping). Training on
both modalities buys it cross-modal structure (see
[modality alignment](modality_alignment.md), where it is far ahead of anything
else) without buying it within-image binding.

**5. Tri-LoRA needs the extra budget, and still does not catch up.** At 4
epochs it is 0.703; continued to match dense's validation loss it reaches
0.761, still 4 points below the dense trunk it was matched to. Matching the
generative loss does not match the representation.

## Reproducing

```bash
python3 evaluate_image_binding.py \
  --checkpoint image_dense=outputs/image_dense_diffusion_1_2m_4e/epoch_003.pt \
  --output-dir outputs/image_binding_eval
```

Results land in `outputs/image_binding_eval/<label>.json`; the pair
construction is recorded once in `construction.json`
(`{scenes: 20000, labels: 72, pairs: 397, probe_train_scenes: 12000, seed: 20260924}`).
Each result file carries `per_pair`, the 397 per-pair scores, for paired
significance tests between two models.

## Related

- [Binding swap](binding_swap_evaluation.md) — the text original this mirrors
- [Semantic d′](semantic_dprime.md) — the effect-size version on text
- [Cross-modal structure](cross_modal_structure.md) — the scene probe this
  replaced as the headline image measure
