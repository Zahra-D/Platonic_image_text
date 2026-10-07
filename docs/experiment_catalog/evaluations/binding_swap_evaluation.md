# Binding-swap evaluation

> **Type:** evaluation, text-only study · **Status:** complete for all final checkpoints, including both gated-predictor runs; epoch-0 results kept for reference  
> **Question:** does a representation know which attribute belongs to which object, beyond which attribute words occur?  
> **Models:** dense, plain Tri-LoRA, all JEPA/HSIC variants, both gated data2vec runs, and (epoch 0) both gated-predictor runs  
> **Scripts:** [binding_swap_captions.py](/home/zd25e122/clevr_discrete_diffusion/binding_swap_captions.py), [evaluate_binding_swap.py](/home/zd25e122/clevr_discrete_diffusion/evaluate_binding_swap.py) · **Results:** `outputs/binding_swap_eval/`, `outputs/binding_swap_eval_epoch000/`  
> **Menu:** [experiment catalog](../README.md)

## Contents

[In one paragraph](#in-one-paragraph) · [Why](#why-this-test-exists) · [Example](#example-held-out-world-6) · [Construction](#how-the-captions-are-built) · [Metric](#primary-metric-minimal-pair-conjunction-probe) · [Final results](#results-final-checkpoints-epoch-3) · [Dense reference, blocks 2–4](#dense-reference-inside-the-jepa-window-blocks-24) · [Epoch 0](#results-epoch-0-interim-after-one-pass-over-the-2m-captions) · [Cosine diagnostic](#secondary-diagnostic-cosine-similarity-is-dominated-by-word-order) · [How to read](#how-to-read-the-binding-results) · [Limitations](#limitations) · [Reproduce](#reproduce)

## In one paragraph

Does a representation know **which attribute belongs to which object**, or
only **which attribute words appear**? The caption "a red cube and a blue
sphere" and the caption "a blue cube and a red sphere" contain exactly the
same words but describe different scenes. This test builds thousands of such
pairs from held-out scenes, trains a simple linear probe on normal training
captions to detect facts like "some object is red **and** a cube" or "a sphere
is right of a cube", and asks how often the probe scores the true caption
above its swapped twin. Anything that only looks at words sees the two
captions as identical and scores **exactly 50%**. Above 50% means the
representation carries binding information; 100% would mean it always tells
the swapped captions apart.

| Score | Meaning |
|---|---|
| 50% | No binding information: word content only (all word-level controls score exactly this) |
| Clearly above 50% | The representation encodes which object has which attribute, or which object is on which side |
| Below 50% | The probe relies on a cue that reverses under the swap; still no binding information |

This page holds the explanation, construction, and results for every
evaluated model. The companion retrieval test, which *can* be solved from word
content alone, is in [cross_pattern_semantic_retrieval.md](cross_pattern_semantic_retrieval.md).

## Why this test exists

The retrieval test above can be solved largely from *which words occur*: plain
word counts reach R@10 = 70.9%. Ordinary semantic probes have the same
weakness. On held-out human captions, a linear probe on word counts detects
"some object is red **and** a cube" with **84.5%** balanced accuracy, without
any notion of which word describes which object.

This evaluation removes that shortcut by construction. Every test compares
two captions with **exactly the same tokens** that describe **different
scenes**, because one *binding* was exchanged: two objects trade a color,
shape, material or size, or a relation's two objects trade roles. A bag of
words, or any order-free average of token embeddings, sees the two captions
as identical and scores **exactly 50%**. Anything above 50% requires knowing
which attribute belongs to which object, or which object is on which side.

## Example (held-out world 6)

Scene: a small red rubber cube, a small red metal sphere, a small blue rubber
cylinder; the sphere is right of the cube; the cylinder is behind the sphere.

> **Query**
> What we have here is a red rubber box that looks tiny; a red glossy metal sphere that looks tiny; and a tube that is tiny, blue, and rubber, for three objects altogether. On top of that, the red glossy metal sphere that looks tiny sits to the right of the box that is tiny, red, and rubber. The blue rubber tube that looks tiny is tucked behind the red glossy metal sphere that looks tiny.

> **Color swap** — same tokens; the cube and the cylinder exchange red and blue
> What we have here is a **blue** rubber box that looks tiny; a red glossy metal sphere that looks tiny; and a tube that is tiny, **red**, and rubber, for three objects altogether. On top of that, the red glossy metal sphere that looks tiny sits to the right of the box that is tiny, **blue**, and rubber. The **red** rubber tube that looks tiny is tucked behind the red glossy metal sphere that looks tiny.
>
> Facts that flip: red cube, blue cylinder → blue cube, red cylinder.

> **Relation swap** — same tokens; the cube and the sphere exchange sides
> What we have here is a red rubber box that looks tiny; a red glossy metal sphere that looks tiny; and a tube that is tiny, blue, and rubber, for three objects altogether. On top of that, the **red rubber box that looks tiny** sits to the right of the **sphere that is tiny, red, and glossy metal**. The blue rubber tube that looks tiny is tucked behind the red glossy metal sphere that looks tiny.
>
> Facts that flip: "a sphere is right of a cube" → "a cube is right of a sphere".

> **Reordered paraphrase** (same scene, same tokens, objects listed in a different order)
> What we have here is a red glossy metal sphere that looks tiny; a blue rubber tube that looks tiny; and a box that is tiny, red, and rubber, for three objects altogether. The blue rubber tube that looks tiny is tucked behind the red glossy metal sphere that looks tiny. On top of that, the red glossy metal sphere that looks tiny sits to the right of the box that is tiny, red, and rubber.

> **Reworded paraphrase** (same scene, different synonyms, styles and sentence order)
> On top of that, the red glossy metal ball that looks smallish ends up to the right of the smallish red matte-textured rubber box. Compared to the ball built from glossy metal, red and smallish, the blue smallish cylinder with a matte-textured rubber finish sits further back. Looking at the image, I count three objects -- a ball built from glossy metal, red and smallish; a matte-textured rubber cylinder in blue, on the smallish side; and a box that is smallish, red, and matte-textured rubber.

## How the captions are built

Captions use the training caption generator's phrase styles, openers,
relation phrases and synonym pools. Its random choices are replaced by a
fixed **phrasing plan** so that exchanging a binding cannot change any word:

- each attribute *value* gets one synonym for the whole caption (all spheres
  are "sphere"), and each material one word that fits both "a rubber box" and
  "made of rubber";
- phrase styles attach to *list positions* and *relation mentions*, not to
  objects; relation phrasing attaches to the relation, not to its sentence
  position;
- the two vowel-initial synonyms ("oversized", "orb") are excluded so "a"
  never becomes "an";
- attribute swaps are only made between objects named equally often in
  relation sentences, between objects that also differ in another attribute
  (otherwise the scene would not change), and never create duplicate objects;
  scenes with duplicate objects are skipped.

The evaluator re-checks, for every item, that the swapped caption and the
reordered paraphrase have exactly the query's token multiset, and refuses to
run otherwise. Within one caption every object of a value uses the same
synonym, which is slightly more uniform than training captions.

Construction: 512 held-out validation worlds, 4234 captions,
2180 minimal pairs (color 492, shape 495, material 382,
size 398, relation 413). No caption has an unknown token; the longest has 179 tokens.

## Primary metric: minimal-pair conjunction probe

1. **Facts.** 74 binding-dependent binary facts. Object facts are "some object has
   A = a and B = b" for every pair of the four attributes, e.g. red + cube,
   metal + small. Relation facts are "some object of shape s is right of / in
   front of some object of shape t", with left and behind rewritten as right
   and front so equivalent statements share a label. Facts with fewer than 50
   positives or negatives in training are dropped.
2. **Training.** One class-balanced multi-label logistic-regression probe per
   model and feature, on 20,000 real training captions (`caption_human`) with
   labels from their scene records.
3. **Test.** For each minimal pair and each fact that flips between its two
   scenes, the probe should give the true caption a higher score than its twin.
   **Pair-ranking accuracy** is the fraction of these comparisons it gets right,
   ties counted as one half, averaged within each pair, then with equal weight
   per swap type. 95% intervals come from a 1,000-repetition world bootstrap.
4. **Sanity.** "Ordinary probe" is the same probe's mean balanced accuracy on
   2,048 held-out human captions (validation rows 10,000–12,047, disjoint from
   the test worlds). That is the conventional probe score, and the lexical
   shortcut works on it.

Features: every one is pooled over content tokens (mean, in float64 so that
permuted identical tokens pool identically) and L2-normalized. They are the
residual stream after each block, the full attention and MLP outputs of each
block (both branches plus bias for Tri-LoRA), the shared and private branch
updates of `attn.out_proj` and `mlp.3` at every block, and the input token
embeddings.

## Results: final checkpoints (epoch 3)

**Controls:**

| Control (no model) | Minimal-pair binding | Ordinary held-out probe | Cosine: reorder | Cosine: reword | Cosine: different scene |
|---|---:|---:|---:|---:|---:|
| Word counts | **50.0 [50.0–50.0]** | 84.5 | 50.0 | 0.0 | 100.0 |
| TF-IDF words | **50.0 [50.0–50.0]** | 85.2 | 50.0 | 0.0 | 100.0 |
| Random 384-d features | **50.1 [48.8–51.6]** | 50.6 | 49.9 | 49.0 | 50.6 |
| Token-embedding mean, Dense | **50.0 [50.0–50.0]** | 84.5 | 50.0 | 0.0 | 100.0 |
| Token-embedding mean, Plain Tri-LoRA | **50.0 [50.0–50.0]** | 84.6 | 50.0 | 0.0 | 100.0 |

The shortcut is removed exactly: every order-free feature scores 50.0% on
minimal pairs, while reaching 84–85% on the ordinary probe.

**Main table** (minimal-pair binding accuracy, %; 95% intervals in brackets):

| Model | Shared `blocks.4.mlp.3` | Private `blocks.4.mlp.3` | Full `mlp.3`, block 4 | Residual, block 4 | Residual, block 7 | Best residual block | Ordinary probe, block 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Dense** | — | — | 93.2 | 77.5 [76.6–78.5] | 82.7 [81.4–83.9] | 93.2 (L6) | 89.7 |
| Plain Tri-LoRA | 62.7 [61.5–63.8] | 62.6 [61.5–63.8] | 63.1 | 62.8 [61.6–63.9] | 78.5 [77.3–79.5] | 79.1 (L6) | 83.4 |
| Average JEPA | 61.5 [60.3–62.6] | 57.7 [56.6–58.9] | 59.3 | 57.2 [56.0–58.4] | 74.5 [73.3–75.5] | 74.5 (L7) | 79.8 |
| Average JEPA + HSIC | 59.4 [58.3–60.5] | 55.2 [54.0–56.4] | 57.1 | 64.1 [63.0–65.2] | 73.0 [71.8–74.2] | 73.0 (L7) | 80.8 |
| Layerwise JEPA | 63.2 [62.3–64.2] | 59.0 [57.9–60.0] | 61.3 | 60.3 [59.3–61.3] | 77.5 [76.5–78.5] | 77.5 (L7) | 80.7 |
| Layerwise JEPA + HSIC | 60.2 [59.1–61.3] | 58.4 [57.3–59.5] | 60.2 | 58.3 [57.2–59.5] | 72.1 [70.8–73.4] | 73.9 (L6) | 81.4 |
| Calibrated average + HSIC | 59.0 [57.9–60.2] | 55.8 [54.7–57.0] | 56.9 | 60.8 [59.6–61.9] | 75.0 [73.8–76.2] | 75.0 (L7) | 81.7 |
| Gated data2vec, average (collapsed) | 45.2 [43.9–46.5] | 44.6 [43.6–45.8] | 46.4 | 64.5 [63.5–65.7] | 78.3 [77.3–79.4] | 78.3 (L7) | 80.4 |
| Gated data2vec, final layer (collapsed) | 50.8 [49.7–51.9] | 51.0 [49.8–52.2] | 51.7 | 63.5 [62.4–64.5] | 76.6 [75.5–77.7] | 77.1 (L6) | 80.7 |
| Gated layerwise JEPA, no HSIC (new) | 61.9 [60.9–62.9] | 58.4 [57.4–59.5] | 60.2 | 61.1 [60.1–62.1] | 73.2 [72.1–74.4] | 73.2 (L7) | 78.5 |
| **Gated layerwise JEPA + HSIC** (new) | 65.4 [64.4–66.3] | 61.3 [60.2–62.2] | 64.0 | 63.1 [62.1–64.3] | 80.5 [79.5–81.6] | 80.5 (L7) | 82.5 |

**Attribute binding vs. relation direction** (%; attribute = equal-weight mean
of color, shape, material and size swaps):

| Model | Residual attribute, best block | Residual relation, best block | Shared L4 attribute | Private L4 attribute | Shared L4 relation | Private L4 relation |
|---|---:|---:|---:|---:|---:|---:|
| **Dense** | 91.7 (L6) | 99.0 (L6) | — | — | — | — |
| Plain Tri-LoRA | 75.8 (L7) | 96.6 (L5) | 57.8 | 57.9 | 82.1 | 81.6 |
| Average JEPA | 68.9 (L7) | 96.6 (L7) | 58.6 | 54.6 | 72.9 | 70.5 |
| Average JEPA + HSIC | 71.9 (L7) | 92.0 (L4) | 54.5 | 52.5 | 79.2 | 65.9 |
| Layerwise JEPA | 72.1 (L7) | 99.3 (L7) | 55.6 | 52.6 | 93.9 | 84.5 |
| Layerwise JEPA + HSIC | 74.4 (L7) | 89.3 (L5) | 54.9 | 53.2 | 81.1 | 78.9 |
| Calibrated average + HSIC | 76.9 (L7) | 83.3 (L5) | 53.8 | 51.1 | 79.7 | 74.8 |
| Gated data2vec, average (collapsed) | 74.7 (L7) | 97.1 (L5) | 49.3 | 49.6 | 29.1 | 24.9 |
| Gated data2vec, final layer (collapsed) | 74.8 (L7) | 95.4 (L5) | 51.3 | 50.2 | 49.2 | 54.5 |
| Gated layerwise JEPA, no HSIC (new) | 69.8 (L7) | 91.3 (L4) | 53.3 | 51.5 | 96.4 | 86.2 |
| **Gated layerwise JEPA + HSIC** (new) | 77.0 (L7) | 96.9 (L5) | 57.6 | 54.9 | 96.4 | 86.7 |

### Dense reference inside the JEPA window (blocks 2–4)

Dense has no shared or private branch: each layer is one weight matrix. The
like-for-like reference for a Tri-LoRA shared update is therefore dense's whole
`mlp.3` output at the same block. For Tri-LoRA the whole output is shared +
private + bias. Binding accuracy in %, attribute = mean of color, shape,
material and size swaps.

| Model | Feature | L2 attribute / relation | L3 attribute / relation | L4 attribute / relation |
|---|---|---:|---:|---:|
| **Dense** | whole `mlp.3` output | **60.3 / 85.0** | **69.0 / 99.0** | **91.5 / 100.0** |
| **Dense** | residual stream | 55.9 / 78.2 | 60.7 / 89.3 | 72.7 / 96.9 |
| Plain Tri-LoRA | shared `mlp.3` | 50.9 / 54.7 | 53.8 / 84.3 | 57.8 / 82.1 |
| | private `mlp.3` | 50.1 / 64.2 | 53.3 / 78.9 | 57.9 / 81.6 |
| | whole `mlp.3` output | 50.5 / 59.8 | 53.6 / 82.3 | 58.4 / 81.8 |
| Average JEPA, no HSIC | shared `mlp.3` | 50.2 / 54.7 | 53.7 / 64.6 | 58.6 / 72.9 |
| | whole `mlp.3` output | 49.8 / 54.0 | 53.6 / 68.3 | 56.9 / 68.5 |
| Layerwise JEPA, no HSIC | shared `mlp.3` | 50.9 / 53.3 | 52.1 / 54.2 | 55.6 / 93.9 |
| | whole `mlp.3` output | 51.3 / 56.4 | 51.9 / 42.6 | 53.9 / 90.8 |
| **Gated layerwise JEPA + HSIC** | shared `mlp.3` | 52.0 / 81.8 | 52.6 / 68.5 | 57.6 / 96.4 |
| | private `mlp.3` | 51.8 / 79.7 | 51.5 / 50.6 | 54.9 / 86.7 |
| | whole `mlp.3` output | 52.4 / 80.9 | 52.3 / 59.3 | 56.8 / 93.0 |

- **Shared vs. private.** In plain Tri-LoRA the two branches are identical at
  block 4 (57.8 / 82.1 vs. 57.9 / 81.6). In gated JEPA + HSIC shared is ahead
  on both parts: relation 96.4% [94.7–98.1] vs. private 86.7% [83.3–89.8], and
  attribute 57.6% vs. 54.9%. The shared gain over plain is relational; the
  attribute gap comes from private dropping below plain (54.9% vs. 57.9%).
- **This is a single-module view.** Summed over all layers and sublayers, the
  shared and private streams are equivalent, even in gated JEPA + HSIC: see
  [decomposed representations](decomposed_representations.md).
- **Relation direction is essentially reached.** Dense's block-4 write is at
  100.0%; gated JEPA + HSIC's shared update is at 96.4%.
- **Attribute binding is the real gap.** Dense's block-4 write binds
  attributes at 91.5% (color 89.5, shape 93.1, material 91.0, size 92.3); the
  best Tri-LoRA shared update is at 58.6% (average JEPA; gated JEPA + HSIC
  57.6%), and even the whole Tri-LoRA write is at most 58.4%. Dense already
  reaches 69.0% at block 3, while no Tri-LoRA `mlp.3` feature exceeds 58.6%
  anywhere in blocks 2–4.
- Unlike retrieval, this gap cannot be closed by keeping more word content:
  word-level features score exactly 50%.

**Residual stream after each block** (binding mean, %):

| Model | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Dense** | 50.5 | 53.6 | 60.4 | 66.4 | 77.5 | 88.5 | 93.2 | 82.7 |
| Plain Tri-LoRA | 52.7 | 55.6 | 55.8 | 59.1 | 62.8 | 77.3 | 79.1 | 78.5 |
| Average JEPA | 52.2 | 52.2 | 51.4 | 55.9 | 57.2 | 65.2 | 69.8 | 74.5 |
| Average JEPA + HSIC | 52.6 | 58.6 | 62.9 | 64.7 | 64.1 | 65.2 | 69.6 | 73.0 |
| Layerwise JEPA | 50.4 | 54.4 | 56.2 | 54.8 | 60.3 | 63.8 | 71.3 | 77.5 |
| Layerwise JEPA + HSIC | 51.9 | 51.8 | 52.4 | 55.5 | 58.3 | 63.2 | 73.9 | 72.1 |
| Calibrated average + HSIC | 50.6 | 50.2 | 53.6 | 54.1 | 60.8 | 62.5 | 70.8 | 75.0 |
| Gated data2vec, average (collapsed) | 50.6 | 52.6 | 58.5 | 63.6 | 64.5 | 72.8 | 73.5 | 78.3 |
| Gated data2vec, final layer (collapsed) | 51.1 | 50.6 | 55.6 | 64.0 | 63.5 | 67.9 | 77.1 | 76.6 |
| Gated layerwise JEPA, no HSIC (new) | 52.0 | 54.6 | 57.7 | 60.5 | 61.1 | 61.7 | 68.5 | 73.2 |
| **Gated layerwise JEPA + HSIC** (new) | 51.9 | 53.0 | 57.3 | 58.1 | 63.1 | 70.0 | 75.1 | 80.5 |

**Shared / private `mlp.3` update at each block** (binding mean, %):

| Model | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---|---|---|---|---|---|---|
| Plain Tri-LoRA | 48.1 / 48.3 | 53.3 / 53.4 | 51.7 / 52.9 | 59.9 / 58.4 | 62.7 / 62.6 | 78.0 / 75.9 | 73.4 / 73.1 | 63.7 / 62.9 |
| Average JEPA | 51.8 / 51.9 | 49.4 / 49.5 | 51.1 / 50.0 | 55.9 / 56.6 | 61.5 / 57.7 | 65.5 / 65.2 | 69.4 / 66.2 | 60.7 / 59.3 |
| Average JEPA + HSIC | 51.4 / 51.3 | 52.9 / 52.5 | 63.2 / 62.6 | 59.9 / 58.3 | 59.4 / 55.2 | 55.8 / 54.6 | 65.7 / 67.4 | 66.3 / 65.1 |
| Layerwise JEPA | 51.1 / 49.0 | 50.7 / 51.1 | 51.4 / 52.1 | 52.5 / 49.8 | 63.2 / 59.0 | 64.4 / 63.9 | 70.9 / 69.7 | 64.0 / 63.2 |
| Layerwise JEPA + HSIC | 52.0 / 51.1 | 49.1 / 49.6 | 49.3 / 49.3 | 55.8 / 55.5 | 60.2 / 58.4 | 63.9 / 62.9 | 72.7 / 75.0 | 62.5 / 59.3 |
| Calibrated average + HSIC | 50.2 / 50.6 | 49.8 / 50.4 | 48.8 / 52.4 | 53.5 / 51.5 | 59.0 / 55.8 | 59.8 / 60.7 | 73.9 / 70.1 | 71.5 / 72.8 |
| Gated data2vec, average (collapsed) | 50.2 / 50.7 | 50.0 / 51.5 | 55.6 / 53.0 | 60.5 / 61.4 | 45.2 / 44.6 | 72.0 / 71.2 | 60.1 / 61.3 | 72.0 / 70.7 |
| Gated data2vec, final layer (collapsed) | 51.3 / 50.1 | 48.4 / 49.8 | 50.2 / 50.1 | 61.2 / 62.2 | 50.8 / 51.0 | 67.0 / 66.2 | 71.9 / 71.0 | 71.6 / 71.1 |
| Gated layerwise JEPA, no HSIC (new) | 49.0 / 49.3 | 51.1 / 51.9 | 58.8 / 57.6 | 59.3 / 58.1 | 61.9 / 58.4 | 56.5 / 56.0 | 66.5 / 66.6 | 63.7 / 63.4 |
| **Gated layerwise JEPA + HSIC** (new) | 50.2 / 49.9 | 50.9 / 51.7 | 58.0 / 57.3 | 55.8 / 51.3 | 65.4 / 61.3 | 70.2 / 68.4 | 72.5 / 72.1 | 70.6 / 71.0 |

## Results: epoch 0 (interim, after one pass over the 2M captions)

Scored on exactly the same captions and probe setup as the final results, so
the two gated-predictor runs can be compared with their layer-local baselines
at the same point in training. Final results for both gated runs are in the
final-checkpoint tables above; this section is kept to show training progress.
**Interim:** one epoch is early, most values
are close to 50%, and the final checkpoints may rank differently.

The gated runs keep the JEPA predictor but send its gradient end-to-end into
the shared LoRA of blocks 2–4; the baselines confine it to each supervised
layer. Otherwise each gated run and its baseline are identically configured.

Training health at epoch 0 (validation loss at t = 0.75; entropy effective rank
of the `blocks.4.mlp.3` shared / private LoRA maps, from the checkpoint):

| Model | Validation loss | Eff. rank shared / private |
|---|---:|---:|
| **Gated layerwise JEPA, no HSIC** (new) | 1.4868 | 18.6 / 13.6 |
| Layerwise JEPA, no HSIC (baseline) | 1.4339 | 11.8 / 15.8 |
| **Gated layerwise JEPA + HSIC** (new) | 1.4580 | 12.1 / 13.7 |
| Layerwise JEPA + HSIC (baseline) | 1.4546 | 11.0 / 17.5 |
| Plain Tri-LoRA | 1.3941 | 16.7 / 18.5 |

Neither gated run collapses (the collapsed gated data2vec run ended at rank
1.8). Gated no-HSIC denoises worse than its baseline; gated + HSIC matches it.

**Minimal-pair binding accuracy** (%; 50 = no binding information; 95%
intervals in brackets; attr = mean over color, shape, material, size swaps;
rel = relation swaps):

| Model | Shared L4 `mlp.3` | Shared attr / rel | Private L4 `mlp.3` | Private attr / rel | Residual, block 7 | Best residual block |
|---|---:|---:|---:|---:|---:|---:|
| **Gated layerwise JEPA, no HSIC** (new) | 52.0 [50.8–53.3] | 50.5 / 57.9 | 49.7 [48.4–51.0] | 49.8 / 49.4 | 52.2 [50.8–53.4] | 52.2 (L7) |
| Layerwise JEPA, no HSIC (layer-local baseline) | 53.5 [52.2–54.7] | 51.5 / 61.5 | 52.1 [50.9–53.2] | 51.4 / 55.0 | 57.8 [56.6–59.0] | 57.8 (L7) |
| **Gated layerwise JEPA + HSIC** (new) | 55.3 [54.1–56.5] | 50.6 / 74.1 | 53.4 [52.1–54.6] | 50.4 / 65.1 | 58.2 [56.8–59.4] | 58.4 (L5) |
| Layerwise JEPA + HSIC (layer-local baseline) | 51.3 [50.1–52.4] | 49.9 / 56.9 | 50.9 [49.8–52.2] | 50.9 / 50.8 | 60.8 [59.5–62.1] | 60.8 (L7) |
| Plain Tri-LoRA | 53.7 [52.6–54.9] | 51.3 / 63.2 | 54.1 [53.1–55.3] | 51.1 / 66.3 | 62.1 [61.1–63.3] | 62.1 (L7) |
| Dense | — | — | — | — | 82.5 [81.4–83.5] | 82.5 (L7) |

**Residual stream after each block** (binding mean, %):

| Model | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Gated layerwise JEPA, no HSIC** (new) | 50.8 | 49.3 | 50.0 | 51.0 | 51.1 | 50.8 | 51.9 | 52.2 |
| Layerwise JEPA, no HSIC (layer-local baseline) | 50.6 | 52.4 | 52.5 | 52.2 | 52.6 | 53.9 | 56.3 | 57.8 |
| **Gated layerwise JEPA + HSIC** (new) | 49.5 | 51.7 | 53.4 | 53.0 | 53.2 | 58.4 | 55.9 | 58.2 |
| Layerwise JEPA + HSIC (layer-local baseline) | 50.3 | 51.6 | 51.7 | 51.3 | 51.5 | 54.2 | 58.1 | 60.8 |
| Plain Tri-LoRA | 53.4 | 53.6 | 50.8 | 56.4 | 55.3 | 61.3 | 61.3 | 62.1 |
| Dense | 49.6 | 53.0 | 57.7 | 62.0 | 70.1 | 77.9 | 81.2 | 82.5 |

**Reading the epoch-0 results:**

1. **Attribute binding is at chance in every LoRA branch** (49.8–51.5%) at
   epoch 0. It only emerges later in training (final plain Tri-LoRA: 57.8%),
   so the runs cannot yet be compared on it.
2. **Gated + HSIC has the strongest shared branch of all epoch-0 LoRA models**,
   55.3% [54.1–56.5], above its baseline's 51.3% [50.1–52.4] with separated
   intervals. All of it comes from relation direction: 74.1% in shared, vs.
   56.9% for its baseline, 63.2% for plain Tri-LoRA's shared branch, and 65.1%
   for its own private branch.
3. **Gated no-HSIC is below its baseline on every binding number** (shared
   52.0% vs. 53.5%, block 7 52.2% vs. 57.8%), and its private branch is at
   chance. Together with its higher validation loss, gating without HSIC is
   so far a cost.
4. **Neither gated run improves the whole model yet.** Block-7 binding is
   52.2% and 58.2%, against 62.1% for plain Tri-LoRA.
5. **Dense is far ahead even after one epoch**: 82.5% at block 7.

Results: `outputs/binding_swap_eval_epoch000/<label>.json`.

## Secondary diagnostic: cosine similarity is dominated by word order

With the same items, preference = how often cos(query, same-scene paraphrase)
is larger than cos(query, binding swap).

| Model | Residual block 4: reorder / reword / different scene | Residual block 7 | Shared `blocks.4.mlp.3` |
|---|---|---|---|
| **Dense** | 6.4 / 0.0 / 99.8 | 29.9 / 0.0 / 99.8 | — |
| Plain Tri-LoRA | 4.6 / 0.0 / 96.1 | 25.6 / 0.2 / 98.0 | 8.6 / 0.0 / 84.8 |
| Average JEPA | 2.8 / 0.0 / 97.7 | 12.4 / 0.0 / 99.2 | 4.0 / 0.0 / 86.9 |
| Average JEPA + HSIC | 10.0 / 0.0 / 95.3 | 17.5 / 0.0 / 99.4 | 8.1 / 0.0 / 75.8 |
| Layerwise JEPA | 2.6 / 0.0 / 93.9 | 13.6 / 0.0 / 96.1 | 8.5 / 0.0 / 88.5 |
| Layerwise JEPA + HSIC | 3.9 / 0.0 / 95.5 | 17.2 / 0.0 / 98.2 | 8.2 / 0.0 / 91.8 |
| Calibrated average + HSIC | 7.2 / 0.0 / 97.3 | 20.3 / 0.0 / 98.6 | 9.1 / 0.0 / 80.7 |
| Gated data2vec, average (collapsed) | 6.3 / 0.0 / 93.9 | 16.0 / 0.0 / 96.3 | 13.4 / 0.0 / 57.8 |
| Gated data2vec, final layer (collapsed) | 5.9 / 0.0 / 95.5 | 16.0 / 0.1 / 99.6 | 12.6 / 2.0 / 52.9 |
| Gated layerwise JEPA, no HSIC (new) | 3.9 / 0.0 / 91.8 | 9.1 / 0.0 / 95.7 | 6.9 / 0.0 / 70.9 |
| **Gated layerwise JEPA + HSIC** (new) | 5.4 / 0.0 / 85.5 | 18.0 / 0.1 / 99.4 | 10.3 / 0.0 / 81.6 |

Every model prefers the binding-swapped caption to the reordered paraphrase
(3–30%) and always prefers it to a reworded paraphrase (0–2%). The swap keeps
every token at the query's position, while a paraphrase moves them. With
learned absolute positions, pooled cosine therefore tracks word order much
more than binding. This is why the probe, which can learn to ignore position,
is the primary metric and raw cosine is not.

## How to read the binding results

**1. The test is valid.** Word counts, TF-IDF and token-embedding averages
are exactly 50.0% on minimal pairs, random features 50.1%, and the ordinary
probe on the same word counts is 84.5%. Ordinary probes mostly measure word
content; this one does not.

**2. Dense binds far better than any Tri-LoRA model.** Dense reaches 91.7%
attribute binding in the residual stream (block 6) and 96.1% in its block-5
MLP output, and 97–99% relation direction at blocks 4–6. Plain Tri-LoRA peaks at
74.9–75.8% attribute binding (blocks 6–7). The retrieval gap between dense and
Tri-LoRA is mostly a gap in compositional binding, not in word content.

**3. Only gated JEPA + HSIC improves binding in the whole model.** Its final
residual stream binds at 80.5% [79.5–81.6] against 78.5% [77.3–79.5] for
plain Tri-LoRA, and its best-block attribute binding is 77.0% vs. 75.8%. Every
other variant, including gated JEPA without HSIC (73.2%), is at or below plain
Tri-LoRA (72.1–78.3%). All remain far below dense (82.7% at block 7, 93.2% at
block 6; attribute binding 91.7%).

**4. Shared vs. private at `blocks.4.mlp.3`.** In plain Tri-LoRA the two
branches are indistinguishable (62.7% vs. 62.6%). Every non-collapsed JEPA
variant makes shared higher than private, by 1.8–4.2 points (gated + HSIC:
4.1) with separated intervals for most. For the layer-local variants **the gap
comes mainly from private losing binding, not shared gaining it**: none of
their shared branches exceeds plain Tri-LoRA's beyond overlapping intervals
(59.0–63.2% vs. 62.7% [61.5–63.8]), while private drops to 55.2–59.0%.

**Gated JEPA + HSIC is the exception.** Its shared branch reaches 65.4%
[64.4–66.3], clearly above plain Tri-LoRA (62.7%) and layerwise JEPA (63.2%
[62.3–64.2]), while its private branch (61.3%) stays close to plain. Gated
JEPA without HSIC does not show this (shared 61.9%, private 58.4%).

The two parts behave differently:
- **Attribute binding** in shared at block 4 is weak everywhere (53.3–58.6%,
  against 57.8% for plain Tri-LoRA; gated + HSIC 57.6%), so no JEPA variant
  raises it in a meaningful way.
- **Relation direction** is where JEPA helps shared: layerwise JEPA reaches
  93.9% and both gated runs 96.4%, vs. 82.1% for plain Tri-LoRA; their private
  branches reach 84.5–86.7%. The gated + HSIC shared-branch gain above comes
  entirely from this.

**5. The collapsed gated data2vec runs have no binding in the collapsed
module.** Their shared and private `blocks.4.mlp.3` updates are at or below
chance (44.6–51.0%). The average-target run is significantly below 50%, which
means its probe relies on a cue that reverses under the swap. Their residual
streams are unaffected (block 7: 76.6–78.3%).

**6. Consequence for the shared/private claim.** At the evaluated module, the
evidence supports "JEPA moves relational information toward shared". Gated
JEPA + HSIC is the first configuration that does this while also improving
whole-model binding and keeping private close to plain Tri-LoRA. No
configuration yet puts more *attribute* binding into shared, so "shared
captures compositional semantics" is not yet supported. Retrieval alone would
have suggested more.

## Limitations

- **Presence, not use.** A probe shows information is linearly decodable, not
  that the model relies on it. The causal swap test is still needed.
- **Relation facts use shapes only.** Relation swaps between objects of the
  same shape flip no fact and are not scored; color-based relation facts were
  too sparse for training.
- **Probe domain shift.** Probes train on generator-sampled human captions
  and test on plan-rendered captions of the same style. That lowers absolute
  accuracy for every model equally, and word-level controls are unaffected by
  construction.
- **Text only, one checkpoint per model, one caption-construction seed.**

## Reproduce

```bash
# All final checkpoints plus model-free controls, on GPUs 2 and 3
./scripts/launch_binding_swap_final_vnode10.sh 2 3

# One new checkpoint (skips labels already present in the output directory)
python evaluate_binding_swap.py --output-dir outputs/binding_swap_eval \
  --checkpoint LABEL=path/to/epoch_003.pt
```

- Caption construction: [binding_swap_captions.py](/home/zd25e122/clevr_discrete_diffusion/binding_swap_captions.py)
- Evaluator: [evaluate_binding_swap.py](/home/zd25e122/clevr_discrete_diffusion/evaluate_binding_swap.py)
- Captions, items, construction audit: `outputs/binding_swap_eval/{captions,items}.jsonl`, `construction.json`
- Per-model results, every feature, per swap type, with intervals: `outputs/binding_swap_eval/<label>.json`; controls: `controls.json`
