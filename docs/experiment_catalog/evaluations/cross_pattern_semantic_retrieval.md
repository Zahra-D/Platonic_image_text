# Cross-pattern semantic retrieval

> **Type:** evaluation, text-only study · **Status:** complete for all final checkpoints, including both gated-predictor runs; epoch-0 results kept for reference  
> **Question:** can a representation find another caption of the same scene written in a different sentence structure?  
> **Models:** dense, plain Tri-LoRA, four original JEPA/HSIC cells, calibrated average JEPA + HSIC, both gated data2vec runs, and (epoch 0) both gated-predictor runs  
> **Scripts:** [evaluate_shared_private_retrieval.py](/home/zd25e122/clevr_discrete_diffusion/evaluate_shared_private_retrieval.py) (branches), [evaluate_layer_retrieval.py](/home/zd25e122/clevr_discrete_diffusion/evaluate_layer_retrieval.py) (whole layers), [evaluate_layer_retrieval_sweep.py](/home/zd25e122/clevr_discrete_diffusion/evaluate_layer_retrieval_sweep.py) (every layer)  
> **Menu:** [experiment catalog](../README.md)

## Summary

- **JEPA shifts retrieval signal from private to shared** at `blocks.4.mlp.3`:
  plain Tri-LoRA shared/private R@10 is 22.9% / 25.5%; layerwise JEPA reaches
  35.4% / 30.5%, and gated JEPA + HSIC 40.1% / 24.9%, the largest shared score
  and shared–private gap. Gated JEPA without HSIC lowers both (20.2% / 17.3%).
- **But this task is largely lexical.** Word counts with no model reach
  R@10 = 70.9%, and every model's average input token embedding 60–63%, above
  every LoRA branch. Branch results show a relative shift, not a strong
  semantic representation.
- **Dense is far stronger** at the same module (67.7%) and in the residual
  stream (block 7 R@1 65.1% vs. 63.8% for plain Tri-LoRA; block 4 50.0% vs.
  31.7%).
- **Every JEPA/HSIC variant lowers whole-model retrieval** relative to plain
  Tri-LoRA (block-7 R@1 37–60% vs. 63.8%).

For tests that remove the lexical shortcut, see the
[hard retrieval](hard_retrieval.md) (same retrieval idea, but every candidate
has exactly the same words) and the [binding-swap evaluation](binding_swap_evaluation.md).

## Contents

[Question](#question) · [Gallery](#gallery) · [Features](#features) · [Metrics](#metrics) · [Results](#results) · [Every layer](#5-every-layer-whole-representations-dense-vs-tri-lora) · [Epoch 0](#6-epoch-0) · [Dense reference, blocks 2–4](#7-dense-reference-inside-the-jepa-window-blocks-24) · [How to read](#how-to-read-these-numbers) · [Limitations](#limitations-and-what-would-make-it-sharper) · [Reproduce](#reproduce)

## Question

Given one caption of a scene, can the model's representation find **a
different caption of the same scene written in a different sentence
structure**, among 2,047 other captions?

For a shared/private Tri-LoRA model the hope is that the shared route does
this better than the private route, because a semantic representation should
not care how the sentence is organized.

## Gallery

The gallery is generated from **held-out validation worlds only** and never
enters training. It has 2,048 captions:

- 256 validation worlds
- × 4 sentence-structure patterns: `inventory_then_relations`,
  `relations_then_inventory`, `reversed_inventory_then_relations`,
  `reversed_relations_then_inventory`
- × 2 independent lexical renderings per pattern, drawn from the same
  human-style phrase pools as the training captions (synonyms such as
  orb / ball / sphere, block / box / cube, glossy / shiny / metallic)

Every one of the 15 stored `variants.jsonl` files is byte-identical
(SHA-256 `a963a06e…`), so all models below saw exactly the same captions.

### Example: world 0

Scene: large blue metal cube, large yellow rubber sphere, small brown rubber
cylinder, small yellow rubber sphere, large yellow metal sphere, small blue
metal sphere; object 2 is right of object 1, object 3 is behind object 2.

> **Query** — pattern 0 `inventory_then_relations`, rendering 0
> The picture shows 6 objects: an orb made of matte-textured rubber, yellow and big; a brown matte cylindrical shape that looks compact; a glossy metal orb in yellow, on the big side; a ball that is small, yellow, and matte-textured rubber; a blue oversized box with a glossy metal finish; plus a blue metal sphere that looks tiny. Position-wise, the ball made of matte-textured rubber, yellow and large is positioned to the right of the blue large box with a metallic finish. Also, the brown rubber cylinder that looks tiny is behind the yellow rubbery ball that looks big.

> **Positive** — pattern 1 `relations_then_inventory`, rendering 0
> The matte ball in yellow, on the big side appears on the right side of the blue oversized box with a shiny metal finish. Position-wise, the cylindrical shape built from matte-textured rubber, brown and tiny is behind the matte-textured rubber sphere in yellow, on the sizable side. Scattered across the floor are a matte orb in yellow, on the sizable side; a blue smallish orb with a metal finish; a yellow metallic ball that looks huge; a rubbery orb in yellow, on the tiny side; a blue metal block that looks huge; plus a matte tube in brown, on the little side -- six objects in total.

> **Positive** — pattern 3 `reversed_relations_then_inventory`, rendering 1
> As for placement, the cylinder that is compact, brown, and rubber appears further from the camera than the sphere made of matte-textured rubber, yellow and huge. Position-wise, the yellow rubber ball that looks huge is off to the right of the blue metallic block that looks large. I can make out 6 shapes: a yellow little orb with a matte finish; a brown smallish cylinder with a rubber finish; a ball that is big, yellow, and matte-textured rubber; a blue little sphere with a glossy metal finish; a big blue metallic cube; and a yellow metal ball that looks sizable.

> **Not a positive, still in the ranking** — pattern 0 `inventory_then_relations`, rendering 1 (same scene, *same* pattern)
> On one side, a little yellow matte orb, a blue big block with a glossy metal finish, plus a tiny blue metallic orb. Elsewhere in the frame, a tiny brown matte-textured rubber cylindrical shape, a sizable yellow matte ball, and a yellow glossy metal ball that looks big. Position-wise, the rubber sphere in yellow, on the huge side sits to the right of the oversized blue metal box. If you look closely, the brown rubber tube that looks compact is behind the large yellow matte-textured rubber sphere.

Across all 256 worlds, a query and its positives share under half their words
(word-multiset Jaccard mean 0.46, 10th–90th percentile 0.36–0.55), because both
sentence order and word choice change.

## Features

Every feature is computed on **clean** text, averaged over content tokens
(padding excluded), then L2-normalized; similarity is cosine.

| Feature | Exists in | What it is |
|---|---|---|
| Shared update | Tri-LoRA | `B_shared A_shared x` of `blocks.4.mlp.3` only (no private, no bias) |
| Private update | Tri-LoRA | `B_text A_text x` of `blocks.4.mlp.3` only |
| `blocks.4.mlp.3` update, `W x` | Dense | dense counterpart of a LoRA update: the layer's weight times input, no bias |
| `blocks.4.mlp.3` full output | Both | everything that linear layer writes (Tri-LoRA: shared + private + bias) |
| Block-4 hidden state | Both | residual stream after Transformer block 4 |
| Input token-embedding mean | Both | the learned token embeddings before any Transformer block: an order-free, context-free bag of embeddings |

## Metrics

For each of the 2,048 captions used as a query (itself excluded):

- **Positives** are the 6 captions of the same world with a *different*
  pattern (3 other patterns × 2 renderings). The other rendering of the
  query's own pattern stays in the ranking as a distractor.
- **R@K** is the fraction of queries with at least one positive among the K
  nearest neighbors. It is reported as a percentage.
- **MRR** is the mean of 1 / rank of the first positive.
- 95% intervals come from a 1,000-repetition bootstrap over worlds; each
  resampled world brings all 8 of its queries. They are stored in the
  `results.json` files.

All values on this page were recomputed from the saved features with the
current metric code, so every row is directly comparable. R@10 reproduces the
earlier catalog values exactly.

## Results

All checkpoints are final (`epoch_003.pt`) text-only models trained on the
same 2M captions for four epochs with the same seed.

### 1. Shared vs. private branch update at `blocks.4.mlp.3`

| Model | Shared R@1 | Shared R@5 | Shared R@10 | Shared MRR | Private R@1 | Private R@5 | Private R@10 | Private MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Plain Tri-LoRA (no JEPA) | 4.4% | 14.3% | **22.9%** | 0.106 | 6.1% | 17.2% | **25.5%** | 0.130 |
| Average JEPA, no HSIC | 7.1% | 20.9% | **30.1%** | 0.149 | 3.6% | 13.0% | **20.4%** | 0.094 |
| Average JEPA + HSIC | 6.7% | 20.7% | **31.9%** | 0.151 | 6.0% | 21.0% | **32.4%** | 0.145 |
| Layerwise JEPA, no HSIC | 9.9% | 25.3% | **35.4%** | 0.184 | 7.5% | 21.5% | **30.5%** | 0.155 |
| Layerwise JEPA + HSIC | 8.4% | 23.4% | **33.0%** | 0.165 | 4.5% | 15.1% | **23.3%** | 0.109 |
| Calibrated average JEPA + HSIC | 4.2% | 13.8% | **22.4%** | 0.103 | 3.5% | 11.6% | **17.3%** | 0.087 |
| Gated data2vec, average target (collapsed) | 1.9% | 6.8% | **10.8%** | 0.055 | 1.5% | 4.8% | **6.4%** | 0.039 |
| Gated data2vec, final-layer target (collapsed) | 0.7% | 1.9% | **3.2%** | 0.021 | 0.5% | 2.5% | **4.3%** | 0.023 |
| Gated layerwise JEPA, no HSIC (new) | 3.8% | 12.8% | **20.2%** | 0.097 | 4.0% | 10.8% | **17.3%** | 0.086 |
| **Gated layerwise JEPA + HSIC** (new) | 10.8% | 28.7% | **40.1%** | 0.203 | 5.7% | 16.9% | **24.9%** | 0.121 |

Paired 95% intervals for R@10: plain shared [20.3, 25.5] vs. private
[22.7, 28.4]; layerwise no-HSIC shared [32.3, 38.5] vs. private [27.5, 33.5].

**Surface-pattern side metrics.** Template MRR retrieves a *different* scene
with the query's sentence pattern (after excluding all same-scene captions);
the last column's chance level is 0.25 for four balanced patterns. Neither is
a required success criterion: private LoRAs also learn the denoising task and
may keep semantic as well as surface information.

| Model | Shared MRR [95% CI] | Private MRR [95% CI] | Template MRR, shared / private | Nearest other-world neighbour has query's pattern, shared / private |
|---|---:|---:|---:|---:|
| Plain Tri-LoRA | 0.106 [0.092, 0.121] | 0.130 [0.114, 0.146] | 0.540 / 0.520 | 0.357 / 0.340 |
| Average JEPA, no HSIC | 0.149 [0.133, 0.167] | 0.094 [0.082, 0.107] | 0.530 / 0.551 | 0.352 / 0.354 |
| Average JEPA + HSIC | 0.151 [0.134, 0.168] | 0.145 [0.130, 0.160] | 0.679 / 0.604 | 0.549 / 0.444 |
| Layerwise JEPA, no HSIC | 0.184 [0.165, 0.205] | 0.155 [0.136, 0.175] | 0.529 / 0.518 | 0.362 / 0.340 |
| Layerwise JEPA + HSIC | 0.165 [0.148, 0.183] | 0.109 [0.096, 0.123] | 0.474 / 0.505 | 0.278 / 0.312 |
| Calibrated average JEPA + HSIC | 0.103 [0.090, 0.117] | 0.087 [0.074, 0.099] | 0.510 / 0.515 | 0.322 / 0.319 |
| Gated data2vec, average target (collapsed) | 0.055 [0.046, 0.064] | 0.039 [0.030, 0.048] | 0.617 / 0.714 | 0.437 / 0.567 |
| Gated data2vec, final-layer target (collapsed) | 0.021 [0.016, 0.027] | 0.023 [0.018, 0.028] | 0.650 / 0.684 | 0.479 / 0.525 |
| Gated layerwise JEPA, no HSIC (new) | 0.097 [0.085, 0.110] | 0.086 [0.075, 0.099] | 0.594 / 0.585 | 0.423 / 0.404 |
| **Gated layerwise JEPA + HSIC** (new) | 0.203 [0.182, 0.226] | 0.121 [0.107, 0.136] | 0.544 / 0.544 | 0.382 / 0.366 |

### 2. Whole linear layer at `blocks.4.mlp.3`, including the dense model

A dense model has no branches, so this is the closest like-for-like location.

| Model | Feature | R@1 | R@5 | R@10 | MRR |
|---|---|---:|---:|---:|---:|
| **Dense (no LoRA)** | `blocks.4.mlp.3` update, `W x` without bias | 24.1% | 52.8% | **67.7%** | 0.380 |
| **Dense (no LoRA)** | `blocks.4.mlp.3` full output | 24.2% | 52.9% | **67.7%** | 0.380 |
| Plain Tri-LoRA (no JEPA) | `blocks.4.mlp.3` full output | 6.3% | 17.0% | **26.1%** | 0.131 |
| Average JEPA, no HSIC | `blocks.4.mlp.3` full output | 5.1% | 15.9% | **23.7%** | 0.115 |
| Average JEPA + HSIC | `blocks.4.mlp.3` full output | 6.6% | 22.9% | **35.5%** | 0.158 |
| Layerwise JEPA, no HSIC | `blocks.4.mlp.3` full output | 9.1% | 24.4% | **34.5%** | 0.177 |
| Layerwise JEPA + HSIC | `blocks.4.mlp.3` full output | 6.9% | 20.1% | **29.6%** | 0.144 |
| Calibrated average JEPA + HSIC | `blocks.4.mlp.3` full output | 5.2% | 14.9% | **21.4%** | 0.109 |
| Gated data2vec, average target (collapsed) | `blocks.4.mlp.3` full output | 2.2% | 6.8% | **9.8%** | 0.054 |
| Gated data2vec, final-layer target (collapsed) | `blocks.4.mlp.3` full output | 1.1% | 3.8% | **5.8%** | 0.032 |
| Gated layerwise JEPA, no HSIC (new) | `blocks.4.mlp.3` full output | 4.0% | 11.9% | **19.3%** | 0.093 |
| **Gated layerwise JEPA + HSIC** (new) | `blocks.4.mlp.3` full output | 7.7% | 21.5% | **30.7%** | 0.155 |

The dense `W x` and full-output rows are identical to three decimals: the bias
has no effect after pooling and normalization.

### 3. Residual stream and the context-free floor

| Model | Block-4 hidden state (R@1 / R@5 / **R@10** / MRR) | Input token-embedding mean (R@1 / R@5 / **R@10** / MRR) |
|---|---|---|
| **Dense (no LoRA)** | 50.0% / 87.3% / **94.3%** / 0.660 | 24.5% / 50.7% / **62.8%** / 0.370 |
| Plain Tri-LoRA (no JEPA) | 31.7% / 67.3% / **81.8%** / 0.479 | 22.6% / 49.1% / **60.8%** / 0.351 |
| Average JEPA, no HSIC | 21.8% / 52.7% / **68.2%** / 0.363 | 22.6% / 48.6% / **60.4%** / 0.349 |
| Average JEPA + HSIC | 31.7% / 71.1% / **83.1%** / 0.490 | 22.6% / 48.6% / **60.8%** / 0.350 |
| Layerwise JEPA, no HSIC | 22.5% / 57.1% / **72.0%** / 0.383 | 22.8% / 48.0% / **60.2%** / 0.351 |
| Layerwise JEPA + HSIC | 21.0% / 51.6% / **66.3%** / 0.356 | 23.2% / 48.9% / **61.2%** / 0.354 |
| Calibrated average JEPA + HSIC | 21.6% / 53.5% / **70.0%** / 0.365 | 23.2% / 50.0% / **61.9%** / 0.359 |
| Gated data2vec, average target (collapsed) | 24.0% / 59.5% / **74.4%** / 0.406 | 23.1% / 48.5% / **60.3%** / 0.351 |
| Gated data2vec, final-layer target (collapsed) | 24.5% / 60.8% / **75.0%** / 0.410 | 22.9% / 47.3% / **60.0%** / 0.349 |
| Gated layerwise JEPA, no HSIC (new) | 16.3% / 44.1% / **59.3%** / 0.299 | 22.6% / 47.5% / **60.0%** / 0.347 |
| **Gated layerwise JEPA + HSIC** (new) | 19.9% / 49.1% / **64.2%** / 0.341 | 22.9% / 49.3% / **62.2%** / 0.355 |

Dense block-4 R@10 95% interval [93.1, 95.5] vs. plain Tri-LoRA [79.1, 84.5].

### 4. Controls without any model

| Control (no model) | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| Word counts (bag of words) | 28.4% | 56.9% | **70.9%** | 0.418 |
| TF-IDF, words | 14.3% | 30.2% | **39.5%** | 0.228 |
| TF-IDF, words + bigrams | 13.7% | 32.8% | **44.1%** | 0.235 |
| Random 384-d features | 0.2% | 1.5% | **3.0%** | 0.017 |

### 5. Every layer: whole representations, dense vs. Tri-LoRA

Same gallery and metric, scored at every block for features that exist in both
model types. For Tri-LoRA the sublayer outputs include **both** branches and the
bias (shared + private + bias). Values are percentages.

**Residual stream after each block, R@1** (R@10 saturates near 99% in the top
layers, so R@1 is the discriminating number there):

| Model | Embeddings | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Dense** | 24.5 | 41.2 | 28.4 | 25.7 | 26.6 | 50.0 | 67.1 | 70.5 | 65.1 |
| Plain Tri-LoRA | 22.6 | 38.9 | 22.2 | 27.0 | 26.6 | 31.7 | 62.9 | 61.7 | 63.8 |
| Average JEPA | 22.6 | 24.2 | 19.8 | 25.0 | 24.5 | 21.8 | 28.6 | 38.9 | 47.1 |
| Average JEPA + HSIC | 22.6 | 43.1 | 35.2 | 36.0 | 30.7 | 31.7 | 35.0 | 55.9 | 54.4 |
| Layerwise JEPA | 22.8 | 30.0 | 33.1 | 22.7 | 21.5 | 22.5 | 25.5 | 37.8 | 49.6 |
| Layerwise JEPA + HSIC | 23.2 | 28.1 | 24.6 | 17.0 | 15.3 | 21.0 | 22.3 | 47.0 | 51.2 |
| Calibrated average + HSIC | 23.2 | 15.4 | 25.3 | 27.0 | 21.6 | 21.6 | 23.5 | 47.2 | 44.5 |
| Gated data2vec, average | 23.1 | 20.7 | 15.7 | 31.6 | 26.9 | 24.0 | 26.9 | 36.4 | 50.0 |
| Gated data2vec, final layer | 22.9 | 34.7 | 21.7 | 29.6 | 30.9 | 24.5 | 26.8 | 62.1 | 59.9 |
| Gated layerwise JEPA, no HSIC (new) | 22.6 | 37.4 | 24.1 | 18.9 | 17.2 | 16.3 | 19.7 | 27.1 | 36.8 |
| **Gated layerwise JEPA + HSIC** (new) | 22.9 | 44.6 | 28.8 | 18.3 | 15.9 | 19.9 | 32.2 | 41.4 | 51.8 |

**Residual stream after each block, R@10:**

| Model | Embeddings | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Dense** | 62.8 | 86.5 | 77.1 | 73.3 | 74.8 | 94.3 | 99.4 | 99.8 | 97.9 |
| Plain Tri-LoRA | 60.8 | 86.8 | 67.9 | 73.8 | 75.3 | 81.8 | 99.4 | 98.8 | 97.6 |
| Average JEPA | 60.4 | 68.5 | 63.9 | 72.6 | 71.7 | 68.2 | 78.2 | 88.6 | 88.7 |
| Average JEPA + HSIC | 60.8 | 91.0 | 84.8 | 84.2 | 79.9 | 83.1 | 85.6 | 97.5 | 93.9 |
| Layerwise JEPA | 60.2 | 73.9 | 81.2 | 70.0 | 68.6 | 72.0 | 76.2 | 88.0 | 91.0 |
| Layerwise JEPA + HSIC | 61.2 | 71.9 | 69.8 | 59.5 | 57.9 | 66.3 | 69.2 | 92.3 | 91.8 |
| Calibrated average + HSIC | 61.9 | 52.4 | 71.0 | 76.7 | 67.6 | 70.0 | 71.9 | 93.6 | 90.0 |
| Gated data2vec, average | 60.3 | 63.3 | 59.1 | 83.4 | 79.5 | 74.4 | 77.5 | 88.5 | 92.0 |
| Gated data2vec, final layer | 60.0 | 82.8 | 64.9 | 80.0 | 81.3 | 75.0 | 78.0 | 98.8 | 97.4 |
| Gated layerwise JEPA, no HSIC (new) | 60.0 | 85.0 | 70.1 | 67.0 | 62.9 | 59.3 | 66.3 | 75.0 | 83.4 |
| **Gated layerwise JEPA + HSIC** (new) | 62.2 | 91.7 | 77.1 | 56.2 | 54.1 | 64.2 | 81.7 | 90.3 | 92.8 |

**Full MLP sublayer output (`mlp.3`), R@1:**

| Model | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Dense** | 50.6 | 16.1 | 11.7 | 14.7 | 24.1 | 58.9 | 61.0 | 39.4 |
| Plain Tri-LoRA | 49.0 | 6.0 | 3.9 | 7.6 | 6.3 | 31.7 | 21.0 | 12.8 |
| Average JEPA | 21.2 | 5.5 | 28.8 | 10.2 | 5.1 | 8.5 | 32.3 | 8.0 |
| Average JEPA + HSIC | 60.5 | 5.2 | 12.6 | 7.7 | 6.6 | 6.7 | 10.3 | 13.7 |
| Layerwise JEPA | 52.6 | 4.1 | 4.4 | 8.7 | 9.1 | 11.7 | 24.4 | 6.3 |
| Layerwise JEPA + HSIC | 57.9 | 3.0 | 3.7 | 6.4 | 6.9 | 8.6 | 23.2 | 9.4 |
| Calibrated average + HSIC | 19.5 | 58.5 | 9.1 | 3.5 | 5.2 | 17.1 | 22.7 | 11.1 |
| Gated data2vec, average | 27.7 | 14.0 | 8.6 | 8.0 | 2.2 | 18.2 | 8.9 | 15.4 |
| Gated data2vec, final layer | 43.7 | 5.6 | 4.9 | 6.6 | 1.1 | 13.2 | 23.7 | 26.2 |
| Gated layerwise JEPA, no HSIC (new) | 39.0 | 3.7 | 4.5 | 6.5 | 4.0 | 7.5 | 9.9 | 18.1 |
| **Gated layerwise JEPA + HSIC** (new) | 56.1 | 13.2 | 5.5 | 6.8 | 7.7 | 15.0 | 14.8 | 19.3 |

**Full attention sublayer output (`attn.out_proj`), R@10:**

| Model | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Dense** | 62.5 | 45.4 | 40.2 | 54.6 | 97.6 | 99.2 | 99.8 | 95.1 |
| Plain Tri-LoRA | 36.7 | 22.9 | 58.2 | 47.4 | 59.7 | 97.6 | 87.3 | 95.0 |
| Average JEPA | 49.2 | 33.3 | 18.0 | 35.9 | 41.9 | 61.5 | 78.2 | 82.7 |
| Average JEPA + HSIC | 40.7 | 47.2 | 54.2 | 49.4 | 73.3 | 61.4 | 92.7 | 90.2 |
| Layerwise JEPA | 20.4 | 61.8 | 44.0 | 31.6 | 43.0 | 48.9 | 75.9 | 88.3 |
| Layerwise JEPA + HSIC | 24.9 | 43.7 | 21.3 | 24.4 | 46.4 | 45.9 | 83.5 | 91.5 |
| Calibrated average + HSIC | 23.3 | 38.1 | 41.0 | 20.8 | 56.2 | 37.2 | 90.0 | 77.9 |
| Gated data2vec, average | 27.2 | 29.1 | 68.2 | 50.0 | 44.3 | 54.8 | 75.9 | 92.5 |
| Gated data2vec, final layer | 33.6 | 17.5 | 59.7 | 52.6 | 46.5 | 56.8 | 97.5 | 93.2 |
| Gated layerwise JEPA, no HSIC (new) | 54.7 | 45.4 | 35.6 | 28.9 | 40.6 | 53.3 | 53.8 | 76.6 |
| **Gated layerwise JEPA + HSIC** (new) | 47.3 | 29.4 | 29.4 | 28.2 | 58.2 | 71.1 | 73.3 | 86.8 |

95% intervals for residual R@1, dense vs. plain Tri-LoRA: L4 [47.1, 52.8] vs.
[28.7, 34.6]; L5 [64.5, 69.6] vs. [60.2, 65.6]; L6 [68.0, 72.9] vs.
[58.9, 64.6]; L7 [62.4, 67.8] vs. [60.9, 66.5].

**Confound check.** On dense, plain Tri-LoRA, layerwise JEPA and average
JEPA + HSIC at blocks 4 and 7: removing the sublayer bias changes nothing (bias
norm 0.5–1.1 against pooled norms of 6–178), and mean-centering or removing
the top principal component moves scores by a few points in either direction
without closing or reversing any gap. The differences are not a bias or
single-direction artifact.

Script: [evaluate_layer_retrieval_sweep.py](/home/zd25e122/clevr_discrete_diffusion/evaluate_layer_retrieval_sweep.py);
results: `outputs/layer_retrieval_sweep/<label>.json`. Its vectorized metric is
checked against `query_statistics` on every model; CPU/GPU float rounding swaps
near-tied neighbors for at most 9 of 2,048 queries, with identical means.

### 6. Epoch 0

Branch retrieval at `blocks.4.mlp.3` after one pass over the 2M captions, same
gallery and metric. Most models are at 9–15% R@10 with overlapping
intervals. The clear exception is average JEPA + HSIC, whose shared update
already reaches 21.1% [18.6, 23.9]. The two gated-predictor runs are within
the range of their layer-local baselines.

| Model (epoch 0) | Shared R@1 | Shared R@5 | Shared R@10 [95% CI] | Shared MRR | Private R@1 | Private R@5 | Private R@10 [95% CI] | Private MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Plain Tri-LoRA | 2.2% | 6.9% | 11.4% [9.7, 13.3] | 0.058 | 2.5% | 7.6% | 12.9% [10.8, 14.9] | 0.064 |
| Average JEPA, no HSIC | 2.5% | 9.3% | 14.6% [12.6, 17.0] | 0.070 | 2.1% | 7.2% | 12.3% [10.5, 14.2] | 0.060 |
| Average JEPA + HSIC | 3.1% | 12.3% | 21.1% [18.6, 23.9] | 0.093 | 2.6% | 9.2% | 15.3% [13.2, 17.6] | 0.072 |
| Layerwise JEPA, no HSIC | 1.3% | 6.9% | 11.0% [9.3, 12.8] | 0.051 | 2.2% | 8.1% | 13.5% [11.7, 15.3] | 0.065 |
| Layerwise JEPA + HSIC | 2.5% | 8.2% | 13.7% [11.8, 15.5] | 0.066 | 1.8% | 7.3% | 12.1% [10.1, 14.2] | 0.056 |
| Calibrated average JEPA + HSIC | 2.3% | 7.5% | 11.9% [10.2, 13.7] | 0.060 | 1.4% | 4.9% | 8.6% [7.2, 10.3] | 0.044 |
| **Gated layerwise JEPA, no HSIC** (epoch 0) | 1.5% | 6.3% | 10.1% [8.4, 11.8] | 0.050 | 1.2% | 4.7% | 8.6% [6.9, 10.5] | 0.042 |
| **Gated layerwise JEPA + HSIC** (epoch 0) | 1.8% | 6.8% | 12.1% [10.3, 14.1] | 0.056 | 2.8% | 7.6% | 12.5% [10.6, 14.5] | 0.064 |

The gated-predictor runs keep the JEPA predictor and send its gradient
end-to-end into the shared LoRA of blocks 2–4 ([plan](../plans/gated_predictor_layerwise_jepa.md)).
Whole-layer epoch-0 results for them, their baselines, plain Tri-LoRA and
dense are in `outputs/layer_retrieval_sweep_epoch000/`.

### 7. Dense reference inside the JEPA window (blocks 2–4)

The same feature as the shared-branch table in section 1 (content-token mean,
L2-normalized `blocks.L.mlp.3` output), scored at every block that receives
JEPA supervision. Dense has no branches, so its whole-layer `W x` is the
reference: what one MLP write of a matched model carries. R@10 in %.

| Model | L2 shared / private / full | L3 shared / private / full | L4 shared / private / full |
|---|---:|---:|---:|
| **Dense** (`W x`, the whole layer) | **42.8** | **52.1** | **67.7** |
| Plain Tri-LoRA | 20.7 / 20.5 / 21.8 | 32.9 / 30.4 / 32.4 | 22.9 / 25.5 / 26.1 |
| Average JEPA, no HSIC | 66.6 / 60.2 / 67.3 | 40.5 / 31.7 / 38.2 | 30.1 / 20.4 / 23.7 |
| Average JEPA + HSIC | 48.5 / 36.6 / 46.7 | 22.2 / 29.3 / 30.7 | 31.9 / 32.4 / 35.5 |
| Layerwise JEPA, no HSIC | 20.7 / 22.6 / 22.8 | 32.0 / 34.4 / 36.0 | 35.4 / 30.5 / 34.5 |
| Layerwise JEPA + HSIC | 20.4 / 16.5 / 18.3 | 28.6 / 25.6 / 30.4 | 33.0 / 23.3 / 29.6 |
| Calibrated average JEPA + HSIC | 34.5 / 36.3 / 37.1 | 16.6 / 16.1 / 18.2 | 22.4 / 17.3 / 21.4 |
| Gated layerwise JEPA, no HSIC | 14.2 / 13.9 / 22.1 | 18.1 / 20.8 / 28.9 | 20.2 / 17.3 / 19.3 |
| **Gated layerwise JEPA + HSIC** | 23.0 / 17.3 / 24.2 | 19.6 / 29.4 / 31.7 | 40.1 / 24.9 / 30.7 |

Residual stream after the same blocks, R@10 (%):

| Model | L2 | L3 | L4 |
|---|---:|---:|---:|
| **Dense** | 73.3 | 74.8 | 94.3 |
| Plain Tri-LoRA | 73.8 | 75.3 | 81.8 |
| Average JEPA, no HSIC | 72.6 | 71.7 | 68.2 |
| Average JEPA + HSIC | 84.2 | 79.9 | 83.1 |
| Layerwise JEPA, no HSIC | 70.0 | 68.6 | 72.0 |
| Layerwise JEPA + HSIC | 59.5 | 58.0 | 66.3 |
| Calibrated average JEPA + HSIC | 76.7 | 67.6 | 70.0 |
| Gated layerwise JEPA, no HSIC | 67.0 | 62.9 | 59.3 |
| **Gated layerwise JEPA + HSIC** | 56.2 | 54.1 | 64.2 |

Reading:

- Dense's own write grows from 42.8% at block 2 to 67.7% at block 4. The
  best Tri-LoRA shared update at block 4 is gated JEPA + HSIC at 40.1%,
  27.6 points below dense.
- **Dense is not a clean ceiling for this metric.** Average JEPA's block-2
  shared update reaches 66.6%, above dense at block 2 (42.8%) and level with
  dense at block 4, because an early layer stays close to the input words:
  the context-free token-embedding average alone scores 60–63%. A branch can
  therefore "reach dense" on retrieval by keeping more word content, without
  becoming more semantic.
- The dense residual stream is 94.3% at block 4, and no Tri-LoRA model is
  above 84.2% in blocks 2–4.

The binding-swap page gives the same comparison for binding, where the gap to
dense is not lexical: [binding at blocks 2–4](binding_swap_evaluation.md#dense-reference-inside-the-jepa-window-blocks-24).

Results: `outputs/layer_retrieval/jepa_window_mlp3_retrieval.json`.

## How to read these numbers

**1. A large part of this task is lexical.** Plain word counts with no model
reach R@10 = 70.9%, and every model's context-free token-embedding average
reaches about 60–63%. Scenes differ mostly in *which* colors, shapes,
materials and sizes they contain, and those words survive any reordering.
(TF-IDF scores lower than raw counts because it down-weights the frequent
attribute words that identify a scene.)

**2. Every LoRA branch update is below that floor.** The best shared update
(40.1%, gated JEPA + HSIC) and every whole `blocks.4.mlp.3` output in a Tri-LoRA model (at most
35.5%) retrieve worse than simply averaging the input embeddings. These
features are *updates written into the residual stream*, not complete
representations, and they keep only part of the word-level identity. The
branch table is therefore evidence of a **relative** shift, where JEPA moves
this signal from private toward shared, not evidence that the shared update is
a strong semantic representation in absolute terms.

**3. Dense is much stronger at the same location.** At `blocks.4.mlp.3`,
dense reaches 67.7% against at most 35.5% for any Tri-LoRA model, and in the
block-4 residual stream 94.3% against 81.8% for plain Tri-LoRA. Dense is also
the better text model at this budget (validation loss at t = 0.75: 1.0758 vs.
1.1638), so part of this gap is simply model quality, not only representation
structure.

**4. Only the residual stream clearly beats the lexical floor.** Dense gains
+31.5 points of R@10 over its own embedding average, plain Tri-LoRA +21.0.
All JEPA/HSIC variants except average JEPA + HSIC (83.1%) have a *lower*
block-4 residual score than plain Tri-LoRA (66–75%): shaping the shared update
is not free for the representation the model actually carries forward.

**5. The collapsed gated data2vec runs are damaged locally.** Their branch
updates fall to 3–11%, near the 3% random floor, yet their block-4 residual
stream still retrieves at 74–75%. The collapse is confined to the supervised
module and the rest of the network routes around it.

**6. Plain Tri-LoRA matches dense in the residual stream; JEPA variants do
not.** Plain Tri-LoRA is within the dense confidence interval at the final
block (R@1 63.8% vs. 65.1%) and at blocks 0, 2 and 3. Its clear gaps are
block 1 (22.2% vs. 28.4%), block 4 (31.7% vs. 50.0%) and block 6 (61.7% vs.
70.5%). Every JEPA/HSIC variant is
far below both across the stack, with final-block R@1 of 37–60% and a mean
per-layer gap to dense of 7–22 R@1 points, against 5 for plain Tri-LoRA. So
the parameterization alone costs little. The auxiliary objectives, which move
branch-level signal toward shared, lower what the whole representation retains
even though diffusion validation loss barely changes.

**7. Adding private does not bring the MLP write near dense.** Except at block
0, the full Tri-LoRA MLP output retrieves far worse than the dense MLP output
at every layer (plain Tri-LoRA R@1 4–32% vs. dense 12–61%). Each Tri-LoRA
sublayer write carries less scene identity on its own, yet the residual stream
it accumulates into catches up with dense by the top of the network.

## Limitations and what would make it sharper

- **It cannot tell compositional semantics from a bag of words.** A
  representation that only knows *which* attribute words occur can score
  highly. The most informative addition is **binding-swap hard negatives**:
  same words, different bindings, such as "a red cube and a blue sphere" vs.
  "a blue cube and a red sphere", or swapped relation arguments. A bag of words
  scores exactly at chance on those by construction, so any gain there is
  compositional.
- **Report the lexical floor next to every model.** A model feature is only
  interesting where it beats the word-count and embedding-average controls.
- **Single location.** Branch results are for `blocks.4.mlp.3` only; other
  layers and modules may differ.
- **Text only.** The shared/private claim is ultimately cross-modal, and this
  test does not touch images.

## Reproduce

```bash
# Shared/private branch features (Tri-LoRA only)
./scripts/launch_shared_private_retrieval_vnode10.sh LABEL CHECKPOINT GPU

# Whole-layer features, dense or Tri-LoRA, on the identical gallery
python evaluate_layer_retrieval.py --checkpoint CHECKPOINT \
  --gallery outputs/shared_private_semantic_template_retrieval/plain_lora_epoch003/variants.jsonl \
  --output outputs/layer_retrieval/LABEL

# All final checkpoints at once
./scripts/run_layer_retrieval_all_final.sh
```

## Artifacts

- Branch features and per-model reports: `outputs/shared_private_semantic_template_retrieval/<label>/`
- Branch metrics recomputed for this page: `outputs/layer_retrieval/branch_metrics_recomputed.json`
- Whole-layer results, dense included: `outputs/layer_retrieval/<label>/results.json`
- Lexical and random controls: `outputs/layer_retrieval/bag_of_words_control/results.json`
- Evaluators: [evaluate_shared_private_retrieval.py](/home/zd25e122/clevr_discrete_diffusion/evaluate_shared_private_retrieval.py), [evaluate_layer_retrieval.py](/home/zd25e122/clevr_discrete_diffusion/evaluate_layer_retrieval.py)

## Original pre-registration

This evaluation was specified before any results existed. Its pre-registered
expectations, kept here for reference:

| Test | Positives | Expectation if the routes specialize |
|---|---|---|
| Cross-pattern semantic R@1/5/10, MRR | Same scene, different pattern | Shared > private |
| Same-pattern semantic MRR | Same scene, same pattern | Both may be high; control |
| Template retrieval | Different scene, same pattern | Private may exceed shared (exploratory) |
| Semantic − template MRR | — | Larger for shared than private |

Guardrails set at the time: report the full shared/private table and
representative neighbours without manual selection; a private route that also
retrieves semantics is not a failure; the gallery must never come from the
training corpus. The lexical-floor controls in section 4 were added later,
after the task turned out to be largely solvable from word content.

