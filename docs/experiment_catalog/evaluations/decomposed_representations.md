# Decomposed representations: layers, sublayers, and shared/private streams

> **Type:** evaluation, text-only study · **Status:** complete for all 11 final checkpoints  
> **Question:** where in the network, and in which branch, do attribute binding, relation binding, and retrieval live? Is the shared branch's contribution to the representation more semantic than the private branch's?  
> **Models:** dense, plain Tri-LoRA, all JEPA/HSIC variants, both gated data2vec runs, both gated-predictor runs  
> **Script:** [evaluate_decomposed_representations.py](/home/zd25e122/clevr_discrete_diffusion/evaluate_decomposed_representations.py) · **Launcher:** [launch_decomposed_eval_final_vnode10.sh](/home/zd25e122/clevr_discrete_diffusion/scripts/launch_decomposed_eval_final_vnode10.sh) · **Results:** `outputs/decomposed_eval/<label>.json`  
> **Menu:** [experiment catalog](../README.md)

## Summary

- **At the level of the representation, shared and private are
  interchangeable in every model.** Summing everything each branch has
  written into the residual stream, the private stream binds attributes at
  least as well as the shared stream in all 10 Tri-LoRA models, at block 4 and
  at block 7 (by 0 to 3.1 points). Relation binding and retrieval
  favour one stream or the other depending on model and layer, with no
  consistent shared advantage.
- **This holds for the best model too.** Gated JEPA + HSIC's single
  `blocks.4.mlp.3` shared update is clearly more relational than its private
  update (96.4% vs. 86.7%), but in its full streams private matches or beats
  shared: relation 87.2% vs. 88.4% at block 4 and 90.1% vs. 94.4% at block 7.
  The module-level split does not survive into what the network carries
  forward.
- **In dense, binding lives in the MLP writes and retrieval in the attention
  writes.** At block 4, dense's MLP output binds attributes at 91.5% (R@10
  67.7%) while its attention output retrieves at 97.6% (attribute binding
  59.3%). Tri-LoRA attention outputs and updates bind attributes near chance
  (49–54%) in blocks 2–4, and no Tri-LoRA sublayer output or branch update in
  blocks 2–4 binds attributes above
  58.6%.
- **Attribute binding appears late in Tri-LoRA.** Plain Tri-LoRA's residual
  stream goes from 58.0% at block 4 to 72.4% at block 5; dense is already at
  72.7% at block 4 and 91.7% at block 6.

## Contents

[What is measured](#what-is-measured) · [Verification](#verification) · [Shared vs. private streams](#1-shared-vs-private-streams-every-model) · [Every layer](#2-every-layer) · [Sublayers](#3-sublayers-and-branches-in-blocks-24) · [How to read](#how-to-read-these-results) · [Reproduce](#reproduce)

## What is measured

The Tri-LoRA models have no frozen base weight, so every attention and MLP
sublayer writes exactly shared update + private update + bias into the
residual stream. The hidden state after block L therefore decomposes exactly:

h_L = h0 + Σ_(i≤L) ( b_att,i + Δshared_att,i + Δprivate_att,i + b_mlp,i + Δshared_mlp,i + Δprivate_mlp,i )

where h0 is the token + position embedding. From this:

| Feature | Definition |
|---|---|
| Residual h_L | hidden state after block L |
| All writes | h_L − h0: every sublayer write up to block L |
| **Shared writes** | Σ_(i≤L) (Δshared_att,i + Δshared_mlp,i): everything the shared branch has written |
| **Private writes** | Σ_(i≤L) (Δprivate_att,i + Δprivate_mlp,i): everything the private branch has written |
| h0 + shared / private writes | the same, plus the embedding |
| Attention / MLP output (blocks 2–4) | the whole sublayer write; for Tri-LoRA shared + private + bias |
| Attention / MLP shared, private (blocks 2–4) | the individual native updates |

Biases are left out of the two streams: they are constant across tokens and
belong to neither branch. Dense has no branches, so only residual, all
writes, and sublayer outputs exist for it.

Each feature is scored on three tests with the same data as the existing
evaluations: **attribute binding** and **relation binding** (the
[binding-swap](binding_swap_evaluation.md) minimal-pair probe, 50% = no
binding information) and **retrieval** R@10 (the
[cross-pattern](cross_pattern_semantic_retrieval.md) gallery). Every feature
is a content-token mean, L2-normalized.

**What the streams mean.** Every update was computed from the full mixed
stream, so the shared stream can contain information private wrote earlier
and the shared branch then read. These features measure what each branch
*contributes* to the representation, not what it would compute on its own.
Adding h0 gives both streams the word content of the input: it raises
retrieval toward the lexical floor and cannot change binding (word-level
features score exactly 50%), so the streams without h0 are the main
comparison.

## Verification

- **The decomposition is exact.** For every model, the evaluator rebuilds
  h_L from h0 and the sublayer writes, and from h0 + shared writes + private
  writes + biases, on real tokens, and refuses to run if they differ. Largest
  difference over all 11 models: 7.2e-05 (float32 rounding).
- **Features that overlap earlier evaluations reproduce them exactly**, e.g.
  plain Tri-LoRA residual block 4 (binding 62.8%, R@10 81.8%), block 7 (78.5%,
  97.6%), `blocks.4.mlp.3` shared (62.7%, 22.9%), and token embeddings (50.0%,
  60.8%).

## Results

Cells are **attribute binding / relation binding / retrieval R@10**, in %.

### 1. Shared vs. private streams, every model

| Model | Block 4: shared writes | Block 4: private writes | Block 7: shared writes | Block 7: private writes | Block 7: residual |
|---|---|---|---|---|---|
| **Dense** | — | — | — | — | 91.3 / 48.2 / 97.9 |
| Plain Tri-LoRA | 56.2 / 84.0 / 74.8 | 58.4 / 78.0 / 79.2 | 74.5 / 85.7 / 97.9 | 76.3 / 90.3 / 96.8 | 75.8 / 89.1 / 97.6 |
| Average JEPA, no HSIC | 54.2 / 61.0 / 60.3 | 56.1 / 66.8 / 62.6 | 66.6 / 95.9 / 88.2 | 68.5 / 96.9 / 87.3 | 68.9 / 96.6 / 88.7 |
| Average JEPA + HSIC | 56.8 / 87.9 / 68.2 | 56.9 / 90.3 / 82.2 | 71.2 / 76.3 / 93.1 | 72.4 / 78.5 / 92.6 | 71.9 / 77.2 / 93.9 |
| Layerwise JEPA, no HSIC | 54.0 / 81.6 / 61.5 | 54.6 / 79.4 / 71.8 | 69.9 / 98.1 / 92.5 | 71.0 / 99.8 / 88.0 | 72.1 / 99.3 / 91.0 |
| Layerwise JEPA + HSIC | 53.1 / 70.0 / 53.1 | 54.0 / 68.5 / 61.8 | 73.0 / 66.1 / 91.0 | 74.8 / 69.2 / 90.1 | 74.4 / 63.0 / 91.8 |
| Calibrated average JEPA + HSIC | 53.4 / 78.2 / 59.0 | 56.3 / 72.9 / 58.5 | 73.6 / 75.3 / 89.2 | 76.6 / 68.0 / 87.1 | 76.9 / 67.1 / 90.0 |
| Gated data2vec, average (collapsed) | 55.7 / 72.9 / 62.1 | 58.3 / 85.0 / 68.0 | 73.1 / 91.3 / 90.5 | 74.8 / 92.0 / 90.4 | 74.7 / 93.0 / 92.0 |
| Gated data2vec, final layer (collapsed) | 56.1 / 82.1 / 60.8 | 56.5 / 88.4 / 68.2 | 73.2 / 88.6 / 97.6 | 75.2 / 81.1 / 96.1 | 74.8 / 83.8 / 97.4 |
| Gated layerwise JEPA, no HSIC | 52.0 / 89.6 / 36.8 | 53.0 / 90.6 / 40.2 | 65.9 / 86.7 / 65.0 | 68.9 / 91.5 / 75.0 | 69.8 / 86.9 / 83.4 |
| **Gated layerwise JEPA + HSIC** | 53.5 / 87.2 / 43.3 | 55.7 / 88.4 / 44.4 | 76.0 / 90.1 / 83.3 | 76.5 / 94.4 / 90.9 | 77.0 / 94.7 / 92.8 |

Attribute binding with 95% intervals:

| Model | Block 4 attribute: shared / private | Block 7 attribute: shared / private |
|---|---|---|
| Plain Tri-LoRA | 56.2 [55.3–57.2] / 58.4 [57.4–59.5] | 74.5 [73.4–75.6] / 76.3 [75.3–77.4] |
| Layerwise JEPA, no HSIC | 54.0 [53.1–54.9] / 54.6 [53.7–55.5] | 69.9 [68.8–70.9] / 71.0 [69.9–72.1] |
| **Gated layerwise JEPA + HSIC** | 53.5 [52.6–54.4] / 55.7 [54.8–56.6] | 76.0 [75.0–77.1] / 76.5 [75.4–77.7] |

Across the 10 Tri-LoRA models, the private stream binds attributes better
than the shared stream in **10 of 10** at block 4 and **10 of 10** at block 7.
The shared stream binds relations better in 4 of 10 at block 4 and 2 of 10 at
block 7; it retrieves better in 1 of 10 at block 4 and 8 of 10 at block 7.

### 2. Every layer

**Attribute binding (%)**

| Model | Stream | Emb. | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Dense** | residual h_L | 49.9 | 49.3 | 53.1 | 55.9 | 60.7 | 72.7 | 85.9 | 91.7 | 91.3 |
|  | all writes h_L − h0 |  | 49.2 | 52.8 | 55.9 | 60.2 | 72.4 | 85.9 | 91.7 | 91.2 |
| Plain Tri-LoRA | residual h_L | 50.5 | 49.9 | 50.1 | 50.2 | 53.9 | 58.0 | 72.4 | 74.9 | 75.8 |
|  | shared writes |  | 50.3 | 50.5 | 50.5 | 52.9 | 56.2 | 70.1 | 73.1 | 74.5 |
|  | private writes |  | 50.4 | 50.0 | 49.8 | 53.6 | 58.4 | 72.7 | 76.0 | 76.3 |
| Layerwise JEPA, no HSIC | residual h_L | 49.9 | 49.5 | 51.4 | 52.9 | 53.4 | 55.1 | 56.6 | 66.4 | 72.1 |
|  | shared writes |  | 49.9 | 50.4 | 52.3 | 52.1 | 54.0 | 55.9 | 63.5 | 69.9 |
|  | private writes |  | 50.0 | 51.2 | 52.1 | 53.0 | 54.6 | 57.2 | 66.9 | 71.0 |
| **Gated JEPA + HSIC** | residual h_L | 49.9 | 50.5 | 50.4 | 51.6 | 52.4 | 56.7 | 63.3 | 71.7 | 77.0 |
|  | shared writes |  | 50.2 | 50.9 | 51.6 | 51.7 | 53.5 | 60.3 | 69.9 | 76.0 |
|  | private writes |  | 50.4 | 50.2 | 51.0 | 51.8 | 55.7 | 62.5 | 71.4 | 76.5 |

**Relation binding (%)**

| Model | Stream | Emb. | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Dense** | residual h_L | 47.9 | 55.2 | 55.7 | 78.2 | 89.3 | 96.9 | 98.8 | 99.0 | 48.2 |
|  | all writes h_L − h0 |  | 56.2 | 55.2 | 78.0 | 89.3 | 97.1 | 99.0 | 99.0 | 47.7 |
| Plain Tri-LoRA | residual h_L | 51.2 | 63.9 | 77.5 | 78.2 | 79.9 | 82.1 | 96.6 | 96.1 | 89.1 |
|  | shared writes |  | 64.9 | 76.8 | 78.9 | 79.2 | 84.0 | 96.6 | 95.6 | 85.7 |
|  | private writes |  | 59.8 | 73.1 | 72.4 | 81.4 | 78.0 | 94.7 | 94.4 | 90.3 |
| Layerwise JEPA, no HSIC | residual h_L | 52.5 | 53.8 | 66.6 | 69.5 | 60.5 | 80.9 | 92.3 | 90.6 | 99.3 |
|  | shared writes |  | 54.7 | 60.5 | 69.5 | 62.7 | 81.6 | 90.3 | 86.7 | 98.1 |
|  | private writes |  | 52.3 | 68.0 | 70.9 | 64.9 | 79.4 | 92.5 | 91.8 | 99.8 |
| **Gated JEPA + HSIC** | residual h_L | 51.5 | 57.6 | 63.4 | 79.9 | 80.9 | 88.9 | 96.9 | 88.6 | 94.7 |
|  | shared writes |  | 59.8 | 64.4 | 80.4 | 79.9 | 87.2 | 95.6 | 83.5 | 90.1 |
|  | private writes |  | 52.5 | 63.7 | 76.0 | 77.7 | 88.4 | 96.1 | 91.8 | 94.4 |

**Retrieval R@10 (%)**

| Model | Stream | Emb. | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Dense** | residual h_L | 82.3 | 86.5 | 77.1 | 73.3 | 74.8 | 94.3 | 99.4 | 99.8 | 97.9 |
|  | all writes h_L − h0 |  | 82.0 | 71.7 | 69.9 | 72.3 | 94.0 | 99.3 | 99.8 | 97.9 |
| Plain Tri-LoRA | residual h_L | 80.9 | 86.8 | 67.9 | 73.8 | 75.3 | 81.8 | 99.4 | 98.8 | 97.6 |
|  | shared writes |  | 63.3 | 42.7 | 51.2 | 61.6 | 74.8 | 98.3 | 98.3 | 97.9 |
|  | private writes |  | 83.9 | 68.4 | 75.9 | 75.3 | 79.2 | 99.0 | 98.5 | 96.8 |
| Layerwise JEPA, no HSIC | residual h_L | 80.9 | 73.9 | 81.2 | 70.0 | 68.6 | 72.0 | 76.2 | 88.0 | 91.0 |
|  | shared writes |  | 49.3 | 65.3 | 56.2 | 55.8 | 61.5 | 66.3 | 82.0 | 92.5 |
|  | private writes |  | 84.9 | 81.4 | 72.0 | 69.9 | 71.8 | 75.3 | 86.3 | 88.0 |
| **Gated JEPA + HSIC** | residual h_L | 81.5 | 91.7 | 77.1 | 56.2 | 54.1 | 64.2 | 81.7 | 90.3 | 92.8 |
|  | shared writes |  | 71.9 | 56.2 | 31.3 | 34.6 | 43.3 | 56.9 | 73.7 | 83.3 |
|  | private writes |  | 91.8 | 79.0 | 34.4 | 37.5 | 44.4 | 64.7 | 82.5 | 90.9 |

### 3. Sublayers and branches in blocks 2–4

| Model | Feature | Block 2 | Block 3 | Block 4 |
|---|---|---|---|---|
| **Dense** | attention output | 53.6 / 59.6 / 40.2 | 53.8 / 69.2 / 54.6 | 59.3 / 89.6 / 97.6 |
|  | MLP output | 60.3 / 85.0 / 42.8 | 69.0 / 99.0 / 52.3 | 91.5 / 100.0 / 67.7 |
| Plain Tri-LoRA | attention output | 50.4 / 64.2 / 58.2 | 51.4 / 48.4 / 47.4 | 51.8 / 65.9 / 59.7 |
|  | attention, shared | 50.4 / 66.6 / 47.0 | 51.1 / 46.5 / 41.8 | 51.7 / 69.7 / 56.8 |
|  | attention, private | 50.3 / 63.0 / 58.5 | 51.0 / 46.2 / 47.7 | 51.6 / 61.5 / 54.5 |
|  | MLP output | 50.5 / 59.8 / 21.8 | 53.6 / 82.3 / 32.4 | 58.4 / 81.8 / 26.1 |
|  | MLP, shared | 50.9 / 54.7 / 20.7 | 53.8 / 84.3 / 32.9 | 57.8 / 82.1 / 22.9 |
|  | MLP, private | 50.1 / 64.2 / 20.5 | 53.3 / 78.9 / 30.4 | 57.9 / 81.6 / 25.5 |
| Layerwise JEPA, no HSIC | attention output | 51.1 / 65.6 / 44.0 | 50.5 / 67.1 / 31.6 | 52.0 / 62.5 / 43.0 |
|  | attention, shared | 51.3 / 72.9 / 44.3 | 50.6 / 69.2 / 31.5 | 52.2 / 62.0 / 50.6 |
|  | attention, private | 50.9 / 55.0 / 40.8 | 51.1 / 68.8 / 31.0 | 51.8 / 61.0 / 36.0 |
|  | MLP output | 51.3 / 56.4 / 22.8 | 51.9 / 42.6 / 36.0 | 53.9 / 90.8 / 34.5 |
|  | MLP, shared | 50.9 / 53.3 / 20.7 | 52.1 / 54.2 / 32.0 | 55.5 / 93.9 / 35.4 |
|  | MLP, private | 51.7 / 54.0 / 22.6 | 51.9 / 41.4 / 34.4 | 52.7 / 84.5 / 30.5 |
| **Gated JEPA + HSIC** | attention output | 50.6 / 70.7 / 29.4 | 51.0 / 70.5 / 28.2 | 50.9 / 80.4 / 58.2 |
|  | attention, shared | 49.8 / 61.5 / 22.5 | 51.2 / 73.6 / 22.5 | 50.7 / 77.5 / 54.6 |
|  | attention, private | 50.0 / 69.0 / 18.7 | 51.4 / 69.5 / 22.9 | 51.2 / 82.3 / 49.1 |
|  | MLP output | 52.4 / 80.9 / 24.2 | 52.3 / 59.3 / 31.7 | 56.8 / 93.0 / 30.7 |
|  | MLP, shared | 52.0 / 81.8 / 23.0 | 52.6 / 68.5 / 19.6 | 57.6 / 96.4 / 40.1 |
|  | MLP, private | 51.8 / 79.7 / 17.3 | 51.5 / 50.6 / 29.4 | 54.9 / 86.7 / 24.9 |

The full grid for all 11 models (68 features for Tri-LoRA, 32 for dense, with
bootstrap intervals and ordinary-probe accuracy) is in
`outputs/decomposed_eval/<label>.json`.

## How to read these results

**1. The shared/private split exists only inside single modules.** JEPA
supervision makes one module's shared update more relational than its
private update, but the attention sublayers and the other blocks write
comparable information through the private branch, so the accumulated
streams end up equivalent. If the goal is a shared *route* that carries the
semantics while private carries modality-specific detail, the objectives so
far do not produce it.

**2. Private is never the less semantic stream.** In every Tri-LoRA model the
private stream is at least as good on attribute binding. Private LoRAs
receive the full denoising gradient, and denoising text requires knowing
which attribute belongs to which object; nothing in the current objectives
discourages private from carrying that.

**3. Dense separates the roles of its sublayers; Tri-LoRA does not yet.**
Dense's block-4 MLP write is highly compositional while its attention write
is highly retrieval-friendly. In Tri-LoRA every sublayer in blocks 2–4 stays
at or below 58.6% on attribute binding.

**4. Consequences for the next objective.** Supervising one module's shared
update is not enough to shape the streams. Candidates: apply the JEPA (and
HSIC) objective to the *cumulative* shared writes instead of a single module;
add pressure that removes binding information from the private stream (e.g.
an adversarial attribute probe with gradient reversal on the private writes);
and move supervision toward blocks 3–5, where binding forms.

## Reproduce

```bash
./scripts/launch_decomposed_eval_final_vnode10.sh
# or one checkpoint:
python evaluate_decomposed_representations.py --output-dir outputs/decomposed_eval \
  --checkpoint LABEL=path/to/epoch_003.pt
```

The evaluator reads captions and minimal pairs from
`outputs/binding_swap_eval/{captions,items}.jsonl` and the retrieval gallery
from `outputs/shared_private_semantic_template_retrieval/plain_lora_epoch003/variants.jsonl`,
so its numbers are directly comparable with those pages.
