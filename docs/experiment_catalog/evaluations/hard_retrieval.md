# Hard retrieval: finding the true scene among lexically identical swaps

> **Type:** evaluation, text-only study · **Status:** complete for the 11 Tri-LoRA / modulewise checkpoints plus an untrained control; **superseded by [semantic d′](semantic_dprime.md) for later families** — see [coverage](#coverage-which-models-are-not-in-this-table) · **Updated:** 2026-09-24  
> **Question:** without any trained probe, does a representation's cosine geometry place a caption closer to the same scene than to scenes that use exactly the same words with swapped bindings?  
> **Models:** dense, plain Tri-LoRA, all JEPA/HSIC variants, both gated data2vec runs, both gated-predictor runs, untrained dense (random weights)  
> **Script:** [evaluate_hard_retrieval.py](/home/zd25e122/clevr_discrete_diffusion/evaluate_hard_retrieval.py) · **Launcher:** [launch_hard_retrieval_final_vnode10.sh](/home/zd25e122/clevr_discrete_diffusion/scripts/launch_hard_retrieval_final_vnode10.sh) · **Results:** `outputs/hard_retrieval_eval/`  
> **Menu:** [experiment catalog](../README.md)

## Summary

- **Shortcuts are removed.** Word counts and token-embedding averages score
  exactly chance, and every one of an untrained model's 24 features
  stays within 10.6–14.2% (attribute, chance 12.5%) and
  41.1–45.3% (relation, chance 43.0%).
- **This is hard even for dense.** Its best feature, the block-7 residual
  stream, picks the true scene 22.8% of the time among 8 attribute
  candidates and 71.5% for relations.
- **Plain Tri-LoRA is second** (19.7% / 62.2%). **Every JEPA/HSIC variant is lower**
  (block 7: 12.5–16.1% / 49.7–58.0%), including gated JEPA + HSIC
  (14.7% / 56.6%).
- **Inside the JEPA window nothing is above chance on attributes.** In
  blocks 2–4, every Tri-LoRA feature and dense's residual stream are at
  chance for attribute swaps; only dense's block-4 MLP output is clearly
  above it (17.9%), and it also leads on relations (62.3%).
- **Shared and private are again interchangeable**, now in raw geometry:
  neither stream is consistently ahead.

Compared with the [binding-swap probe](binding_swap_evaluation.md), which asks
whether binding is *linearly decodable*, this test asks whether binding
*organizes the representation's similarity structure*. A probe can learn to
ignore word position and pick out small binding signals; cosine similarity
cannot, so these scores are much lower.

## Contents

[Why](#why-a-hard-version) · [Construction](#construction) · [Example](#example) · [Metric](#metric) · [Controls](#controls) · [Results](#results) · [Shared/private in blocks 2–4](#inside-the-jepa-window-shared-and-private-at-blocks-24) · [Every block and sublayer](#every-block-every-sublayer-dense-plain-tri-lora-gated-jepa--hsic) · [How to read](#how-to-read) · [Reproduce](#reproduce)

## Why a hard version

In the [cross-pattern retrieval](cross_pattern_semantic_retrieval.md) every
other scene is a distractor. Different scenes mention different attribute
words, so plain word counts already reach R@10 = 70.9%. Here every candidate
uses the same words, so the only way to find the true scene is to know which
attribute belongs to which object, or which object is on which side.

## Construction

For each of 2,000 held-out validation scenes (duplicate objects skipped):

- **Query:** one caption in the human caption style, with a random object
  order, relation order and sentence pattern.
- **Candidates:** the true scene plus binding-swapped scenes, all written with
  one shared phrasing plan that is *independent of the query's* (different
  synonyms, phrase styles and relation wording). Each candidate gets its own
  random object order, relation order and sentence pattern.
  - **Attribute task:** 7 swapped scenes, each exchanging one color, shape,
    material or size between two objects. Chance R@1 = 1/8 = 12.5%.
  - **Relation task:** every distinct scene made by swapping one relation's
    two objects (1–3 per scene). Chance R@1 = 43.0% on average.
- Every candidate set is **verified to contain exactly the same tokens**
  (same words, same counts), and the phrasing plan is built so that swaps
  never change a word (one synonym per value, styles tied to positions, no
  vowel-initial synonyms, as in the [binding-swap captions](binding_swap_evaluation.md#how-the-captions-are-built)).

The independent phrasing and per-candidate random order matter. An earlier
version wrote all candidates with the query's own phrasing and one fixed
reordering. That left systematic positional overlaps: an untrained model then
scored 1–3% on attributes (far *below* chance) and 64% on relations (far
*above*). The final design removes both biases, as the controls show.

Scale: 1,162 attribute tasks and 1,588 relation tasks from 2,000 scenes
(838 scenes have fewer than 7 valid attribute swaps; 412 have no relation);
15,080 distinct captions.

## Example

> **Query:** Scattered across the floor are a sizable red rubber cylindrical shape; a sphere that is sizable, purple, and glossy metal; and a cube that is little, red, and rubber -- three objects in total.
>
> **True scene:** Present in the scene: a red sizable tube with a rubber finish; a box that is little, red, and rubber; plus a purple sizable sphere with a glossy metal finish.
>
> **Swap 1** (the little red rubber cube and the big purple metal sphere exchange shapes): Present in the scene: a red sizable tube with a rubber finish; a box that is sizable, purple, and glossy metal; plus a red little sphere with a rubber finish.
>
> **Swap 2** (tube and box exchange size): Present in the scene: a red little tube with a rubber finish; a box that is sizable, red, and rubber; plus a purple sizable sphere with a glossy metal finish.

All eight candidates contain the same words; only the bindings differ.

## Metric

Cosine similarity between the query and each candidate (content-token mean,
L2-normalized, same features as the [decomposed evaluation](decomposed_representations.md)).
**R@1** is the rate at which the true scene is the most similar candidate;
tied candidates are scored by the expected value under random tie-breaking,
so a feature that cannot separate the candidates scores exactly chance. 95%
intervals from a 1,000-repetition bootstrap over scenes.

## Controls

| Control | Attribute R@1 | Relation R@1 |
|---|---:|---:|
| Chance | 12.5 | 43.0 |
| Word counts | 12.5 [12.5–12.5] | 43.0 [42.6–43.4] |
| Word pairs (bigrams) | 14.0 [12.1–15.9] | 41.4 [39.1–43.8] |
| Random 384-d features | 12.7 [10.8–14.5] | 43.5 [41.0–46.0] |
| Token-embedding average (untrained model) | 12.5 [12.5–12.5] | 43.0 [42.6–43.4] |
| **Untrained model, all 24 features** (range) | **10.6–14.2** | **41.1–45.3** |
| Untrained model, residual block 7 | 11.2 [9.5–13.1] | 44.5 [42.3–46.9] |

## External pretrained models as a check on the metric

A worry worth testing directly: is this task measuring scene structure, or is
it broken? If it were broken, a model that plainly understands English -- above
all one trained to make sentence cosine meaningful -- ought to score well on it.
So four Hugging Face models were run on the **identical items**, read from the
`texts.jsonl` and `tasks.jsonl` this evaluator wrote, with the same pooling,
cosine ranking and expected-rank tie handling
(`evaluate_hard_retrieval_external.py`). Every layer was scored under two
readouts, `mean` over content tokens and `last`/`cls`, and the table reports
the best of all of them.

| Model | Layers | Best attribute R@1 | Best relation R@1 | Candidate sets no longer multiset-identical |
|---|---:|---:|---:|---:|
| Chance | — | 12.5 | 43.0 | — |
| Untrained model (this repo) | 8 | 11.2 | 44.5 | 0 / 2750 |
| `gpt2` | 12 | 14.2 | 44.8 | **1778 / 2750** |
| `google/bert_uncased_L-8_H-512_A-8` | 8 | 13.8 | 45.7 | 0 / 2750 |
| `all-MiniLM-L6-v2` (sentence-similarity trained) | 6 | 14.5 | 48.0 | 0 / 2750 |
| `all-mpnet-base-v2` (sentence-similarity trained) | 12 | 14.4 | 48.9 | 0 / 2750 |
| `facebook/data2vec-text-base` | 12 | 14.3 | 41.8 | 1778 / 2750 |
| `roberta-base` | 12 | 13.9 | 43.1 | 1778 / 2750 |
| **Dense model (this repo)** | 8 | **22.8** | **71.5** | 0 / 2750 |

`data2vec-text-base` is the published NLP checkpoint of the objective this
study replicates, and `roberta-base` is its controlled comparison: same
architecture, same data, same scale, differing only in the pretraining
objective. Both land at chance here, as does every other pretrained model.

What it establishes: the task is not solvable by generic English competence,
and specifically not by generic sentence similarity. The two
sentence-transformer models exist to make pooled cosine meaningful, and they
gain about two points over chance. If the metric were an artifact of the
scoring code, or reachable through surface statistics, those two would have
scored high.

Two limits of this check, stated plainly:

* **GPT-2's number is not clean.** Its BPE splits a word differently depending
  on whether it follows a space, so reordering the same words changes the token
  multiset -- 1,778 of 2,750 candidate sets lose the lexical identity this task
  depends on. GPT-2 therefore had a purely lexical difference available and
  still scored 14.2%. BERT and both sentence models keep the identity intact,
  so theirs are the numbers to quote.
* **Domain is confounded with capability.** These models never saw a CLEVR
  caption; the models in this catalog trained on two million of them. The check
  shows the task is hard, not that the models here are good at language. The
  control that separates the two is the *easy* version of the same retrieval
  (find the scene among unrelated scenes rather than among binding swaps) on
  these same models and captions; it has not been run.

## Results

R@1 in %, 95% intervals in brackets.

### Main table

| Model | Residual block 7: attribute | Residual block 7: relation | Block 4 MLP: attribute | Block 4 MLP: relation |
|---|---:|---:|---:|---:|
| **Dense** | 22.8 [20.6–25.0] | 71.5 [69.5–73.8] | output 17.9 [15.7–20.0] | output 62.3 [60.1–64.9] |
| Plain Tri-LoRA | 19.7 [17.3–21.9] | 62.2 [59.8–64.6] | shared 11.7 / private 12.6 | shared 47.0 / private 43.7 |
| Average JEPA, no HSIC | 14.4 [12.5–16.4] | 49.7 [47.2–52.1] | shared 13.1 / private 12.9 | shared 42.7 / private 42.9 |
| Average JEPA + HSIC | 15.1 [12.9–17.2] | 52.2 [49.7–54.7] | shared 12.4 / private 12.7 | shared 42.1 / private 40.6 |
| Layerwise JEPA, no HSIC | 13.9 [11.9–15.8] | 51.5 [49.0–54.0] | shared 11.9 / private 12.5 | shared 48.2 / private 45.3 |
| Layerwise JEPA + HSIC | 16.1 [14.1–18.2] | 57.6 [55.4–60.1] | shared 11.6 / private 13.1 | shared 44.3 / private 43.9 |
| Calibrated average JEPA + HSIC | 15.2 [13.4–17.5] | 58.0 [55.6–60.2] | shared 12.2 / private 11.8 | shared 47.2 / private 45.2 |
| Gated data2vec, average (collapsed) | 12.5 [10.6–14.4] | 52.9 [50.5–55.3] | shared 12.4 / private 12.6 | shared 44.3 / private 43.9 |
| Gated data2vec, final layer (collapsed) | 14.0 [12.0–16.1] | 57.0 [54.7–59.4] | shared 13.1 / private 13.7 | shared 44.6 / private 44.2 |
| Gated layerwise JEPA, no HSIC | 13.4 [11.4–15.5] | 53.3 [50.9–55.9] | shared 11.8 / private 10.2 | shared 44.8 / private 42.6 |
| **Gated layerwise JEPA + HSIC** | 14.7 [12.7–16.6] | 56.6 [54.1–59.0] | shared 14.0 / private 12.6 | shared 47.8 / private 47.1 |

### Coverage: which models are *not* in this table

The eleven rows above are the Tri-LoRA / modulewise family on the 2M text
corpus, all at 0.73B tokens. **None of the later families was scored here**, and
that is deliberate rather than an oversight:

| Family | Scored on hard retrieval? |
|---|---|
| Tri-LoRA + modulewise JEPA (11 runs) | yes — the table above |
| data2vec from a dense trunk | no |
| data2vec from scratch | no |
| SIGReg / LeJEPA | no |
| Stage 2 (dense shared + private LoRA) | no |
| Multimodal paired / unpaired | no |

**Why it was retired as the primary comparison.** This metric runs from a 12.5%
floor to 22.8% for the best model, so the whole usable range is ten points and
model differences sit inside a few points of bootstrap noise. It also
separated *none* of the four from-dense data2vec variants, which
[semantic d′](semantic_dprime.md) orders cleanly and monotonically (+0.293 →
+0.395 → +0.445 → +0.527). d′ measures the same comparison as a signed,
normalized effect size with no ceiling, on the same minimal pairs, and it is
what new models are scored on.

Hard retrieval remains the right page for **what the construction guarantees** —
the shortcut controls, the candidate sets, the exact-template counterfactuals —
and those sections are unaffected. An attempt to backfill the six later
runs was made on 2026-09-24 and is not complete; see `outputs/hard_retrieval_eval`.

### Inside the JEPA window: shared and private at blocks 2–4

JEPA supervises only blocks 2–4, so these are the features it was meant to
shape. Every Tri-LoRA shared and private feature there was scored: the `mlp.3`
and `attn.out_proj` updates, and the accumulated shared and private streams
(with and without h0). That is 240 features across the 10 models for
each task.

**Attribute task, MLP `mlp.3` update, R@1 % (chance 12.5)**

| Model | L2 shared / private | L3 shared / private | L4 shared / private |
|---|---:|---:|---:|
| **Dense** (whole MLP output) | **12.4** | **13.9** | **17.9** |
| Plain Tri-LoRA | 12.0 / 12.3 | 12.4 / 13.2 | 11.7 / 12.6 |
| Average JEPA, no HSIC | 12.9 / 14.0 | 12.7 / 11.3 | 13.1 / 12.9 |
| Average JEPA + HSIC | 11.9 / 14.0 | 12.2 / 13.0 | 12.4 / 12.7 |
| Layerwise JEPA, no HSIC | 12.4 / 12.6 | 13.4 / 13.1 | 11.9 / 12.5 |
| Layerwise JEPA + HSIC | 12.0 / 14.3 | 12.0 / 11.9 | 11.6 / 13.1 |
| Calibrated average JEPA + HSIC | 12.6 / 13.4 | 12.3 / 11.4 | 12.2 / 11.8 |
| Gated data2vec, average (collapsed) | 12.7 / 11.3 | 11.6 / 12.9 | 12.4 / 12.6 |
| Gated data2vec, final layer (collapsed) | 12.5 / 13.2 | 11.7 / 10.8 | 13.1 / 13.7 |
| Gated layerwise JEPA, no HSIC | 13.0 / 10.4 | 12.8 / 10.7 | 11.8 / 10.2 |
| **Gated layerwise JEPA + HSIC** | 13.0 / 11.8 | 11.5 / 13.2 | 14.0 / 12.6 |

**Relation task, MLP `mlp.3` update, R@1 % (chance 43.0)**

| Model | L2 shared / private | L3 shared / private | L4 shared / private |
|---|---:|---:|---:|
| **Dense** (whole MLP output) | **42.9** | **50.7** | **62.3** |
| Plain Tri-LoRA | 44.5 / 47.2 | 46.8 / 44.6 | 47.0 / 43.7 |
| Average JEPA, no HSIC | 43.0 / 44.1 | 45.7 / 43.6 | 42.7 / 42.9 |
| Average JEPA + HSIC | 44.6 / 47.9 | 45.0 / 44.8 | 42.1 / 40.6 |
| Layerwise JEPA, no HSIC | 44.3 / 43.6 | 42.9 / 44.5 | 48.2 / 45.3 |
| Layerwise JEPA + HSIC | 45.3 / 43.1 | 40.7 / 43.1 | 44.3 / 43.9 |
| Calibrated average JEPA + HSIC | 43.2 / 42.3 | 42.7 / 41.9 | 47.2 / 45.2 |
| Gated data2vec, average (collapsed) | 42.3 / 43.0 | 43.5 / 44.0 | 44.3 / 43.9 |
| Gated data2vec, final layer (collapsed) | 41.9 / 42.8 | 44.4 / 42.8 | 44.6 / 44.2 |
| Gated layerwise JEPA, no HSIC | 48.2 / 46.3 | 40.7 / 43.4 | 44.8 / 42.6 |
| **Gated layerwise JEPA + HSIC** | 41.1 / 42.9 | 45.3 / 46.3 | 47.8 / 47.1 |

- **Attributes: every shared and private feature in blocks 2–4 is at chance**
  (10.2–15.2%, against 10.6–13.4% for the untrained model's features
  at the same blocks). A handful of the 240 features have intervals just
  above 12.5%, about the number expected by chance across so many tests; none
  is a shared branch.
- **Relations: a small signal, mostly in MLP updates, shared and private
  alike.** About 20 of the 240 features have intervals above 43.0% (roughly 6
  would be expected by chance), scattered over blocks 2–4, at most 5 points
  above chance. Shared and private rise together. Examples at block 4: layerwise JEPA shared
  48.2 [45.8–50.7] vs. private 45.3 [43.0–47.7];
  gated JEPA + HSIC shared 47.8 [45.5–50.3] vs. private
  47.1 [44.8–49.6]; plain Tri-LoRA shared
  47.0 [44.5–49.4] vs. private 43.7 [41.4–46.3]. The full range is
  40.0–48.2%.
- **Dense at the same block is well above both** (block-4 MLP output
  17.9 [15.7–20.0] attribute, 62.3 [60.1–64.9] relation).
- **Contrast with the probe.** The binding-swap probe finds gated JEPA +
  HSIC's block-4 shared update 96.4% correct on relation swaps. That
  information is linearly decodable, but it barely shows in cosine geometry:
  here the same update is at 47.8%, with chance at 43.0%. JEPA has made
  relation information *readable* from the shared update, not *dominant* in
  it.

### Every block, every sublayer: dense, plain Tri-LoRA, gated JEPA + HSIC

Four features at each of the 8 blocks, where h_(L−1) is the stream entering block L:

| Feature | Definition |
|---|---|
| attention output | what the attention sublayer writes |
| attention + residual | the stream after attention: h_(L−1) + attention output |
| MLP output | what the MLP sublayer writes |
| layer output | the stream after the block: h_L |

Same tasks and captions as the main table (verified byte-identical); the dense
residual and block-4 MLP numbers reproduce it exactly. The best cell for each
model is in bold. The untrained model's 42 features stay at chance throughout
(10.2–14.2% attribute, 41.1–45.8% relation).

**Attribute task, R@1 % (chance 12.5)**

| Model | Feature | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Dense** | attention output | 11.6 | 11.8 | 10.5 | 12.5 | 11.7 | 14.5 | 20.5 | 24.3 |
|  | attention + residual | 11.4 | 12.3 | 11.7 | 12.1 | 13.7 | 13.0 | 19.1 | **24.3** |
|  | MLP output | 12.1 | 11.2 | 12.4 | 13.9 | 17.9 | 20.6 | 22.8 | 17.9 |
|  | layer output | 13.9 | 12.2 | 13.1 | 12.8 | 14.3 | 15.7 | 20.9 | 22.8 |
| Plain Tri-LoRA | attention output | 11.3 | 12.7 | 12.1 | 12.0 | 12.9 | 12.0 | 14.7 | **22.5** |
|  | attention + residual | 12.1 | 12.5 | 13.0 | 10.9 | 11.7 | 13.0 | 13.6 | 21.8 |
|  | MLP output | 11.4 | 12.7 | 12.2 | 13.6 | 12.5 | 17.4 | 13.2 | 11.8 |
|  | layer output | 11.7 | 11.9 | 11.8 | 12.6 | 11.7 | 14.0 | 15.0 | 19.7 |
| Gated JEPA + HSIC | attention output | 14.5 | 11.4 | 12.0 | 13.3 | 11.6 | 13.7 | 11.9 | 17.0 |
|  | attention + residual | 13.9 | 11.7 | 12.5 | 12.5 | 12.1 | 12.5 | 11.3 | 16.3 |
|  | MLP output | 11.4 | 13.0 | 14.0 | 13.0 | 13.3 | 10.6 | **17.4** | 13.7 |
|  | layer output | 12.8 | 12.1 | 12.0 | 11.7 | 12.0 | 13.2 | 11.1 | 14.7 |
| Untrained | attention output | 12.0 | 10.2 | 10.8 | 10.6 | 12.1 | 13.0 | 11.4 | 12.1 |
|  | attention + residual | 10.7 | 11.2 | 12.7 | 12.2 | 11.6 | 11.5 | 11.4 | 12.0 |
|  | MLP output | 13.1 | 11.0 | 12.0 | 13.3 | 11.8 | 12.5 | 11.5 | 12.7 |
|  | layer output | 11.8 | 12.0 | 11.5 | 12.4 | 11.3 | 11.7 | 12.0 | 11.2 |

**Relation task, R@1 % (chance 43.0)**

| Model | Feature | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Dense** | attention output | 45.0 | 44.1 | 42.2 | 44.1 | 44.3 | 50.2 | 65.2 | 70.2 |
|  | attention + residual | 44.9 | 44.2 | 42.4 | 42.7 | 43.8 | 48.3 | 59.7 | 69.3 |
|  | MLP output | 43.1 | 43.1 | 42.9 | 50.7 | 62.3 | 66.6 | **78.8** | 69.1 |
|  | layer output | 45.1 | 44.3 | 42.9 | 44.6 | 48.6 | 53.0 | 67.5 | 71.5 |
| Plain Tri-LoRA | attention output | 45.4 | 43.0 | 43.2 | 42.6 | 43.3 | 52.5 | 44.3 | 69.3 |
|  | attention + residual | 45.7 | 43.7 | 42.6 | 42.7 | 42.1 | 47.3 | 47.6 | 66.8 |
|  | MLP output | 43.3 | 43.2 | 44.8 | 47.1 | 46.0 | 52.3 | **69.5** | 48.9 |
|  | layer output | 45.7 | 42.5 | 42.6 | 42.8 | 43.3 | 49.1 | 52.0 | 62.2 |
| Gated JEPA + HSIC | attention output | 43.8 | 43.8 | 41.6 | 44.2 | 46.3 | 41.0 | 44.0 | 62.5 |
|  | attention + residual | 43.0 | 42.4 | 42.5 | 40.7 | 41.7 | 43.1 | 44.0 | 58.0 |
|  | MLP output | 42.5 | 44.3 | 43.6 | 45.2 | 45.9 | 52.5 | **62.7** | 51.9 |
|  | layer output | 43.1 | 42.2 | 42.6 | 41.3 | 42.8 | 45.3 | 46.2 | 56.6 |
| Untrained | attention output | 43.8 | 43.1 | 43.7 | 43.6 | 41.5 | 45.0 | 45.5 | 43.3 |
|  | attention + residual | 44.1 | 44.2 | 44.9 | 45.8 | 45.6 | 44.8 | 44.1 | 44.0 |
|  | MLP output | 43.7 | 41.7 | 41.1 | 42.5 | 44.9 | 44.4 | 42.5 | 43.4 |
|  | layer output | 44.1 | 44.4 | 45.3 | 45.3 | 44.9 | 44.5 | 43.5 | 44.5 |

Reading:

- **Before block 3, every model stays within about a point of the untrained
  model's range.** In dense, binding
  structure appears first in the **MLP writes**: relations from block 3
  (50.7%) and attributes from block 4 (17.9%). The MLP write
  peaks at block 6 (22.8 [20.6–25.0] attribute, 78.8 [76.7–80.7] relation, the
  best relation score anywhere).
- **At the top, dense's attention takes over.** Its attention write is weak
  through block 4 and then rises at blocks 6–7. At block 7 the attention output
  and the stream after attention both reach 24.3%, dense's best attribute score.
- **Plain Tri-LoRA's top attention nearly matches dense** (block 7 attention
  output 22.5 [20.1–24.8] attribute, 69.3 [66.9–71.7] relation), but its MLP writes
  carry little attribute structure (at most 17.4%), even though its block-6
  MLP reaches 69.5% on relations.
- **Gated JEPA + HSIC is weaker than plain Tri-LoRA at the top** (block 7
  attention output 17.0 [14.9–19.3] / 62.5 [60.1–64.9]). Its best attribute score is
  its block-6 MLP write (17.4 [15.2–19.5]).
- **What dense has that Tri-LoRA lacks is compositional MLP writes in the
  middle of the network** (blocks 4–6). That is the gap to target.

Results: `outputs/hard_retrieval_eval_all_sublayers/<label>.json`.

### Accumulated shared vs. private streams at block 7

| Model | Shared stream: attribute / relation | Private stream: attribute / relation |
|---|---:|---:|
| Plain Tri-LoRA | 16.7 / 60.5 | 19.0 / 61.6 |
| Average JEPA, no HSIC | 15.0 / 49.0 | 14.1 / 48.0 |
| Average JEPA + HSIC | 13.9 / 53.9 | 15.8 / 53.4 |
| Layerwise JEPA, no HSIC | 15.1 / 48.7 | 14.2 / 51.6 |
| Layerwise JEPA + HSIC | 14.5 / 53.4 | 16.3 / 57.9 |
| Calibrated average JEPA + HSIC | 15.3 / 55.2 | 14.9 / 57.5 |
| Gated data2vec, average (collapsed) | 12.3 / 52.2 | 13.0 / 52.8 |
| Gated data2vec, final layer (collapsed) | 15.2 / 56.4 | 14.3 / 55.6 |
| Gated layerwise JEPA, no HSIC | 13.0 / 47.2 | 13.9 / 47.8 |
| **Gated layerwise JEPA + HSIC** | 13.7 / 48.6 | 14.7 / 54.3 |

### Residual stream at every block (attribute / relation)

| Model | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---|---|---|---|---|---|---|
| Untrained | 11.8 / 44.1 | 12.0 / 44.4 | 11.5 / 45.3 | 12.4 / 45.3 | 11.3 / 44.9 | 11.7 / 44.5 | 12.0 / 43.5 | 11.2 / 44.5 |
| **Dense** | 13.9 / 45.1 | 12.2 / 44.3 | 13.1 / 42.9 | 12.8 / 44.6 | 14.3 / 48.6 | 15.7 / 53.0 | 20.9 / 67.5 | 22.8 / 71.5 |
| Plain Tri-LoRA | 11.7 / 45.7 | 11.9 / 42.5 | 11.8 / 42.6 | 12.6 / 42.8 | 11.7 / 43.3 | 14.0 / 49.1 | 15.0 / 52.0 | 19.7 / 62.2 |
| Average JEPA, no HSIC | 12.6 / 44.0 | 11.8 / 43.8 | 14.1 / 42.7 | 12.9 / 43.2 | 13.2 / 43.2 | 13.5 / 45.3 | 14.2 / 50.4 | 14.4 / 49.7 |
| Average JEPA + HSIC | 11.4 / 44.2 | 12.4 / 44.0 | 13.6 / 44.8 | 13.3 / 46.1 | 12.2 / 44.1 | 12.1 / 45.1 | 13.1 / 43.8 | 15.1 / 52.2 |
| Layerwise JEPA, no HSIC | 12.7 / 43.0 | 11.6 / 44.1 | 13.2 / 42.5 | 13.1 / 42.9 | 14.1 / 43.0 | 13.8 / 44.9 | 15.7 / 49.4 | 13.9 / 51.5 |
| Layerwise JEPA + HSIC | 12.3 / 40.1 | 12.4 / 41.4 | 12.2 / 41.5 | 11.6 / 42.2 | 12.6 / 42.8 | 12.1 / 44.3 | 12.7 / 49.4 | 16.1 / 57.6 |
| Calibrated average JEPA + HSIC | 11.6 / 42.3 | 12.3 / 41.8 | 13.1 / 42.5 | 13.7 / 41.6 | 11.8 / 42.6 | 11.5 / 41.9 | 12.5 / 48.0 | 15.2 / 58.0 |
| Gated data2vec, average (collapsed) | 12.5 / 42.5 | 10.2 / 42.0 | 11.8 / 44.7 | 13.2 / 44.4 | 12.1 / 44.8 | 13.2 / 48.0 | 11.9 / 49.2 | 12.5 / 52.9 |
| Gated data2vec, final layer (collapsed) | 11.4 / 42.6 | 12.7 / 44.2 | 13.1 / 43.6 | 13.0 / 42.9 | 12.2 / 45.4 | 11.9 / 46.3 | 13.0 / 47.4 | 14.0 / 57.0 |
| Gated layerwise JEPA, no HSIC | 13.1 / 43.5 | 11.1 / 44.9 | 12.7 / 44.8 | 13.3 / 44.4 | 13.9 / 42.4 | 12.4 / 43.1 | 14.8 / 44.6 | 13.4 / 53.3 |
| **Gated layerwise JEPA + HSIC** | 12.8 / 43.1 | 12.1 / 42.2 | 12.0 / 42.6 | 11.7 / 41.3 | 12.0 / 42.8 | 13.2 / 45.3 | 11.1 / 46.2 | 14.7 / 56.6 |

## How to read

1. **The structure of the representation barely reflects binding in any
   model before the last blocks.** Even dense only separates attribute swaps
   from the true scene at blocks 6–7 (20.9% and 22.8%), although probes find
   binding information much earlier. Binding exists in the representation but
   does not dominate its geometry.
2. **The JEPA objectives make this worse, not better.** Every variant is below
   plain Tri-LoRA at block 7 on both tasks. Combined with the other results,
   JEPA reshapes where information sits in individual modules but does not
   organize the representation around compositional scene structure.
3. **Use this as the strict target.** Reaching dense here (22.8% / 71.5%) would
   mean the shared route's geometry encodes scene structure, not just word
   content. The cross-pattern retrieval gap to dense, by contrast, can be
   closed by keeping more word content.

## Reproduce

```bash
./scripts/launch_hard_retrieval_final_vnode10.sh
# one checkpoint, reusing the stored captions and tasks:
python evaluate_hard_retrieval.py --num-worlds 2000 --output-dir outputs/hard_retrieval_eval \
  --checkpoint LABEL=path/to/epoch_003.pt
```

Files: `texts.jsonl` (all captions), `tasks.jsonl` (query, candidates, task),
`construction.json`, `controls.json`, and one JSON per model with every
feature of the [decomposed evaluation](decomposed_representations.md).
