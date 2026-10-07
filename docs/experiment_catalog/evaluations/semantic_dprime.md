# Semantic effect sizes (d′): is the representation about the scene or the words?

> **Type:** evaluation · **Status:** 38 model/route cells, including six external pretrained models · **Updated:** 2026-09-24  
> **Menu:** [experiment catalog](../README.md) · **Code:** `evaluate_semantic_dprime.py`

## Why this exists

The [binding swap](binding_swap_evaluation.md) probe and [hard retrieval](hard_retrieval.md)
each ask one binary question — was this one binding flipped? — and both compress
model differences into a narrow band: hard retrieval runs from a 12.5% floor to
22.8% for the best model, so rankings live inside a few points of noise. Neither
measures *how much* of a scene a representation holds.

This evaluation replaces the preference rate with a signed, normalized effect
size, on the same kind of minimal pairs.

## The two quantities

Both are computed from cosines between pooled, L2-normalized representations,
paired per scene, with a bootstrap over scenes:

    d' = (mean_positive - mean_negative) / sqrt((var_positive + var_negative) / 2)

**`d_semantic`** — invariance to wording, sensitivity to scene.
* positive: the **same scene** rendered by a *different* phrasing plan (different
  synonyms, phrase styles, relation wording, sentence order);
* negative: a **different scene** rendered in the query's *own* template — same
  plan, same object and relation order, same sentence pattern, every value
  changed.

The negative therefore shares the wording and differs in content, so a
representation that follows the words rather than the scene scores *negative*.

**`d_binding`** — sensitivity to which object has which attribute.
* positive: the same paraphrase as above;
* negative: the query scene with exactly one binding exchanged between two
  objects, rendered by the **paraphrase's** plan.

Positive and negative are then word-for-word identical and differ only in
binding; the evaluator verifies the token multisets match and drops any pair
where they do not.

## Controls, and why the scale is informative

| Control | `d_semantic` | `d_binding` |
|---|---:|---:|
| Bag of words | **−2.53** | **0.000** |
| Untrained model, L7 | −2.56 | 0.000 |

`d_semantic` is strongly *negative* for any surface measure: these captions are
long enumerations in which wording tokens outnumber content tokens, so the
template twin shares more words with the query than the paraphrase does. A
positive value therefore means the representation actively overcame the lexical
pull of the template. The usable range runs from about −2.6 to +0.8, far wider
than hard retrieval's 12.5–22.8%.

`d_binding` is exactly 0.000 for any order-free function of the words, by
construction — the same clean floor hard retrieval has, but continuous.

## The lexicon-matched construction

> **Added 2026-09-25**, in response to a question that turned out to identify a
> real confound: *why should a model be expected to know that "tube" and
> "cylinder" name the same shape?*

### The problem

In the original construction the four captions are built from two independently
drawn phrasing plans, and **a plan fixes both the sentence structure and which
synonym stands for each attribute value**:

| caption | plan | attribute vocabulary |
|---|---|---|
| query | `plan` | **A** |
| paraphrase (positive) | `other` | **B** |
| template twin (`d_sem` negative) | `plan` | **A** |
| binding swap (`d_bind` negative) | `other` | **B** |

For `d_binding` this is fine — the positive and the negative both use `other`,
so they share their vocabulary and differ only in binding. **`d_binding` is
unaffected by any of this.**

For `d_semantic` it is a confound. The positive used vocabulary B while the
negative got the query's own vocabulary A, so preferring the positive required
knowing that *tube ≡ cylinder ≡ cylindrical shape* and *large ≡ big ≡ huge ≡
sizable* — a lexical fact about the generator, not a fact about the scene — and
the negative was handed the query's exact words for free.

### The fix

`make_plan(..., lexicon=other_plan)` copies the four synonym dictionaries
(`size`, `color`, `shape`, `material`) from another plan and resamples
everything else — phrase styles, relation wording, openers, object order,
sentence order. The synonym draws still happen and are then overwritten, so
every structural field is **bit-identical** to the same call without `lexicon`;
the two conditions differ in the vocabulary and in nothing else.

```bash
python3 evaluate_semantic_dprime.py --lexicon-matched --num-worlds 2000 \
  --output-dir outputs/semantic_dprime_lexmatched --checkpoint LABEL=path/to/epoch_00N.pt
```

Every result file records `"lexicon_matched": true|false` in its protocol block.

### It makes the metric better behaved, not easier

The worry was that holding the vocabulary fixed would let word overlap identify
the positive, flipping the bag-of-words control strongly positive and making
`d_semantic` lexically solvable. Measured on the same 1994 items, it does not:

| | query ∩ positive | query ∩ negative | **bag-of-words `d_sem`** |
|---|---:|---:|---:|
| original (free lexicon) | 64.0% | 86.8% | **−2.528** |
| **lexicon matched** | 80.2% | 86.8% | **−0.118** |

The negative is a *different scene in the query's own template*, so it keeps its
86.8% overlap from the framing words; the positive rises to 80.2% but changes
every structural word. The two effects cancel and the control lands essentially
at zero.

That is a strictly better property. Under the original construction `d_semantic`
had to be read as "distance above −2.53"; under the matched construction it has
a real zero — **positive means the representation followed the scene, negative
means it followed the wording**, with no arithmetic. `d_binding` stays exactly
+0.000 for bag of words in both, as it must.

## Results

Every model, at the L7 residual stream and at its own best feature. External
models are reported at their final-layer mean and their best non-degenerate
feature (the layer-0 CLS vector is constant across sequences, giving an
exactly-zero effect size that is not a measurement).

### Reference points

| Model | d_semantic (L7 residual) | best d_semantic | d_binding (L7) | best d_binding |
|---|---:|---|---:|---|
| Bag of words (control) | -2.53 | — | +0.000 | — |
| Untrained model | -2.56 | `L1.mlp_out` -2.14 | +0.000 | `L3.mlp_out` +0.004 |
| Dense, 4 epochs (0.73B tokens) | +0.38 | `L6.attn_out` +2.05 | +0.122 | `L6.mlp_out` +0.244 |
| **Dense, 6 epochs (1.10B — budget-matched)** | +0.44 | — | +0.229 | `L6.mlp_out` **+0.443** |
| Dense, 8 epochs (1.47B) | +0.34 | — | +0.239 | `L6.mlp_out` +0.438 |
| Plain Tri-LoRA | +0.47 | `L5.attn_shared` +1.71 | +0.108 | `L7.attn_shared` +0.293 |

> **Read the JEPA runs against dense at 6 epochs, not 4.** A from-dense JEPA run
> costs its parent's 0.73B plus its own 0.37B = 1.10B tokens. Dense at the same
> budget reaches +0.443, not +0.244. An earlier version of this page compared
> against the 4-epoch number and reported that JEPA "more than doubles" binding;
> the honest figure is **+19% relative** (+0.527 vs +0.443). That claim is
> withdrawn.

### data2vec from a pretrained dense trunk

| Model | d_semantic (L7) | best d_semantic | d_binding (L7) | best d_binding |
|---|---:|---|---:|---|
| avg, all 8 | -0.76 | `L5.attn_out` +1.44 | +0.037 | `L6.mlp_out` +0.293 |
| avg, L4–7 | +0.11 | `L5.attn_out` +1.22 | +0.052 | `L6.mlp_out` +0.395 |
| layerwise, L4–7 | +0.66 | `L6.attn_out` +1.63 | +0.087 | `L7.mlp_out` +0.445 |
| **layerwise, all 8** | **+0.75** | `L6.attn_out` +1.77 | +0.094 | `L5.mlp_out` **+0.527** |

### JEPA from scratch (no pretrained init) — all collapse

| Model | d_semantic (L7) | best d_semantic | d_binding (L7) | best d_binding |
|---|---:|---|---:|---|
| window 12–24, avg | +0.42 | `L6.attn_plus_residual` +1.14 | +0.018 | `L7.mlp_out` +0.198 |
| window 12–24, layerwise | -1.72 | `L3.attn_out` +0.19 | +0.001 | `L4.mlp_out` +0.085 |
| window 4–8, avg | -0.53 | `L4.attn_out` +0.71 | +0.020 | `L7.mlp_out` +0.248 |
| window 4–8, layerwise | -1.27 | `L0.attn_out` +0.53 | +0.001 | `L4.mlp_out` +0.081 |
| layerwise all-8, random-t | -1.64 | `L3.attn_out` -1.30 | +0.004 | `L7.mlp_out` +0.022 |

### SIGReg / LeJEPA — collapse in both initializations

| Model | d_semantic (L7) | d_binding (L7) | best d_binding |
|---|---:|---:|---|
| LeJEPA layerwise, from dense | **-2.68** | +0.006 | `L5.mlp_out` +0.013 |
| LeJEPA layerwise, from scratch | -0.82 | -0.002 | `L0.mlp_out` +0.016 |

Both are at the untrained model's level on binding (+0.004) and the from-dense
run is *below* the bag-of-words control on `d_semantic`. At λ = 0.05 the sliced
Epps–Pulley term does not rescue the objective; it destroys the trunk it starts
from. See [the JEPA survey](../jepa_model_survey.md#sigreg--lejepa).

### Multimodal unpaired family (text side)

| Model | d_semantic (L7) | d_binding (L7) | best d_binding |
|---|---:|---:|---|
| unpaired dense, 4 ep | -1.46 | +0.005 | `L7.attn_out` +0.007 |
| JEPA from **text** dense | **+0.30** | +0.065 | `L5.mlp_out` **+0.381** |
| JEPA from **image** dense | -0.93 | -0.002 | `L1.mlp_out` +0.010 |
| JEPA from unpaired dense, EMA | -3.19 | -0.001 | `L3.mlp_out` +0.021 |
| JEPA from unpaired dense, SIGReg | -2.14 | +0.009 | `L1.mlp_out` +0.022 |

The unpaired dense trunk has **no text binding at all** (+0.007, at the
untrained floor) even though it trains on both modalities — interleaved
unpaired batches do not produce the text representation that text-only dense
produces. Only the run initialized from the *text* dense trunk carries binding
forward, and it carries roughly what its parent had.

### Stage 2: dense trunk as shared route + private LoRA

Reported three ways, because these models have two routes and the question is
what each one holds. **Trunk only** suppresses the private LoRA at every layer;
**private only** suppresses the trunk contribution; **full** is the model as
trained. All rows are the **final** epoch (`epoch_001`); the epoch-0 checkpoints
of these runs score 1–4 points higher on `d_bind` full (frozen avg +0.356,
trainable avg +0.426, frozen layerwise +0.508), i.e. the second epoch of stage-2
training slightly *degrades* the full model while leaving the trunk untouched.

| Model | route | d_semantic (L7) | d_binding (L7) | best d_binding |
|---|---|---:|---:|---|
| stage 2 frozen, avg trunk | full | -0.51 | +0.069 | `L6.mlp_shared` +0.347 |
| | trunk only | -0.76 | +0.037 | `L6.mlp_out` +0.293 |
| | private only | -1.08 | -0.003 | `L0.writes` +0.003 |
| stage 2 trainable, avg trunk | full | -0.24 | +0.112 | `L6.mlp_shared` +0.396 |
| | trunk only | +0.36 | +0.120 | `L6.mlp_out` +0.433 |
| | private only | -1.74 | +0.003 | `L6.mlp_out` +0.014 |
| stage 2 frozen, **layerwise** trunk | full | +0.10 | +0.100 | `L5.mlp_shared` +0.529 |
| | trunk only | **+0.75** | +0.094 | `L5.mlp_out` +0.527 |
| | private only | -1.54 | -0.001 | `L7.attn_out` +0.024 |
| **stage 2 trainable, layerwise trunk** | full | *(pending — OOM, requeued)* | | |
| | **trunk only** | **+0.63** | **+0.163** | `L5.mlp_out` **+0.555** |
| Tri-LoRA controls | | | | |
| plain Tri-LoRA | full | +0.47 | +0.108 | `L7.attn_shared` +0.293 |
| | trunk only | -0.98 | +0.000 | `L6.attn_out` +0.019 |
| | private only | +0.27 | +0.003 | `L1.mlp_out` +0.009 |
| gated layerwise + HSIC | full | +0.18 | +0.057 | `L7.attn_private` +0.148 |
| | trunk only | -0.52 | +0.021 | `L4.mlp_out` +0.036 |
| | private only | -0.56 | -0.005 | `L2.mlp_out` +0.007 |

### External pretrained models, on identical items

| Model | d_semantic (L7/final) | best d_semantic | d_binding | best d_binding |
|---|---:|---|---:|---|
| `facebook/data2vec-text-base` | -1.21 | `L12.last` -0.60 | +0.012 | `L9.cls` +0.027 |
| `roberta-base` | -2.33 | `L11.cls` -0.38 | -0.002 | `L6.cls` +0.016 |
| `all-mpnet-base-v2` | -2.83 | `L7.last` -0.52 | +0.016 | `L10.cls` +0.042 |
| `gpt2` | — | — | — | `L10.last` +0.025 |
| `bert` (8-layer) | — | — | — | `L7.last` +0.007 |
| `all-MiniLM-L6-v2` | — | — | — | `L2.cls` +0.008 |

## What the numbers say

1. **JEPA helps binding, and the effect is ordered by target design.** Across the
   four from-dense data2vec variants the best `d_binding` rises monotonically:
   averaged all-8 +0.293, averaged 4-7 +0.395, layerwise 4-7 +0.445, layerwise
   all-8 **+0.527**. Against the budget-matched dense baseline (+0.443) the best
   configuration is **+19%**; against the undertrained 4-epoch dense it would look
   like +116%, which is why the budget has to be matched. Neither hard retrieval
   nor the binding probe separated these runs at all.
2. **Averaging the target is the damage, not which layers are averaged.** `avg
   L4–7` (+0.395) and `avg all-8` (+0.293) both sit below their layerwise
   counterparts at the *same* layer window (+0.445, +0.527). The earlier
   hypothesis that shallow blocks poison the average is refuted: restricting the
   average to L4–7 recovers only part of the gap. [Image binding](image_binding.md)
   reproduces this ordering independently.
3. **Dense still owns the most scene-sensitive single feature**: `L6.attn_out` at
   +2.05, ahead of the best JEPA model's +1.77. JEPA reorganizes *where* binding
   lives and improves the residual stream (+0.75 vs +0.38); it does not produce a
   more semantic feature than dense already has.
4. **Layerwise targeting needs a pretrained trunk.** From dense it is the best
   configuration; from scratch it collapses to the surface — −1.72 and −1.27 with
   window masking, −1.64 with the *identical* random-t masking of the from-dense
   run, and −0.82 with SIGReg instead of an EMA teacher, with binding at the
   untrained floor in every case. Masking was not the variable, and neither was
   the anti-collapse mechanism; **initialization is**.
5. **The private route is essentially empty; the trunk carries everything.**
   Best private-only `d_binding` across the six two-route models: +0.003, +0.014,
   +0.024, +0.007, +0.009, +0.019 — against +0.004 for an untrained model and
   +0.293 to +0.555 for the same models intact. The largest private reading is
   1/22 of its own model's full score. Trunk-only, meanwhile, *keeps* or
   *exceeds* the full model's binding in the dense_private runs (+0.433 vs +0.396
   full; +0.555 trunk-only). The private LoRA is not an information store; at
   best it is a decoder-side adapter the trunk routes around.
6. **The best single configuration in the whole text study is the stage-2
   trainable layerwise trunk, read trunk-only: `d_binding` +0.555 at
   `L5.mlp_out`, `d_semantic` +0.63.** It beats the layerwise JEPA trunk it
   started from (+0.527) and budget-matched dense (+0.443). Notably it is *better
   with the private route switched off* — the private branch actively dilutes it.
7. **Letting the trunk train beats freezing it, on every matched comparison.**
   Same parent, same budget: avg-trunk frozen +0.347 vs trainable +0.396 (full)
   and +0.293 vs +0.433 (trunk-only); layerwise-trunk frozen +0.527 vs trainable
   **+0.555** (trunk-only). Validation loss agrees (1.0617 frozen vs 1.0553
   trainable). This **corrects an earlier reading of these runs** in which
   freezing looked better — that comparison put a frozen *layerwise*-parent run
   against a trainable *avg*-parent run, so it was measuring the parent, not the
   freeze. The "freeze the shared route to protect it" intuition is not supported
   here.
8. **The LoRA gated runs rank worst of the trained models** (+0.18 / +0.06 and
   −0.23 / +0.04), below plain Tri-LoRA. The catalog's earlier claim that gated
   layerwise + HSIC is the best JEPA configuration came from binding-probe
   numbers; on effect sizes that ranking inverts.
9. **Pretrained English models are on the surface side of zero.** `data2vec-text`,
   `roberta-base`, `all-mpnet-base-v2`, `gpt2`, 8-layer BERT and MiniLM all score
   negative `d_semantic` at every feature and essentially zero `d_binding`
   (0.007–0.042). They behave like the bag-of-words control on these captions.
   This is a statement about the captions — long templated enumerations whose
   lexical statistics are dominated by wording — not about the models, and it
   means the effect sizes our models reach are genuine in-domain effects rather
   than an artifact of an easy metric.

## Limits

* Both quantities are measured on pooled content-token means, so anything the
  pooling discards is invisible.
* `d_semantic` is solvable in part by word *content* (the paraphrase shares the
  scene's attribute words), which is why the bag-of-words control matters: the
  quantity to read is the distance above −2.53, not the raw sign.
* Captions come from the same generator as the training data, so no model is
  penalized for unfamiliar wording — and no model is credited for generalizing
  beyond it.

## Reproduce

```bash
python evaluate_semantic_dprime.py --num-worlds 2000 \
  --output-dir outputs/semantic_dprime_eval \
  --checkpoint LABEL=path/to/epoch_00N.pt \
  --untrained-from outputs/text_dense_diffusion_2m_4e_matched/epoch_003.pt

# external Hugging Face models on the same items
python evaluate_semantic_dprime.py --num-worlds 2000 \
  --output-dir outputs/semantic_dprime_eval \
  --tokenizer-from outputs/text_dense_diffusion_2m_4e_matched/epoch_003.pt \
  --hf-model hf_data2vec_text=facebook/data2vec-text-base
```

Route isolation, for any model with a shared and a private route:

```bash
# trunk / shared route only -- private LoRA suppressed at every layer
python evaluate_semantic_dprime.py --ablate-private --num-worlds 2000 \
  --output-dir outputs/semantic_dprime_trunk_only \
  --checkpoint LABEL_trunkonly=path/to/epoch_00N.pt

# private route only -- shared contribution suppressed at every layer
python evaluate_semantic_dprime.py --ablate-shared --num-worlds 2000 \
  --output-dir outputs/semantic_dprime_private_only \
  --checkpoint LABEL=path/to/epoch_00N.pt
```

`--ablate-private` sets `route_ids = -1`, which suppresses the private branch;
`--ablate-shared` drives the per-modality private-only route id
(`TEXT_PRIVATE_ONLY_ROUTE_ID = 3`, `IMAGE_PRIVATE_ONLY_ROUTE_ID = 2`). The
isolation was validated end-to-end: a frozen-trunk stage-2 model read trunk-only
reproduces its parent exactly (`d_semantic` −0.763 in both), so the ablation is
recovering the parent's computation rather than a perturbed version of it.

Files: `texts.jsonl`, `items.jsonl` (query, paraphrase, template twin, swap),
`controls.json`, and one JSON per model with both effect sizes, their 95%
intervals, the paired preference rates and the underlying cosine means, for
every feature of the [decomposed evaluation](decomposed_representations.md).
