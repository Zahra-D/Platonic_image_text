# Exact-template counterfactual control

> **Type:** evaluation, text-only study · **Status:** complete for all final checkpoints; superseded as a semantic measure by the [binding-swap evaluation](binding_swap_evaluation.md)  
> **Question:** does a representation prefer the same scene in different wording, or the same wording with every scene value changed?  
> **Models:** plain Tri-LoRA, the four original JEPA/HSIC cells, calibrated average JEPA + HSIC, both gated data2vec runs, both gated-predictor runs  
> **Script:** [evaluate_exact_template_conflict.py](/home/zd25e122/clevr_discrete_diffusion/evaluate_exact_template_conflict.py) · **Launcher:** [launch_exact_template_conflict_vnode10.sh](/home/zd25e122/clevr_discrete_diffusion/scripts/launch_exact_template_conflict_vnode10.sh) · **Results:** `outputs/exact_template_conflict/`  
> **Menu:** [experiment catalog](../README.md)

## Summary

Most original cells' shared and private `blocks.4.mlp.3` updates prefer the
same scene (87–97%), and so does plain Tri-LoRA (96.1% / 94.1%). The
exceptions are average JEPA + HSIC (shared 34.0%) and calibrated average
JEPA + HSIC (shared 62.9%). The control separates models poorly: changing
every value at once is a very large change, and the counterfactual always has
one more object. Its preference rate is also a sign test with no effect size. The one
striking row, gated data2vec with average target (shared 70.3% semantic,
private 82.0% template), rests on shared cosines of 0.9986 vs. 0.9980 from a
collapsed representation, so it is not evidence of a clean shared/private
split. For semantic claims use the [binding-swap evaluation](binding_swap_evaluation.md),
which removes lexical and order shortcuts by construction.

## Contents

[Why](#why-this-control-exists) · [Construction](#construction) · [Examples](#worked-example--world-0) · [Verification](#verification) · [Metric](#what-is-measured) · [Results](#results-final-checkpoints) · [Interpretation](#interpretation) · [Limitations](#limitations) · [How to run](#how-to-run)

## Question

If two captions use **exactly the same canonical wording** but describe
**completely different scene values**, does the representation consider them
more similar than two captions describing the **same scene in different
wording**? A semantic route should prefer the same-scene candidate; a
surface-form route should prefer the same-wording candidate.

## Why this control exists

Cross-pattern nearest-neighbor retrieval
([cross-pattern semantic retrieval](cross_pattern_semantic_retrieval.md)) cannot tell a
semantic feature from one that benefits incidentally from template regularity,
because in that gallery wording and scene identity are never placed in direct
opposition. This control forces the conflict: every query is scored against
exactly one semantic candidate and exactly one template candidate, and the
decision metric is a **preference rate**, not retrieval accuracy.

## Construction

For each of 256 held-out validation worlds the evaluator builds **four** clean
captions, so the gallery is 1,024 captions:

| Row | Scene | Wording |
|---|---|---|
| 1 | source | template 0 |
| 2 | source | template 1 |
| 3 | counterfactual | template 0 |
| 4 | counterfactual | template 1 |

### The two canonical templates

Non-slot wording is fixed verbatim. Only the bracketed slots vary.

```
template 0:  Scene summary: there are {N} objects. {object clauses} Spatial facts: {relation clauses}
template 1:  Description of a scene. {object clauses} The scene contains {N} objects. Spatial description: {relation clauses}
```

```
object clause:    Object {i} is a {size} {color} {material} {shape}.
relation clause:  Object {a} is {to the left of|to the right of|in front of|behind} object {b}.
no relations:     No spatial relation is stated.
```

The two templates differ only in sentence order and three fixed connective
phrases. The object clauses are byte-identical between them for a given scene.

### The counterfactual transformation

`counterfactual_world()` changes every indexed slot deterministically:

| Slot | Rule |
|---|---|
| size | `small ↔ large` |
| material | `metal ↔ rubber` |
| color | index + 3 (mod 8) over `gray red blue green brown purple cyan yellow` |
| shape | index + 1 (mod 3) over `cube sphere cylinder` |
| count | append one object if `< 10`, otherwise drop the last one |
| relations | each relation flipped to its opposite (`left↔right`, `front↔behind`) |
| no relations | insert `Object 1 is to the left of object 2.` when count ≥ 2 |

The appended object is derived from object 1 with size flipped, material
flipped, color + 5, shape + 2, so it cannot reintroduce a source slot value.

## Worked example — world 0

**Semantic candidate pair** (same scene, wording changes):

> **source / template 0**
> Scene summary: there are 6 objects. Object 1 is a large blue metal cube. Object 2 is a large yellow rubber sphere. Object 3 is a small brown rubber cylinder. Object 4 is a small yellow rubber sphere. Object 5 is a large yellow metal sphere. Object 6 is a small blue metal sphere. Spatial facts: Object 2 is to the right of object 1. Object 3 is behind object 2.

> **source / template 1**
> Description of a scene. Object 1 is a large blue metal cube. Object 2 is a large yellow rubber sphere. Object 3 is a small brown rubber cylinder. Object 4 is a small yellow rubber sphere. Object 5 is a large yellow metal sphere. Object 6 is a small blue metal sphere. The scene contains 6 objects. Spatial description: Object 2 is to the right of object 1. Object 3 is behind object 2.

**Template candidate** (wording identical to the query, every value changed):

> **counterfactual / template 0**
> Scene summary: there are 7 objects. Object 1 is a small purple rubber sphere. Object 2 is a small blue metal cylinder. Object 3 is a large yellow metal cube. Object 4 is a large blue metal cylinder. Object 5 is a small blue rubber cylinder. Object 6 is a large purple rubber cylinder. Object 7 is a large blue metal cube. Spatial facts: Object 2 is to the left of object 1. Object 3 is in front of object 2.

> **counterfactual / template 1**
> Description of a scene. Object 1 is a small purple rubber sphere. Object 2 is a small blue metal cylinder. Object 3 is a large yellow metal cube. Object 4 is a large blue metal cylinder. Object 5 is a small blue rubber cylinder. Object 6 is a large purple rubber cylinder. Object 7 is a large blue metal cube. The scene contains 7 objects. Spatial description: Object 2 is to the left of object 1. Object 3 is in front of object 2.

Tracing object 1: `large blue metal cube` → `small purple rubber sphere`
(size flipped, blue + 3 = purple, metal → rubber, cube + 1 = sphere). The
relations flip `right → left` and `behind → front`.

## Worked example — world 10, source with no relations

> **source / template 0**
> Scene summary: there are 3 objects. Object 1 is a small red rubber cube. Object 2 is a large purple metal sphere. Object 3 is a large red rubber cylinder. Spatial facts: No spatial relation is stated.

> **counterfactual / template 0**
> Scene summary: there are 4 objects. Object 1 is a large brown metal sphere. Object 2 is a small gray rubber cylinder. Object 3 is a small brown metal cube. Object 4 is a small red rubber cube. Spatial facts: Object 1 is to the left of object 2.

This is the branch that inserts a default relation. 46 of the 256 source
worlds take it.

## Verification

Two checks run every time, and both are recorded in `results.json`:

- **Indexed slot overlap.** `slot_atoms()` builds position-aware atoms
  (`object:3:color:yellow`, `count:6`, `relation:0:{...}`) so a changed slot
  cannot hide inside an unordered set. The Jaccard overlap between source and
  counterfactual atom sets is reported as `mean_contrast_atom_jaccard` and
  `max_contrast_atom_jaccard`. Every completed run reports **0.0 for both**.
- **Caption collision.** The evaluator refuses to run if any two of the 1,024
  captions are byte-identical, which would make the contrast ambiguous.

## What is measured

Features are the clean native `blocks.4.mlp.3` update, content-token mean,
L2-normalized, computed separately for the shared and the text-private LoRA
route. Same feature definition as the retrieval protocol.

Each world contributes **two** queries — template 0 and template 1 — so both
wording directions are used. For a query in template *t*:

- semantic similarity = cos(query, source in the *other* template)
- template similarity = cos(query, counterfactual in the *same* template)

The two directions are averaged per world before bootstrapping, then a
1,000-repetition paired bootstrap over the 256 worlds gives 95% intervals for
five quantities: the two cosines, their gap, and the two preference rates.

`semantic_preference_rate` is the fraction of worlds where the gap is positive.
Since the two rates are complements, only one is independent.

## Alternative contrast mode

`--contrast-mode heldout_max_dissimilar` replaces the synthetic counterfactual
with the **real** held-out world of lowest atom-Jaccard overlap, found by an
all-pairs search over the 256 worlds. It trades guaranteed zero slot overlap
for naturalness. Every run recorded in the catalog uses the default
`counterfactual` mode.

## Results: final checkpoints

Feature: clean `blocks.4.mlp.3` shared or private update, content-token mean,
L2-normalized; 256 held-out worlds; 1,000-repetition world bootstrap.

### Original cells

| Final checkpoint | Shared semantic preference | Shared template preference | Private semantic preference | Private template preference | Shared gap | Private gap |
|---|---:|---:|---:|---:|---:|---:|
| Plain Tri-LoRA | 96.1% | 3.9% | 94.1% | 5.9% | +0.015 | +0.018 |
| Average JEPA, no HSIC | 87.1% | 12.9% | 90.2% | 9.8% | +0.034 | +0.062 |
| Average JEPA + HSIC | **34.0%** | **66.0%** | 54.7% | 45.3% | **-0.003** | +0.000 |
| Layerwise JEPA, no HSIC | 97.3% | 2.7% | 88.3% | 11.7% | +0.033 | +0.022 |
| Layerwise JEPA + HSIC | 93.8% | 6.2% | 97.3% | 2.7% | +0.052 | +0.036 |
| Earlier calibrated average JEPA + HSIC | 62.9% | 37.1% | 87.1% | 12.9% | +0.004 | +0.008 |

The gap is mean cosine(`same semantics, different template`) minus
cosine(`same template, changed values`). Positive favors semantics; negative
favors the exact wording template.

The gap is mean cosine(same scene, different template) minus
cosine(same template, changed values). Positive favors semantics.

### Gated data2vec runs

| Run | Route | Semantic preference | Template preference | Same-scene cosine | Same-template cosine | Gap |
|---|---|---:|---:|---:|---:|---:|
| Average teacher target | Shared | 70.3% | 29.7% | 0.9986 | 0.9980 | +0.0006 |
|  | Private | 18.0% | **82.0%** | 0.555 | 0.866 | −0.311 |
| Final-layer teacher target | Shared | 42.2% | 57.8% | 0.969 | 0.965 | +0.004 [−0.012, +0.019] |
|  | Private | 12.1% | **87.9%** | 0.326 | 0.864 | −0.539 |

Both rows use final `epoch_003.pt` checkpoints; the final-layer-target row
replaces the earlier provisional epoch-2 result.

### Gated-predictor runs

Same layerwise JEPA as the original layerwise cells, with the JEPA gradient
routed end-to-end into the shared LoRA of blocks 2–4
([plan](../plans/gated_predictor_layerwise_jepa.md)).

| Run | Route | Semantic preference | Template preference | Same-scene cosine | Same-template cosine | Gap [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| Gated layerwise JEPA, no HSIC | Shared | 42.6% | 57.4% | 0.981 | 0.982 | -0.002 [-0.002, -0.001] |
|  | Private | 59.0% | 41.0% | 0.957 | 0.952 | +0.005 [+0.002, +0.007] |
| Gated layerwise JEPA + HSIC | Shared | 88.7% | 11.3% | 0.969 | 0.930 | +0.039 [+0.034, +0.044] |
|  | Private | 80.9% | 19.1% | 0.974 | 0.952 | +0.022 [+0.019, +0.026] |

## Interpretation

### Original cells

Private is **not generally a template-only route**. In five of six final
checkpoints it prefers the same semantic scene over the exact-template,
all-values-changed counterfactual at least 87% of the time. This is coherent:
private LoRAs are optimized by the text-denoising loss and therefore retain
semantic information.

However, final average JEPA+HSIC is a real exception: its **shared** update
prefers the exact-template counterfactual 66% of the time, and its private
update is nearly indifferent. This warns that its positive ordinary semantic
retrieval score should not be interpreted as clean semantic invariance. It is
the configuration most vulnerable to this canonical-wording control.

The counterfactual is intentionally synthetic and text-only; it isolates the
representation's treatment of wording from values. It does not replace the
held-out real-world retrieval evaluation, which remains the main semantic
generalization measure. Together, the two tests say: layerwise JEPA variants
are strongest in real cross-pattern retrieval and remain semantic under this
strict conflict, whereas average JEPA+HSIC deserves further investigation for
template sensitivity.

### Gated data2vec runs

Both runs look like the intended split at first glance: private strongly
prefers the exact template despite every scene value changing. That reading
does not hold up:

- **Shared is collapsed.** Its cosines are all near 1 (0.9986 vs. 0.9980 for
  the average-target run), and the effective rank of its `blocks.4.mlp.3`
  update is 3.0 (vs. 16.4 for layerwise JEPA). A 70.3% preference rate on a
  0.0006 cosine gap is not a semantic result; the final-layer run's shared
  preference is at chance (42.2%, gap interval includes 0).
- **Private is template-locked because it is damaged, not specialized.** Its
  update at the same module also collapsed (effective rank 7.3 and 5.5), and it
  probes worse for scene attributes than any healthy model.

Details: [gated data2vec results and collapse diagnosis](data2vec_gated_collapse_diagnosis.md).

### Gated-predictor runs

Gated JEPA + HSIC behaves like the healthy layerwise cells: both routes prefer
the same scene (88.7% shared, 80.9% private). Gated JEPA without HSIC is the
only healthy model near chance on both routes (shared 42.6%, private 59.0%;
both gaps within ±0.01 of zero); its representations are not collapsed
(shared effective rank 37.4) but lean more on sentence pattern: its template MRR in the
[retrieval evaluation](cross_pattern_semantic_retrieval.md) (0.594 shared,
0.585 private) is higher than any other layerwise model's (0.474–0.544).

## Limitations

These are properties of the construction, not bugs, but they bound how far the
preference rate can be pushed:

1. **The counterfactual is never the same length as the source.** In the 256
   evaluated worlds the object count ranges 3–8, so the `< 10` branch always
   fires and the counterfactual always has **exactly one more object** than the
   source. Object count and caption length are therefore a systematic cue that
   makes the template candidate easier to reject, which inflates semantic
   preference for every checkpoint alike. The `pop()` branch is never exercised
   at this world count.
2. **Sources without relations gain one.** For the 46 no-relation worlds the
   counterfactual always acquires the same fixed clause, another systematic
   asymmetry.
3. **Preference rate ignores effect size.** It is a sign test. A route whose
   two cosines are 0.9986 and 0.9980 produces a preference rate on the fifth
   decimal of a near-degenerate feature. Always read
   `semantic_minus_template_cosine` alongside the rate — see the
   gated data2vec rows below, where a 70.3% shared preference
   rests on a gap of +0.00057 from a representation with effective rank ~3.
4. **Text-only and synthetic.** It isolates wording from values; it does not
   replace held-out cross-pattern retrieval as the semantic generalization
   measure.

## How to run

```bash
./scripts/launch_exact_template_conflict_vnode10.sh LABEL CHECKPOINT [GPU]
```

Defaults: 256 held-out worlds from
`platonic_text_only_v1_1m/val_text_only_human.jsonl`, batch size 64, 1,000
bootstrap repetitions, seed 20260917. The launcher refuses to start if
`outputs/exact_template_conflict/LABEL/results.json` already exists or a tmux
session of the same name is live. Each run writes `variants.jsonl` (the exact
1,024 captions), `features.pt`, `results.json`, and `REPORT.md`.

## Artifacts

Each result directory keeps the generated captions (`variants.jsonl`), exact
features (`features.pt`), bootstrap intervals (`results.json`), and `REPORT.md`:
`outputs/exact_template_conflict/<label>_counterfactual_epoch003/` for
`plain_lora`, `avg_no_hsic`, `avg_hsic`, `layerwise_no_hsic`, `layerwise_hsic`,
`avg_hsic_calibrated_old`, `data2vec_avg_gated`, and `data2vec_noavg_gated`.
