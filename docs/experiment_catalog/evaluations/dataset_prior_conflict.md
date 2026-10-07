# Dataset-prior conflict: scene binding or training statistics?

> **Type:** checkpoint-only evaluation · **Status:** complete, four models  
> **Training:** none · **New dataset:** none; exactly reuses the existing semantic-d′ items  
> **Code:** `evaluate_dataset_prior_conflict.py` · **Results:** `outputs/dataset_prior_conflict/`

## Question

The from-scratch JEPA models might not represent the binding in a particular
scene. They might instead learn the empirical training joint

```text
p_train(color, shape, material, size)
```

and prefer whichever object combination is statistically typical. An IID probe
cannot distinguish these explanations because the dataset prior is usually
correct on IID data.

## Test

This evaluation reuses all 7,976 captions and 1,994 items already stored by
`semantic_dprime_eval`; an exact-content audit fails if even one caption or item
differs. No checkpoint or probe is trained.

Each item contains:

- a query caption;
- a positive: the same held-out scene in different wording;
- a counterfactual: exactly the positive's words, but with one attribute
  exchanged between two objects.

The positive and counterfactual therefore have identical words and identical
per-attribute multisets. Only their attribute binding differs.

The existing 2M-caption training manifest is streamed once to estimate a
Laplace-smoothed object prior. For each minimal pair, the summed object log
probability determines whether the prior supports the true scene or conflicts
with it. Of 1,692 attribute swaps, the prior supports truth in 1,298 and favors
the false counterfactual in 394. Relation swaps are excluded because the tested
prior is specifically the object-factor joint.

The predeclared primary representation is the final residual, `L7.residual`.
Other layers and sublayers are exploratory.

## Results

True-scene cosine preference; chance is 0.500. Intervals are a 1,000-draw
bootstrap over held-out worlds.

| Model | all | prior supports truth | prior conflicts | support − conflict | correlation with prior margin |
|---|---:|---:|---:|---:|---:|
| scratch JEPA, layerwise | 0.498 [0.475, 0.522] | 0.513 [0.489, 0.540] | **0.447 [0.396, 0.496]** | +0.066 [0.014, 0.123] | +0.017 [−0.028, +0.064] |
| scratch JEPA, averaged windows | 0.543 [0.521, 0.569] | 0.564 [0.539, 0.591] | 0.475 [0.427, 0.525] | +0.089 [0.034, 0.146] | +0.122 [+0.074, +0.168] |
| dense diffusion | **0.641 [0.619, 0.665]** | 0.663 [0.640, 0.690] | **0.569 [0.521, 0.615]** | +0.095 [0.044, 0.147] | +0.153 [+0.106, +0.198] |
| layerwise JEPA from dense | **0.648 [0.626, 0.671]** | 0.681 [0.657, 0.707] | 0.541 [0.496, 0.589] | +0.140 [0.086, 0.191] | +0.203 [+0.157, +0.247] |

The empirical prior by itself chooses the true scene in 1,298 / 1,692 = 76.7%
of these IID pairs, illustrating why a high undivided IID score would be
misleading. By construction it is wrong on every prior-conflict item.

## Interpretation

**The simple claim “scratch JEPA only stores the joint prior” is too strong.**
The layerwise scratch model has almost no final binding signal at all: its
overall result is chance and its continuous correlation with prior strength is
also indistinguishable from zero. It is better described as a failed or nearly
empty final representation with a small categorical prior-following bias.

**The averaged scratch model shows the hypothesized shortcut more clearly.** It
has weak above-chance binding overall, but the gain occurs when the training
prior supports truth. When the statistically typical binding is deliberately
wrong, performance falls to chance/slightly below it. Its positive correlation
with prior margin confirms that its cosine decision moves with training
typicality.

**Sensitivity to the prior is not unique to scratch JEPA.** Dense and
from-dense JEPA also score better when the prior agrees, and have equal or
larger prior-margin correlations. The qualitative difference is conflict
resolution: dense remains significantly above chance when the prior is wrong,
whereas neither scratch model does. Dense therefore contains sample-specific
binding information in addition to dataset statistics.

**The final residual is not the whole from-dense story.** Exploratorily,
`L6.mlp_out` reaches 0.685 [0.641, 0.734] on prior conflicts for layerwise JEPA
from dense, versus dense's best `L5.mlp_out` at 0.607 [0.558, 0.653]. This
matches the earlier d′ result that layerwise refinement concentrates binding in
an internal MLP feature rather than improving every readout. It is exploratory
because the feature was selected after examining all layers.

The supported conclusion is therefore:

> From-scratch JEPA does not build a robust sample-specific binding
> representation. The averaged variant learns a weak signal dominated by
> training-distribution typicality when the prior and the actual scene
> disagree; the layerwise variant largely fails to learn the signal at all.
> Diffusion training supplies binding evidence that can override the prior.

## Reproduce

```bash
python3 evaluate_dataset_prior_conflict.py \
  --checkpoint scratch_layerwise=outputs/text_data2vec_scratch_layerwise_all_randt_2m_4e/epoch_003.pt \
  --checkpoint scratch_average=outputs/text_data2vec_scratch_window12_24_30pct_avg_l4to7_2m_4e/epoch_003.pt \
  --checkpoint dense=outputs/text_dense_diffusion_2m_4e_matched/epoch_003.pt \
  --checkpoint from_dense_layerwise=outputs/text_data2vec_from_dense_layerwise_all_2m_2e/epoch_001.pt \
  --output-dir outputs/dataset_prior_conflict
```

