# Modulewise EMA-JEPA + HSIC representation, specialization, and causal-use evaluation

> **Type:** evaluation plan, text-only study · **Status:** partly carried out  
> **Menu:** [experiment catalog](../README.md)

## What has been done so far

| Plan level | Status | Where |
|---|---|---|
| Level 1: semantic accessibility (frozen probes) | Done, with a stricter replacement | Ordinary probes: `outputs/modulewise_jepa_hsic_epoch000_modality_probe/REPORT.md` and [collapse diagnosis](../evaluations/data2vec_gated_collapse_diagnosis.md#probes-on-the-collapsed-module). Because ordinary probes are largely solvable from word content, the [binding-swap evaluation](../evaluations/binding_swap_evaluation.md) is now the primary semantic measure. |
| Level 1: nearest-neighbour agreement | Done as retrieval | [Cross-pattern semantic retrieval](../evaluations/cross_pattern_semantic_retrieval.md) |
| Level 2: private usefulness, incremental information, dependence | **Not done** | Planned: incremental probes (does private add semantic information beyond shared? does it add wording information?) |
| Level 3: causal use (text shared-update intervention) | **Not done** for the text study | Script exists: `evaluate_causal_shared_private_swap_text.py`; image version: [causal swap](../evaluations/causal_shared_private_swap.md) |
| Diffusion-quality constraint | Done | Validation losses in the [text-study run registry](../models/text_study_run_registry.md) |

The original plan follows unchanged.

## Decision question

Did modulewise EMA-JEPA make the middle **shared** text LoRA route encode scene semantics that are more accessible, less redundant with the **text-private** route, and actually used during text denoising?

The evaluation has three ordered levels:

$$
\boxed{\text{representation accessibility} \;\rightarrow\; \text{specialization / dependence} \;\rightarrow\; \text{causal use}.}
$$

Fixed-mask diffusion validation is a constraint.  It is not the representation-selection objective by itself.

## Checkpoints and comparisons

All primary comparisons use each run's `best.pt`, selected by the same fixed-$t=0.75$ validation metric.  Evaluation never chooses a best layer or module after looking at the held-out test set.

| ID | Checkpoint | Role in the inference |
|---|---|---|
| `plain_lora` | `outputs/text_lora_diffusion_2m_4e_matched/best.pt` | **Primary diffusion-only control.** It has the same no-base Tri-LoRA shared/private structure, dataset, seed, duration, and optimizer schedule, but no JEPA or HSIC. |
| `avg_no_hsic` | `outputs/text_module_jepa_avg_no_hsic_2m_4e/best.pt` | Effect of averaged-target JEPA relative to `plain_lora`. |
| `avg_hsic` | `outputs/text_module_jepa_avg_hsic_2m_4e/best.pt` | Original weak-HSIC comparison within averaged-target JEPA. |
| `avg_hsic_calibrated` | `outputs/text_module_jepa_avg_hsic_calibrated_2m_4e/best.pt` | Main HSIC test.  This run raises the adaptive HSIC upper bound from 1 to 50 so its realized gradient ratio can approach the intended 5%. |
| `layerwise_no_hsic` | `outputs/text_module_jepa_layerwise_no_hsic_2m_4e/best.pt` | Effect of layerwise JEPA relative to `plain_lora`. |
| `layerwise_hsic` | `outputs/text_module_jepa_layerwise_hsic_2m_4e/best.pt` | HSIC comparison within layerwise JEPA. |
| `dense` | `outputs/text_dense_diffusion_2m_4e_matched/best.pt` | Supporting diffusion/model-capacity control only.  It has no shared/private LoRA split, so it is not used for branch-gap or HSIC claims. |

The causal contrasts are therefore:

$$
\begin{aligned}
&\text{JEPA effect:} && \texttt{avg\_no\_hsic},\;\texttt{layerwise\_no\_hsic} - \texttt{plain\_lora},\\
&\text{HSIC effect:} && \texttt{avg\_hsic} - \texttt{avg\_no\_hsic},\quad
\texttt{layerwise\_hsic} - \texttt{layerwise\_no\_hsic},\\
&\text{Adequately calibrated HSIC:} && \texttt{avg\_hsic\_calibrated} - \texttt{avg\_no\_hsic},\\
&\text{Target design:} && \texttt{layerwise\_no\_hsic} - \texttt{avg\_no\_hsic}.
\end{aligned}
$$

The original HSIC runs saturated their coefficient at 1 and realized only about 0.2--0.4% of the diffusion gradient.  They remain useful as weak-penalty ablations, but a negative result from them must not be interpreted as evidence against HSIC.

## Representations under test

For clean or corrupted text caption $x$, selected block $l\in\{2,3,4\}$, module $b\in\{\mathrm{att},\mathrm{mlp}\}$, and eligible caption token $i$, record the native adapter output updates:

$$
S_{b,i}^{(l)}=\Delta h_{b,\mathrm{shared},i}^{(l)},\qquad
P_{b,i}^{(l)}=\Delta h_{b,\mathrm{text-private},i}^{(l)}.
$$

Here `att` is `attn.out_proj` and `mlp` is `mlp.3`, exactly as in the training objective.  These are not full hidden states and are not obtained by subtracting a private-free forward pass.

For scene-level semantic probes, mean-pool only eligible content-token updates after L2 normalizing each token update:

$$
\bar S_b^{(l)}(x)=\frac{1}{|E_x|}\sum_{i\in E_x}N(S_{b,i}^{(l)}),\qquad
\bar P_b^{(l)}(x)=\frac{1}{|E_x|}\sum_{i\in E_x}N(P_{b,i}^{(l)}),
$$

where $E_x$ excludes padding and special tokens.  Retain the full hidden-state pooled readout as a descriptive reference, but do not mistake it for evidence about either adapter route.

Evaluate every $(l,b)$ separately.  Do not pool layers or mix attention with MLP before the primary analysis: either operation could hide the specialization that the modulewise objective is intended to produce.

## Fixed data and randomness

| Item | Setting |
|---|---|
| Probe-fitting split | A fixed, manifest-indexed subset of 2,048 training captions. |
| Model-selection split | A disjoint fixed subset of 512 training captions used only to select logistic-regression regularization. |
| Held-out scoring split | A fixed, manifest-indexed subset of 2,048 validation captions. |
| Semantic labels | Object count and scene-level presence labels for color, shape, material, size, and spatial relations, derived from the CLEVR scene manifest. |
| Clean pass | `eval()` mode, no corruption, same examples for every checkpoint. |
| Robustness pass | Fixed $t=0.8$ corruption; reset the generator for every batch so all checkpoints see identical masks. |
| Probe seeds | Five fixed seeds for probe fitting and k-NN tie handling; report mean, seed range, and paired bootstrap confidence intervals over held-out examples. |
| Feature preprocessing | Fit L2 normalization and standardization on each probe-fitting split only; apply the fitted transforms unchanged to validation. |

The exact manifest row indices, corruption seeds, tokenizer checksum, checkpoint hash, and feature-extraction configuration must be stored in the result directory.  This avoids a changed split or changed mask realization being confused with a representation improvement.

## Level 1 — semantic accessibility

### Frozen linear probes

For every representation $R\in\{\bar S_b^{(l)},\bar P_b^{(l)},[\bar S_b^{(l)},\bar P_b^{(l)}],H^{(l)}\}$, fit only a regularized linear classifier:

$$
\hat y=WR+c.
$$

Fit one multinomial, class-balanced logistic probe for object count and one class-balanced binary probe for every semantic label.  The regularization value is selected on the model-selection split, then frozen before the held-out test split is scored.  Score count and binary labels with balanced accuracy; report macro averages only in addition to the complete per-label table.

The primary table has one row for each checkpoint, layer, and module:

| Model | Layer | Module | Route | Count | Color | Shape | Material | Size | Relations | Semantic macro avg. |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| `plain_lora` | 2/3/4 | att/mlp | shared/private |  |  |  |  |  |  |  |
| each JEPA/HSIC run | 2/3/4 | att/mlp | shared/private |  |  |  |  |  |  |  |

Run this table once on clean captions and once on 80%-masked captions.  Clean probes answer whether semantic information is linearly accessible; masked probes answer whether it remains accessible under the JEPA-relevant corruption condition.

### k-nearest-neighbor agreement

For the same standardized frozen features, compute $k\in\{5,15,31\}$ nearest neighbors in the fitting split for each held-out caption.  Use the same class-balanced scores as the linear probes.  Pre-register $k=15$ as the headline value; the other two values are sensitivity checks.

Linear and k-NN results answer different questions:

- Linear probes test whether semantics are readily linearly extractable.
- k-NN tests whether semantically similar captions occupy nearby regions of the representation space.

An improvement supported by both is stronger than an improvement in only one.

### Primary representation criterion

For semantic macro accuracy $A_{S,b,l}$ and $A_{P,b,l}$, define the specialization gap:

$$
G_{b,l}=A_{S,b,l}-A_{P,b,l}.
$$

The main result is the change in this gap relative to plain LoRA:

$$
\Delta G_{b,l}^{m}=G_{b,l}^{m}-G_{b,l}^{\texttt{plain\_lora}}.
$$

Evidence that JEPA improves the shared route requires more than a high shared score.  The preferred pattern is:

$$
A_S\uparrow,\qquad G\uparrow,\qquad \text{and } A_P \text{ remains useful rather than collapsing.}
$$

Report every layer and module, plus an explicitly pre-registered equal-weight average across the six $(b,l)$ cells.  Never report only the best held-out cell.

## Level 2 — specialization and non-redundancy

### Private usefulness: lexical/local probes

Semantic separation is not a success if the private route is simply zero.  On token-level features before scene pooling, evaluate private and shared updates for:

1. exact token identity at the same eligible position;
2. relative position bucket within the caption;
3. local left/right token identity, using only positions for which the target neighbor exists; and
4. caption lexical-form features (for example, the surface form of color, shape, and relation words).

Use the same linear-only fitting protocol.  For targets that are predictable directly from the current input token, score both clean and masked passes and label the result as a *local-retention diagnostic*, not as an abstract-semantic result.  The desired qualitative pattern is shared semantic accessibility increasing while private retains comparatively stronger lexical/local information.

### Incremental information

Use capacity-matched linear probes for $S$, $P$, and their concatenation $[S,P]$.  Define:

$$
\Delta_{S\mid P}=A([S,P])-A(P),\qquad
\Delta_{P\mid S}=A([S,P])-A(S).
$$

For semantic labels, the intended pattern is:

$$
\Delta_{S\mid P}>\Delta_{P\mid S}.
$$

This is an operational unique-information diagnostic, not a formal partial-information-decomposition estimate.  It asks whether shared adds semantic value beyond what private already carries.

### Held-out dependence

For each $(b,l)$ and each checkpoint, recompute the exact biased RBF-HSIC estimator used in training on fixed held-out masked positions.  Use 512 positions per deterministic batch, the same per-batch detached-median bandwidth rule, and aggregate equally across batches.

Report:

$$
\operatorname{HSIC}(S_b^{(l)},P_b^{(l)}),
$$

its equal-weight module/layer average, and the paired-bootstrap difference from the matched no-HSIC checkpoint.  Also report shared and private update norms and their effective rank.  A lower HSIC is evidence of reduced dependence only when paired with preserved norms, retained private local information, and useful shared semantic probes.

### Level-2 success condition

The calibrated-HSIC run supports the intended decomposition only if all of the following hold relative to `avg_no_hsic`:

1. realized training HSIC/diffusion gradient ratio is materially closer to the 5% target than in the original HSIC run;
2. held-out HSIC is lower with a paired-bootstrap interval excluding zero in the expected direction;
3. shared semantic score and the shared-minus-private gap do not materially regress; and
4. private norms/effective rank and local-retention probes rule out a trivial private collapse.

## Level 3 — causal use by denoising

Run this only after Level 1 produces a repeatable shared semantic advantage.  It distinguishes “the shared route contains semantic information” from “the denoiser uses the shared route to control semantics.”

### Text shared-update intervention

Construct fixed held-out source/target caption pairs $(A,B)$ with deliberately different scene labels.  Cache clean source shared native updates $S_b^{(l)}(A)$.  During the denoising of a fixed corrupted target $B$, replace only selected source updates:

$$
S_b^{(l)}(B)\leftarrow S_b^{(l)}(A),\qquad
P_b^{(l)}(B)\text{ remains unchanged}.
$$

Include four conditions for every pair:

| Condition | Shared updates | Private updates | Purpose |
|---|---|---|---|
| Self | $S(B)$ | $P(B)$ | Normal target reconstruction. |
| Shared swap | $S(A)$ | $P(B)$ | Primary causal intervention. |
| Private swap | $S(B)$ | $P(A)$ | Branch-specific control. |
| Joint swap | $S(A)$ | $P(A)$ | Positive transfer/reference condition. |

Decode generated captions with a frozen, held-out semantic parser.  Score each output against $Y_A$ and $Y_B$; report source advantage, target retention, and source-win rate.  The parser itself must meet a pre-specified held-out quality threshold before causal claims are made.  Also publish representative source/target/generated triples, selected before scoring.

The intended result is that the shared-swap condition preferentially transfers source scene semantics while the target-private route remains fixed.  A private swap that produces the same result would weaken the route-specific interpretation.

## Diffusion-quality constraint

For every checkpoint, report final and best fixed-$t=0.75$ validation loss, plus generation/reconstruction diagnostics using the same held-out captions.  Representation scores do not excuse severe denoising degradation.

Before inspecting probes, set the practical acceptability rule to: no more than 3% relative degradation in fixed-mask validation loss versus `plain_lora`, unless the result is explicitly labelled a representation--generation trade-off rather than a successful all-around model.  This threshold is a decision aid, not a statistical test.

## Statistical reporting and interpretation rules

- Use the same held-out examples for every checkpoint and paired bootstrap resampling over examples (10,000 resamples, fixed seed).
- Report effect size and 95% confidence interval for every primary contrast, not only a $p$-value.
- A headline semantic result requires the expected sign in at least four of the six semantic groups (count, color, shape, material, size, relations) and an interval excluding zero for the pre-registered six-cell average.
- A module-specific finding is exploratory unless it repeats across both clean and 80%-masked evaluations.
- Do not claim disentanglement from HSIC alone, or semantic specialization from a shared probe alone.
- Keep the complete per-label, per-layer, per-module table in the result artifact; do not select the strongest layer after test scoring.

## Deliverables

Create `outputs/modulewise_jepa_hsic_representation_evaluation/` containing:

1. `protocol.json` with every split index, seed, checkpoint hash, feature name, and preprocessing choice;
2. `semantic_probes_clean.json` and `semantic_probes_masked_t080.json` with all linear-probe and k-NN values;
3. `local_retention_probes.json`, `incremental_information.json`, and `heldout_hsic.json`;
4. bootstrap samples/intervals and a compact `REPORT.md` containing the pre-registered summary tables; and
5. after the gating criteria are met, `causal_text_shared_private_swap/` with parser metrics, pair list, generations, and source/target scores.

## Execution order

1. Wait for `plain_lora`, `dense`, and `avg_hsic_calibrated` to finish; verify the calibrated run's realized HSIC gradient ratio before evaluation.
2. Implement one frozen feature extractor that records native shared/private updates for every $(l,b)$ cell, and write a one-batch equivalence test against the training recorder.
3. Freeze and persist the split indices; run Level 1 on clean and 80%-masked captions for every completed checkpoint.
4. Run Level 2 only from the same saved features and publish all dependence, norm, effective-rank, local-retention, and incremental-information diagnostics together.
5. Decide whether the shared route is meaningfully better and non-redundant.
6. Only then implement and run the causal text-swap evaluation.

This sequence keeps the claim calibrated: probes establish accessibility, specialization tests establish non-redundancy without collapse, and swaps test whether the branch is causally used.
