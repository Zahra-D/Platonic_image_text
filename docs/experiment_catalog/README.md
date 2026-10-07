# CLEVR discrete diffusion — experiment catalog

> **Updated:** 2026-09-24 · **Main question:** can a shared route carry scene semantics while private routes carry modality-specific detail?
>
> **Start here:** [every experiment](all_experiments.md) (62 runs, 9 families, generated) · [key findings](#key-findings-so-far) · [what was withdrawn](#findings-that-were-withdrawn)

This page is the single menu for every model, experiment plan, evaluation, and
result in the project. Every page it links to starts with a short box giving
its type, status, and a link back here.

## Menu

1. [Current study: shared route vs private route, across text and images](#1-current-study-shared-route-vs-private-route-across-text-and-images)
   - [Status at a glance](#status-at-a-glance) · [Key findings](#key-findings-so-far) · [Withdrawn findings](#findings-that-were-withdrawn) · [Runs](#runs) · [Experiment plans](#experiment-plans) · [Evaluations](#evaluations) · [Open questions](#open-questions-and-next-steps)
2. [Earlier phase: multimodal pretraining and single-modality pilots](#2-earlier-phase-multimodal-pretraining-and-single-modality-pilots)
   - [Earlier-phase evaluations](#earlier-phase-evaluations) · [Earlier-phase models](#earlier-phase-models) · [Other reports](#other-reports)
3. [Method reference pages](#3-method-reference-pages)
4. [Glossary](#4-glossary)
5. [Adding a new page](#5-adding-a-new-page)

---

## 1. Current study: shared route vs private route, across text and images

**The question.** Can a *shared* route carry scene semantics — which object has
which attribute, independent of how it is worded or rendered — while *private*
routes carry modality-specific detail?

Three designs have been tried for the split, on three corpora:

| Design | Shared route | Private route | Where it is written up |
|---|---|---|---|
| **Tri-LoRA** | `B_s A_s x`, rank 256 | `B_p A_p x`, rank 128, one per modality; no frozen base | [four-run sweep](plans/modulewise_jepa_hsic_four_run_plan.md) |
| **Stage 2 / dense_private** | the dense weight `Wx` itself | `B_p A_p x`, rank 128 = d/3 | [the plan](plans/dense_shared_private_lora_plan.md) |
| **One dense trunk, two modalities** | the whole trunk | none | [multimodal registry](models/multimodal_run_registry.md) |

**62 runs in 9 families** are trained and catalogued in
[every experiment](all_experiments.md).

### Status at a glance

| | |
|---|---|
| Best text model | `text_dense_private_trainable_hsic_from_d2v_lw_all_2m_2e`, **read trunk-only**: `d_bind` **+0.555**, `d_sem` +0.63 — [stage 2](plans/dense_shared_private_lora_plan.md#the-full-22-both-trunks-frozen-and-trainable) |
| Best image model | `image_data2vec_from_dense_layerwise_all_block2d_1_2m_2e`, binding **0.822** vs dense's 0.802 — [image binding](evaluations/image_binding.md) |
| Best cross-modal model | `multimodal_paired_dense_1_2m_4e`, **CKA 0.622 at L6** — nothing unpaired is within 3× — [modality alignment](evaluations/modality_alignment.md) |
| Running | 4 harder-masking from-scratch runs (image 65%, text 60%), one d′ backfill |
| Primary semantic measure | [Semantic effect sizes (d′)](evaluations/semantic_dprime.md) for text, [image binding](evaluations/image_binding.md) for images |
| The empty cell | **unpaired stage 2** — dense unpaired trunk as shared route + per-modality private LoRA. Never run, and every result points at it. |

### Key findings so far

1. **Layerwise data2vec on a *pretrained* trunk is the only JEPA configuration
   that beats its baseline — in both modalities.** Text: `d_bind` +0.527 against
   budget-matched dense's +0.443, a **+19%** gain. Images: binding 0.822 against
   dense's 0.802. The gain is real and modest.
   ([survey](jepa_model_survey.md), [d′](evaluations/semantic_dprime.md),
   [image binding](evaluations/image_binding.md))
2. **Averaging the target destroys that gain, and it is the averaging, not the
   layer window.** `avg L4–7` loses to `layerwise L4–7` at the *same* layers, in
   text (+0.395 vs +0.445) and in images (0.744 vs 0.814). An earlier hypothesis
   that shallow blocks poison the average is refuted in both modalities.
3. **From scratch, JEPA collapses — at every masking rate and both anti-collapse
   mechanisms.** Random-t, 1-D windows (4–8 and 12–24), 2-D image blocks, EMA
   teacher, SIGReg: `d_bind` lands at the untrained floor (+0.004) every time.
   The diagnosis is that the objective is *too easy* — image cosine to target
   reaches 0.978 after one epoch — not that it is undertrained.
4. **The private route is essentially empty; the shared route carries
   everything.** Route isolation over six two-route models: private-only
   `d_bind` spans +0.003 to +0.024, against +0.004 for an untrained model and
   +0.293 to +0.555 for the same models intact — the largest private reading is
   1/22 of its own model's full score. Trunk-only, meanwhile, matches or
   *exceeds* the full model. The isolation is validated, not assumed: a frozen
   trunk read trunk-only reproduces its parent to three decimals.
   ([stage 2](plans/dense_shared_private_lora_plan.md#route-isolation-the-private-branch-is-empty))
5. **Only paired training produces cross-modal structure, and it is not close.**
   Paired dense reaches CKA 0.554 / 0.622, three times any unpaired model, and
   is the only model whose CKA *climbs* with depth. No self-supervised objective
   on unpaired data reproduced it.
   ([per-layer CKA](evaluations/modality_alignment.md#per-layer-cka-the-depth-profile))
6. **SIGReg / LeJEPA collapses at λ = 0.05, in all four runs.** It is the only
   mechanism that closed the modality gap (0.050, AUC 0.576) — by emptying both
   representations rather than finding shared content. λ was never swept.
   ([survey](jepa_model_survey.md#sigreg--lejepa))
7. **Validation loss is not a proxy for representation quality, twice over.**
   Unpaired dense has the *better* val loss than paired dense (2.1990 vs 2.2596)
   and a 3× worse CKA. Image Tri-LoRA continued until it *beats* dense on val
   loss (3.6214 vs 3.6340) is still 4 points below it on binding.
8. **Pretrained English models score at the bag-of-words control on these
   captions.** GPT-2, RoBERTa, BERT, MPNet, MiniLM and data2vec-text all give
   negative `d_semantic` and `d_bind` ≤ 0.042, which is what makes the effect
   sizes our models reach meaningful rather than an artifact of an easy metric.

### Findings that were withdrawn

Kept visible on purpose, because each was reported before it was checked:

| Claim | What was wrong | Where |
|---|---|---|
| "JEPA more than doubles dense's binding" | compared against the *undertrained* 4-epoch dense (+0.244); budget-matched dense is +0.443, so the gain is +19% | [d′](evaluations/semantic_dprime.md#results) |
| "Averaging shallow blocks is what causes the damage" | `avg L4–7` ≈ `avg all-8`; averaging itself is the problem | [the plan](plans/dense_shared_private_lora_plan.md) |
| "Freezing the trunk beats training it" | compared a frozen *layerwise*-parent run against a trainable *averaged*-parent run — it was measuring the parent | [the 2×2](plans/dense_shared_private_lora_plan.md#the-full-22-both-trunks-frozen-and-trainable) |
| "Gated layerwise JEPA + HSIC is the best configuration" | true on the binding probe, inverted on effect sizes; it is among the worst trained models on d′ | [d′](evaluations/semantic_dprime.md) |

### Earlier Tri-LoRA phase findings

These concern the 11 Tri-LoRA / modulewise runs on the 2M text corpus and still
stand *for that family*; they were the state of the study before the from-dense
data2vec and stage-2 families existed.

1. **Ordinary retrieval is mostly lexical; the hard version is not.** Word
   counts retrieve same-scene paraphrases at R@10 = 70.9%
   ([retrieval](evaluations/cross_pattern_semantic_retrieval.md)). With
   lexically identical candidates, word counts and an untrained model score
   chance, dense reaches 22.8% / 71.5% (attribute / relation, chance 12.5% /
   43.0%), plain Tri-LoRA 19.7% / 62.2%, and every JEPA variant is lower.
   ([hard retrieval](evaluations/hard_retrieval.md))
2. **JEPA moves relational information into shared, not attribute binding.**
   At `blocks.4.mlp.3`, shared beats private under every non-collapsed JEPA
   variant. Shared relation binding rises to 93.9–96.4% (plain 82.1%), but
   shared attribute binding stays at 53–59% in every non-collapsed model
   (plain 57.8%).
   ([binding](evaluations/binding_swap_evaluation.md))
3. **The shared/private split does not reach the representation.** Summing
   everything each branch writes into the residual stream, the private stream
   binds attributes at least as well as the shared stream in all 10 Tri-LoRA
   models. The module-level split is local.
   ([decomposed representations](evaluations/decomposed_representations.md))
4. **Removing the predictor collapsed the shared branch.** The gated data2vec
   runs collapsed at the supervised module (effective rank 1.8 and 1.1) while
   diffusion loss stayed normal; standard probes and masked–clean stability
   did not detect it. ([collapse diagnosis](evaluations/data2vec_gated_collapse_diagnosis.md))
5. **Pure-JEPA scratch tuning helps attributes but not relation binding.** In
   eight matched pilots, slow EMA and faithful 15% BERT corruption beat fast
   EMA/global pooling; best attribute R@1 reached 18.9%, but relation remained
   at or below 48.3% versus 71.5% in dense diffusion.
   ([sweep](evaluations/pure_jepa_scratch_hyperparameter_sweep.md))

### Runs

Three registries, one per corpus, plus the generated master index:

| Registry | Covers |
|---|---|
| **[Every experiment](all_experiments.md)** | **all 62 runs, generated from configs and results — start here** |
| [Text-study run registry](models/text_study_run_registry.md) | the 2M-caption text runs: status, validation loss, effect sizes, W&B, configs |
| [Image-study run registry](models/image_study_run_registry.md) | the 1.2M-image runs, 2-D block masking, and the four from-dense target designs |
| [Multimodal run registry](models/multimodal_run_registry.md) | the 7 paired/unpaired 1.2M runs — one model, both modalities |

| Group | Runs | Status |
|---|---|---|
| Controls | dense [4v57nzhs](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/4v57nzhs), plain Tri-LoRA [4q137e90](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/4q137e90) | finished |
| Original modulewise JEPA sweep | average / layerwise × with / without HSIC, calibrated average + HSIC | finished |
| All-shared-module JEPA | average, layerwise | stopped, no checkpoints |
| Gated data2vec | average and final-layer targets | finished, collapsed |
| Gated-predictor layerwise JEPA | [itm48bxb](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/itm48bxb) no HSIC, [y0eklen2](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/y0eklen2) + HSIC | finished; + HSIC is the best JEPA variant |

### Results surveys

| Page | What it covers |
|---|---|
| [Every experiment](all_experiments.md) | One row per trained run: method in plain terms, what it started from, tokens (own and cumulative through stages), and every evaluation |
| [JEPA model survey](jepa_model_survey.md) | Every JEPA run with its training budget in tokens, effect sizes split by route, the image models, the compute-matched dense control, JEPA-to-JEPA cross-modal agreement, the [SIGReg/LeJEPA family](jepa_model_survey.md#sigreg--lejepa) and the [multimodal family](jepa_model_survey.md#the-multimodal-family-paired-vs-unpaired) |

### Runbooks

| Page | What it covers |
|---|---|
| [Running these runs on another cluster](running_on_another_cluster.md) | Files to copy, dependencies, commands, Slurm script, and how to check a run is correct |

### Experiment plans

| Plan | Question | Status |
|---|---|---|
| [Modulewise EMA-JEPA + HSIC four-run sweep](plans/modulewise_jepa_hsic_four_run_plan.md) | Does average vs. layerwise targeting, with or without HSIC, move semantics into the shared route? | complete |
| [All-shared-module JEPA](plans/modulewise_jepa_all_shared_correction.md) | Does supervising all four adapter types change the result? | not completed |
| [Gated data2vec JEPA](plans/data2vec_gated_shared_lora_plan.md) | Direct regression without a predictor, gradient end-to-end | complete, collapsed |
| [Gated-predictor layerwise JEPA](plans/gated_predictor_layerwise_jepa.md) | End-to-end gradient routing alone, with the predictor kept | complete; + HSIC best so far |
| [Representation evaluation plan](plans/modulewise_jepa_hsic_representation_evaluation_plan.md) | Which evaluations decide the shared/private claim? | partly carried out |
| [Dense shared route + rank-128 private LoRA](plans/dense_shared_private_lora_plan.md) | Does a full-rank dense shared route with only the private branch rank-limited beat the two-adapter split? | **complete, all four cells: yes — this design holds the best text model in the study (`d_bind` +0.555 trunk-only), and route isolation shows the private branch is empty** |
| [Grounded global JEPA from scratch](plans/scratch_grounded_global_jepa.md) | Can structured denoising bootstrap a global JEPA target without a pretrained checkpoint? | cancelled; replaced by pure JEPA |

### Evaluations

| Evaluation | Question | Status |
|---|---|---|
| [Binding-swap evaluation](evaluations/binding_swap_evaluation.md) | Does a representation know which attribute belongs to which object? Removes lexical shortcuts by construction. Preference rates saturate, so [d′](evaluations/semantic_dprime.md) is now the primary text measure. | final: the 11 Tri-LoRA runs |
| [Cross-pattern semantic retrieval](evaluations/cross_pattern_semantic_retrieval.md) | Can a representation find the same scene written in a different sentence structure? Includes dense, every layer, and lexical controls. | final: all models |
| [Exact-template counterfactual](evaluations/exact_template_counterfactual.md) | Same scene in new wording, or same wording with every value changed? | complete; superseded by binding |
| **[Semantic effect sizes (d′)](evaluations/semantic_dprime.md)** | How much is a representation about the scene rather than the wording, and how sensitive is it to binding? Signed effect sizes with a usable range; 38 model/route cells including six pretrained baselines, with shared/private route isolation. **Primary, text.** | final |
| [Dataset-prior conflict](evaluations/dataset_prior_conflict.md) | Does scratch JEPA encode the held-out scene's binding, or follow the empirical training joint when the two disagree? Checkpoint-only, no trained probe, exact reuse of existing d′ items. | complete: 4 models |
| **[Image binding](evaluations/image_binding.md)** | Does an image model know *which* object has which attribute? Built from 397 content-matched scene pairs that occur naturally in the corpus; chance 50%. **Primary, images.** | final: 11 models |
| **[Modality alignment](evaluations/modality_alignment.md)** | For one model encoding both modalities: modality gap, AUC, paired R@1, and [CKA at every layer](evaluations/modality_alignment.md#per-layer-cka-the-depth-profile) for 15 models | final |
| **[Cross-modal structure](evaluations/cross_modal_structure.md)** | Do independently trained text and image models converge on the same scene structure? CKA, RSA and the per-modality scene probe. | final: 26 models |
| [Hard retrieval](evaluations/hard_retrieval.md) | Among candidates with exactly the same words, does cosine similarity find the true scene rather than a binding swap? **Superseded by d′** — it compresses model differences into a 10-point range and separated none of the from-dense variants. | final for the 11 Tri-LoRA runs only |
| [Decomposed representations](evaluations/decomposed_representations.md) | Attribute binding, relation binding and retrieval for every layer, every sublayer in blocks 2–4, and the accumulated shared and private streams | final: all models |
| [Gated data2vec collapse diagnosis](evaluations/data2vec_gated_collapse_diagnosis.md) | What happened to the representation when the predictor was removed? | complete |
| **[Pure-JEPA scratch hyperparameter sweep](evaluations/pure_jepa_scratch_hyperparameter_sweep.md)** | Do EMA, masking, global pooling, and target depth make JEPA work from random initialization without diffusion? | complete: 8 pilots |

### Open questions and next steps

**Ranked by what the current results point at.**

1. **Unpaired stage 2 — the decisive experiment, never run.** Take the unpaired
   dense trunk as the shared route and add one private LoRA per modality
   (rank 128 = d/3), exactly the design that produced the best text model. This
   is the shared/private hypothesis stated on the multimodal data it was
   formulated for, and it is the one empty cell in the grid. Everything else
   below is secondary to it.
2. **Why is the private route empty?** Three confounds are untested and all
   predict the same data: HSIC may be *pushing* content out rather than the
   private branch failing to attract it (a stage-2 run with HSIC disabled would
   separate these); capacity (rank 128 against full rank); and the trunk's head
   start (joint training from scratch in this layout).
   [Details](plans/dense_shared_private_lora_plan.md#is-the-shared-advantage-just-private-trained-less).
3. **A SIGReg λ sweep.** λ = 0.05 collapsed all four runs and was never varied.
   The per-layer CKA profile shows the first block switching the representation
   off, which is the signature of a penalty that is simply too strong. An
   image-only SIGReg run is the other missing cell.
4. **Does harder masking rescue from-scratch JEPA?** Four runs at ~60–65%
   realized masking are training. The diagnosis says the objective converges in
   one epoch because it is too easy; if that is right, these should move and
   nothing else will.
5. **A paired JEPA run.** Every JEPA run on the multimodal corpus is unpaired.
   Since paired training is the only thing that produced cross-modal structure,
   layerwise data2vec on the *paired* dense trunk is an obvious and cheap test.
6. **Backfill the evaluations that lag.** Six hard-retrieval cells and one d′
   cell are outstanding; 11 pilots and 2 MS-COCO runs have never been scored.
7. **Fix confidence-order sampling.** `topk` without Gumbel noise makes every
   `reveal_order: confidence` configuration produce blank images. This affects
   generation only, not any representation measurement reported here.

### Older open questions, from the Tri-LoRA phase

- Shared *route*, not just one module: apply JEPA/HSIC to the accumulated
  shared writes, and discourage binding in the private stream (e.g. an
  adversarial attribute probe with gradient reversal).
- Attribute binding: no configuration puts it into shared. Blocks 2–4 have the
  lowest text–image gradient conflict, but attribute binding forms at blocks
  5–6. Candidate ablation: gated + HSIC with JEPA on blocks 3–5.
- Not yet run for the text study: the causal shared/private swap and
  incremental probes (does private add semantic information beyond shared?).
- If collapse reappears: add a variance floor on the shared updates.

---

## 2. Earlier phase: multimodal pretraining and single-modality pilots

### Earlier-phase evaluations

| Evaluation | Question | Status |
|---|---|---|
| [Modality-local semantic probe at 80% masking](evaluations/modality_local_jepa_semantic_80pct.md) | Does the shared representation keep scene semantics with one modality 80% masked? | complete |
| [Causal shared/private swap (image-only)](evaluations/causal_shared_private_swap.md) | Do substituted shared updates redirect the generated image's scene? | complete |

### Earlier-phase models

Status is the W&B run state checked on 2026-09-17. Each model page lists its
configuration, objective, and evaluated values.

| Model | Family | W&B | Status | Checkpoint |
|---|---|---|---|---|
| [Dense paired](models/dense_paired.md) | Multimodal | [h7u81qb6](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_shared_lora_alignment/runs/h7u81qb6) | finished | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_paired_dense_absolute_70e/best.pt) |
| [LoRA paired](models/lora_paired.md) | Multimodal | [30fi6whi](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_shared_lora_alignment/runs/30fi6whi) | finished | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_paired_lora_absolute_70e/best.pt) |
| [Dense unpaired](models/dense_unpaired.md) | Multimodal | [coyf0dx3](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_shared_lora_alignment/runs/coyf0dx3) | finished | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_unpaired_dense_absolute_70e/best.pt) |
| [Plain LoRA unpaired](models/plain_lora_no_stage.md) | Multimodal | [zvle4hpq](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_shared_lora_alignment/runs/zvle4hpq) | finished | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_unpaired_lora_absolute_no_stage0_70e/best.pt) |
| [LoRA stage 0](models/lora_stage0.md) | Multimodal | [smua88cz](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_shared_lora_alignment/runs/smua88cz) | finished | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_unpaired_lora_absolute_stage0_70e/best.pt) |
| [LoRA + DANN](models/lora_dann.md) | Multimodal | [vn430zje](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_shared_lora_alignment/runs/vn430zje) | finished | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_unpaired_lora_absolute_dann_70e/best.pt) |
| [LoRA stage 0 → DANN](models/lora_stage0_then_dann.md) | Multimodal | [yglwleep](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_shared_lora_alignment/runs/yglwleep) | finished | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_unpaired_lora_absolute_stage0_then_dann_70e/best.pt) |
| [LoRA + SIGReg](models/sigreg.md) | Multimodal | [1ambo2ko](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_shared_lora_alignment/runs/1ambo2ko) | finished | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_unpaired_lora_sigreg_absolute_no_stage0_70e/best.pt) |
| [LoRA stage 0 → SIGReg](models/stage0_sigreg.md) | Multimodal | [29sd132d](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_shared_lora_alignment/runs/29sd132d) | finished | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_unpaired_lora_absolute_stage0_then_sigreg_70e/best.pt) |
| [LoRA + per-layer SIGReg](models/per_layer_sigreg.md) | Multimodal | [ldbop141](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_shared_lora_alignment/runs/ldbop141) | finished | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_unpaired_lora_sigreg_per_layer_absolute_no_stage0_70e/best.pt) |
| [LoRA stage 0 → per-layer SIGReg](models/stage0_per_layer_sigreg.md) | Multimodal | [t443ovp6](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_shared_lora_alignment/runs/t443ovp6) | finished | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_unpaired_lora_absolute_stage0_then_sigreg_per_layer_70e/best.pt) |
| [LoRA + per-layer JEPA/SIGReg](models/jepa_per_layer.md) | Multimodal | [ttztvt9n](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_shared_lora_alignment/runs/ttztvt9n) | finished | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_unpaired_lora_shared_jepa_sigreg_per_layer_absolute_70e/best.pt) |
| [LoRA + normalized JEPA/SIGReg](models/jepa_normalized.md) | Multimodal | [s6159f54](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_shared_lora_alignment/runs/s6159f54) | finished | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_unpaired_lora_middle_l2_l4_normalized_jepa_sigreg_absolute_70e/best.pt) |
| [LoRA + strong normalized JEPA/SIGReg](models/jepa_strong_normalized.md) | Multimodal | [pz54tjy5](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_shared_lora_alignment/runs/pz54tjy5) | finished | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_unpaired_lora_middle_l2_l4_strong_balanced_normalized_jepa_sigreg_absolute_70e/best.pt) |
| [LoRA + strong raw-L2 JEPA/SIGReg](models/jepa_strong_raw_l2.md) | Multimodal | [hv18e0fd](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_shared_lora_alignment/runs/hv18e0fd) | finished | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_unpaired_lora_middle_l2_l4_strong_balanced_raw_l2_jepa_sigreg_absolute_70e/best.pt) |
| [Text diffusion-only](models/text_diffusion.md) | Text-only pilot | [v1gr53y0](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_single_modality_ema_jepa/runs/v1gr53y0) | finished | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/text_diffusion_only_ema_jepa_pilot/best.pt) |
| [Text EMA-JEPA fixed](models/text_ema_fixed.md) | Text-only pilot | [kdo2g035](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_single_modality_ema_jepa/runs/kdo2g035) | finished | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/text_ema_jepa_fixed_lambda050/best.pt) |
| [Text EMA-JEPA dynamic .10](models/text_ema_r010.md) | Text-only pilot | [s7ii512c](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_single_modality_ema_jepa/runs/s7ii512c) | stopped early (checkpoint and report retained) | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/text_ema_jepa_dynamic_r010/best.pt) |
| [Text EMA-JEPA dynamic .25](models/text_ema_r025.md) | Text-only pilot | [ykppwfrp](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_single_modality_ema_jepa/runs/ykppwfrp) | stopped early (checkpoint and report retained) | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/text_ema_jepa_dynamic_r025/best.pt) |
| [Text EMA-JEPA dynamic .50](models/text_ema_r050.md) | Text-only pilot | [xlzkx2ev](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_single_modality_ema_jepa/runs/xlzkx2ev) | stopped early (checkpoint and report retained) | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/text_ema_jepa_dynamic_r050/best.pt) |
| [Image diffusion-only](models/image_diffusion.md) | Image-only pilot | [rp9rbau8](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_single_modality_ema_jepa/runs/rp9rbau8) | failed (checkpoint and report retained) | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/image_diffusion_only_ema_jepa_pilot/best.pt) |
| [Image EMA-JEPA fixed](models/image_ema_fixed.md) | Image-only pilot | [el5xn2y3](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_single_modality_ema_jepa/runs/el5xn2y3) | finished | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/image_ema_jepa_fixed_lambda050_retry/best.pt) |
| [Image EMA-JEPA dynamic .10](models/image_ema_r010.md) | Image-only pilot | [6q6cywma](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_single_modality_ema_jepa/runs/6q6cywma) | finished | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/image_ema_jepa_dynamic_r010/best.pt) |
| [Image EMA-JEPA dynamic .25](models/image_ema_r025.md) | Image-only pilot | [d2lp63ha](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_single_modality_ema_jepa/runs/d2lp63ha) | finished | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/image_ema_jepa_dynamic_r025/best.pt) |
| [Image EMA-JEPA dynamic .50](models/image_ema_r050.md) | Image-only pilot | [6ncr6t5q](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_single_modality_ema_jepa/runs/6ncr6t5q) | finished | [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/image_ema_jepa_dynamic_r050/best.pt) |

### Other reports

- [All multimodal-LoRA 80% semantic values: text and image, every attribute](/home/zd25e122/clevr_discrete_diffusion/outputs/LORA_MULTIMODAL_SEPARATE_MODALITY_ATTRIBUTE_REPORT.md)
- [First-epoch text-only shared/private probe report](../../outputs/modulewise_jepa_hsic_epoch000_modality_probe/REPORT.md)
- [First-epoch dense versus plain-LoRA layer-representation report](../../outputs/dense_lora_epoch000_layer_probe/REPORT.md)
- [Multimodal LoRA experiment notebook](/home/zd25e122/clevr_discrete_diffusion/notebooks/multimodal_lora_experiment_record.ipynb)

---

## 3. Method reference pages

Short descriptions of diagnostic tools, mostly from the multimodal phase.

| Tool | What it measures |
|---|---|
| [Diffusion training and validation loss](evaluations/reference/diffusion_training_validation.md) | Weighted and unweighted masked-token loss and accuracy logged during training |
| [Generation and reconstruction quality](evaluations/reference/generation_reconstruction_quality.md) | Marginal and conditional generation diagnostics |
| [Paired conditional validation](evaluations/reference/paired_conditional_validation.md) | Matched vs. shuffled vs. null context loss on reserved text–image pairs |
| [Layerwise branch recall](evaluations/reference/branch_layerwise_recall.md) | Paired text–image retrieval per layer from shared-only or private-only branches |
| [No-base branch ablation](evaluations/reference/no_base_branch_ablation.md) | Loss with full, shared-only, or private-only routing |
| [Layerwise text–image gradient conflict](evaluations/reference/layer_gradient_conflict.md) | Which layers' text and image gradients agree (basis for the blocks 2–4 choice) |
| [JEPA objective checkpoint test](evaluations/reference/jepa_objective_checkpoint_test.md) | Masked-position prediction against clean targets |
| [JEPA training trajectories](evaluations/reference/jepa_training_trajectories.md) | JEPA loss and cosine over training from W&B |
| [Semantic clustering and probes](evaluations/reference/semantic_clustering_probes.md) | K-means and frozen probes on shared representations |
| [SIGReg geometry](evaluations/reference/sigreg_geometry.md) | Distribution and Gaussianity of shared representations |

---

## 4. Glossary

| Term | Meaning |
|---|---|
| Block / layer `L0`–`L7` | Zero-based Transformer block index |
| Tri-LoRA, no base | Each adapted linear layer outputs shared update + private update + bias; there is no frozen dense weight |
| Shared / private update | Output of one LoRA branch (`B·A·x`) of one linear layer, e.g. shared `blocks.4.mlp.3` |
| `out_proj`, `mlp.3` | Attention output projection and MLP down-projection: the two layers that write into the residual stream |
| Residual stream / block output | Hidden state after a Transformer block |
| EMA-JEPA | Masked student predicts the clean exponential-moving-average teacher's representation |
| Average vs. layerwise target | Teacher target averaged over blocks 2–4, or taken from the same block |
| Gated gradient | JEPA gradient routed end-to-end into selected shared LoRA parameters only |
| HSIC | Kernel dependence penalty between shared and private updates |
| Effective rank | exp(entropy of normalized squared singular values); about 1 means collapse |
| R@K, MRR | Retrieval recall at K and mean reciprocal rank of the first correct match |
| Binding accuracy | Minimal-pair probe accuracy; 50% means no binding information |

---

## 5. Adding a new page

- Put protocols with their results in `evaluations/`, training designs in
  `plans/`, and run records in `models/`.
- Start every page with a box: `**Type:**`, `**Status:**`, the question, the
  script and results location, and a `**Menu:**` line pointing back at this page.
- Add one row to the matching table on this page. Each folder also has a
  short index: [evaluations/](evaluations/README.md), [plans/](plans/README.md),
  [models/](models/README.md).
- Mark interim results as interim, and replace them when the final checkpoint
  is evaluated.
