# CLEVR multimodal experiment audit and run ledger

Last updated: 2026-08-25 16:13 CEST

## 1. Research objective

The project is testing whether text/image pretraining can produce a useful
shared representation and reduce the amount of paired supervised data needed
during later bidirectional instruction tuning.

The main comparisons developed into:

1. dense paired training;
2. dense strictly-unpaired pretraining;
3. Tri-LoRA strictly-unpaired pretraining;
4. Tri-LoRA with modality-adversarial distribution alignment;
5. Tri-LoRA without a frozen random base and with a shared-only stage 0;
6. paired bidirectional instruction tuning from the different pretrained
   checkpoints.

The current full-pair instruction runs do **not** yet answer the final
data-efficiency question. Every completed instruction run used all 90,000
training pairs. Reduced-data runs such as 1k/3k/10k/30k/90k are still required.

## 2. Dataset and tokenization

Primary dataset:

`/home/zd25e122/clevr-dataset-gen_clone/output/platonic_clevr_v1_100k_gpu_visible_s20260825`

The dataset was generated using the project-specific CLEVR rule set and the
corrected visibility check. Its split is:

| Split | Rows |
|---|---:|
| Train | 90,000 |
| Validation | 5,000 |
| Test | 5,000 |
| Total | 100,000 |

The training and validation manifests were audited for missing images,
captions, duplicates, and train/validation overlap. The token caches match the
manifest fingerprints. Image tokens have shape `16 x 24`, with valid code IDs
from 0 through 511.

The active text field is `caption_human`. The image tokenizer is the VQ-VAE at:

`outputs/vqvae_training_bs128/best.pt`

## 3. What “strictly unpaired” means in these runs

The source manifest contains the original image-caption association so that the
two complete modality pools can be loaded. For unpaired training, that
association is discarded before model input.

`BalancedUnpairedDataset` independently permutes all 90,000 text indices and
all 90,000 image indices. It rejects permutations until no caption is carried
beside its source image. A direct audit found:

| Check | Result |
|---|---:|
| Carriers | 90,000 |
| Same-index text/image carriers | 0 |
| Unique text indices | 90,000 |
| Unique image indices | 90,000 |

The collator then separates each carrier into a text-only batch and an
image-only batch. The trainer performs two independent model forwards. The
model never receives a text/image pair in the same sequence during unpaired
training.

Paired validation is intentionally still performed after each unpaired epoch,
under `torch.no_grad()`. It is a diagnostic for emergent alignment and does not
provide a training signal or select the best unpaired checkpoint. The primary
unpaired checkpoint metric is the marginal fixed-`t=0.75` validation loss.

## 4. Validation metrics

The paired diagnostic evaluates both directions:

- text context -> masked image target;
- image context -> masked text target.

For each direction it compares:

- `matched_loss`: correct context and target;
- `shuffled_loss`: context from a different row with the same target;
- `null_loss`: target without cross-modal context;
- `shuffle_gap = shuffled_loss - matched_loss`;
- `context_gain = null_loss - matched_loss`.

Matched loss alone does not demonstrate alignment. A model with a strong image
or text marginal can obtain a reasonable matched loss while ignoring context.
The shuffle gap is the more direct correspondence test because matched and
shuffled inputs contain the same modalities and differ only in correspondence.

Near-zero shuffle gap in unpaired pretraining means the model is not using the
specific pairing. Negative context gain means adding the other modality is
currently worse than providing no context, which is possible because an
unpaired model has never processed a mixed sequence during training.

## 5. First corrected-dataset pretraining comparison

### Dense strictly-unpaired pretraining

- Output: `outputs/multimodal_unpaired_dense_correct_dataset_visible_check`
- W&B: `92g8ltfg`
- Status: complete, valid for its configured architecture
- Schedule: 50 epochs, 703 updates/epoch
- Best marginal validation loss: 1.7483 at epoch 48

Purpose: establish how well a single dense backbone learns the two independent
modality marginals when every weight is shared.

### Original Tri-LoRA strictly-unpaired pretraining

- Output: `outputs/multimodal_unpaired_lora_correct_dataset_visible_check`
- W&B: `c386f2nw`
- Status: complete, valid for the original frozen-random-base formulation
- Schedule: 50 epochs, 703 updates/epoch
- Best marginal validation loss: 1.7888 at epoch 49

Purpose: reproduce the shared/text-private/image-private Tri-LoRA idea on the
small CLEVR transformer.

Important limitation: this was not conventional adaptation of a pretrained
dense backbone. The target linear matrices were randomly initialized and then
frozen. Tri-LoRA learned factorized deltas around that random base. Rank 384
also resulted in 25.86M trainable parameters versus 14.88M for dense training,
so this was not a parameter-efficiency comparison.

This design made dense optimization easier: dense directly learned each matrix,
whereas Tri-LoRA had to construct a useful matrix through factor products around
a frozen random transformation. Private branches also allowed the two modality
marginals to specialize without forcing all learning into the shared branch.

### First adversarial Tri-LoRA pretraining

- Output: `outputs/multimodal_unpaired_lora_adversarial_correct_dataset_visible_check_incorrect`
- W&B: `dgyzre9n`
- Status: complete but invalid as an adversarial result
- Best marginal validation loss: 1.7994 at epoch 47

Purpose: encourage modality invariance in the shared Tri-LoRA representation
using a gradient-reversal discriminator.

Discovered defect: `SharedActivationRecorder` overwrote its representation at
every eligible layer. Only the last eligible adapter, `blocks.7.mlp.3`, remained
connected to the adversarial loss. Because adapter input was detached, the DANN
gradient could not propagate into earlier adapters.

Checkpoint evidence showed a pathological concentration of scale in that last
module: the adversarial shared-delta/base norm ratio reached about 122.9x,
versus about 5.35x in ordinary Tri-LoRA. Therefore this run must not be used to
judge whether adversarial alignment works.

## 6. First full-pair instruction-tuning comparison

All instruction runs used:

- 90,000 paired training rows;
- bidirectional text-target and image-target objectives;
- 10 epochs;
- full-target masking with probability 0.15;
- matched/shuffled/null paired evaluation.

### Dense instruction tuning

- Output: `outputs/instruction_tune_dense_correct_dataset_visible_check`
- W&B: `6rata6go`
- Initialization: dense unpaired checkpoint `92g8ltfg`
- Status: complete and valid
- Best paired full-mask selection loss: 4.0155 at epoch 8
- Epoch-8 full-mask shuffle gaps: 0.2132 text->image and 0.2294 image->text

### Original Tri-LoRA instruction tuning

- Output: `outputs/instruction_tune_lora_correct_dataset_visible_check`
- W&B: `9h2wakez`
- Initialization: original Tri-LoRA unpaired checkpoint `c386f2nw`
- Status: invalid because of paired token routing
- Best recorded selection loss: 4.0648 at epoch 9

### Original adversarial-initialized Tri-LoRA instruction tuning

- Output: `outputs/instruction_tune_lora_adversarial_correct_dataset_visible_check`
- W&B: `m3v84zgh`
- Initialization: defective adversarial checkpoint `dgyzre9n`
- Status: invalid for two reasons
- Best recorded selection loss: 4.0677 at epoch 9

This run inherited the adversarial-recorder defect and also used the incorrect
paired routing described below.

## 7. Instruction-tuning routing defect and corrected rerun

### The defect

Tri-LoRA originally accepted one route ID per batch row. During paired
instruction tuning, the route was selected from the prediction objective:

- predicting text selected the text-private branch for the entire sequence;
- predicting images selected the image-private branch for the entire sequence.

As a result, when predicting text, untouched image-context tokens incorrectly
passed through text-private LoRA. When predicting images, untouched text-context
tokens incorrectly passed through image-private LoRA.

This was especially harmful because each private branch had only processed its
own modality during unpaired pretraining. The paired stage forced context tokens
through a private branch that had never seen that token modality.

### The fix

Routing is now token-level:

- every text token, including control tokens, uses text-private LoRA;
- every image token, including control tokens, uses image-private LoRA;
- padding uses neither private branch;
- mixed paired rows use both private branches simultaneously.

The same routing is used in training, paired validation, shuffled/null controls,
and generation. A regression test proves that a paired row gives gradients to
both private branches. The complete suite currently passes 12 tests.

### Corrected Tri-LoRA instruction tuning

- Output: `outputs/instruction_tune_lora_token_routed_correct_dataset_visible_check`
- W&B: `a6nn0atl`
- Status: complete and valid for the original frozen-random-base pretraining
- Best paired full-mask selection loss: 4.0185 at epoch 8
- Epoch-8 full-mask shuffle gaps: 0.2197 text->image and 0.2190 image->text
- Epoch-9 full-mask shuffle gaps: 0.2175 text->image and 0.2404 image->text

This rerun materially changed the conclusion. The best selection losses are:

| Instruction model | Best selection loss | Difference from dense |
|---|---:|---:|
| Dense | 4.0155 | -- |
| Correct token-routed Tri-LoRA | 4.0185 | +0.0030 |
| Incorrect row-routed Tri-LoRA | 4.0648 | +0.0493 |

The corrected Tri-LoRA result is almost tied with dense and its bidirectional
shuffle gaps are also approximately equal to dense. Most of the earlier
downstream gap was therefore caused by the instruction-routing defect, not by an
inherent inability of Tri-LoRA to instruction-tune.

The two incorrect W&B instruction runs are tagged `invalid`,
`wrong_token_routing`, and `exclude_from_comparison`. Their output directories
also contain `INVALID_TOKEN_ROUTING.md` markers.

## 8. Corrected adversarial activation aggregation

`SharedActivationRecorder` now retains every shared LoRA activation. Each is
token-pooled, adaptively reduced to `d_model` width when required, and the
representations are averaged. DANN therefore supplies gradients to every
intended shared adapter rather than only the final MLP projection.

### Gradient-fixed adversarial run with retained random base

- Output: `outputs/multimodal_unpaired_lora_adversarial_correct_dataset_visible_check_gradient_fixed`
- W&B: `74zdxt32`
- Status at report time: active, epoch 29
- Best marginal validation loss so far: 1.9188 at epoch 28

Purpose: isolate the recorder fix while retaining the rest of the original
Tri-LoRA formulation. This is a valid test of corrected DANN for the
frozen-random-base architecture, but it is not the final no-base staged design.

Its current paired shuffle gaps remain near zero, as expected for strictly
unpaired training. A low matched loss by itself must not be interpreted as
alignment.

## 9. No-base and shared-stage experiments

The next hypothesis was that the frozen random base itself was undesirable.
These runs delete the target linear base weights so that shared/private
low-rank branches form the complete transformation.

They also use a 10-epoch shared-only stage 0. Private branch tensors exist but
are frozen during epochs 0-9. This initially forces both modalities to establish
a common transformation before private specialization begins.

### Non-adversarial no-base staged run

- Output: `outputs/multimodal_unpaired_lora_shared_stage0_10e_no_base_correct_dataset_visible_check`
- W&B: `cqjz8vx5`
- Schedule: epochs 0-9 shared-only; epochs 10-49 shared+private
- Status at report time: active, epoch 31
- Best marginal validation loss so far: 1.9643 at epoch 30

Purpose: measure the effect of deleting the random base and staging private
specialization without the additional DANN intervention.

### Cancelled adversary-from-epoch-0 staged ablation

- Output: `outputs/multimodal_unpaired_lora_adversarial_shared_stage0_10e_no_base_correct_dataset_visible_check`
- W&B: `z4mqm6yq`
- Status: cancelled during epoch 11
- Retained role: adversary-from-epoch-0 shared-only ablation

Problem: DANN was active during shared-only epochs 0-9. This mixed two
interventions in stage 0: learning the initial common transformation and
simultaneously forcing discriminator invariance. The intended hypothesis is
cleaner if stage 0 first establishes the shared transformation without DANN,
then starts DANN when private specialization begins.

This run is tagged `cancelled_by_design`, `ablation`, `adversary_from_epoch0`,
`shared_only_stage0_with_adversary`, and `not_mainline_schedule`. It has a local
`CANCELLED_ABLATION.md` explanation and must not be reported as the mainline
staged-adversarial result.

### Correct mainline staged-adversarial run

- Output: `outputs/multimodal_unpaired_lora_adversarial_after_shared_stage0_10e_no_base_correct_dataset_visible_check`
- W&B: `dfzusn3u`
- Status at report time: active, early stage 0
- GPU process: PID 1320908

Schedule:

| Epochs | Shared LoRA | Private LoRA | DANN |
|---|---|---|---|
| 0-9 | train | frozen | disabled |
| 10-49 | train | train | enabled |

DANN starts at optimizer step 7,030. Its 1,000-step warmup is local to that
transition, so the adversarial weight grows from step 7,030 rather than being
fully warmed up before it is enabled. Initial logs contain no adversarial loss,
confirming that stage 0 is non-adversarial.

## 10. Paired-dense batch/step mismatch

### Original paired-dense run

- Output: `outputs/multimodal_dense_correct_dataset_visible_check`
- W&B: `zd2g1lil`
- Status: intentionally stopped during epoch 91 to launch the matched control
- Best marginal validation loss so far: 1.5194 at epoch 90
- Configuration: microbatch 16, accumulation 2, effective batch 32 pairs
- Optimizer updates per epoch: approximately 2,813

This run is valid for its configured optimization regime and has learned strong
paired correspondence. However, it is not a fair primary batch/step-matched
control for the unpaired runs.

The unpaired runs use 128 text targets and 128 image targets per optimizer
update and make 703 updates per epoch. The original paired run uses only 32 of
each per update, so it makes approximately four times as many parameter updates
over the same one-epoch data exposure. It also runs for 100 epochs rather than
50.

It is now tagged as:

- `valid_result`;
- `paired_dense_small_batch_more_updates_control`;
- `not_exposure_matched`;
- `effective_batch_32_pairs`;
- `2813_steps_per_epoch`;
- `exclude_from_batch_matched_primary_comparison`.

It must be retained, not discarded: it measures the benefit of smaller batches
and more updates. Validation curves should be plotted against epoch/data
exposure rather than raw W&B `_step`.

### New exposure-matched paired-dense control

- Config: `configs/multimodal_dense_paired_exposure_matched_correct_dataset_visible_check.yaml`
- Output: `outputs/multimodal_dense_paired_exposure_matched_correct_dataset_visible_check`
- W&B: `pzysz0ur`
- Status at report time: active on the GPU released by the original paired run
- Training PID: 1322448
- Target CUDA ID: 2, the same device released by the original paired run

Matched configuration:

| Quantity | Paired matched control | Unpaired control |
|---|---:|---:|
| Text targets/update | 128 | 128 |
| Image targets/update | 128 | 128 |
| Rows per epoch | 89,984 | 89,984 |
| Optimizer steps/epoch | 703 | 703 |
| Epochs | 50 | 50 |

The paired run uses microbatch 16 and accumulation 8, giving 128 paired rows per
update. `max_train_samples=89984` makes the number of microbatches divisible by
8 and exactly matches the examples retained by the unpaired drop-last loader.

Initially all four GPUs were occupied, so the new run was queued instead of
being co-located on a saturated GPU. The original paired-dense run was then
intentionally stopped during epoch 91 at the user's request. The queue detected
the released GPU and launched the matched run successfully at 16:10 CEST. Its
startup and training stream is also recorded in:

`outputs/multimodal_dense_paired_exposure_matched_correct_dataset_visible_check/queue.log`

## 11. Step-axis interpretation

Raw optimizer step is not a data-exposure-matched axis when effective batch
sizes differ.

| Run type | Effective per-modality batch | Updates/epoch |
|---|---:|---:|
| Original paired dense | 32 | 2,813 |
| Exposure-matched paired dense | 128 | 703 |
| Unpaired pretraining | 128 | 703 |
| Full-pair instruction tuning | 256 | 352 |

For completed validation curves, `epoch` is the appropriate x-axis because each
epoch exposes approximately one full pass of every modality. For new training
runs, `train/epoch_progress` is also logged to support within-epoch normalized
plots.

The original paired dense, non-adversarial staged, and corrected staged-
adversarial W&B runs have summary metadata recording optimizer steps per epoch,
targets per modality per epoch, and `recommended_validation_x_axis=epoch`.

## 12. Current run ledger

| Run | W&B | State | Interpretation |
|---|---|---|---|
| Dense paired, small batch | `zd2g1lil` | Stopped during epoch 91 | Valid small-batch/more-updates control; stopped to free GPU for replacement |
| Dense unpaired | `92g8ltfg` | Complete | Valid dense unpaired baseline |
| Original Tri-LoRA unpaired | `c386f2nw` | Complete | Valid frozen-random-base Tri-LoRA baseline |
| Original adversarial Tri-LoRA | `dgyzre9n` | Complete, invalid | Recorder sent DANN only to final eligible adapter |
| Gradient-fixed adversarial Tri-LoRA | `74zdxt32` | Active | Correct recorder, retained random base |
| No-base shared-stage Tri-LoRA | `cqjz8vx5` | Active | Non-adversarial staged no-base experiment |
| Adversary-from-epoch-0 no-base stage | `z4mqm6yq` | Cancelled ablation | DANN active during shared-only stage 0 |
| DANN-after-stage0 no-base mainline | `dfzusn3u` | Active | Intended staged-adversarial schedule |
| Dense instruction tuning | `6rata6go` | Complete, valid | Dense full-pair instruction reference |
| Old Tri-LoRA instruction tuning | `9h2wakez` | Complete, invalid | Wrong whole-row target-dependent routing |
| Old adversarial-init instruction | `m3v84zgh` | Complete, invalid | Wrong routing plus defective adversarial initialization |
| Correct token-routed instruction | `a6nn0atl` | Complete, valid | Correct mixed-sequence private routing; nearly ties dense |
| Exposure-matched paired dense | `pzysz0ur` | Active | Primary 703-step/50-epoch paired control |

Older outputs without the `correct_dataset_visible_check` naming are development
or superseded runs. They should not be mixed into the primary comparison tables
unless a specific historical ablation is being discussed.

## 13. Other implementation findings and remaining caveats

### Adapter-only checkpoints are incomplete for scratch Tri-LoRA

`best_adapter.pt` contains only LoRA A/B tensors. Scratch Tri-LoRA also trains
token, position, and modality embeddings, LayerNorm parameters, and the output
head. Therefore adapter-only files cannot reproduce these trained models by
themselves. The current downstream runs correctly initialize from full
`best.pt` checkpoints, so completed results were not affected.

### One seed is insufficient for small differences

Most comparisons currently use seed 13 only. Once configurations are finalized,
the key dense/Tri-LoRA comparisons should be repeated with at least three seeds.
A 0.003 selection-loss difference between dense and corrected Tri-LoRA is too
small to interpret confidently from one seed.

### Same nominal seed does not guarantee identical stochastic batches

Different architectures consume different amounts of random state during model
construction. Unless DataLoader and mask corruption use dedicated generators,
the same seed does not guarantee identical shuffled order and masks across
architectures. This is a moderate control issue, not evidence of current data
leakage.

### Checkpoint selection and grounding are different objectives

Instruction checkpoints are selected using average matched full-mask loss.
That measures conditional prediction quality but does not directly maximize
correspondence sensitivity. Future reports should retain both the best matched-
loss checkpoint and the best shuffle-gap checkpoint.

### Pretraining cannot establish instance correspondence by itself

Strictly-unpaired training can learn shared distributional structure, but it
never observes which caption belongs to which image. Near-zero pretraining
shuffle gaps are therefore not automatically a failure. The decisive question
is whether that initialization learns paired grounding faster or with fewer
supervised examples during instruction tuning.

## 14. Experiments still required to answer the original question

After the active pretraining runs finish:

1. instruction-tune each valid initialization with correct token routing;
2. use paired subsets such as 1k, 3k, 10k, 30k, and 90k;
3. keep effective batch, optimizer steps, masks, validation items, and training
   budget matched;
4. include an instruction-from-scratch control for each downstream
   architecture;
5. repeat the decisive settings across multiple seeds;
6. report matched, shuffled, and null losses together;
7. compare sample efficiency using examples seen or epoch, not raw optimizer
   step when batches differ.

Only that learning-curve experiment can establish whether unpaired pretraining,
Tri-LoRA staging, or adversarial alignment reduces the amount of supervised
paired data needed.

## 15. Unpaired back-translation continuation (2026-08-26)

The corrected all-layer DANN checkpoint
`outputs/multimodal_unpaired_lora_adversarial_correct_dataset_visible_check_gradient_fixed/best.pt`
was preserved and used only as an initialization checkpoint for a new stage.
The continuation therefore has fresh optimizer/counters and writes to a
separate output directory.

The new auxiliary stage still uses the strictly deranged unpaired loader. Every
scheduled auxiliary update performs both directions:

1. real text -> fully sampled pseudo-image -> randomly masked real-text
   reconstruction;
2. real image -> fully sampled pseudo-text -> randomly masked real-image
   reconstruction.

The initial full-sampling leg uses shared-only routing on source positions and
shared+target-private routing on generated positions. The cycle leg likewise
disables the pseudo-source private branch and enables the reconstructed target's
private branch. No full denoising chain is used for the reconstruction leg.

In addition to cycle loss, generated-target shared representations are aligned
to stop-gradient real-source shared representations. A short cosine pilot
(`zbpwrx1c`) was stopped because its alignment loss began near `0.0005`: DANN
had already made the pooled shared vectors nearly parallel, so positive-only
cosine alignment was saturated and could not resist collapse. That pilot is
tagged `cosine_alignment_saturated` and `superseded_pilot` in W&B.

The active main run uses in-batch contrastive alignment instead:

- W&B: `ik5yw8jv`
- output: `outputs/multimodal_unpaired_lora_adversarial_backtranslation_contrastive_from_gradient_fixed_correct_dataset_visible_check`
- pseudo-batch: 8 strictly unpaired carriers every 10 microbatches
- pseudo-generation: 12 confidence-order steps, greedy token choice
- confidence threshold: 0.10
- cycle weight: 0.10
- contrastive alignment weight: 0.05, temperature 0.07
- auxiliary warmup: 1,000 optimizer steps
- original marginal denoising and corrected all-layer DANN remain active
- checkpoint selection remains marginal validation loss, so paired validation
  is diagnostic rather than a hidden supervised model-selection signal

At step 10, both directions accepted all eight pseudo-pairs. Contrastive loss
was `2.08 ~= ln(8)` and retrieval top-1 was `0.125 = 1/8`, the exact random
baseline. This confirms that the new alignment objective is initially unsolved
and measurable. It does not yet demonstrate semantic translation; improvement
must be judged by held-out matched-vs-shuffled gaps, rule-sensitive probes, and
qualitative generation without using paired examples for gradients.

One unavoidable current limitation is pseudo-caption length: because EOS was
not trained as a denoising target, image-to-text generation borrows only the
length/EOS slots from the independently sampled text row. All semantic text
tokens are masked, so their content is not leaked, but length is a nuisance
template rather than a generated variable.

## 16. Matched 1,000-pair instruction-tuning study (2026-08-26)

Four instruction-tuning runs were launched to measure supervised sample
efficiency in the limited-pair regime. They use the same train manifest, the
same 1,000 unique paired rows, the same captions, the same masking objective,
and the same validation set. The selected row indices are generated with an
RNG independent of model construction using subset seed `20260826`. Every run
records and uploads `train_subset_indices.json`; the shared index SHA-256 is
`1df563f284b18ea6b41c01bdcb56759c0360be5ce1a7ec2e59984921d39f381a`.

The schedule is batch 50 with five-way gradient accumulation, 100 epochs, and
therefore exactly four optimizer steps per epoch, 400 optimizer steps total,
and 100,000 paired-example presentations. Validation and checkpoint selection
use the same 512 held-out paired rows and average matched full-mask loss in both
directions. Each exact YAML config and the subset-index artifact are uploaded
to W&B under group
`instruction_tuning_1k_same_pairs_data_constraint_seed20260826`.

The four initializations are:

- dense unpaired pretraining: W&B `phjocxse`;
- Tri-LoRA unpaired pretraining without DANN: W&B `d8tisu4t`;
- Tri-LoRA corrected all-layer DANN pretraining: W&B `dw1jgvjf`;
- the corrected-DANN checkpoint after back-translation/contrastive
  continuation: W&B `5cn1cc31`.

All four started successfully on separate GPUs. DANN and back-translation are
initialization labels only: the downstream supervised objective is identical
and neither auxiliary loss is active during instruction tuning. Text and image
tokens retain corrected per-token private-adapter routing in every Tri-LoRA
run.

Important architecture caveat: the three Tri-LoRA checkpoints in this matched
ablation all belong to the older `delete_base_weights: false` lineage, so their
frozen dense base tensors remain present. This is stated explicitly in their
run names as `frozen_base`. The newer no-base/stage-0 lineage is a separate
architecture and is not mixed into this four-way comparison; doing so would
confound the effect of adversarial or translation pretraining with deleting the
base weights.
