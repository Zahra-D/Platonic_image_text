# Multimodal CLEVR discrete diffusion: code and experiment guide

This guide explains the multimodal part of this repository: how captions and
images become one Transformer input, how the masked-diffusion loss works, what
the shared/private Tri-LoRA branches do, and what each alignment experiment is
actually testing.

It is intentionally explicit about a central limitation of the unpaired
setting: learning the two modality **marginals** is not the same as learning
which caption belongs to which image.  That distinction is essential when
interpreting matched, shuffled, and null validation loss.

## 1. The experiment in one picture

```text
caption ----------------------------------> text tokens
                                                    \
                                                     +--> joint Transformer --> vocabulary logits
                                                    /
VQ-VAE image tokens ---------------------> image tokens

Paired training:     the two items in a row correspond.
Unpaired training:   text and image are deliberately deranged; they do not
                     correspond and are denoised in separate model forwards.
```

The model is a masked-token denoiser, not an autoregressive language model.
For any chosen target modality, some or all of its content tokens are replaced
with `<mask>`.  The model predicts the original token at those positions.

The important code entry points are:

| File | Responsibility |
| --- | --- |
| `train_multimodal.py` | Configuration, data loading, training loop, checkpointing, W&B logging. |
| `data/multimodal_dataset.py` | Paired/unpaired data handling and construction of token sequences. |
| `data/text_tokenizer.py` | Small vocabulary learned from the CLEVR captions. |
| `models/multimodal_transformer.py` | Transformer, shared-activation extraction, DANN head and gradient reversal. |
| `models/lora.py` | Standard LoRA, Tri-LoRA, routes, branch masking, and shared-activation recorder. |
| `multimodal_diffusion.py` | Mask corruption, masked cross-entropy, and iterative generation. |
| `alignment_evaluation.py` | Matched/shuffled/null paired validation. |
| `unpaired_backtranslation.py` | Optional pseudo-pair translation, cycle loss, and contrastive/cosine alignment loss. |
| `configs/*.yaml` | Fully specified experiment recipes. |
| `tests/test_multimodal.py` | Unit tests for data routing, loss, DANN, generation, and controls. |

## 2. Tokens and one input row

### Text

`ClevrTextTokenizer` lowercases and splits words/punctuation.  The vocabulary
is learned only from the training captions.  It reserves:

```text
<pad> <mask> <bos> <eos> <unk> <text> <image>
```

A text segment is:

```text
<text> <bos> word_1 ... word_N <eos>
```

Only `word_1 ... word_N` are eligible for masking and prediction.  The marker,
BOS, and EOS give structure but are never reconstruction targets.

### Image

Images are first encoded once with the frozen VQ-VAE.  The default 64 x 96
image becomes a 16 x 24 grid, therefore 384 discrete codebook indices.  The
image codes have an offset equal to the text vocabulary size, so text and image
content occupy non-overlapping vocabulary IDs.

An image segment is:

```text
<image> <bos> code_1 ... code_384 <eos>
```

Again, only the 384 code positions are eligible targets.

### Paired sequence

For a paired item `i`, the collator concatenates the two segments:

```text
<text> <bos> T_i <eos> <image> <bos> I_i <eos>
```

When `model.use_modality_embeddings: true`, every token receives three
embeddings which are summed before the Transformer:

1. token embedding;
2. position embedding (positions restart inside each modality segment);
3. modality embedding (`text`, `image`, or padding).

The modality marker is therefore redundant with the modality embedding.  This
is useful for ordinary modality-aware modelling, but it is an explicit modality
cue.  For the alignment-focused no-base experiments it is disabled with
`model.use_modality_embeddings: false`; then the input is only token embedding
+ position embedding.  The `<text>`/`<image>` markers and non-overlapping image
code vocabulary remain, so the model can still infer modality indirectly.

## 3. Paired versus strictly unpaired data

`ClevrMultimodalDataset` checks that the VQ-token cache and image manifest have
the same order.  In paired mode, row `i` has caption `T_i` and image `I_i`.

In unpaired mode, `BalancedUnpairedDataset` constructs two separately shuffled
index lists and keeps resampling the image list until it is a derangement:

```text
text carrier:  T_i
image carrier: I_j, where j != i
```

At the start of every epoch, the trainer calls `set_epoch(epoch)` and redraws
the carrier.  For the normal dataset size (more than two items), it additionally
rejects a draw that repeats any text-to-image carrier from the immediately
previous epoch.  Thus a caption and an image that co-occur in an artificial
carrier in one epoch will not co-occur again in the next one.  The DataLoader
still shuffles carrier order within the epoch, so their optimizer step also
varies.

The collator returns `text_batch` and `image_batch`, not a joint sequence.  The
trainer performs a text-only forward and an image-only forward independently.
There is no attention between a random caption and a random image and no loss
that treats the deranged carrier as a real pair.

With configured batch size `B`, each unpaired loader batch contains `B/2` text
carriers and `B/2` image carriers.  The two losses are averaged for the
optimizer update.  Thus unpaired pretraining is valid marginal modelling, but
it contains no instance-level correspondence supervision.

## 4. Transformer and masked diffusion objective

`MultimodalMaskedTransformer` is an 8-layer Transformer by default:

```text
embedding sum
  -> [LayerNorm -> self-attention -> residual
      LayerNorm -> MLP            -> residual] x 8
  -> output LayerNorm -> linear vocabulary head
```

The attention is bidirectional among all non-padding tokens.  It is not causal.
In a paired conditional forward, clean context tokens can attend to the masked
target and vice versa.

### Corruption

`corrupt_batch` chooses one masking probability `t` per row:

\[
t \sim U(\epsilon,1), \qquad m_{r,k}\sim\operatorname{Bernoulli}(t_r)
\]

but only at eligible positions in the selected objective.  `objective=image`
masks only image codes; `text` masks only text words; `both` is used as two
separate modality forwards.  If random masking selects zero targets in a row,
the code forces one eligible position to be masked so every row contributes a
loss.

The optional `full_mask_probability` replaces `t` with 1 for a subset of rows.
It is useful when the downstream task includes fully masked generation.

### Loss

For each masked target position, the ground truth is the *original clean token*
from `input_ids`, before corruption.  The main loss is masked token
cross-entropy:

\[
L_{\mathrm{task}} =
\frac{1}{\sum m_{r,k}}
\sum_{r,k}m_{r,k}\;\operatorname{CE}(z_{r,k},x_{r,k}).
\]

By default, each selected token is additionally divided by `t_r`.  This is the
default `1/t`-weighted objective.  Set `diffusion.unweighting: true` to ignore
the `1/t` factor and use a plain equal-weight average over masked tokens.

Older YAMLs may contain `weight_by_t`.  They remain supported for reproducible
old runs (`weight_by_t: false` maps to `unweighting: true`), but new recipes
should use the clearer `unweighting` setting.

## 5. The two training modes: dense or no-base Tri-LoRA

### Dense model

With `train_mode: dense`, every Transformer linear map has one ordinary dense
weight matrix (and bias when configured).  It is shared by all tokens and both
modalities, and is optimized directly.

### No-base Tri-LoRA

With `train_mode: lora`, the multimodal training path uses Tri-LoRA only.  It
does **not** retain a frozen dense base matrix.  The selected attention/MLP
linear modules are replaced by:

\[
y = b_{shared} + \Delta_{shared}(x) + \Delta_{text}(x) + \Delta_{image}(x).
\]

The original dense weight and its frozen base bias are removed.  Each delta is
a low-rank `B @ A` projection.  A new **trainable shared bias** is initialized
like an ordinary linear bias, so text uses `shared_bias + shared + text-private`
and image uses `shared_bias + shared + image-private`.  The selected modules
are `qkv`, `out_proj`, `mlp.0`, and `mlp.3` by default.

where the private deltas are masked by a route ID at every token:

| Route ID | Active branches |
| --- | --- |
| `0`: text | shared bias + shared + text-private |
| `1`: image | shared bias + shared + image-private |
| `2`: image private-only | shared bias + image-private; shared delta suppressed |
| `3`: text private-only | shared bias + text-private; shared delta suppressed |
| `-1`: shared-only | shared bias + shared; neither private branch |

`train_mode: lora` always means this strict no-base Tri-LoRA path.  There is no
separate architecture or `delete_base_weights` switch in the current trainer;
those fields in old YAMLs are ignored legacy metadata, not experimental choices.

The LoRA rank budget is split approximately 2/3 shared and 1/3 private for
each modality.  The three branches are separate parameter tensors; “shared”
means the same adapter parameters are used by both modalities, not that their
activations are forced to be equal.

### What receives gradients?

For normal text rows, shared + text-private adapters receive gradients.
For normal image rows, shared + image-private adapters receive gradients.  The
other private branch is inactive and gets no task gradient.

In no-base Tri-LoRA experiments, shared/private adapters, input/output
embeddings, LayerNorms, the vocabulary head, and (when enabled) the DANN
discriminator are trainable.  There is no frozen attention/MLP weight or bias
under the selected modules.

## 6. Stage 0

`train.shared_only_epochs: N` does exactly this for Tri-LoRA:

```text
epochs 0 ... N-1: shared LoRA trainable; text/private and image/private frozen
epochs N ... end: shared + both private LoRAs trainable
```

Embeddings, LayerNorms, and output head remain trainable in both stages.  Stage
0 does **not** make captions and images corresponding examples; it only forces
the same shared adapter parameters to serve both marginal denoising tasks.

For the normalized Stage-0 DANN configuration, epochs 0--9 run this shared-only
stage without DANN.  At epoch 10, both private branches are enabled and DANN is
enabled.  See
`configs/multimodal_unpaired_lora_adversarial_l2_normalized_after_shared_stage0_10e_no_base_correct_dataset_visible_check.yaml`.

## 7. DANN modality adversary and normalization

The optional DANN head asks, “can a small classifier identify whether a pooled
shared LoRA activation came from a text row or an image row?”

### Shared representation supplied to DANN

While a model forward is running with `return_shared=True`, every injected
Tri-LoRA layer recomputes its shared delta from a **detached input**, pools it
over all non-padding positions, and reduces it to `d_model=384` if necessary.
The model averages these pooled vectors across all injected adapters.  This
creates one 384-D representation per row.

The detached adapter input is important: DANN gradients can update the shared
LoRA A/B tensors, but cannot travel backward through the activations into
earlier layers or private branches.  DANN directly acts on the shared branch.

The pool includes all non-padding sequence positions, including marker/BOS/EOS
tokens.  It is a representation of the full routed row, not a semantic-word or
image-code-only pool.

### DANN loss and gradient reversal

The discriminator is:

```text
Linear(384, hidden) -> GELU -> Linear(hidden, 2)
```

It gets target label `0` for text rows and `1` for image rows.  Its loss is:

\[
L_{DANN}=\operatorname{CE}(D(h_{shared}),m).
\]

The gradient reversal layer is identity in the forward pass but multiplies the
gradient with respect to `h_shared` by `-lambda` in the backward pass.  Thus:

```text
discriminator parameters: minimize L_DANN, become better modality classifiers
shared LoRA parameters:  receive -lambda * dL_DANN/dh, become harder to classify
```

The total per-modality training loss while DANN is active is:

\[
L = L_{task} + w_{DANN}\,p\,L_{DANN},
\]

where `p` linearly warms from 0 to 1 over
`modality_adversarial_warmup_steps` after the configured start epoch.

### Why normalize before DANN?

Without normalization, the discriminator could classify modality by the norm
of the shared representation alone.  The new option:

```yaml
alignment:
  modality_adversarial_representation_normalization: l2
```

feeds this to DANN instead:

\[
\hat h = \frac{h}{\max(\lVert h\rVert_2,10^{-6})}.
\]

The raw representation continues unchanged through the main task.  Only DANN
sees `h_hat`, so this removes radial magnitude as a shortcut without adding a
constraint to the denoising path.  A zero vector remains zero due to the
epsilon; this avoids NaNs but means a collapsed shared representation can still
be diagnosed from the logs.

W&B logs, for each modality, include:

```text
adversarial/text_raw_shared_l2
adversarial/text_dann_input_l2
adversarial/image_raw_shared_l2
adversarial/image_dann_input_l2
```

With L2 normalization, nonzero `*_dann_input_l2` values should be approximately
1.  Raw norms are intentionally left visible as a collapse/scale diagnostic.

The two current no-base normalized recipes are:

| Recipe | DANN timing |
| --- | --- |
| `multimodal_unpaired_lora_adversarial_l2_normalized_no_base_no_stage0_correct_dataset_visible_check.yaml` | Starts in epoch 0. |
| `multimodal_unpaired_lora_adversarial_l2_normalized_after_shared_stage0_10e_no_base_correct_dataset_visible_check.yaml` | Starts at epoch 10 after shared-only stage 0. |

Launch both with:

```bash
scripts/launch_normalized_dann_no_base_pretraining.sh
```

DANN enforces modality indistinguishability, not text--image pair identity.
It can remove modality-specific information, leave permutation ambiguity
untouched, or be satisfied by suppressing the shared branch.  Always evaluate
it with the paired matched/shuffled/null controls below.

## 8. Gradient balancing

`SharedGradientBalancer` optionally records gradient norms from text and image
losses on the shared LoRA tensors only.  It keeps an EMA of each norm and scales
the next shared gradient inversely to its modality’s EMA, clipped to configured
minimum/maximum weights.  Private LoRA gradients are *not* rescaled.

The goal is to prevent one modality’s marginal denoising loss from dominating
updates to the shared branch.  The logged `shared_grad/*` values are diagnostic;
they do not demonstrate cross-modal alignment by themselves.

## 9. Paired fine-tuning and asymmetric condition/target routing

In ordinary paired bidirectional training, the model runs twice on each paired
batch:

```text
image target: clean text context + masked image, normal text/image routes
text target:  clean image context + masked text, normal text/image routes
```

With `lora.asymmetric_condition_target: true`, the mini-batch is split in half:

```text
first half:  image target <- text condition
second half: text target  <- image condition
```

For each direction, condition tokens take their full route (shared + their own
private branch).  Target tokens take the private-only route.  For example,
image-from-text uses:

```text
text condition: shared + text-private
image target:   image-private only
```

The target image does not independently create a shared LoRA delta.  It can
still attend to the text-produced shared representation inside the joint
Transformer.  The opposite direction uses the symmetric text-private-only
target route.  This implements the “condition modality supplies shared;
target modality supplies private” hypothesis in both directions.

## 10. Conditional validation: matched, shuffled, and null

`alignment_evaluation.py` evaluates only true held-out pairs.  It rebuilds a
batch so the **target and ground truth always remain from the original row**.
Only the context is changed.

For text-to-image at a mask ratio `r`:

| Control | Model input | Tokens scored |
| --- | --- | --- |
| Matched | clean `T_i` + masked `I_i` | original image tokens of `I_i` |
| Shuffled | clean `T_{i+1}` + the same masked `I_i` | original image tokens of `I_i` |
| Null | masked `I_i` only | original image tokens of `I_i` |

The shuffled caption uses the next batch row with wraparound, so it is
intentionally wrong.  Image-to-text is exactly symmetric.

For every row and direction, the same sampled target-local mask positions are
reused for the three controls.  Therefore their loss difference tests context,
not a different masking draw.

The reported metrics are:

\[
\text{shuffle gap}=L_{shuffled}-L_{matched}
\]

\[
\text{context gain}=L_{null}-L_{matched}.
\]

Lower loss is better.  Positive values mean the correct paired context helps;
negative values mean the supposedly correct context is not helping relative to
that control.  A high matched loss and high null loss mean the model is weak at
the target marginal reconstruction task.  They do **not** by themselves prove
it is generating mismatched images.  Alignment is evidenced by matched beating
the *same target under shuffled and null context*.

`t1` means every target content token is masked.  `t0.75` masks 75% of target
content tokens.  The validation output also reports target masked-token
accuracy, but loss is usually more sensitive.

There are two validation cadences:

* fast step validation uses a smaller fixed prefix of the validation set;
* epoch validation uses the broader configured validation set and controls
  best-checkpoint selection.

Do not interpret a sawtooth in a reused metric key as pure optimization change
without checking which cadence produced each point.

## 11. Iterative sampling and `val/samples`

`generate_masked_modality` starts by replacing all target positions with
`<mask>`.  Context is kept clean.  For each generation iteration it:

1. runs the Transformer on the currently partially filled sequence;
2. obtains a distribution over target vocabulary IDs at every still-masked
   position;
3. samples each candidate token (or uses argmax when temperature is zero);
4. selects positions to commit;
5. repeats until no target masks remain or the step budget is exhausted.

The number of positions revealed at iteration `s` is approximately
`ceil(remaining / steps_left)`, so all positions are filled by the requested
number of steps.

`reveal_order: confidence` commits the positions whose selected-token
probabilities are highest first.  `reveal_order: random` chooses commit
positions uniformly at random.  `temperature > 0` samples token identities;
`temperature: 0` is greedy argmax.  In the training configs, epoch sample grids
use the `generate` section (currently 50 steps, temperature 1.0, confidence
reveal).  They are examples, not a validation loss.

## 12. Optional unpaired translation / back-translation

This path is in `unpaired_backtranslation.py`.  It is optional and separate
from the ordinary unpaired marginal losses and optional DANN loss.

The input is a strict deranged carrier `(T_i,I_j)` with `i != j`.  It never uses
a real pair as supervision.

For text-to-image translation:

```text
source text T_i:       shared-only route (-1)
target image I_j:      shared + image-private route (1), then fully mask image
generation:            iteratively produce pseudo-image I_hat(T_i)
```

The token choices during generation are under `torch.no_grad()` and are
discrete.  There is no gradient through “which token was sampled.”  Rows whose
mean committed-token confidence is below `minimum_confidence` are ignored for
the subsequent losses.

### Cycle reconstruction

Make pseudo-pair `(T_i,I_hat)` and randomly mask the original source text.
Route the source target as shared + text-private and compute ordinary masked
cross-entropy against the original text tokens `T_i`:

\[
L_{cycle}^{T\to I}=
\operatorname{CE}(\text{masked }T_i\mid I_{hat},T_i, T_i).
\]

Image-to-text uses the symmetric construction.

### Pseudo-pair representation alignment

The original source is encoded shared-only in a no-gradient teacher pass.  The
generated target is encoded shared-only in a differentiable student pass.
After L2 normalization it uses either cosine distance or batch InfoNCE.  The
contrastive version is:

\[
S_{ab}=\frac{\hat h_{target,a}^\top\hat h_{source,b}}{\tau},\qquad
L_{align}=\operatorname{CE}(S,\operatorname{diag}).
\]

Only the student/generated-target side receives an alignment gradient.  The
source teacher is detached.  Within a translation direction the combined loss
is:

\[
L_{BT}=w_{cycle}L_{cycle}+w_{align}L_{align}.
\]

The two directions are averaged and warmed up over the configured number of
optimizer steps before they are backpropagated.  Translation can impose an
additional useful learning signal, but it is still based on model-generated
pseudo-pairs and must be judged by held-out matched/shuffled/null controls.

## 13. Checkpoints, configurations, and reproducibility

Every checkpoint stores the model state, optimizer state, epoch, step,
best-validation score, parsed arguments, tokenizer vocabulary, and the list of
injected LoRA modules.  `best.pt` is selected by the configured broad epoch
validation metric; `last.pt` is the most recent resumable state.

For LoRA runs, an adapter-only diagnostic state may also be saved.  Use the
complete checkpoint for inference: it also contains embeddings, LayerNorms,
the vocabulary head, optimizer state, and the tokenizer vocabulary.

* `--resume CHECKPOINT` restores model, optimizer, epoch, step, and best score.
* `--init-checkpoint CHECKPOINT` copies only compatible model weights and starts
  a fresh optimizer/run.
* The exact source YAML and, when a limited training subset is used, the
  deterministic subset-index JSON are uploaded to W&B.
* `wandb.tags` is a YAML list.  Use it for orthogonal experiment facts such as
  `pretrained`, `unpaired`, `tri-lora`, `no-base`, `stage0`, `dann`, or
  `translated`; use `wandb.group` for the broader experiment family.

Useful checks before comparing experiments:

1. identical tokenizer/VQ-VAE, image-code count, grid size, and Transformer;
2. the same dense-versus-no-base-Tri-LoRA mode;
3. identical data split, caption field, and paired subset;
4. controls measured on the same paired validation rows and mask ratios;
5. distinguish a completed run from an interrupted ablation.

## 14. Common commands

Run the unit tests:

```bash
python -m unittest tests/test_multimodal.py
```

Train with a fully specified recipe:

```bash
python train_multimodal.py --config configs/NAME.yaml
```

Evaluate a checkpoint without training:

```bash
python train_multimodal.py --config configs/NAME.yaml \
  --resume outputs/RUN/best.pt --eval-only
```

Resume a run:

```bash
python train_multimodal.py --config configs/NAME.yaml \
  --resume outputs/RUN/last.pt
```

The default GPU launchers are in `scripts/`.  They deliberately refuse to
overwrite an output directory that already has a log or checkpoint.

## 15. What evidence would support the shared-LoRA hypothesis?

The hypothesis is not “DANN accuracy becomes low.”  The meaningful downstream
claim is that unpaired shared pretraining makes it easier to establish
*instance-level* correspondence with limited paired data.

Strong evidence would be a controlled comparison where, after the same small
paired fine-tuning budget, a shared-LoRA initialization has:

1. lower matched conditional loss;
2. larger positive shuffled gap and positive null-context gain in **both**
   directions;
3. the effect at high masking ratios, especially `t1`, where the condition is
   actually needed;
4. stable results across seeds and no decline in marginal target modelling;
5. better conditional samples, inspected in addition to the loss controls.

Good marginal samples or low unpaired reconstruction loss are useful, but they
only show that the model learned the text and image distributions separately.
They cannot resolve the caption--image permutation ambiguity without a signal
that couples the two modalities.
