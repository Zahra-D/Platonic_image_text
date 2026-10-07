# Dense shared route, rank-128 private LoRA

**Status:** complete — all four stage-2 cells trained and scored, with route isolation and a compute-matched control. **This design contains the best text model in the study** (`d_bind` +0.555, trunk-only). See [results](#results), [the full 2×2](#the-full-22-both-trunks-frozen-and-trainable), [route isolation](#route-isolation-the-private-branch-is-empty), and [what survives compute matching](#compute-matched-control).

## Why this design

Every Tri-LoRA model in this catalog splits each adapted linear into a shared
adapter and a private adapter, with no dense base at all. Two findings pushed
us off that parameterization:

1. **LoRA is worse than dense at equal rank.** The dense text model reaches
   validation 1.0758 with 14.87M parameters; the matched Tri-LoRA model reaches
   1.1638 with 25.88M. The two adapter ranks sum to the dense rank, so this is
   not a capacity limit — the factored parameterization simply optimizes worse.
   The realized effective rank of the trained shared delta was only 25–46 out of
   a nominal 256.
2. **The split does not survive summation.** Per module the shared and private
   writes differ, but summed over each branch's writes the private stream binds
   attributes at least as well as the shared one in all ten Tri-LoRA models
   ([decomposed representations](../evaluations/decomposed_representations.md)).

The purpose of LoRA here was never efficiency — it was a way to give the shared
and private routes *different subspaces* without forcing them to be orthogonal.
That goal does not require factoring the shared route too. So: keep the dense
weight as the shared route, at full rank and with dense optimization behavior,
and rank-limit only the private route.

## Stage 1 — faithful data2vec on the dense trunk (`text_data2vec_from_dense_2m_2e`)

Deliberately the *original* data2vec recipe, so that if their idea works, this
one should too:

- initialization: the trained dense diffusion checkpoint (validation 1.0758),
- only loss: Smooth-L1 (beta 2.0) between the student's final hidden state
  through one prediction head and the mean of the **parameter-free
  LayerNorm-ed EMA-teacher hidden states of all 8 blocks**, at masked positions,
- **`diffusion.weight: 0`** — no reconstruction term at all,
- EMA decay 0.999 ramping to 0.9999 over 5000 steps, 2 epochs of 2M captions.

This is JEPA **averaged at the end**, not layerwise: one student (the final
block) against one averaged target. The layerwise variants supervise each
block's own adapter writes against its own teacher target; both modes exist in
the code and the run registry records which each run used.

## Stage 2 — add the private LoRA, HSIC and the diffusion loss

Both conditions start from stage 1's `epoch_001.pt`, keep the dense trunk as the
shared route, and add rank-128 (`d_model/3`) private adapters to `qkv`,
`out_proj`, `mlp.0` and `mlp.3` in every block, with `alpha = 128` so the
private scaling is 1.0, exactly as in the earlier Tri-LoRA runs.

| | Condition A `text_dense_private_frozen_hsic_2m_2e` | Condition B `text_dense_private_trainable_hsic_2m_2e` |
|---|---|---|
| Dense trunk | frozen | trains |
| Private adapters (rank 128) | train | train |
| Diffusion loss | on (weight 1) | on (weight 1) |
| HSIC (dense write vs. private write) | on, target gradient ratio 5% | on, target gradient ratio 5% |
| data2vec | off (a frozen trunk cannot move) | off |
| Trainable parameters | 12.58M of 27.45M | 27.45M |

Condition A is the sharp test: the semantic route cannot move, so every bit of
modality-specific detail the diffusion objective needs must be carried by the
private branch. Condition B measures how much of stage 1's structure survives
once the diffusion loss is allowed to rewrite the trunk.

Keeping data2vec running *alongside* diffusion in stage 2 would need the two
loss paths merged (the data2vec branch currently short-circuits the training
step); it is not part of these two runs.

## Implementation

`lora.train_mode: dense_private` (`models/lora.py`, `train_multimodal.py`):

- `delete_base_weights=False` and `shared_branch=False`, so the dense weight
  stays and no shared adapter is created; the private rank is set by
  `lora.private_rank` and scaled by `alpha / private_rank`.
- The activation recorder reports the **dense write** as the module's shared
  native update, so HSIC, the JEPA losses and every evaluator read the shared
  route without changes.
- `lora.freeze_base` selects condition A or B.
- HSIC no longer requires a modulewise JEPA mode; it can be the only alignment
  term in a run.
- The effective-rank diagnostic reports the dense weight's spectrum for the
  shared route. At initialization from stage 1 it is 96–202 across blocks,
  against 14–68 for the private adapters — a far wider shared subspace than the
  25–46 the trained shared LoRA ever realized.
- A dense checkpoint's `blocks.N.attn.qkv.weight` keys are remapped onto
  `...qkv.base.weight` when initializing an adapter-injected model, and the
  freshly created adapter tensors are permitted to be absent from it.

Unit test: `tests.test_multimodal.MultimodalTest.test_dense_private_keeps_base_as_shared_route`.

## Checkpoints

Each run writes `epoch_000.pt`, `epoch_001.pt`, `best.pt`, `last.pt` and the
matching `*_adapter.pt` into **its own** `outputs/<run_name>/` directory
(`save_every: 1`). Stage 2 reads stage 1's checkpoint through
`train.init_checkpoint` and writes elsewhere, so no stage-1 checkpoint is
touched.

Launch: `scripts/launch_dense_private_stage2_vnode10.sh <frozen|trainable> <gpu>`;
`scripts/start_dense_private_stage2_when_ready.sh` waits for stage 1 and for a
free GPU and then starts both.

## Results

### Reconstruction (masked-token loss at t = 0.75, 2,048 held-out captions)

| Run | Epoch 0 | Epoch 1 | Epoch 3 |
|---|---:|---:|---:|
| Dense, diffusion only (4 epochs) | 1.1852 | 1.1137 | **1.0758** |
| Plain Tri-LoRA (4 epochs) | 1.3941 | 1.2327 | 1.1638 |
| Stage 1: data2vec only, `diffusion.weight 0` | 1.3232 | 1.4488 | — |
| Stage 2A: frozen trunk + rank-128 private LoRA + HSIC | **1.0433** | 1.0677 | — |
| Stage 2B: trainable trunk + private LoRA + HSIC | 1.0480 | 1.0648 | — |

Stage 1 trained cleanly — JEPA loss 0.133 → 0.015, prediction/target cosine
0.418 → 0.964, `target_position_spread` steady at 0.032–0.040, so no collapse
— and with no reconstruction term its diffusion loss rose away from the dense
optimum, as expected. Adding only the rank-128 private adapters on top of that
drifted trunk recovered it to 1.0433 **with the trunk frozen** (12.58M of
27.45M parameters trainable). Both stage-2 conditions then got worse in epoch 1
under the constant 3e-4 learning rate, so `best.pt` is epoch 0 in both. Total
compute is not matched against the dense baseline: stage 2 sits on top of the
dense 4 epochs plus stage 1's 2.

HSIC behaved: its adaptive weight settled at 1.7–2.2, well under the cap of 10,
so the 5% gradient-ratio target was actually met, and the shared/private
dependence fell from 0.0029 to 0.00085.

### Does the data2vec phase improve semantics? No.

Both tests use the protocols and candidate sets the dense baseline used
(hard retrieval: 2,000 worlds, all sublayers, 1,162 attribute and 1,588
relation items — identical counts confirm identical data).

[Hard retrieval](../evaluations/hard_retrieval.md), R@1 %, residual stream
(chance 12.5 attribute / 43.0 relation):

| Block | ATTR dense | ATTR stage 1 | ATTR untrained | REL dense | REL stage 1 | REL untrained |
|---|---:|---:|---:|---:|---:|---:|
| L4 | 14.3 | 13.6 | 11.3 | 48.6 | 47.6 | 44.9 |
| L5 | 15.7 | 16.7 | 11.7 | 53.0 | 49.6 | 44.5 |
| L6 | 20.9 | 18.1 | 12.0 | 67.5 | 61.7 | 43.5 |
| L7 | **22.8** | 17.4 | 11.2 | **71.5** | 60.0 | 44.5 |

[Binding swap](../evaluations/binding_swap_evaluation.md), pair ranking
accuracy %, where the word-level control scores exactly 50.0:

| Feature | ATTR dense | ATTR plain LoRA | ATTR stage 1 (e0 / e1) | REL dense | REL stage 1 (e1) |
|---|---:|---:|---:|---:|---:|
| block4 | 72.9 | 57.9 | 65.6 / 66.4 | 96.9 | 98.3 |
| block5 | 85.9 | 72.9 | 75.9 / 76.7 | 98.8 | 95.6 |
| block6 | 91.5 | 75.2 | 77.9 / 79.3 | 99.0 | 92.7 |
| block7 | **91.1** | 76.0 | 75.8 / 76.2 | 48.2 | **69.7** |
| mlp4 | 91.5 | 58.5 | 85.8 / 87.6 | 100.0 | 98.8 |
| mlp6 | 95.8 | 70.0 | 94.1 / 94.5 | 95.6 | 100.0 |

The data2vec phase **subtracts** binding information rather than adding it. On
attributes it costs 6–15 points in the upper blocks and lands roughly where
plain Tri-LoRA sits (block7 76.2 vs. 76.0), while the mid-network MLP writes
survive almost intact (mlp6 94.5 vs. 95.8). Hard retrieval shows the same
shape: still well above the untrained control, but below dense everywhere
except L5 attributes. The one gain is relation binding in the last block,
48.2 → 69.7, where dense is unusually weak.

Both stage-1 epochs agree (attributes 75.8 → 76.2 at block7), so the loss
happens in the first epoch and then plateaus. The reading is that regressing
the mean of all blocks' LayerNorm-ed teacher states smooths the trunk toward a
layer-averaged code, which keeps global scene structure but discards the
upper-block detail that separates lexically identical binding swaps — the same
direction as its reconstruction loss rising to 1.4488.

## Stage-1 variants: averaged vs. layerwise targets

Four stage-1 variants were run from the same dense checkpoint, JEPA only, 2
epochs, to separate *which blocks* are supervised from *whether the target
mixes depth*. Protocols and candidate sets are the dense baseline's.

| Variant | Target | Blocks | Val ep0 / ep1 |
|---|---|---|---:|
| original | average | all 8 | 1.3232 / 1.4488 |
| avg 4-7 | average | 4-7 | 1.3625 / 1.5243 |
| layerwise 4-7 | each block's own | 4-7 | 1.6409 / 1.9885 |
| layerwise all | each block's own | all 8 | 1.6065 / running |

Hard retrieval, R@1 % on the residual stream (chance 12.5 attribute, 43.0
relation):

| Block | dense | avg all-8 | avg 4-7 | layerwise 4-7 | untrained |
|---|---:|---:|---:|---:|---:|
| ATTR L4 | 14.3 | 13.6 | 13.5 | **15.6** | 11.3 |
| ATTR L5 | 15.7 | 16.7 | 15.7 | **17.2** | 11.7 |
| ATTR L6 | **20.9** | 18.1 | 17.1 | 19.1 | 12.0 |
| ATTR L7 | **22.8** | 17.4 | 17.3 | 19.5 | 11.2 |
| REL L5 | 53.0 | 49.6 | 47.9 | **54.7** | 44.5 |
| REL L7 | **71.5** | 60.0 | 57.7 | 63.5 | 44.5 |

Binding swap, pair-ranking accuracy % (the word-level control scores exactly
50.0):

| Feature | dense | avg all-8 | avg 4-7 | layerwise 4-7 | plain LoRA |
|---|---:|---:|---:|---:|---:|
| ATTR block4 | **72.9** | 66.4 | 66.9 | 72.6 | 57.9 |
| ATTR block6 | **91.5** | 79.3 | 79.3 | 84.5 | 75.2 |
| ATTR block7 | **91.1** | 76.2 | 75.3 | 85.8 | 76.0 |
| ATTR mlp7 | 88.0 | 88.0 | 89.2 | **97.7** | 61.3 |
| ATTR attn7 | **82.1** | 70.8 | 69.0 | 80.3 | 62.9 |
| REL block7 | 48.2 | 69.7 | **81.1** | 78.0 | 89.1 |

Three conclusions:

1. **The layer window is not what mattered.** `avg 4-7` sits within a point of
   `avg all-8` everywhere (block7 attributes 75.3 vs. 76.2; hard retrieval L7
   17.3 vs. 17.4). The earlier hypothesis in this document -- that averaging the
   shallow lexical blocks into the target caused the upper-block binding loss --
   is **wrong**. Averaging itself is the problem, not which blocks are averaged.
2. **Layerwise targets recover most of the loss and exceed dense in places.**
   Attribute binding at block 7 rises 76.2 -> 85.8 against dense's 91.1, `mlp7`
   reaches 97.7 against dense's 88.0, and hard retrieval is above dense at L4
   and L5 for both attributes and relations. Only the top two blocks' hard
   retrieval remains below dense.
3. **Reconstruction loss and binding are dissociated.** Layerwise drifts the
   diffusion loss much further from the optimum (1.9885 vs. 1.5243) while
   preserving binding much better, so validation loss is not a proxy for how
   much scene structure survived a JEPA phase.

## From-scratch runs: window masking and the averaged/layerwise reversal

Four runs with **no pretrained initialization**, JEPA only
(`diffusion.weight: 0`), blocks 4-7, 4 epochs of 2M captions, differing in the
target mode and in the masking. Masking selects contiguous **windows** rather
than scattered tokens: each window's size is drawn uniformly from the stated
range and enough windows are placed to reach the target fraction, so a whole
phrase is hidden and the gap cannot be closed from neighbouring words.

Hard retrieval R@1 % (chance 12.5 attribute / 43.0 relation):

| Block | w12-24 avg | w12-24 layerwise | w4-8 avg | w4-8 layerwise | dense | untrained |
|---|---:|---:|---:|---:|---:|---:|
| ATTR L6 | 17.1 | 11.0 | 16.2 | 12.9 | **20.9** | 12.0 |
| ATTR L7 | 16.3 | 12.3 | 14.9 | 13.4 | **22.8** | 11.2 |
| REL L6 | 55.5 | 48.4 | 64.4 | 45.6 | **67.5** | 43.5 |
| REL L7 | 53.4 | 47.2 | 56.2 | 46.6 | **71.5** | 44.5 |

Binding swap %, with the probe's clean held-out accuracy after the slash:

| Feature | w12-24 avg | w12-24 layerwise | w4-8 avg | w4-8 layerwise | dense |
|---|---|---|---|---|---|
| ATTR block4 | **79.5** / 86.1 | 70.2 / 83.5 | 72.4 / 85.6 | 65.4 / 82.7 | 72.9 / 85.0 |
| ATTR block6 | 86.6 / 87.0 | 58.7 / 80.8 | 73.6 / 85.2 | 57.5 / 79.3 | **91.5** / 89.3 |
| ATTR block7 | 78.7 / 85.4 | 55.7 / 80.0 | 68.8 / 83.8 | 55.5 / 77.9 | **91.1** / 89.7 |
| ATTR mlp7 | **90.7** / 86.0 | 73.5 / 81.5 | 82.2 / 84.0 | 70.4 / 79.2 | 88.0 / 83.7 |
| REL block7 | **95.2** / 85.4 | 91.0 / 80.0 | 82.1 / 83.8 | 80.4 / 77.9 | 48.2 / 89.7 |

1. **Layerwise fails from scratch, at both window sizes.** Its upper blocks sit
   at chance on hard retrieval (11.0 and 12.9 at L6) and its attribute binding
   *falls* with depth, to five points above the 50% floor. The tell is that the
   probe's clean accuracy also falls with depth (82.7 -> 77.9), while it rises
   in every averaged and every pretrained model: depth is losing information,
   not merely failing to add binding. Training diagnostics look healthy
   throughout (cosine 0.89, target spread 0.039), so this is not the collapse
   signature.
2. **Averaged with large windows is the best from-scratch model** and beats
   dense at block 4 (79.5 vs. 72.9) and `mlp7` (90.7 vs. 88.0), trailing at
   blocks 6-7. Window size matters within the averaged mode: 86.6 vs. 73.6 at
   block 6.
3. **The reversal is the finding.** From a pretrained trunk layerwise was
   clearly better than averaged (85.8 vs. 76.2 at block 7); from scratch it is
   the opposite and layerwise collapses. The better target mode is therefore
   not a property of the objective but of what the trunk already contains. A
   per-block target may refine an existing hierarchy while being unable to
   create one -- consistent with both halves, though not established by them.
   The gradient distribution is a plausible mechanism: in layerwise mode block 0
   receives gradient from all four losses and block 7 only from its own
   (measured norms 1.64e-3 down to 1.06e-3), whereas averaged mode routes a
   single top-level loss through the whole stack.

All four runs are ordinary full backpropagation, everything trainable
(`21,831,303 / 21,831,303`); the gated gradient routing belongs to the LoRA
modulewise objective and is not used here. With `diffusion.weight: 0` nothing
trains the output head, so these runs' validation loss is meaningless and
`best.pt` is arbitrary.

## Image runs with 2D block masking

The image analogue, from scratch on the 1.2M-image cache, JEPA only, blocks
4-7: the 384 eligible tokens are the 16x24 VQ grid and each mask is a
rectangle covering 10-25% of the grid with aspect ratio in [0.75, 1.5], enough
rectangles to hide ~30% (realized 30.4%). A VQ grid is locally redundant, so
scattered codes are recoverable from their neighbours and a contiguous region
is not. Runs: `image_data2vec_scratch_block2d_30pct_{avg,layerwise}_l4to7_1_2m_4e`.

Reference points from the image diffusion baselines, which reproduce the text
ordering exactly: dense 3.6340 vs. Tri-LoRA 3.9252 at epoch 3 (text: 1.0758
vs. 1.1638). Note that **no image-side probe exists yet** -- hard retrieval and
binding swap both construct caption minimal pairs -- so these two runs cannot
be scored until one is built.

## Compute-matched control

The stage-2 numbers were first compared against the four-epoch dense baseline,
which is not a fair comparison: stage 2 has seen **eight** epochs of data (four
as the dense baseline, two of stage-1 data2vec, two of stage 2). Continuing the
dense baseline to eight epochs (`text_dense_diffusion_2m_8e_continued`, resumed
with optimizer state, constant learning rate, no schedule to restart):

| Run | Total epochs | Val loss |
|---|---:|---:|
| Dense baseline | 4 | 1.0758 |
| Dense continued | 6 | 1.0520 |
| Dense continued | 7 | 1.0555 |
| **Dense continued** | **8** | **1.0475** |
| stage 2 frozen, averaged trunk | 7 | **1.0433** |
| stage 2 frozen, averaged trunk | 8 | 1.0677 |
| stage 2 frozen, layerwise trunk | 7 | 1.0440 |
| stage 2 frozen, layerwise trunk | 8 | 1.0617 |

The gap falls from 0.032 to **0.004**. The earlier claim in this document, that
training only a rank-128 private branch beats dense, does not survive the
control and is withdrawn. What survives: a frozen trunk plus a rank-128 private
branch **matches** full dense training while updating 12.58M of 27.45M
parameters, reaching its best value one epoch earlier. The shared/private
separation results are unaffected, since they compare routes *within* a model.

Both stage-2 runs degrade in their second epoch (1.0433 → 1.0677), as does dense
at epoch 7 before recovering, which is consistent with the constant 3e-4 learning
rate; `best.pt` is epoch 0 for both.

## The full 2×2: both trunks, frozen and trainable

The grid is two stage-1 trunks (averaged targets vs layerwise targets) × two
stage-2 conditions (trunk frozen vs trunk trains). All four cells are trained.
Every row is the **final** epoch, 2 epochs of stage 2 on top of a 1.10B-token
parent, for 1.47B total.

| Stage-1 trunk | Trunk | Val e0 | Val e1 | `d_sem` (L7) | best `d_bind` (full) | best `d_bind` (trunk only) |
|---|---|---:|---:|---:|---|---|
| averaged, all 8 | frozen | **1.0433** | 1.0677 | −0.51 | `L6.mlp_shared` +0.347 | `L6.mlp_out` +0.293 |
| averaged, all 8 | trains | 1.0480 | 1.0648 | −0.24 | `L6.mlp_shared` +0.396 | `L6.mlp_out` +0.433 |
| **layerwise, all 8** | frozen | 1.0440 | 1.0617 | +0.10 | `L5.mlp_shared` +0.529 | `L5.mlp_out` +0.527 |
| **layerwise, all 8** | **trains** | 1.0466 | **1.0553** | *(pending)* | *(pending)* | `L5.mlp_out` **+0.555** |

Effect sizes are the **final** epoch (e1). Note that **the second epoch of
stage 2 makes every cell worse on validation loss** (1.0433 → 1.0677 and so on)
and worse on the full model's binding in two cells of three measured (frozen
averaged +0.356 → +0.347, trainable averaged +0.426 → +0.396; frozen layerwise
is the exception, +0.508 → +0.529). Stage 2 converges inside one epoch and the
second epoch is at best neutral, so anything comparing these runs must say which
epoch it means.

Three things fall out of the grid, and two of them correct earlier readings of
this page.

**1. The layerwise trunk dominates the averaged trunk in every cell**, on both
validation loss and both effect sizes. The trunk's semantic advantage survives
into the split rather than being washed out by stage 2. Reconstruction is
essentially unaffected by which trunk it started from (1.0617 vs 1.0677),
although the layerwise trunk's *own* reconstruction before stage 2 was far worse
(1.9656 vs 1.4488) — the private branch absorbed the whole difference.

**2. On binding, letting the trunk train beats freezing it — correcting an
earlier claim on this page.** Matched on parent: averaged-trunk frozen +0.347
full / +0.293 trunk-only against trainable +0.396 / +0.433; layerwise-trunk
frozen +0.527 trunk-only against trainable **+0.555**. The earlier "freezing is
better" reading compared a frozen *layerwise*-parent run against a trainable
*averaged*-parent run, so it was measuring the parent, not the freeze.

**Validation loss does not settle it and changes sides.** At epoch 0 frozen wins
both pairs (1.0433 < 1.0480, 1.0440 < 1.0466); at epoch 1 trainable wins both
(1.0648 < 1.0677, 1.0553 < 1.0617). Binding is consistent across epochs and
validation loss is not, which is why the conclusion is stated on binding.

**3. `text_dense_private_trainable_hsic_from_d2v_lw_all_2m_2e`, read trunk-only,
is the best text model measured anywhere in this study**: `d_bind` **+0.555** at
`L5.mlp_out` and `d_sem` +0.63, against budget-matched dense's +0.443 and the
layerwise JEPA trunk's own +0.527. It is *better with the private route switched
off*.

## Route isolation: the private branch is empty

Every stage-2 model was re-scored twice more, with one route suppressed at every
layer by route id — an actual forward pass, not a decomposition, so later blocks
also receive a branch-free input.

| Model | full | trunk only | private only |
|---|---:|---:|---:|
| frozen, averaged trunk | +0.347 | +0.293 | **+0.003** |
| trainable, averaged trunk | +0.396 | +0.433 | **+0.014** |
| frozen, layerwise trunk | +0.529 | +0.527 | **+0.024** |
| trainable, layerwise trunk | *(pending)* | **+0.555** | *(pending)* |
| *untrained model (floor)* | *+0.004* | | |
| *plain Tri-LoRA, for contrast* | *+0.293* | *+0.019* | *+0.009* |

Best `d_binding` over all features, so the columns are directly comparable with
the [d′ tables](../evaluations/semantic_dprime.md).

**Private-only is one to six times the untrained floor and one to two orders of
magnitude below the intact model.** The largest private-only reading, +0.024, is
1/22 of the same model's +0.529. The private LoRA holds essentially no binding
information that survives the trunk being removed, while trunk-only *matches or
exceeds* the full model in three of the four cells. The shared route is carrying
the representation; the private branch is, at best, a decoder-side adapter that
the trunk routes around.

**The isolation is validated, not assumed.** The frozen-trunk run read
trunk-only reproduces its stage-1 parent to three decimals — `d_semantic` −0.763
against the parent's −0.76, `d_binding` +0.293 against +0.293. The ablation
recovers the parent's computation exactly.

This also explains why the same test is *not* informative for plain Tri-LoRA,
where each adapted linear is `shared_delta + private_delta` with no base weight:
there, trunk-only scores +0.019 and private-only +0.009 against +0.293 intact.
Removing either branch does not isolate a route, it breaks the layer.

## Is the shared advantage just "private trained less"?

A fair objection: in the frozen condition the trunk arrives with six epochs of
training while the private branch gets two, starts from a zero-initialized `B`,
and has rank 128 against the trunk's full 384. The frozen condition tests it
directly, because the shared route is *identical* between epoch 0 and epoch 1,
so any change comes from further private training:

| Feature | d_bind e0 → e1 | d_sem e0 → e1 |
|---|---|---|
| L7 accumulated shared | 0.084 → 0.065 | −0.22 → −0.36 |
| L7 accumulated **private** | 0.063 → **0.035** | −0.86 → −0.89 |
| `L6.mlp_shared` | 0.356 → 0.347 | −0.01 → −0.02 |
| `L6.mlp_private` | 0.109 → **0.095** | −0.86 → −0.91 |

With more training the private branch's binding content **falls** rather than
catching up, and the gap widens from 0.021 to 0.030; the trainable condition
shows the same, more sharply (`L6.mlp_private` 0.296 → 0.129). So the amount of
training does not explain the separation.

One alternative remains open and predicts the same data: **HSIC** explicitly
penalizes dependence between the shared and private writes, and its weight ramps
from 0 to about 2 over exactly this period (the measured dependence falls
0.0029 → 0.00085). Whether the private branch is specializing or being pushed out
needs a stage-2 run with HSIC disabled, which has not been done. Two further
confounds are untested: capacity (rank 128 against full rank) and the trunk's
head start (joint training from scratch in this layout).
