# Grounded global JEPA from scratch

> **Type:** experiment plan, text-only study · **Status:** cancelled during epoch 0 at step 2,280; no checkpoint (the user required pure JEPA with no diffusion bootstrap)  
> **Config:** [`text_data2vec_scratch_grounded_global_window12_24_30pct_avg_l4to7_2m_4e.yaml`](/home/zd25e122/clevr_discrete_diffusion/configs/text_data2vec_scratch_grounded_global_window12_24_30pct_avg_l4to7_2m_4e.yaml)  
> **Menu:** [experiment catalog](../README.md)

> This plan is retained as a record, but its premise was rejected: starting
> with diffusion would bias the representation before JEPA begins. It was
> replaced by the completed [pure-JEPA scratch sweep](../evaluations/pure_jepa_scratch_hyperparameter_sweep.md).

## Diagnosis being tested

The successful layerwise data2vec models start from a diffusion-trained dense
trunk. The failed scratch models set `diffusion.weight: 0`, so a random student
is trained only against a moving random teacher. Layerwise same-depth targets
then preserve a hierarchy that does not yet exist. The strongest scratch result
instead used an averaged top-level target and 12--24-token windows, but it still
had no grounding or generation objective.

This run asks whether JEPA can work without a pretrained checkpoint when the
semantic hierarchy is bootstrapped inside the same training run.

## Recipe

* Dense model initialized randomly; no checkpoint is loaded.
* Epoch 0: structured masked-token denoising only. The EMA teacher follows the
  student during this epoch but supplies no loss.
* Epochs 1--3: denoising stays active and two JEPA terms are added:
  1. the existing masked-position data2vec loss, predicting the mean clean EMA
     target of blocks 4--7 from the final student block;
  2. a new global loss predicting the clean content-token-pooled target from
     the corrupted content-token-pooled student state.
* A small VICReg-style variance floor acts on the pooled pre-predictor student
  state, so a nearly constant scene code cannot minimize the EMA objective.
* Text corruption uses 12--24-token contiguous windows covering about 30% of a
  caption. BERT 80/10/10 replacement remains disabled so the hidden phrase is
  never leaked through unchanged positions.
* EMA starts at 0.99 and ramps to 0.9999. This lets the target track rapid early
  learning, then stabilize after the representation is grounded.

The new loss is disabled by default (`data2vec_global_weight: 0`,
`data2vec_variance_weight: 0`), so every existing configuration remains
reproducible.

## Why the global term is necessary

The tokenwise loss can be minimized with lexical and positional features. The
global term presents a different problem: from a caption with an entire phrase
hidden, produce the pooled code of the complete clean scene. It therefore
rewards information shared across visible objects and relations rather than a
local replacement at one position.

This is still not cross-modal alignment. It is a test of whether a semantic
within-modality JEPA representation can be learned from random initialization.

## Primary comparisons and success criteria

Compare the final checkpoint with:

* scratch averaged, windows 12--24: `d_semantic` L7 +0.42, best `d_binding`
  +0.198, hard retrieval L7 16.3 / 53.4 (attribute / relation);
* dense diffusion: `d_semantic` L7 +0.38, best `d_binding` +0.244, hard
  retrieval L7 22.8 / 71.5;
* scratch layerwise controls, which are close to the surface/untrained floor.

The experiment succeeds if it simultaneously:

1. exceeds the prior scratch-averaged model on L7 `d_semantic` and best
   `d_binding`;
2. exceeds it on both hard-retrieval tasks, with bootstrap intervals reported;
3. retains useful masked-token validation and generation, which the
   `diffusion.weight: 0` scratch models cannot provide; and
4. keeps pooled effective rank and variance away from collapse.

## Run

```bash
scripts/launch_data2vec_stage1_variant_vnode10.sh \
  text_data2vec_scratch_grounded_global_window12_24_30pct_avg_l4to7_2m_4e 0
```

After training, run `evaluate_semantic_dprime.py`, `evaluate_hard_retrieval.py`,
the binding-swap evaluation, and ordinary validation/generation. Do not select
the checkpoint from a semantic test set; retain the fixed validation selection
rule in the config.

## Verification completed

* Python compilation passes.
* All 33 tests in `tests.test_multimodal` pass, including two new tests for the
  global/variance objective and its pooling-mask validation.
* A two-epoch 64-caption end-to-end smoke run completed epoch 0 denoising,
  activated JEPA in epoch 1, validated, and wrote checkpoints.
* The full run initialized correctly on 2026-09-22, allocated GPU 0, and began
  epoch 0 at about 1,480 captions/s after warm-up.
