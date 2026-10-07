# Pure-JEPA-from-scratch hyperparameter sweep

> **Type:** evaluation and run record, text-only study · **Status:** complete (8 matched pilots, 2026-09-22)  
> **Question:** Can EMA, masking, target depth, and pooled-scene regularization make JEPA learn CLEVR semantics from random initialization with no diffusion loss?  
> **Results:** [`outputs/text_jepa_scratch_pilot_*`](/home/zd25e122/clevr_discrete_diffusion/outputs) and [`outputs/text_jepa_scratch_faithful_*`](/home/zd25e122/clevr_discrete_diffusion/outputs)  
> **Menu:** [experiment catalog](../README.md)

## Answer

Only partly. The sweeps improved the strongest attribute signal, from the
previous scratch model's 16.3% hard-retrieval R@1 at L7 to 18.9% at L6, but
they did not improve relation binding. The strongest relation result in these
pilots was 48.3%; the previous longer scratch run reached 53.4%, and the dense
diffusion baseline reaches 71.5%. Chance is 12.5% for attributes and about
42.9% for relations.

The best balanced pilot was faithful BERT-15% masking with blocks 4--7 as the
teacher target. Its final step-4000 L4 residual scored 16.2% attribute and
46.9% relation R@1. This is a useful pilot improvement, not evidence that the
from-scratch problem is solved.

## Controlled protocol

Every arm used the same random seed, same deterministic 512,000-caption subset,
batch size 64 with four-step accumulation, two epochs (4,000 optimizer steps),
and a randomly initialized dense 8-block Transformer. `diffusion.weight` was
exactly zero and JEPA started at epoch 0. Hard retrieval used 500 generated
worlds at steps 1,000, 2,000, 3,000, and 4,000.

The training log still prints `text_loss` and masked-token accuracy as
diagnostics. They are not optimized: `optimized_loss` is the pure data2vec/JEPA
loss in these runs.

## Sweep 1: EMA and global-scene objective

All arms used 30% masking in 12--24-token windows, averaged teacher blocks
4--7, Smooth-L1 beta 2, and LR 3e-4.

| Arm | Best balanced residual checkpoint | Attribute | Relation | Finding |
|---|---:|---:|---:|---|
| EMA .999 | step 2000, L4 | 14.8% | 47.4% | strongest sweep-1 balance; L6 attribute alone reached 18.9% |
| EMA .99 | step 2000, L7 | 13.8% | 46.2% | faster target tracking did not help |
| EMA .99 + global 1.0 + variance .1 | step 1000, L5 | 13.9% | 45.6% | pooled target became unstable and later hurt relation |
| same, token weight .25 | step 1000, L0 | 13.6% | 46.5% | lowering token pressure did not recover semantics |

The global arms often reached prediction/target cosine near 1 and tiny JEPA
loss while retrieval got worse. A variance floor kept nonzero feature variance,
but arbitrary sample variation is not the same thing as semantic variation.

## Sweep 2: closer to the published text data2vec recipe

The [original data2vec paper](https://proceedings.mlr.press/v162/baevski22a.html)
uses 15% BERT 80/10/10 corruption for NLP, beta 4, EMA 0.999 to 0.9999 over
100,000 updates, peak LR 2e-4, top-K contextual targets, batch size 256, and
one million updates. The second sweep adopted the corruption, beta, EMA ramp,
LR, and batch size, then varied mask structure and target depth.

| Arm | Best balanced residual checkpoint | Attribute | Relation | Finding |
|---|---:|---:|---:|---|
| BERT 15%, blocks 4--7 | step 4000, L4 | 16.2% | 46.9% | best balanced pilot |
| BERT 15%, blocks 0--7 | step 1000, L3 | 14.1% | 48.0% | best relation in sweep 2; peaks very early |
| 15% four-token spans, blocks 0--7 | step 1000, L6 | 13.8% | 47.0% | no advantage over BERT corruption |
| 30% 12--24 windows, blocks 0--7 | step 3000, L3 | 12.5% | 47.0% | worst balance; old corruption is too hard for bootstrap |

Values above select one layer/checkpoint by a chance-normalized mean of the two
retrieval tasks. They are pilot screening values, not final test selection;
the 500-world confidence intervals are broad.

## What was wrong

1. **The previous scratch recipe was much harder than text data2vec.** It hid
   roughly twice as many tokens in very long contiguous spans, used beta 2,
   ramped EMA over only 5,000 updates, and optimized at 3e-4.
2. **EMA .99 is too reactive here.** It rapidly copies the student's arbitrary
   code and makes self-agreement easy without making the code semantic. EMA
   .999 was consistently safer.
3. **Pooled agreement plus a variance floor is not a semantic constraint.** It
   prevents a constant vector but permits high-variance nuisance codes.
4. **JEPA loss is not a checkpoint-selection metric.** Loss and cosine improve
   while hard retrieval oscillates or degrades. Semantic probes must select
   pilot checkpoints.
5. **The implementation is still not paper-faithful in three material ways.**
   It targets post-block residual states rather than the FFN output before the
   final residual connection; it EMA-updates token/position embeddings rather
   than sharing the input encoder between teacher and student; and it uses a
   constant LR rather than the paper's 5% warmup / 80% hold / 15% decay.
6. **Scale is radically different.** These pilots ran 4,000 updates and the
   earlier full CLEVR run about 31,000; published text data2vec used one million.

## Decision

Do not spend a four-GPU full run on the global-pooled variants. The next proper
experiment should first implement the three fidelity gaps above, preserve
probe-time checkpoints, and then run BERT-15% / blocks-4--7 from random init
with the long EMA schedule. Compare it against BERT-15% / all blocks. Keep
diffusion at zero throughout. If relation retrieval still cannot exceed the
existing 53.4% scratch result, the remaining limitation is the self-targeting
objective or dataset/evaluation, not a simple EMA setting.

## Fidelity follow-up: d-prime-selected pilots

Implemented on 2026-09-22:

* clean EMA targets now optionally use each block's FFN output before the final
  residual addition (`data2vec_target_type: ffn_output`);
* clean teacher token/position/modality embeddings can come directly from the
  online input encoder (`data2vec_share_input_encoder: true`), and those unused
  private teacher copies are excluded from EMA updates;
* a checkpointed/resumable data2vec tri-stage LR schedule provides 5% linear
  warmup, 80% hold, and 15% linear decay;
* periodic probes can run `evaluate_semantic_dprime.py` instead of hard
  retrieval and preserve the exact probe-time checkpoint.

Two matched 4,000-update pilots used 1,000-world, 500-bootstrap d-prime probes
every 1,000 steps. The blocks-4--7 arm improved monotonically at L7 from
`d_semantic=-1.611` to `-1.206`; its best binding feature rose from +0.009 to
+0.028. The all-8 arm peaked earlier, ending at L7 `-1.554` with best binding
+0.042. Against the old BERT-15% pilots scored on identical items, the fixes
improved the K=4 L7 residual (`-1.366` to `-1.206`) and best binding (+0.023 to
+0.028), but the old model still had a better single semantic sublayer (`-0.793`
versus `-0.994`). The change is modest and mixed, not a solved result.

The K=4 fidelity arm was promoted to a full 2M-caption/four-epoch run:
[`text_jepa_scratch_fidelity_dprime_bert15_k4_2m_4e.yaml`](/home/zd25e122/clevr_discrete_diffusion/configs/text_jepa_scratch_fidelity_dprime_bert15_k4_2m_4e.yaml).
It uses 2,000-world d-prime probes every 4,000 updates and retains every probe
checkpoint. Hard retrieval is secondary and does not select the model.

## Artifacts

All eight configs are under `configs/text_jepa_scratch_{pilot,faithful}_*.yaml`.
Each output directory contains `last.pt`, full training logs, and JSON probe
results for all four probe times. W&B run IDs are recorded in each launcher
log. Unit tests: 38/38 passed after adding independently weighted token/global/
variance components and the fidelity paths above.
