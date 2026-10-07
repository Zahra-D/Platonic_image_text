# Gated-predictor layerwise JEPA

> **Type:** experiment plan, text-only study · **Status:** complete; both runs finished four epochs and are fully evaluated  
> **Menu:** [experiment catalog](../README.md) · **Run registry:** [text-study runs](../models/text_study_run_registry.md)

## Why this run exists

The [gated data2vec runs](data2vec_gated_shared_lora_plan.md) changed three
things at once: they removed the JEPA predictor, replaced the direction-only
loss with Smooth-L1 on raw updates, and routed the JEPA gradient end-to-end
through the shared LoRA of blocks 2–4. Both
[collapsed](../evaluations/data2vec_gated_collapse_diagnosis.md). This run keeps
only the gradient routing and restores everything else, so it tests end-to-end
gating on its own.

## Design

Each run is identical to one completed layerwise cell of the
[four-run sweep](modulewise_jepa_hsic_four_run_plan.md) except for one setting.

| | Layer-local baseline | Gated predictor (this run) |
|---|---|---|
| JEPA loss | `modulewise_shared_jepa_loss`, `normalized_mse` | same |
| Predictor | one MLP per (module, layer) | same |
| Target | same-layer clean EMA teacher update (`layerwise`) | same |
| Supervised modules | `out_proj`, `mlp.3` at blocks 2, 3, 4 | same |
| JEPA coefficient | fixed 0.5, 1,000-step warm-up | same |
| **Gradient reach** | the supervised layer's own `out_proj`/`mlp.3` shared A/B (adapter input detached) | **end-to-end** into shared A/B of `qkv`, `out_proj`, `mlp.0`, `mlp.3` at blocks 2–4, plus the predictors |

Private LoRAs, embeddings, LayerNorms, and blocks 0–1 and 5–7 receive no JEPA
gradient in either design.

**Implementation.** New option `alignment.modulewise_jepa_gated_predictor: true`
in [train_multimodal.py](/home/zd25e122/clevr_discrete_diffusion/train_multimodal.py).
The student native update keeps its real forward graph, the JEPA gradient is
taken with `torch.autograd.grad` over the gated parameters and added manually,
and the predictor parameters are included in that set (the gated term never
reaches `backward()`, so predictors would otherwise stay at initialization).
Before launch, a unit check confirmed that the gated path gives non-zero
gradient to all four shared modules at every block 2–4 and to all 24
predictor tensors, while the layer-local path reaches only `out_proj` and
`mlp.3`.

**New diagnostic.** Every 250 optimizer steps the trainer logs the entropy
effective rank of each supervised shared and private LoRA map
(`delta_effective_rank/...` in W&B), so collapse is visible early.

## Runs

| Run | Compare with | Config | W&B | Output |
|---|---|---|---|---|
| Gated, no HSIC | layerwise JEPA, no HSIC | [text_module_jepa_layerwise_gated_no_hsic_2m_4e.yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_module_jepa_layerwise_gated_no_hsic_2m_4e.yaml) | [itm48bxb](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/itm48bxb) | `outputs/text_module_jepa_layerwise_gated_no_hsic_2m_4e/` |
| Gated + HSIC | layerwise JEPA + HSIC | [text_module_jepa_layerwise_gated_hsic_2m_4e.yaml](/home/zd25e122/clevr_discrete_diffusion/configs/text_module_jepa_layerwise_gated_hsic_2m_4e.yaml) | [y0eklen2](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/y0eklen2) | `outputs/text_module_jepa_layerwise_gated_hsic_2m_4e/` |

Launcher: [launch_modulewise_jepa_gated_predictor_vnode10.sh](/home/zd25e122/clevr_discrete_diffusion/scripts/launch_modulewise_jepa_gated_predictor_vnode10.sh) (GPUs 0 and 1, 2026-09-17 13:02 to 19:51 and 21:18).

## Results: final checkpoints (epoch 3)

| | Gated, no HSIC | Layerwise baseline | **Gated + HSIC** | Layerwise baseline | Plain Tri-LoRA | Dense |
|---|---:|---:|---:|---:|---:|---:|
| Validation loss (t = 0.75) | 1.1705 | 1.1595 | 1.1778 | 1.1677 | 1.1638 | 1.0758 |
| Effective rank, `blocks.4.mlp.3` shared / private | 37.4 / 20.2 | 45.7 / 27.8 | 42.3 / 20.3 | 37.3 / 28.8 | 25.5 / 28.2 | — |
| Binding, shared L4 `mlp.3` | 61.9 | 63.2 | **65.4** | 60.2 | 62.7 | — |
| Binding, private L4 `mlp.3` | 58.4 | 59.0 | 61.3 | 58.4 | 62.6 | — |
| Binding, shared L4: attribute / relation | 53.3 / 96.4 | 55.6 / 93.9 | 57.6 / 96.4 | 54.9 / 81.1 | 57.8 / 82.1 | — |
| Binding, residual block 7 | 73.2 | 77.5 | **80.5** | 72.1 | 78.5 | 82.7 |
| Retrieval R@10, shared / private | 20.2 / 17.3 | 35.4 / 30.5 | **40.1** / 24.9 | 33.0 / 23.3 | 22.9 / 25.5 | — |
| Retrieval R@1, residual block 7 | 36.8 | 49.6 | 51.8 | 51.2 | 63.8 | 65.1 |
| Counterfactual semantic preference, shared / private | 42.6 / 59.0 | 97.3 / 88.3 | 88.7 / 80.9 | 93.8 / 97.3 | 96.1 / 94.1 | — |

All values in %, except loss and rank. Binding intervals: gated + HSIC shared
65.4 [64.4–66.3] vs. plain 62.7 [61.5–63.8] and layerwise JEPA 63.2
[62.3–64.2]; gated + HSIC block 7 80.5 [79.5–81.6] vs. plain 78.5
[77.3–79.5]. Shared-rank trajectory at `blocks.4.mlp.3` over epochs 0–3: gated,
no HSIC 18.6 → 25.5 → 31.6 → 37.4; gated + HSIC 12.1 → 22.9 → 34.4 → 42.3.

### Conclusion

1. **End-to-end gating with the predictor kept does not collapse.** Shared rank
   grows every epoch, as in the layer-local baselines.
2. **Gated JEPA + HSIC is the best JEPA configuration so far.** It has the
   strongest shared branch on both retrieval (R@10 40.1%) and binding (65.4%),
   the largest shared–private gap in retrieval, and it is the first variant to
   *raise* whole-model binding above plain Tri-LoRA (80.5% vs. 78.5%). Its
   private branch stays close to plain Tri-LoRA on binding.
3. **The gain is relational, not attribute binding.** Shared relation binding
   rises to 96.4% (plain 82.1%); shared attribute binding stays at 57.6%
   (plain 57.8%).
4. **Gated JEPA without HSIC is worse than its layer-local baseline on
   everything:** retrieval, binding, counterfactual, residual stream. HSIC is
   what makes the end-to-end gradient useful here.
5. **The split is local to the supervised module.** In the accumulated
   residual streams, private matches or beats shared on every binding measure
   ([decomposed representations](../evaluations/decomposed_representations.md)).
6. **Costs remain.** Both gated runs denoise slightly worse than their
   baselines (+0.011 and +0.010 validation loss), whole-model retrieval stays
   below plain Tri-LoRA (block-7 R@1 51.8% vs. 63.8%), and dense still binds far
   better than any Tri-LoRA model.

Full tables: [binding-swap evaluation](../evaluations/binding_swap_evaluation.md),
[cross-pattern semantic retrieval](../evaluations/cross_pattern_semantic_retrieval.md),
[exact-template counterfactual](../evaluations/exact_template_counterfactual.md).

## Results at epoch 0 (interim, kept for reference)

| | Gated, no HSIC | Baseline | Gated + HSIC | Baseline |
|---|---:|---:|---:|---:|
| Validation loss (t = 0.75) | 1.4868 | 1.4339 | 1.4580 | 1.4546 |
| Effective rank, `blocks.4.mlp.3` shared / private | 18.6 / 13.6 | 11.8 / 15.8 | 12.1 / 13.7 | 11.0 / 17.5 |
| Binding, shared L4 `mlp.3` | 52.0% | 53.5% | **55.3%** | 51.3% |
| Binding, relation swaps only, shared L4 | 57.9% | 61.5% | **74.1%** | 56.9% |
| Binding, residual block 7 | 52.2% | 57.8% | 58.2% | 60.8% |
| Retrieval R@10, shared / private | 10.1% / 8.6% | 11.0% / 13.5% | 12.1% / 12.5% | 13.7% / 12.1% |

- Neither run collapses.
- Gated without HSIC costs denoising (+0.053 validation loss) and is below its
  baseline on every representation measure.
- Gated + HSIC matches its baseline's loss and has the strongest shared binding
  of any epoch-0 LoRA model, entirely from relation direction; attribute
  binding is still at chance in every LoRA model at epoch 0.

Details: [binding-swap evaluation, epoch 0](../evaluations/binding_swap_evaluation.md#results-epoch-0-interim-after-one-pass-over-the-2m-captions),
[retrieval, epoch 0](../evaluations/cross_pattern_semantic_retrieval.md#6-epoch-0).

## Next

- Candidate ablation: gated + HSIC with JEPA on blocks 3–5.
  Blocks 2–4 have the lowest text–image gradient conflict
  ([gradient conflict](../evaluations/reference/layer_gradient_conflict.md)), but
  in the text study attribute binding only forms at blocks 5–6.
