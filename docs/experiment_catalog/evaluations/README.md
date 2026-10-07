# Evaluations

The full, annotated list is in the [experiment catalog menu](../README.md#evaluations).

## Primary

- [Semantic effect sizes (d′)](semantic_dprime.md) — **primary text measure.** Scene-vs-wording and binding as signed effect sizes, with shared/private route isolation
- [Image binding](image_binding.md) — **primary image measure.** Content-matched scene pairs, chance 50%
- [Modality alignment](modality_alignment.md) — modality gap, separability, paired retrieval and [per-layer CKA](modality_alignment.md#per-layer-cka-the-depth-profile) inside one shared space
- [Cross-modal structure](cross_modal_structure.md) — CKA/RSA between separately trained text and image models, and the per-modality scene probe

## Superseded, but the construction still matters

- [Binding-swap evaluation](binding_swap_evaluation.md) — the probe d′ replaced; preference rates saturate
- [Hard retrieval](hard_retrieval.md) — lexically identical candidates; ten-point usable range, so it could not separate the later families
- [Cross-pattern semantic retrieval](cross_pattern_semantic_retrieval.md)
- [Exact-template counterfactual](exact_template_counterfactual.md)

## Diagnostics and single-purpose pages

- [Decomposed representations](decomposed_representations.md) — every layer, sublayer, and shared/private stream
- [Gated data2vec collapse diagnosis](data2vec_gated_collapse_diagnosis.md)
- [Pure-JEPA scratch hyperparameter sweep](pure_jepa_scratch_hyperparameter_sweep.md) — 8 matched EMA/masking/target-depth pilots with diffusion exactly zero
- [Modality-local semantic probe at 80% masking](modality_local_jepa_semantic_80pct.md) — earlier phase
- [Causal shared/private swap (image-only)](causal_shared_private_swap.md) — earlier phase
- [Method reference pages](reference/) — see the [menu](../README.md#3-method-reference-pages)
