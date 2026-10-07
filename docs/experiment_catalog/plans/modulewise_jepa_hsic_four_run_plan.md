# Modulewise EMA-JEPA + HSIC four-run sweep

> **Type:** experiment plan, text-only study · **Status:** complete; all four runs finished four epochs  
> **Menu:** [experiment catalog](../README.md) · **Run registry:** [text-study runs](../models/text_study_run_registry.md)

## Runs

| Cell | W&B | Output | Final validation loss |
|---|---|---|---:|
| Average target, no HSIC | [xzskwqhp](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/xzskwqhp) | `outputs/text_module_jepa_avg_no_hsic_2m_4e/` | 1.1435 |
| Average target + HSIC | [6nhfkmyl](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/6nhfkmyl) | `outputs/text_module_jepa_avg_hsic_2m_4e/` | 1.1613 |
| Layerwise target, no HSIC | [muq3kc5f](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/muq3kc5f) | `outputs/text_module_jepa_layerwise_no_hsic_2m_4e/` | 1.1595 |
| Layerwise target + HSIC | [sxpwb1cb](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/sxpwb1cb) | `outputs/text_module_jepa_layerwise_hsic_2m_4e/` | 1.1677 |
| Matched plain Tri-LoRA control | [4q137e90](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/4q137e90) | `outputs/text_lora_diffusion_2m_4e_matched/` | 1.1638 |
| Matched dense control | [4v57nzhs](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/4v57nzhs) | `outputs/text_dense_diffusion_2m_4e_matched/` | 1.0758 |

A fifth follow-up, average target + HSIC with a calibrated HSIC coefficient
(maximum 50 instead of 1), ran as [ayxc635a](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_modulewise_ema_jepa_hsic/runs/ayxc635a)
(`outputs/text_module_jepa_avg_hsic_calibrated_2m_4e/`, final loss 1.1489).

Results: [binding-swap evaluation](../evaluations/binding_swap_evaluation.md),
[cross-pattern semantic retrieval](../evaluations/cross_pattern_semantic_retrieval.md),
[exact-template counterfactual](../evaluations/exact_template_counterfactual.md).

**Implementation note.** As first implemented, the average-target cells
supervised only the layer-4 student; blocks 2 and 3 contributed to the target
but received no JEPA gradient, as described below. The current code supervises
every selected layer in average mode as well.

## Question

Can EMA-JEPA make the **middle shared LoRA route** encode contextual / abstract information, while the private route remains useful but non-redundant?  This is a text-only first test.  It deliberately does not test cross-modal alignment yet.

The four runs cross two target designs with HSIC on or off:

| ID | JEPA target design | HSIC shared-private decorrelation |
|---|---|---|
| `text_module_jepa_avg_no_hsic` | average clean teacher layers 2–4, predict it from student layer 4 | no |
| `text_module_jepa_avg_hsic` | average clean teacher layers 2–4, predict it from student layer 4 | yes |
| `text_module_jepa_layerwise_no_hsic` | clean and masked representations matched at each of layers 2, 3, 4 | no |
| `text_module_jepa_layerwise_hsic` | clean and masked representations matched at each of layers 2, 3, 4 | yes |

The comparison is paired: the two `_no_hsic` runs ask whether the target design alone helps; comparing each of them to its `_hsic` counterpart asks whether explicitly reducing shared/private dependence helps.

The executable configurations are:

- [`text_module_jepa_avg_no_hsic_2m_4e.yaml`](../../../configs/text_module_jepa_avg_no_hsic_2m_4e.yaml)
- [`text_module_jepa_avg_hsic_2m_4e.yaml`](../../../configs/text_module_jepa_avg_hsic_2m_4e.yaml)
- [`text_module_jepa_layerwise_no_hsic_2m_4e.yaml`](../../../configs/text_module_jepa_layerwise_no_hsic_2m_4e.yaml)
- [`text_module_jepa_layerwise_hsic_2m_4e.yaml`](../../../configs/text_module_jepa_layerwise_hsic_2m_4e.yaml)

## Common training conditions

All four must be fresh, directly comparable models:

- **modality:** text only;
- **architecture:** 8-block Tri-LoRA Transformer, width \(d=384\), rank \(r=384\), no dense/frozen base map, no modality embeddings, learned 1-D absolute positions;
- **data:** the verified 2M human-style CLEVR caption corpus, four epochs; each corpus row is seen once per epoch and four times in total;
- **batch:** microbatch 64 with four-way gradient accumulation (nominal global batch 256).  The 2M corpus gives 7,813 optimizer updates per epoch, with the final update containing two rather than four microbatches; 31,252 updates per run;
- **masking:** exactly the same configured corruption distribution and seeded data/mask stream in every run;
- **EMA:** teacher decay \(0.999\), teacher sees clean tokens, student sees the normal corrupted tokens;
- **selected blocks:** \(\mathcal L=\{2,3,4\}\), using zero-based Transformer block indices;
- **selected shared modules only:** attention `out_proj` and MLP `mlp.3` (the MLP down-projection).  We intentionally do **not** put JEPA on `qkv` or `mlp.0` in this sweep;
- **predictors:** independent predictor for each supervised module and, in the layerwise design, each block.  Thus average-target has two predictors; layerwise has six;
- **evaluation:** fixed held-out captions and fixed corruption seed at each validation point, plus the same frozen semantic probes afterward.

The normal reconstruction/diffusion loss remains unchanged and is optimized over the masked text tokens:

$$
\mathcal L_{\rm diff}=\frac{1}{|\mathcal M|}\sum_{i\in\mathcal M}\frac{1}{t_b}\,{\rm CE}(\ell_i,x_i).
$$

Here \(x_i\) is the original discrete token, \(\ell_i\) are the student logits, \(\mathcal M\) is the masked-position set, and \(t_b\) is the row's sampled mask fraction.  `unweighting=False`, so the \(1/t_b\) factor is active.

## Exact shared quantities

For block \(l\), module \(b\in\{\mathrm{att},\mathrm{mlp}\}\), and token \(i\), define

$$
u^{(l)}_{b,s,i}=\Delta h^{(l)}_{b,\mathrm{shared},i},\qquad
u^{(l)}_{b,p,i}=\Delta h^{(l)}_{b,\mathrm{text-private},i}.
$$

They are the **native output updates** of the corresponding LoRA linear map: `att` is `out_proj`; `mlp` is `mlp.3`.  Each has width \(d=384\).  They are not the full block output, and the shared update is not formed by subtracting a private-free forward pass.

The shared map receives the regular block input.  That input can already contain effects of earlier private modules; the JEPA gradient itself, however, is attached only to the selected shared adapter output and its predictor.  The clean EMA-teacher target is stop-gradient.

Let

$$
N(v)=\frac{\sqrt d\,v}{\lVert v\rVert_2+10^{-6}}
$$

be the tokenwise normalization.  This is the existing `normalized_mse` convention and prevents the JEPA objective from being driven only by magnitude.

## Run A: averaged-teacher target

For each module separately, form a clean teacher target by averaging the normalized shared updates over the selected middle blocks:

$$
y_{b,i}=\operatorname{sg}\!\left[N\!\left(\frac{1}{3}\sum_{l=2}^{4}N\!\left(u^{(l)}_{b,s,i,T}\right)\right)\right].
$$

Only the last selected masked student block is directly supervised:

$$
\mathcal L_b=\frac{1}{|\mathcal M|}\sum_{i\in\mathcal M}
\operatorname{MSE}\!\left(N\!\left(P_b\!\left(u^{(4)}_{b,s,i,S}\right)\right),y_{b,i}\right),
$$

$$
\mathcal L_{\rm JEPA}=0.5\,\mathcal L_{\rm att}+0.5\,\mathcal L_{\rm mlp}.
$$

This asks whether the end of the middle shared route can recover information persistent across blocks 2–4.  Blocks 2 and 3 are sources of the target but do not receive a direct JEPA gradient.

## Run B: layerwise same-layer target

For every selected block and module, a separate predictor recovers that block's clean teacher update from the corresponding masked student update:

$$
\mathcal L_b^{(l)}=\frac{1}{|\mathcal M|}\sum_{i\in\mathcal M}
\operatorname{MSE}\!\left(N\!\left(P_b^{(l)}\!\left(u^{(l)}_{b,s,i,S}\right)\right),
\operatorname{sg}\!\left[N\!\left(u^{(l)}_{b,s,i,T}\right)\right]\right),
$$

$$
\mathcal L_{\rm JEPA}=\frac{1}{3}\sum_{l=2}^{4}
\left(0.5\,\mathcal L_{\rm att}^{(l)}+0.5\,\mathcal L_{\rm mlp}^{(l)}\right).
$$

This does **not** force blocks 2, 3, and 4 to be equal.  It separately requires masked block \(l\) to predict clean teacher block \(l\), so every selected shared module receives a direct contextual-prediction signal.

## HSIC condition

The `_hsic` runs add a branchwise shared/private independence penalty.  It never compares attention to MLP, and it never pools different blocks before measuring dependence.

For each \((l,b)\), collect up to 512 masked tokens deterministically and uniformly from the current update.  Let \(S\) and \(P\) be their normalized shared and text-private updates.  With RBF Gram matrices

$$
K_{ij}=\exp\!\left(-\frac{\lVert S_i-S_j\rVert_2^2}{2\sigma_S^2}\right),\qquad
G_{ij}=\exp\!\left(-\frac{\lVert P_i-P_j\rVert_2^2}{2\sigma_P^2}\right),
$$

where each bandwidth is that batch's detached median nonzero pairwise distance, define \(H=I-\frac1n\mathbf1\mathbf1^\top\) and

$$
\mathcal L_{\rm HSIC}=\frac{1}{|\mathcal L|\,2}
\sum_{l\in\mathcal L}\sum_{b\in\{\rm att,mlp\}}
\frac{\operatorname{tr}(KHG H)}{(n-1)^2}.
$$

The formula is computed only in the two HSIC runs.  Its gradient reaches both selected shared and private LoRA adapters.  Diffusion and JEPA remain present, so minimizing HSIC alone cannot obtain a useful solution by collapsing either branch.

## Coefficients and why

The final objective is

$$
\boxed{\mathcal L=\mathcal L_{\rm diff}+r_J\lambda_J\mathcal L_{\rm JEPA}+r_H\lambda_H\mathcal L_{\rm HSIC}.}
$$

| Term | Value / schedule | Reason |
|---|---:|---|
| \(\lambda_J\) | **0.50**, fixed | The completed previous text EMA-JEPA run with fixed 0.50 was the best stable JEPA pilot.  Dynamic 0.10/0.25/0.50 pilots did not beat it and ended early after W&B failures. |
| JEPA warm-up \(r_J\) | \(\min(1,(u+1)/1000)\) | The earlier pilot used 5000 steps, but this one-pass run has only 7812 updates; 1000 gives the model 12.8% of the pass to establish diffusion behavior first. |
| attention / MLP balance | \(0.5/0.5\) | Adding two branches should not silently double the old total JEPA scale. |
| \(\lambda_H\) | adaptive, initially 0 | HSIC has not been measured in this codebase, so a copied fixed constant would be arbitrary. |
| HSIC target gradient ratio | 0.05 of diffusion gradient | Strong enough to be visible, deliberately smaller than the generative objective. |
| HSIC \(\lambda_H\) update | every 100 updates; EMA 0.95; clamp \([10^{-4},1]\) | Makes the unknown RBF-HSIC numerical scale comparable across training, while preventing spikes. |
| HSIC warm-up \(r_H\) | \(\min(1,(u+1)/1000)\) | Avoids enforcing branch separation before either branch has a signal. |

For the adaptive HSIC multiplier, using the union of selected shared and private adapter parameters \(\theta_{s,p}\), the proposed update is

$$
\lambda_H=\operatorname{clip}_{[10^{-4},1]}\!\left(
\frac{0.05\;\operatorname{EMA}\!\left[\lVert\nabla_{\theta_{s,p}}\mathcal L_{\rm diff}\rVert_2\right]}
{\operatorname{EMA}\!\left[\lVert\nabla_{\theta_{s,p}}\mathcal L_{\rm HSIC}\rVert_2\right]+10^{-12}}
\right).
$$

The fixed \(\lambda_J\) is intentional: it makes the target-design comparison scientific.  We will log the actual selected-adapter gradient norms for diffusion, raw JEPA, and raw HSIC every 250 updates.  If the realized JEPA contribution is clearly negligible or dominates diffusion, we stop before treating the sweep as evidence and revise the coefficient rather than silently accepting it.

## Evidence used to choose the JEPA strategy

The prior text-only pilots were the only directly relevant coefficient evidence:

| Previous run | Best fixed-mask validation loss | Status |
|---|---:|---|
| diffusion-only | 1.1612 | completed 70 epochs |
| EMA-JEPA, fixed \(\lambda_J=0.50\) | 1.1759 | completed 70 epochs |
| EMA-JEPA, dynamic target 0.10 | 1.1988 | ended at epoch 61 after W&B failure |
| EMA-JEPA, dynamic target 0.25 | 1.1978 | ended at epoch 58 after W&B failure |
| EMA-JEPA, dynamic target 0.50 | 1.1816 | ended at epoch 55 after W&B failure |

The fixed-0.50 run logged raw JEPA around 0.20–0.24 late in training, hence a weighted contribution around 0.10–0.12 against diffusion around 1.8–2.0.  It was stable, but it did not improve marginal diffusion validation loss; that is acceptable here because the primary outcome is frozen representation quality.  There is no existing HSIC pilot, which is why HSIC uses measured gradient calibration rather than an invented fixed weight.

## Required outputs before judging the idea

For every run, save best, final, and every-epoch checkpoints.  Report:

1. diffusion loss and raw/weighted JEPA terms by module and layer;
2. raw/weighted HSIC and realized gradient ratios (HSIC runs);
3. frozen linear probe and k-NN on the shared update, private update, and full hidden state, separately for count, color, shape, material, size, and relations;
4. same held-out examples and same masking seed across all four runs;
5. a direct comparison to the already-running 2M text diffusion-only control.

The first scientific decision is not generation quality.  It is whether shared-update probes improve over the diffusion-only control **without** private-update probes simply rising by the same amount.  If that fails, an MLP-only JEPA ablation is the next clean experiment; it is intentionally not mixed into this four-run factorial sweep.

## Implementation work after approval

1. Extend the adapter recorder to retain native `out_proj`/`mlp.3` **shared and private** updates.
2. Add the two target modes, per-module/per-layer predictors, and the optional HSIC term.
3. Add strict configuration validation, fixed data/evaluation seeds, W&B tags, and the diagnostics above.
4. Run a one-update CPU/GPU smoke test and verify that the no-HSIC paths reproduce the diffusion-plus-JEPA objective exactly.
5. Only then launch these four jobs.
