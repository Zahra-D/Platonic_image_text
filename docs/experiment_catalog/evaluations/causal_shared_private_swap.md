# Causal shared/private swap (image-only)

> **Type:** evaluation, earlier single-modality phase · **Status:** complete  
> **Question:** if a clean scene's shared LoRA updates are substituted into the generation of a different, fully masked image, does the output follow the substituted scene?  
> **Models:** image diffusion-only control and four image EMA-JEPA pilots  
> **Script:** [evaluate_causal_shared_private_swap.py](/home/zd25e122/clevr_discrete_diffusion/evaluate_causal_shared_private_swap.py) · **Results:** [results.json](/home/zd25e122/clevr_discrete_diffusion/outputs/causal_shared_private_swap_image/results.json)  
> **Menu:** [experiment catalog](../README.md)

## Summary

Every image EMA-JEPA pilot follows the substituted source scene in 81–86% of
pairs, against 45% for the diffusion-only control. The selected middle-layer
shared updates are therefore used causally for image semantics. The frozen
scene parser is only moderately accurate (count accuracy 0.434), so treat this
as an intervention diagnostic rather than a final semantic score. It does not
test text–image alignment. A text version exists
([evaluate_causal_shared_private_swap_text.py](/home/zd25e122/clevr_discrete_diffusion/evaluate_causal_shared_private_swap_text.py))
but has not been run on the text-only study models.

## Contents

- [Question](#question) · [Adapter intervention](#notation-and-adapter-intervention) · [Pairing and generation](#pairing-and-generation) · [Scene parser](#frozen-scene-parser-and-ground-truth) · [Results](#results) · [Interpretation](#interpretation) · [Artifacts](#reproducibility-artifacts)

## Question

This is an **image-only causal intervention**. It asks whether the shared LoRA updates of a clean source scene can change the semantics of a fully masked target image while the target's image-private route remains active. It is not a text-image alignment evaluation.

The evaluated models are all no-base Tri-LoRA image models. Their selected linear maps have no frozen dense $Wz$ term.

## Notation and adapter intervention

Let $A$ be a clean source image and $B$ a different target image. Both are image-only sequences: no caption is passed to the model. Each image has a $16\times24=384$ VQ-code grid.

For Transformer block $l$, selected linear module $a$, and local module input $z^{l,a}$, the shared and image-private updates are

$$
\Delta_s^{l,a}(z)=\frac{\alpha}{\rho_s}B_s^{l,a}A_s^{l,a}D(z),
\qquad
\Delta_I^{l,a}(z)=\frac{\alpha}{\rho_s}B_I^{l,a}A_I^{l,a}D(z).
$$

Here $a\in\mathcal A=\{\mathrm{qkv},\mathrm{out\_proj},\mathrm{mlp.0},\mathrm{mlp.3}\}$. $D$ is LoRA dropout; it is zero for these checkpoints, so $D(z)=z$. The implementation uses the shared-rank denominator $\rho_s$ for both branches' scaling; it does not use a separate $\rho_I$ scaling factor. The ordinary image route is

$$
y^{l,a}=b_s^{l,a}+\Delta_s^{l,a}(z)+\Delta_I^{l,a}(z).
$$

The source is run once without masking. For every $l\in\{2,3,4\}$ and every $a\in\mathcal A$, the evaluator records its native, token-aligned shared update:

$$
\delta_A^{l,a}=\Delta_s^{l,a}(z_A^{l,a}).
$$

The target begins fully masked. During every generation forward pass, the selected target module becomes

$$
y_{\mathrm{swap}}^{l,a}=b_s^{l,a}+\delta_A^{l,a}+\Delta_I^{l,a}(z_B^{l,a}).
$$

So only the selected **shared additive update** is sourced from $A$. The private update is still computed from B's current local input, and B keeps its own tokens, attention, private LoRAs, and every unselected shared layer. This is neither hidden-state replacement nor an ablation that removes the private branch.

The symmetric reference is

$$
G_{\mathrm{self}}=G(\Delta_s(B_{\mathrm{clean}}),P_B),
\qquad
G_{\mathrm{swap}}=G(\Delta_s(A_{\mathrm{clean}}),P_B).
$$

`self` deliberately supplies clean-B shared deltas as a side channel. It is not a normal unconditional sample; it measures what the retained B-private route can do when the shared route comes from the correct scene. The swap tests whether replacing that guide with A redirects the result.

## Pairing and generation

There are 64 held-out pairs. A seeded random permutation of validation examples is cyclically shifted by one position to form the source indices, ensuring $A\ne B$ for every pair.

Initially, all 384 target codes are masks. At iteration $k$, for each still-masked position $i\in R_k$, logits $\ell_{k,i}$ give

$$
\hat x_i=\arg\max_c\operatorname{softmax}(\ell_{k,i})_c,
\qquad
p_i=\max_c\operatorname{softmax}(\ell_{k,i})_c.
$$

With $K=64$ total iterations, the evaluator commits

$$
n_k=\min\left(|R_k|,\left\lceil\frac{|R_k|}{K-k}\right\rceil\right)
$$

remaining positions with the largest confidence $p_i$. The generated token is $\hat x_i$. Thus the procedure is deterministic, greedy confidence sampling: it has no temperature sampling or random draws.

## Frozen scene parser and ground truth

The evaluator trains a separate CNN parser only on 20,000 clean training VQ-code grids, then freezes it. Its target scene label is $Y=(c,a)$:

- $c$: number of objects in the scene;
- $a\in\{0,1\}^{17}$: presence of colors, shapes, materials, sizes, and the `right`/`behind` relation labels from the manifest.

For code grid $X$, parser heads $g_c(X)$ and $g_a(X)$ are trained with

$$
\mathcal L_{\mathrm{parser}}=
\operatorname{CE}(g_c(X),c)+
\frac{1}{17}\sum_{j=1}^{17}\operatorname{BCEWithLogits}(g_{a,j}(X),a_j).
$$

On 1,024 held-out clean code grids, count accuracy is **0.434** and ordinary attribute accuracy is **0.810**. The count score is modest and the attribute score is not balanced accuracy; use this as a causal diagnostic rather than a final semantic benchmark.

For a generated code grid, the frozen-parser NLL against scene $Y$ is

$$
\operatorname{NLL}(X;Y)=
\operatorname{CE}(g_c(X),c_Y)+
\frac{1}{17}\sum_{j=1}^{17}\operatorname{BCEWithLogits}(g_{a,j}(X),a_{Y,j}).
$$

We report self NLL against $Y_B$, swap NLL against $Y_A$ and $Y_B$, source advantage

$$
\operatorname{Adv}=\operatorname{NLL}(G_{\mathrm{swap}};Y_B)-
\operatorname{NLL}(G_{\mathrm{swap}};Y_A),
$$

and source-win rate

$$
\frac1N\sum_{b=1}^{N}\mathbf1\left[
\operatorname{NLL}(G_{\mathrm{swap},b};Y_{A,b})<
\operatorname{NLL}(G_{\mathrm{swap},b};Y_{B,b})\right].
$$

Positive advantage and source-win rate above 0.5 mean that the swapped output is more source-like than target-like under the frozen parser.

## Results

| Model | Self NLL vs $Y_B$ | Swap NLL vs $Y_A$ | Swap NLL vs $Y_B$ | Source advantage | Source win rate |
|---|---:|---:|---:|---:|---:|
| [Image diffusion-only (no JEPA)](../models/image_diffusion.md) | 115.374 | 115.374 | 117.685 | 2.312 | 0.453 |
| [Image EMA-JEPA fixed](../models/image_ema_fixed.md) | 3.826 | 3.826 | 19.196 | 15.370 | 0.812 |
| [Image EMA-JEPA dynamic .10](../models/image_ema_r010.md) | 3.541 | 3.541 | 18.653 | 15.112 | 0.812 |
| [Image EMA-JEPA dynamic .25](../models/image_ema_r025.md) | 3.765 | 3.765 | 15.952 | 12.187 | 0.859 |
| [Image EMA-JEPA dynamic .50](../models/image_ema_r050.md) | 3.297 | 3.297 | 17.624 | 14.327 | 0.828 |

## Interpretation

The no-JEPA diffusion-only Tri-LoRA control has source-win rate 0.453: it does not reliably move toward the substituted source. Every EMA-JEPA variant has 0.812–0.859 source win rate and a large source advantage (12.187–15.370). Within this controlled intervention, the selected shared updates therefore have a causal semantic effect for the EMA-JEPA models.

The equality of mean self NLL vs $Y_B$ and swap NLL vs $Y_A$ for the EMA-JEPA models is not sample reuse. Under cyclic pairing, it is the expected pattern if swapping makes the output as source-like as the self side-channel makes it target-like.

This result is evidence for causal use of shared image features. It does **not** prove that the shared route is modality-invariant or aligned with text. A cross-modal version and a stronger balanced scene parser are required for that claim.

## Reproducibility artifacts

- [Evaluation script](/home/zd25e122/clevr_discrete_diffusion/evaluate_causal_shared_private_swap.py)
- [Raw result JSON](/home/zd25e122/clevr_discrete_diffusion/outputs/causal_shared_private_swap_image/results.json)
- [Run report](/home/zd25e122/clevr_discrete_diffusion/outputs/causal_shared_private_swap_image/REPORT.md)
- [Frozen parser checkpoint](/home/zd25e122/clevr_discrete_diffusion/outputs/causal_shared_private_swap_image/frozen_image_code_scene_parser.pt)
