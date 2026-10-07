# Shared-latent JEPA pretraining

This experiment adds a masked-to-clean prediction objective to the shared
branch of strict no-base Tri-LoRA. It is designed to test whether the shared
branch can learn information that is stable under diffusion corruption, while
SIGReg prevents the representation from solving prediction by collapsing.

## What one training update does

Unpaired text and image batches are processed separately. They never become
pseudo-pairs and no text row is treated as the target of an image row. Both
modalities nevertheless use the same shared LoRA parameters and the same JEPA
predictors.

For each modality `m`:

1. Sample the ordinary diffusion mask `M` and replace those tokens with the
   mask token.
2. Run the corrupted sequence through shared + modality-private Tri-LoRA and
   compute the standard masked-token cross entropy.
3. Record a tokenwise shared-LoRA readout `z_masked[l]` at every Transformer
   block `l`. The four shared adapters in a block (`qkv`, `out_proj`, `mlp.0`,
   and `mlp.3`) are reduced to `d_model` and averaged.
4. Run the original clean sequence through the same online model in evaluation
   mode and without autograd, producing `z_clean[l]`.
5. At exactly the positions in `M`, the layer-specific predictor `P_l` predicts
   `z_clean[l]` from `z_masked[l]`.

The JEPA term is

```text
L_JEPA(m) = mean_l MSE(P_l(z_masked[l][M]), stopgrad(z_clean[l][M]))
```

The complete per-modality objective is

```text
L(m) = L_diffusion(m)
     + lambda_JEPA(step) * L_JEPA(m)
     + lambda_SIGReg(step) * mean_l SIGReg(pool(z_masked[l]))
```

Text and image objectives are averaged by the existing balanced unpaired
training loop. The configured weights are `0.1` for JEPA and `0.01` for
SIGReg, each linearly warmed up over 1,000 optimizer updates.

## Gradient routing

- The clean target is inside `torch.no_grad()` and explicitly detached.
- The shared-activation recorder recomputes each shared delta from a detached
  module input.
- Therefore JEPA gradients update only the shared LoRA branch at the block
  where the readout was recorded and that block's JEPA predictor.
- JEPA cannot directly update private LoRA, token/position embeddings, or an
  earlier block through the recorded input.
- The diffusion loss still updates the ordinary model path, including the
  modality-private branch.
- SIGReg operates on the same per-layer shared readouts and supplies an
  anti-collapse distributional signal.

This is an online, stop-gradient JEPA target, not a separate EMA teacher. That
choice follows the requested masked/clean shared-LoRA construction and keeps
the ablation focused. If it is unstable, an EMA target encoder is a reasonable
next experiment.

## What it can and cannot prove

JEPA teaches invariance to masking within text and within image. Since the same
shared parameters and predictors serve both modalities, it gives the shared
branch pressure to represent predictable scene information. Per-layer SIGReg
prevents a constant shared vector from being the easy solution.

It still does not identify caption `i` with image `i`: there is no paired
anchor in this pretraining loss. Consequently, success must be checked with
both distribution diagnostics and instance-level paired retrieval / matched
versus shuffled validation. Good Gaussianity or similar marginal geometry
alone is not evidence of semantic cross-modal alignment.

## Run

```bash
./scripts/run_shared_jepa_sigreg_per_layer_node10.sh 0
```

The launcher refuses to overwrite an existing output directory or reuse an
active tmux session.
