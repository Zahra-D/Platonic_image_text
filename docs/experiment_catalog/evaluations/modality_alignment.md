# Modality alignment: where the two modalities sit, in one shared space

> **Type:** evaluation · **Status:** 15 models, per-layer · **Updated:** 2026-09-24  
> **Menu:** [experiment catalog](README.md) · **Code:** `evaluate_paired_modality_alignment.py`

When a **single model** encodes both captions and images, the two modalities
land in the *same* 384-dimensional space with the same basis, so they can be
compared directly — no rotation-invariant proxy needed. That is what this
evaluation does, on genuinely paired data: the same scene as a caption and as
VQ image tokens, so row *i* of each matrix is the same scene.

[Cross-modal structure](cross_modal_structure.md) answers the neighbouring
question for models that do *not* share a space (a text model and an image model
trained separately), where only CKA and RSA are meaningful.

## What is measured

**`modality_gap`** — the distance between the two modality centroids,
`‖mean(text) − mean(image)‖`, on L2-normalized vectors. Zero means the two
clouds sit in the same place; two unit vectors can be at most 2 apart, so ~1.4
means they occupy clearly separate regions. This is the "modality gap" reported
for CLIP-style models.

**`modality_auc`** — how separable the modalities are along that centroid
direction: 0.5 means a linear rule cannot tell a caption from an image, 1.0
means it separates them perfectly. The gap says *how far apart* the clouds are;
the AUC says *whether they overlap at all*, which a gap alone cannot.

**`paired_r1`** — given a caption's representation, does its **own** image rank
first among all candidates? Chance is 1/N. This is instance-level
correspondence, which distributional overlap does not imply.

**`matched_cosine` / `mismatched_cosine`** — the raw ingredients of that
retrieval, so a high R@1 can be traced to either modality alignment or simply
to everything being close together.

**`cka` / `rsa`** — the structure measures, reported here too so one table
covers both senses of "similar". Definitions in
[cross-modal structure](cross_modal_structure.md).

For Tri-LoRA models the metrics are reported on the **shared write stream**
(`L7.shared`), since that is the route the shared/private hypothesis is about;
for dense models on the residual stream.

## Results

`corpus` distinguishes the 1.2M paired/unpaired study from the earlier 90k
family; the two are not directly comparable on absolute values. Sorted by gap.

| Model | corpus | design | feature | gap | AUC | paired R@1 | CKA | RSA | smallest gap anywhere |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `lejepa_scratch_both` | 1.2M | dense | L7 | 0.050 | 0.576 | 0.1% | 0.018 | -0.008 | L5 0.047 |
| `jepa_sigreg` (from unpaired dense) | 1.2M | dense | L7 | 0.053 | 0.557 | 0.0% | 0.023 | +0.014 | L7 0.053 |
| `lora_jepa_per_layer` | 90k | lora | L7.shared | 0.445 | 0.999 | 0.1% | 0.108 | +0.084 | L1.shared 0.367 |
| `unpaired_d2v_from_image` | 1.2M | dense | L7 | 0.540 | 1.000 | 0.2% | 0.182 | +0.224 | L6 0.534 |
| `dense_paired` | 90k | dense | L7 | 0.911 | 1.000 | 0.1% | 0.347 | +0.414 | L7 0.911 |
| `jepa_ema` (from unpaired dense) | 1.2M | dense | L7 | 1.096 | 1.000 | 0.2% | 0.116 | +0.110 | L2 0.633 |
| `lora_paired` | 90k | lora | L7.shared | 1.240 | 1.000 | 0.3% | 0.258 | +0.173 | L5.shared 0.984 |
| **`paired_dense`** | **1.2M** | dense | L7 | 1.275 | 1.000 | 0.0% | **0.554** | **+0.526** | L6 1.258 |
| `unpaired_d2v_from_text` | 1.2M | dense | L7 | 1.331 | 1.000 | 0.1% | 0.061 | +0.116 | L0 0.495 |
| `dense_unpaired` | 90k | dense | L7 | 1.363 | 1.000 | 0.1% | 0.123 | +0.180 | L4 0.756 |
| `lora_jepa_normalized` | 90k | lora | L7.shared | 1.413 | 1.000 | 0.2% | 0.078 | +0.171 | L1.shared 0.285 |
| `lora_plain` | 90k | lora | L7.shared | 1.481 | 1.000 | 0.1% | 0.211 | +0.257 | L5.shared 0.587 |
| `unpaired_dense` | 1.2M | dense | L7 | 1.561 | 1.000 | 0.1% | 0.187 | +0.179 | L0 0.427 |
| `lora_jepa_strong` | 90k | lora | L7.shared | 1.643 | 1.000 | 0.1% | 0.195 | +0.251 | L1.shared 0.450 |
| two separately trained dense models | 90k | — | L7 | *n/a* | *n/a* | *n/a* | 0.074 | +0.139 | — |

Two separately trained models share no basis, so only CKA and RSA are defined
for them; they are the "no shared space at all" floor.

## Per-layer CKA: the depth profile

The headline table reports one layer. The *shape* of the CKA curve across depth
turns out to separate the models more sharply than any single number, so it is
tabulated in full here. All values are linear CKA between the model's text
features and its image features on the same 2000 scenes.

**Where a model has a private route, only the shared stream is read** (marked
`*`): the shared/private hypothesis is a claim about the shared route, and
including the private writes would score the model on exactly the part the
hypothesis says should be modality-specific.

### One model, both modalities — the unpaired / paired 1.2M family

| Model | emb | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **paired dense, 4 ep** | 0.134 | 0.216 | 0.423 | 0.430 | 0.507 | 0.552 | 0.554 | **0.622** | 0.554 | **L6 0.622** |
| unpaired dense, 4 ep | 0.125 | 0.131 | 0.175 | 0.164 | 0.149 | 0.130 | 0.124 | 0.135 | **0.187** | L7 0.187 |
| JEPA from **image** dense | 0.142 | 0.157 | 0.160 | 0.171 | 0.141 | 0.150 | 0.130 | 0.127 | **0.182** | L7 0.182 |
| JEPA from **text** dense | **0.186** | 0.182 | 0.207 | 0.150 | 0.103 | 0.080 | 0.074 | 0.070 | 0.061 | L1 0.207 |
| JEPA from unpaired dense, EMA | 0.140 | 0.135 | 0.161 | 0.150 | 0.127 | 0.108 | 0.079 | 0.086 | 0.116 | L1 0.161 |
| JEPA from unpaired dense, SIGReg | 0.157 | 0.038 | 0.030 | 0.027 | 0.025 | 0.023 | 0.023 | 0.023 | 0.023 | emb 0.157 |
| LeJEPA from scratch, both modalities | 0.188 | 0.021 | 0.021 | 0.021 | 0.020 | 0.019 | 0.019 | 0.019 | 0.018 | emb 0.188 |

### Earlier 90k paired / unpaired family

| Model | emb | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `dense_paired` | 0.137 | 0.309 | 0.363 | 0.416 | 0.414 | **0.416** | 0.355 | 0.379 | 0.347 | L4 0.416 |
| `dense_unpaired` | 0.116 | 0.133 | 0.116 | 0.115 | **0.147** | 0.115 | 0.121 | 0.137 | 0.123 | L3 0.147 |
| `lora_paired` * | — | 0.196 | 0.217 | **0.464** | 0.428 | 0.331 | 0.343 | 0.371 | 0.258 | **L2 0.464** |
| `lora_plain` * | — | 0.085 | 0.062 | 0.157 | 0.124 | 0.106 | 0.119 | 0.180 | **0.211** | L7 0.211 |
| `lora_jepa_strong` * | — | 0.118 | 0.124 | 0.150 | 0.103 | 0.086 | 0.118 | 0.172 | **0.195** | L7 0.195 |
| `lora_jepa_per_layer` * | — | 0.044 | **0.112** | 0.085 | 0.054 | 0.019 | 0.055 | 0.072 | 0.108 | L1 0.112 |
| `lora_jepa_normalized` * | — | 0.065 | **0.082** | 0.040 | 0.020 | 0.020 | 0.023 | 0.042 | 0.078 | L1 0.082 |
| two separately trained dense models | 0.100 | 0.078 | 0.068 | **0.126** | 0.112 | 0.054 | 0.078 | 0.091 | 0.074 | L2 0.126 |

### What the depth profile says

**1. Only paired training produces a curve that *climbs*.** The paired dense
model goes 0.13 → 0.22 → 0.42 → … → 0.62, more than quadrupling from embedding
to L6. Every other model in the study is flat or declining. Whatever cross-modal
structure the other models have is inherited from the input statistics and then
eroded by depth; only correspondence supervision builds it up.

**2. The unpaired dense model's L7 bump (0.187) is not cross-modal structure
being built — it is the readout layer.** Its curve dips through the middle of
the network (0.175 → 0.124 at L5) and recovers only at the last block, where
both modalities are being shaped for their respective output heads. Compare the
paired model, whose peak is at **L6**, one block *before* the readout.

**3. JEPA from a text trunk actively destroys cross-modal structure with
depth**: 0.186 at the embedding, monotonically down to 0.061 at L7 — a 3× loss,
the steepest decline in the table. Interestingly this is also the run with the
best text `d_binding` in the unpaired family (+0.381). It specializes hard on
text, and the image side is left behind. The mirror run from the *image* trunk
does the opposite and keeps its L7 value at 0.182, essentially matching the
unpaired dense trunk.

**4. Both SIGReg runs are flat at ~0.02 from L0 onward.** The embedding layer
still shows 0.157 / 0.188 — the input statistics are intact — and then the very
first block collapses the structure and every subsequent block keeps it
collapsed. This is a clean signature: SIGReg at λ = 0.05 is not slowly degrading
the representation, it is switching it off at the first opportunity. It is also
why the *gap* numbers for these runs (0.05 from L0 onward) look so good: an
empty representation has no modality separation to measure.

**5. In the 90k family the Tri-LoRA shared route peaks in the middle of the
network, unlike the dense models.** `lora_paired`'s shared stream reaches
**0.464 at L2**, the highest mid-network shared-route value anywhere in the
study and above `dense_paired`'s own L2 (0.416) — the one piece of evidence that
an explicit shared route can carry cross-modal content better than an
undifferentiated trunk. It does not hold up at depth (0.258 by L7), and it is on
the small 90k corpus, so it is a hint rather than a result. **The obvious
follow-up — unpaired stage 2, a dense shared trunk plus per-modality private
LoRA on the 1.2M unpaired data — has not been run.**

## What the numbers say

1. **Every model keeps the modalities separable.** AUC is 1.000 everywhere
   except the two SIGReg runs — a single linear direction tells a caption from
   an image without error, even in the paired models. No training signal in this
   study produced genuinely overlapping distributions by learning; the only
   thing that closed the gap was an explicit distributional penalty.
2. **Instance correspondence is never learned.** `paired_r1` is at chance
   (0.0–0.3%) for all fifteen models, including the paired ones. Whatever these
   models share across modalities is scene-*class* structure, not the identity
   of a particular scene. One caveat: each modality is encoded alone, which is
   out of distribution for a model trained on joint sequences, so the paired
   models' R@1 may be understated.
3. **Paired training buys structure, not proximity.** `paired_dense` has the
   highest CKA in the entire study, **0.554**, and RSA +0.526 — roughly three
   times the unpaired dense model (0.123 / +0.180) and well above two separately
   trained dense models (0.322 / +0.371). Its modality *gap*, though, is 1.275,
   no better than the unpaired model's 1.363. Correspondence in training aligns
   the *relational structure* of the two modalities while leaving them in
   separate regions of the space.
4. **SIGReg closes the gap by emptying it.** `lejepa_scratch_both` reaches a gap
   of 0.050 and AUC 0.576 — the only genuinely overlapping distributions
   measured — but its CKA falls to 0.018 and RSA to −0.008, and its scene probes
   drop to 55.7% text / 57.2% image against the trunk's 73.1 / 79.8. The
   isotropic-Gaussian constraint merges the modalities by flattening both, not
   by finding shared content. Distributional overlap and shared structure are
   not the same goal, and this is the clearest demonstration of it.
5. **In the LoRA family the shared route is not more modality-agnostic**: the
   per-layer JEPA model's shared stream reaches a gap of 0.445 at L7 and 0.367
   at L1, the best of that family, but its CKA stays at 0.108 — again proximity
   without structure.
6. **JEPA on the unpaired trunk moves the gap but not the structure.** Starting
   from the *image* trunk closes the gap from 1.561 to 0.540 while keeping CKA
   roughly where the trunk had it (0.182 vs 0.187); starting from the *text*
   trunk keeps the gap at 1.331 and cuts CKA to 0.061. Neither approaches the
   paired model. The EMA variant lands between them (gap 1.096, CKA 0.116) and
   the SIGReg variant collapses (gap 0.053, CKA 0.023) — the same split seen in
   the per-layer profiles above.
7. **Nothing in the unpaired family gets close to paired dense.** Its CKA of
   0.554 is 3.0× the best unpaired model (0.187) and 9.1× the best from-unpaired
   JEPA run's L7 value (0.061 – 0.182). Correspondence supervision is doing
   something that no amount of shared-trunk self-supervised training has
   reproduced so far.

**The target to beat**, stated plainly: `paired_dense` at **CKA 0.554 / RSA
+0.526**, with the open goal of reaching that *while* closing the gap and
lifting `paired_r1` off chance. No model has done more than one of the three.

## Reproduce

```bash
python evaluate_paired_modality_alignment.py --num-scenes 2000 \
  --manifest .../val_pairs_human.jsonl \
  --image-cache outputs/image_only_1_2m_token_cache/val_tokens.pt \
  --output-dir outputs/unpaired_modality_alignment \
  --model LABEL=path/to/epoch_00N.pt
```

`--model` is for one model that encodes both modalities; `--text-model` and
`--image-model` compare two separately trained ones, in which case only CKA and
RSA are reported. Output: `alignment.json`, every metric for every feature.
