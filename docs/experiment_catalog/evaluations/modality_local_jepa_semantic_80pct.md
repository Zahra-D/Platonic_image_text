# Modality-local semantic probe at 80% masking

> **Type:** evaluation, earlier multimodal and single-modality phase · **Status:** complete  
> **Question:** does the shared representation keep scene semantics when one modality is shown alone with 80% of its tokens masked?  
> **Models:** eight multimodal LoRA pretraining variants; text-only and image-only EMA-JEPA pilots  
> **Script:** [evaluate_jepa_modality.py](/home/zd25e122/clevr_discrete_diffusion/evaluate_jepa_modality.py)  
> **Menu:** [experiment catalog](../README.md)

**Caution when reading.** These are ordinary attribute probes, which the later
[binding-swap evaluation](binding_swap_evaluation.md) shows are largely
solvable from word content alone (84.5% balanced accuracy from word counts on
text). In the text-only study, 80%-masked held-out probes were near chance for
every model, and masked–clean cosine stability rewards collapsed
representations (see the [collapse diagnosis](data2vec_gated_collapse_diagnosis.md)).

## Question answered

Does a pretrained model retain scene semantics in its **shared** representation when only one modality is presented and 80% of that modality's discrete tokens are randomly masked?

This is a within-modality robustness/semantic-decoding evaluation.  It is **not** a text-image retrieval or paired-alignment score.

## Controlled protocol

| Item | Fixed for every compared checkpoint? | Actual setting |
|---|---|---|
| Training examples for the probe | Yes | First 512 rows of the training manifest, fixed order |
| Held-out test examples | Yes | First 256 rows of the validation manifest, fixed order |
| Evaluated modality | Yes within a table | Text-only or image-only; the other modality is removed |
| Masking rate | Yes | `t=0.8` (80% of eligible discrete tokens) |
| Exact mask positions | Yes | The seed is reset before every validation batch and mask ratio, so every checkpoint sees identical corrupted tokens |
| Semantic labels | Yes | Object count; color, shape, material, size, and relation presence from the CLEVR scene manifest |
| Probe type | Same procedure | L2-normalization + standardization + class-balanced logistic regression; a new probe is fitted per model because features differ |
| Layer selection in summary tables | Same rule | Select L2, L3, or L4 with the highest shared attribute-group average for that model/modality |

The deterministic corruption seed is `seed + batch_index × 1009 + ratio_index × 31 + 500000` on validation.  Models run in `eval()` mode, so they have no dropout/random inference.  Therefore the plain-LoRA and JEPA rows below see exactly the same scenes and the same masked positions.

## What is fitted and scored

1. Run each modality alone through the final **student** checkpoint.
2. Pool eligible-token shared representations from clean training examples.
3. Fit one class-balanced logistic-regression classifier for object count and one binary classifier per attribute.
4. Apply the deterministic 80% token mask to each held-out example.
5. Pool its masked shared representations and score the clean-trained classifiers against ground-truth scene labels.

`Count` is balanced accuracy for number of objects. `Attribute avg` equally averages five group averages: color, shape, material, size, and relation. A value near 0.5 for a binary attribute is chance-level under balanced accuracy.

## Results: every evaluated multimodal LoRA checkpoint

All rows below used the exact same train/held-out examples and deterministic masks within each modality. `L` is the selected shared layer, chosen independently for every row as the one with the largest attribute average among layers 2, 3, and 4.

| Model | Text L | Text count | Text attr. avg | Image L | Image count | Image attr. avg |
|---|---:|---:|---:|---:|---:|---:|
| [Plain LoRA unpaired](../models/plain_lora_no_stage.md) | 4 | 0.264 | 0.560 | 4 | 0.253 | 0.547 |
| [LoRA stage 0](../models/lora_stage0.md) | 3 | 0.216 | 0.561 | 4 | 0.402 | 0.594 |
| [LoRA + DANN](../models/lora_dann.md) | 3 | 0.271 | 0.563 | 4 | 0.322 | 0.564 |
| [LoRA stage 0 → DANN](../models/lora_stage0_then_dann.md) | 2 | 0.220 | 0.540 | 4 | 0.167 | 0.556 |
| [LoRA + per-layer JEPA/SIGReg](../models/jepa_per_layer.md) | 4 | 0.348 | 0.613 | 4 | 0.310 | 0.564 |
| [LoRA + normalized JEPA/SIGReg](../models/jepa_normalized.md) | 3 | 0.376 | 0.634 | 3 | 0.175 | 0.539 |
| [LoRA + strong normalized JEPA/SIGReg](../models/jepa_strong_normalized.md) | 4 | 0.392 | 0.554 | 4 | 0.335 | 0.602 |
| [LoRA + strong raw-L2 JEPA/SIGReg](../models/jepa_strong_raw_l2.md) | 4 | 0.366 | 0.607 | 3 | 0.269 | 0.536 |

For example, plain LoRA versus strong normalized JEPA is `0.253 → 0.335` in image count and `0.547 → 0.602` in image attribute average.  This difference cannot be explained by a changed held-out split or easier mask realization.  It says the selected masked image shared representation supports more robust semantic decoding.  It does **not**, by itself, prove text-image representations are aligned.

## Results: single-modality EMA-JEPA pilots

These models are trained and evaluated on one modality only.  The same clean-to-masked, 80%-mask probe protocol is used.  `Shared–clean cosine` measures the masked student shared readout against its clean target at masked positions; `Predictor–teacher cosine` is present only for JEPA models.

### Text-only pilots

| Model | Selected layer | Count | Attribute avg. | Shared–clean cosine | Predictor–teacher cosine |
|---|---:|---:|---:|---:|---:|
| [Text diffusion-only](../models/text_diffusion.md) | 4 | 0.209 | 0.576 | 0.832 | — |
| [Text EMA-JEPA fixed](../models/text_ema_fixed.md) | 3 | 0.206 | 0.579 | 0.682 | 0.854 |
| [Text EMA-JEPA dynamic .10](../models/text_ema_r010.md) | 4 | 0.425 | 0.579 | 0.780 | 0.904 |
| [Text EMA-JEPA dynamic .25](../models/text_ema_r025.md) | 4 | 0.450 | 0.575 | 0.767 | 0.914 |
| [Text EMA-JEPA dynamic .50](../models/text_ema_r050.md) | 4 | 0.530 | 0.640 | 0.664 | 0.831 |

### Image-only pilots

| Model | Selected layer | Count | Attribute avg. | Shared–clean cosine | Predictor–teacher cosine |
|---|---:|---:|---:|---:|---:|
| [Image diffusion-only](../models/image_diffusion.md) | 2 | 0.167 | 0.500 | 0.482 | — |
| [Image EMA-JEPA fixed](../models/image_ema_fixed.md) | 3 | 0.300 | 0.609 | 0.793 | 0.919 |
| [Image EMA-JEPA dynamic .10](../models/image_ema_r010.md) | 4 | 0.370 | 0.630 | 0.787 | 0.891 |
| [Image EMA-JEPA dynamic .25](../models/image_ema_r025.md) | 4 | 0.286 | 0.616 | 0.725 | 0.886 |
| [Image EMA-JEPA dynamic .50](../models/image_ema_r050.md) | 4 | 0.321 | 0.640 | 0.775 | 0.888 |

## Results: private-LoRA readout, single-modality EMA-JEPA pilots

This uses the same completed forward passes, examples, labels, mask ratios, and exact mask positions as the shared table above.  The only change is the readout: each selected module contributes its private-LoRA update $\Delta_T$ for text or $\Delta_I$ for image, rather than $\Delta_s$.  The selected layer is now the layer with the highest **private** attribute average.

`Predictor–teacher cosine` is not applicable: the JEPA predictor and target are defined only for the shared branch.

### Text-only private branch

| Model | Selected layer | Count | Attribute avg. | Private–clean cosine | Predictor–teacher cosine |
|---|---:|---:|---:|---:|---:|
| [Text diffusion-only](../models/text_diffusion.md) | 4 | 0.236 | 0.585 | 0.738 | — |
| [Text EMA-JEPA fixed](../models/text_ema_fixed.md) | 4 | 0.244 | 0.568 | 0.728 | — |
| [Text EMA-JEPA dynamic .10](../models/text_ema_r010.md) | 4 | 0.281 | 0.563 | 0.791 | — |
| [Text EMA-JEPA dynamic .25](../models/text_ema_r025.md) | 4 | 0.319 | 0.576 | 0.871 | — |
| [Text EMA-JEPA dynamic .50](../models/text_ema_r050.md) | 4 | 0.314 | 0.564 | 0.839 | — |

### Image-only private branch

| Model | Selected layer | Count | Attribute avg. | Private–clean cosine | Predictor–teacher cosine |
|---|---:|---:|---:|---:|---:|
| [Image diffusion-only](../models/image_diffusion.md) | 3 | 0.214 | 0.518 | 0.659 | — |
| [Image EMA-JEPA fixed](../models/image_ema_fixed.md) | 4 | 0.292 | 0.577 | 0.877 | — |
| [Image EMA-JEPA dynamic .10](../models/image_ema_r010.md) | 4 | 0.384 | 0.566 | 0.890 | — |
| [Image EMA-JEPA dynamic .25](../models/image_ema_r025.md) | 2 | 0.167 | 0.525 | 0.878 | — |
| [Image EMA-JEPA dynamic .50](../models/image_ema_r050.md) | 3 | 0.167 | 0.549 | 0.898 | — |

## Implementation and raw artifacts

- Evaluation implementation: [/home/zd25e122/clevr_discrete_diffusion/evaluate_jepa_modality.py](/home/zd25e122/clevr_discrete_diffusion/evaluate_jepa_modality.py)
- Image raw results: [/home/zd25e122/clevr_discrete_diffusion/outputs/jepa_modality_eval_lora_family_image/results.json](/home/zd25e122/clevr_discrete_diffusion/outputs/jepa_modality_eval_lora_family_image/results.json)
- Text raw results: [/home/zd25e122/clevr_discrete_diffusion/outputs/jepa_modality_eval_lora_family_text/results.json](/home/zd25e122/clevr_discrete_diffusion/outputs/jepa_modality_eval_lora_family_text/results.json)
- Text-only EMA-pilot raw results: [/home/zd25e122/clevr_discrete_diffusion/outputs/jepa_modality_eval_text_ema/results.json](/home/zd25e122/clevr_discrete_diffusion/outputs/jepa_modality_eval_text_ema/results.json)
- Image-only EMA-pilot raw results: [/home/zd25e122/clevr_discrete_diffusion/outputs/jepa_modality_eval_image_ema/results.json](/home/zd25e122/clevr_discrete_diffusion/outputs/jepa_modality_eval_image_ema/results.json)
- Full per-attribute table: [/home/zd25e122/clevr_discrete_diffusion/outputs/LORA_MULTIMODAL_SEPARATE_MODALITY_ATTRIBUTE_REPORT.md](/home/zd25e122/clevr_discrete_diffusion/outputs/LORA_MULTIMODAL_SEPARATE_MODALITY_ATTRIBUTE_REPORT.md)
- Pretraining and checkpoint links: [experiment catalog](../README.md)
