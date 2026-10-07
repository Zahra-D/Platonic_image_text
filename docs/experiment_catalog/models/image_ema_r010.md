# Image EMA-JEPA dynamic .10

> **Type:** model page, earlier phase · **Menu:** [experiment catalog](../README.md)

**Family:** Image-only pilot

**Purpose:** EMA teacher; dynamic JEPA gradient target .10

## Artifacts

- W&B: [6q6cywma](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_single_modality_ema_jepa/runs/6q6cywma) — `finished`
- Configuration: [image_ema_jepa_pilot.yaml](/home/zd25e122/clevr_discrete_diffusion/configs/image_ema_jepa_pilot.yaml)
- Best checkpoint: [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/image_ema_jepa_dynamic_r010/best.pt)
- Latest/resume checkpoint: [last.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/image_ema_jepa_dynamic_r010/last.pt)
- Output directory: `/home/zd25e122/clevr_discrete_diffusion/outputs/image_ema_jepa_dynamic_r010`

## Saved configuration summary

- `data`: `{'mode': 'unpaired', 'balanced_modalities': True}`
- `model`: `{'d_model': 384, 'n_layers': 8, 'use_modality_embeddings': False}`
- `lora`: `{'train_mode': 'lora', 'rank': 384, 'target_modules': ['qkv', 'out_proj', 'mlp.0', 'mlp.3']}`
- `train`: `{'epochs': 70, 'shared_only_epochs': 0, 'batch_size': 64, 'gradient_accumulation_steps': 4}`
- `alignment`: `{'modality_adversarial_enabled': False, 'sigreg_enabled': False, 'shared_jepa_enabled': True, 'shared_jepa_layers': [2, 3, 4], 'shared_jepa_loss': 'normalized_mse', 'shared_jepa_ema_enabled': True}`

## JEPA placement

- Direct JEPA target layers: **2, 3, 4** (Transformer layer indices).
- Layers without their own direct JEPA target: **0, 1, 5, 6, 7**.
- Target: clean EMA teacher; student input is the masked modality.
- Loss: `normalized_mse` on the shared representation, through a JEPA predictor.
- The regular diffusion/reconstruction loss trains all layers.  A JEPA loss at a selected layer also backpropagates through earlier layers that produced its representation, but those earlier layers have no separate JEPA target unless listed above.

## Exact forward pass and optimized objective

### Shared notation

- $x_i$ is the clean discrete token at position $i$; $\tilde x_i$ is its corrupted value (the mask token when $i\in M$).
- $p_i$ is the within-modality position; $m_i$ is the modality identifier; $\ell_i$ is the output-logit vector.
- Input: $h_i^0=E_{tok}(\tilde x_i)+E_{pos}(p_i)$.  There is **no modality embedding**.
- A mask rate $t_b\sim U(0.001,1)$ is drawn per row; each eligible token is independently included in $M$ with probability $t_b$, with at least one eligible token forced into $M$.  Full-mask probability: **0**.

### Forward pass

Every selected linear map has **no frozen dense base weight**.  For branch $r\in\{s,T,I\}$:

$$\Delta_r^{(m)}(z)=\frac{\alpha}{\rho_s}\cdot B_r^{(m)}\cdot A_r^{(m)}\cdot D(z),\qquad \rho_s=\max(1,\lfloor2R/3\rfloor).$$

Here $D$ is adapter dropout.  Its configured probability is **0**, so $D(z)=z$ in this run.
For ordinary text and image routing, respectively, $y_T=b_s+\Delta_s(z)+\Delta_T(z)$ and $y_I=b_s+\Delta_s(z)+\Delta_I(z)$.  $b_s$ is a learned shared bias; there is no $Wz$ term.
- Requested rank $R$: **384**; shared rank $\rho_s$: **256**; text-private rank: **128**; image-private rank: **128**; $\alpha$: **256**; adapter dropout: **0**.
- Thus the shared and private branches do **not** have equal rank.  The requested rank determines roughly two-thirds shared and one-third **for each** private branch; consequently, the three branch ranks together sum to roughly $4R/3$, not $R$.
- Replaced linear modules: `qkv, out_proj, mlp.0, mlp.3`.

The Transformer maps the corrupted sequence to logits: $\ell=F_\theta(\tilde x,p,m)$.

### Diffusion/reconstruction loss

$$\mathcal L_{\mathrm{diff}}^{(q)}=\frac{1}{|M_q|}\sum_{i\in M_q} w_{b(i)}\cdot\operatorname{CE}(\ell_i,x_i).$$

where $w_{b(i)}=1/t_{b(i)}$.  `unweighting=False`; the diffusion coefficient is **1.0**.  The optimized step averages active modality sub-losses ($q=I$).
### JEPA term

At direct target layers **2, 3, 4**, $J_l=\operatorname{MSE}(\sqrt d\,\widehat{P_l(s_l^{mask})},\sqrt d\,\widehat{\operatorname{sg}(s_l^{clean})})$, averaged over selected layers and exactly masked positions.  $P_l$ is the learned JEPA predictor and $\widehat{v}=v/(\lVert v\rVert_2+10^{-6})$.
- Teacher: **EMA teacher** (EMA decay **0.999**).
- Coefficient: $\lambda_J=\operatorname{clip}(r\,\operatorname{EMA}\|g_{diff}\|/\operatorname{EMA}\|g_J\|, 0.0001, 10)$; warm-up: **5000** steps beginning at epoch **0**.  With local post-start step $u$, $r_J=\min(1,(u+1)/5000)$ (or $1$ when warm-up is zero).
- Dynamic-gradient target ratio $r$: **0.1**; coefficient EMA decay: **0.95**.

### Final optimized objective

$$\boxed{\mathcal L_{\mathrm{step}}=\frac1{|\mathcal Q|}\sum_{q\in\mathcal Q}\left[\mathcal L_{\mathrm{diff}}^{(q)}+\lambda_J\cdot r_J\cdot\mathcal L_{\mathrm{JEPA}}^{(q)}\right]}$$

$\mathcal Q$ is the set of active modality sub-batches: $q=I$.  This outer average is the code's `len(sub_batches)` division before gradient accumulation.

## Available evaluated values

### Image EMA pilot only, 80% masking

Selected shared layer: **4** (highest attribute average among L2–L4).

| Count | Attribute avg | Attributes |
|---:|---:|---|
| 0.370 | 0.630 | gray=0.538; red=0.613; blue=0.686; green=0.683; brown=0.487; purple=0.526; cyan=0.593; yellow=0.678; cube=0.602; sphere=0.550; cylinder=0.590; metal=0.580; rubber=0.613; small=0.787; large=0.964; right=0.496; behind=0.500 |

## Causal shared/private swap result

Image-only, layers 2–4; clean source shared deltas replace the masked target's selected shared deltas while its image-private route remains active.

| Self NLL vs B | Swapped NLL vs A | Swapped NLL vs B | Source advantage | Source win rate |
|---:|---:|---:|---:|---:|
| 3.541 | 3.541 | 18.653 | 15.112 | 0.812 |

The parser's count accuracy is modest (0.434), so see the complete protocol and limitation before interpreting this as a final semantic result.

- [Detailed causal-swap protocol and comparison](../evaluations/causal_shared_private_swap.md)

## Evaluation pointers

- [Experiment catalog menu](../README.md)
- [Multimodal separate-modality attribute table](/home/zd25e122/clevr_discrete_diffusion/outputs/LORA_MULTIMODAL_SEPARATE_MODALITY_ATTRIBUTE_REPORT.md)
- [Multimodal LoRA experiment notebook](/home/zd25e122/clevr_discrete_diffusion/notebooks/multimodal_lora_experiment_record.ipynb)
