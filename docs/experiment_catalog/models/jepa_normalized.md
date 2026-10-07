# LoRA + normalized JEPA/SIGReg

> **Type:** model page, earlier phase · **Menu:** [experiment catalog](../README.md)

**Family:** Multimodal

**Purpose:** Normalized-MSE JEPA at layers 2–4 plus per-layer SIGReg

## Artifacts

- W&B: [s6159f54](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_shared_lora_alignment/runs/s6159f54) — `finished`
- Configuration: [pretraining_unpaired_lora_middle_l2_l4_normalized_jepa_sigreg_absolute_70e.yaml](/home/zd25e122/clevr_discrete_diffusion/configs/pretraining_unpaired_lora_middle_l2_l4_normalized_jepa_sigreg_absolute_70e.yaml)
- Best checkpoint: [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_unpaired_lora_middle_l2_l4_normalized_jepa_sigreg_absolute_70e/best.pt)
- Latest/resume checkpoint: [last.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_unpaired_lora_middle_l2_l4_normalized_jepa_sigreg_absolute_70e/last.pt)
- Output directory: `/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_unpaired_lora_middle_l2_l4_normalized_jepa_sigreg_absolute_70e`

## Saved configuration summary

- `data`: `{'mode': 'unpaired', 'balanced_modalities': True}`
- `model`: `{'d_model': 384, 'n_layers': 8, 'use_modality_embeddings': False}`
- `lora`: `{'train_mode': 'lora', 'rank': 384, 'target_modules': ['qkv', 'out_proj', 'mlp.0', 'mlp.3']}`
- `train`: `{'epochs': 70, 'shared_only_epochs': 0, 'batch_size': 256, 'gradient_accumulation_steps': 1}`
- `alignment`: `{'modality_adversarial_enabled': False, 'sigreg_enabled': True, 'shared_jepa_enabled': True, 'shared_jepa_layers': [2, 3, 4], 'shared_jepa_loss': 'normalized_mse'}`

## JEPA placement

- Direct JEPA target layers: **2, 3, 4** (Transformer layer indices).
- Layers without their own direct JEPA target: **0, 1, 5, 6, 7**.
- Target: clean stop-gradient online target; student input is the masked modality.
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

where $w_{b(i)}=1/t_{b(i)}$.  `unweighting=False`; the diffusion coefficient is **1.0**.  The optimized step averages active modality sub-losses ($q\in\{T,I\}$).
### JEPA term

At direct target layers **2, 3, 4**, $J_l=\operatorname{MSE}(\sqrt d\,\widehat{P_l(s_l^{mask})},\sqrt d\,\widehat{\operatorname{sg}(s_l^{clean})})$, averaged over selected layers and exactly masked positions.  $P_l$ is the learned JEPA predictor and $\widehat{v}=v/(\lVert v\rVert_2+10^{-6})$.
- Teacher: **online clean stop-gradient student**.
- Coefficient: $\lambda_J=0.5$; warm-up: **1000** steps beginning at epoch **0**.  With local post-start step $u$, $r_J=\min(1,(u+1)/1000)$ (or $1$ when warm-up is zero).
- Per-modality coefficient overrides: text **0.1**, image **0.5**.
### SIGReg term

$$R(z)=\frac1S\sum_{j=1}^S N\sum_k w_k[(\overline{\cos(t_k u_j^\top z)}-e^{-t_k^2/2})^2+\overline{\sin(t_k u_j^\top z)}^2]$$

Random unit directions $u_j$ compare the empirical projected characteristic function with $N(0,1).$
- Applied to **2, 3, 4**; the selected layer losses are averaged.
- Coefficient $\lambda_S$: **0.002**; warm-up: **1000** steps from epoch **0**, with $r_S=\min(1,(u+1)/1000)$ (or $1$ when warm-up is zero); slices **256**, points **17**, $t_{max}$ **3**.

### Final optimized objective

$$\boxed{\mathcal L_{\mathrm{step}}=\frac1{|\mathcal Q|}\sum_{q\in\mathcal Q}\left[\mathcal L_{\mathrm{diff}}^{(q)}+\lambda_J\cdot r_J\cdot\mathcal L_{\mathrm{JEPA}}^{(q)}+\lambda_S\cdot r_S\cdot\mathcal L_{\mathrm{SIGReg}}^{(q)}\right]}$$

$\mathcal Q$ is the set of active modality sub-batches: $q\in\{T,I\}$.  This outer average is the code's `len(sub_batches)` division before gradient accumulation.

## Available evaluated values

### Text only, 80% masking

Selected shared layer: **3** (highest attribute average among L2–L4).

| Count | Attribute avg | Attributes |
|---:|---:|---|
| 0.376 | 0.634 | gray=0.538; red=0.571; blue=0.573; green=0.543; brown=0.541; purple=0.498; cyan=0.558; yellow=0.560; cube=0.649; sphere=0.652; cylinder=0.674; metal=0.757; rubber=0.709; small=0.464; large=0.585; right=0.776; behind=0.633 |

### Image only, 80% masking

Selected shared layer: **3** (highest attribute average among L2–L4).

| Count | Attribute avg | Attributes |
|---:|---:|---|
| 0.175 | 0.539 | gray=0.500; red=0.514; blue=0.513; green=0.536; brown=0.500; purple=0.500; cyan=0.500; yellow=0.505; cube=0.519; sphere=0.544; cylinder=0.511; metal=0.498; rubber=0.461; small=0.494; large=0.863; right=0.510; behind=0.500 |

## Evaluation pointers

- [Experiment catalog menu](../README.md)
- [Multimodal separate-modality attribute table](/home/zd25e122/clevr_discrete_diffusion/outputs/LORA_MULTIMODAL_SEPARATE_MODALITY_ATTRIBUTE_REPORT.md)
- [Multimodal LoRA experiment notebook](/home/zd25e122/clevr_discrete_diffusion/notebooks/multimodal_lora_experiment_record.ipynb)
