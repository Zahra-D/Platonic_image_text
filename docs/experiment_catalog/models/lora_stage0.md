# LoRA stage 0

> **Type:** model page, earlier phase · **Menu:** [experiment catalog](../README.md)

**Family:** Multimodal

**Purpose:** Shared-only for epochs 0–9; private branches enabled at epoch 10

## Artifacts

- W&B: [smua88cz](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_shared_lora_alignment/runs/smua88cz) — `finished`
- Configuration: [pretraining_unpaired_lora_absolute_stage0_70e.yaml](/home/zd25e122/clevr_discrete_diffusion/configs/pretraining_unpaired_lora_absolute_stage0_70e.yaml)
- Best checkpoint: [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_unpaired_lora_absolute_stage0_70e/best.pt)
- Latest/resume checkpoint: [last.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_unpaired_lora_absolute_stage0_70e/last.pt)
- Output directory: `/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_unpaired_lora_absolute_stage0_70e`

## Saved configuration summary

- `data`: `{'mode': 'unpaired', 'balanced_modalities': True}`
- `model`: `{'d_model': 384, 'n_layers': 8, 'use_modality_embeddings': False}`
- `lora`: `{'train_mode': 'lora', 'rank': 384, 'target_modules': ['qkv', 'out_proj', 'mlp.0', 'mlp.3']}`
- `train`: `{'epochs': 70, 'shared_only_epochs': 10, 'batch_size': 256, 'gradient_accumulation_steps': 1}`
- `alignment`: `{'modality_adversarial_enabled': False}`

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
- Stage 0 (epochs 0–9): private $A_T,B_T,A_I,B_I$ are frozen, so $y=b_s+\Delta_s(z)$.  From epoch 10: full modality routing above is trainable.

The Transformer maps the corrupted sequence to logits: $\ell=F_\theta(\tilde x,p,m)$.

### Diffusion/reconstruction loss

$$\mathcal L_{\mathrm{diff}}^{(q)}=\frac{1}{|M_q|}\sum_{i\in M_q} w_{b(i)}\cdot\operatorname{CE}(\ell_i,x_i).$$

where $w_{b(i)}=1/t_{b(i)}$.  `unweighting=False`; the diffusion coefficient is **1.0**.  The optimized step averages active modality sub-losses ($q\in\{T,I\}$).

### Final optimized objective

$$\boxed{\mathcal L_{\mathrm{step}}=\frac1{|\mathcal Q|}\sum_{q\in\mathcal Q}\left[\mathcal L_{\mathrm{diff}}^{(q)}\right]}$$

$\mathcal Q$ is the set of active modality sub-batches: $q\in\{T,I\}$.  This outer average is the code's `len(sub_batches)` division before gradient accumulation.

## Available evaluated values

### Text only, 80% masking

Selected shared layer: **3** (highest attribute average among L2–L4).

| Count | Attribute avg | Attributes |
|---:|---:|---|
| 0.216 | 0.561 | gray=0.500; red=0.532; blue=0.605; green=0.497; brown=0.500; purple=0.500; cyan=0.503; yellow=0.500; cube=0.500; sphere=0.621; cylinder=0.581; metal=0.500; rubber=0.718; small=0.500; large=0.562; right=0.500; behind=0.658 |

### Image only, 80% masking

Selected shared layer: **4** (highest attribute average among L2–L4).

| Count | Attribute avg | Attributes |
|---:|---:|---|
| 0.402 | 0.594 | gray=0.522; red=0.690; blue=0.654; green=0.520; brown=0.496; purple=0.513; cyan=0.500; yellow=0.649; cube=0.500; sphere=0.506; cylinder=0.589; metal=0.500; rubber=0.650; small=0.693; large=0.873; right=0.524; behind=0.498 |

## Evaluation pointers

- [Experiment catalog menu](../README.md)
- [Multimodal separate-modality attribute table](/home/zd25e122/clevr_discrete_diffusion/outputs/LORA_MULTIMODAL_SEPARATE_MODALITY_ATTRIBUTE_REPORT.md)
- [Multimodal LoRA experiment notebook](/home/zd25e122/clevr_discrete_diffusion/notebooks/multimodal_lora_experiment_record.ipynb)
