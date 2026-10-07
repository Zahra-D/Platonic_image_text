# LoRA paired

> **Type:** model page, earlier phase · **Menu:** [experiment catalog](../README.md)

**Family:** Multimodal

**Purpose:** No-base Tri-LoRA, true paired data

## Artifacts

- W&B: [30fi6whi](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_shared_lora_alignment/runs/30fi6whi) — `finished`
- Configuration: [pretraining_paired_lora_absolute_70e.yaml](/home/zd25e122/clevr_discrete_diffusion/configs/pretraining_paired_lora_absolute_70e.yaml)
- Best checkpoint: [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_paired_lora_absolute_70e/best.pt)
- Latest/resume checkpoint: [last.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_paired_lora_absolute_70e/last.pt)
- Output directory: `/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_paired_lora_absolute_70e`

## Saved configuration summary

- `data`: `{'mode': 'paired'}`
- `model`: `{'d_model': 384, 'n_layers': 8, 'use_modality_embeddings': False}`
- `lora`: `{'train_mode': 'lora', 'rank': 384, 'target_modules': ['qkv', 'out_proj', 'mlp.0', 'mlp.3']}`
- `train`: `{'epochs': 70, 'shared_only_epochs': 0, 'batch_size': 128, 'gradient_accumulation_steps': 1}`
- `alignment`: `{'modality_adversarial_enabled': False}`

## Exact forward pass and optimized objective

### Shared notation

- $x_i$ is the clean discrete token at position $i$; $\tilde x_i$ is its corrupted value (the mask token when $i\in M$).
- $p_i$ is the within-modality position; $m_i$ is the modality identifier; $\ell_i$ is the output-logit vector.
- Input: $h_i^0=E_{tok}(\tilde x_i)+E_{pos}(p_i)$.  There is **no modality embedding**.
- A mask rate $t_b\sim U(0.001,1)$ is drawn per row; each eligible token is independently included in $M$ with probability $t_b$, with at least one eligible token forced into $M$.  Full-mask probability: **0.15**.

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

### Final optimized objective

$$\boxed{\mathcal L_{\mathrm{step}}=\frac1{|\mathcal Q|}\sum_{q\in\mathcal Q}\left[\mathcal L_{\mathrm{diff}}^{(q)}\right]}$$

$\mathcal Q$ is the set of active modality sub-batches: $q\in\{T,I\}$.  This outer average is the code's `len(sub_batches)` division before gradient accumulation.

## Available evaluated values

No modality-local value is available for this model yet.

## Evaluation pointers

- [Experiment catalog menu](../README.md)
- [Multimodal separate-modality attribute table](/home/zd25e122/clevr_discrete_diffusion/outputs/LORA_MULTIMODAL_SEPARATE_MODALITY_ATTRIBUTE_REPORT.md)
- [Multimodal LoRA experiment notebook](/home/zd25e122/clevr_discrete_diffusion/notebooks/multimodal_lora_experiment_record.ipynb)
