# LoRA + DANN

> **Type:** model page, earlier phase · **Menu:** [experiment catalog](../README.md)

**Family:** Multimodal

**Purpose:** L2-normalized shared readout; modality adversary from epoch 0

## Artifacts

- W&B: [vn430zje](https://wandb.ai/zahra-delbari-university-of-bern/Platonic_CLEVR_shared_lora_alignment/runs/vn430zje) — `finished`
- Configuration: [pretraining_unpaired_lora_absolute_dann_70e.yaml](/home/zd25e122/clevr_discrete_diffusion/configs/pretraining_unpaired_lora_absolute_dann_70e.yaml)
- Best checkpoint: [best.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_unpaired_lora_absolute_dann_70e/best.pt)
- Latest/resume checkpoint: [last.pt](/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_unpaired_lora_absolute_dann_70e/last.pt)
- Output directory: `/home/zd25e122/clevr_discrete_diffusion/outputs/pretraining_unpaired_lora_absolute_dann_70e`

## Saved configuration summary

- `data`: `{'mode': 'unpaired', 'balanced_modalities': True}`
- `model`: `{'d_model': 384, 'n_layers': 8, 'use_modality_embeddings': False}`
- `lora`: `{'train_mode': 'lora', 'rank': 384, 'target_modules': ['qkv', 'out_proj', 'mlp.0', 'mlp.3']}`
- `train`: `{'epochs': 70, 'shared_only_epochs': 0, 'batch_size': 256, 'gradient_accumulation_steps': 1}`
- `alignment`: `{'modality_adversarial_enabled': True, 'modality_adversarial_start_epoch': 0}`

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
### DANN term

For a block $l$ and its shared adapter module $a$, let $\Delta^{(l,a)}_{s,b,j}$ be the shared adapter output for example $b$ and token $j$.  Let $P_{l,a}$ reduce its output width to $d$ when necessary (adaptive average pooling), and let $A_{b,j}\in\{0,1\}$ be the attention mask.
$$\Delta^{(l,a)}_{s,b,j}=\frac{\alpha}{\rho_s}\,B_s^{(l,a)}A_s^{(l,a)}D\!\left(z^{(l,a)}_{b,j}\right).$$
This is the **raw additive shared-LoRA update of that one linear module**.  It is not the full Transformer-block output, not the hidden state $x^l$, and not a separate forward pass in which private LoRAs are disabled or subtracted.
For example, $z^{(l,\mathrm{qkv})}=\operatorname{LN}_1(x^{l-1})$; $z^{(l,\mathrm{out\_proj})}$ is the attention output; $z^{(l,\mathrm{mlp.0})}=\operatorname{LN}_2(x^{l-1}+\operatorname{Attn}(\cdot))$; and $z^{(l,\mathrm{mlp.3})}$ is the GELU/dropout output of `mlp.0`.
Consequently, later module inputs can contain effects caused by private branches in earlier modules.  However, the recorder uses $\operatorname{stopgrad}(z^{(l,a)})$ when forming this DANN readout, so the DANN gradient cannot flow back through those inputs into private branches or earlier blocks.
$$r^{(l,a)}_{s,b}=\frac{\sum_j A_{b,j}\,P_{l,a}\!\left(\Delta^{(l,a)}_{s,b,j}\right)}{\sum_j A_{b,j}},\qquad h^l_{s,b}=\frac1{|\mathcal A_l|}\sum_{a\in\mathcal A_l}r^{(l,a)}_{s,b},\qquad s_b=\frac1L\sum_{l=0}^{L-1}h^l_{s,b}.$$
Here $\mathcal A_l=\{\mathrm{qkv},\mathrm{out\_proj},\mathrm{mlp.0},\mathrm{mlp.3}\}$ and $L=8$ in these runs.  Thus $h^l_s$ is the named per-layer shared readout, while $s$ is their global average.
$$\bar s_b=\operatorname{norm}(s_b),\qquad \mathcal L_{\mathrm{DANN}}^{(q)}=\frac1B\sum_{b=1}^B\operatorname{CE}\!\left(D\!\left(\operatorname{GRL}_{\gamma}(\bar s_b)\right),q\right).$$

There is **one** DANN cross-entropy per homogeneous text or image sub-batch, not one independent CE per layer.  Because $s$ averages all $h_s^l$, this one loss sends a direct gradient to every recorded shared adapter in every layer.
- L2 input normalization: $\bar s=s/(\lVert s\rVert_2+10^{-6})$ independently for each row.  Therefore the discriminator cannot classify the modality from the length of $s$; it can only use its direction.
- The discriminator is an MLP $D: d\rightarrow 128\rightarrow 2$ with GELU; its target $q$ is text (0) or image (1).
- Loss strength: $\lambda_D=0.1$.  DANN turns on at epoch **0**.  With $u=0,1,\ldots$ counting optimizer steps since activation, $r_D=\min(1,(u+1)/1000)$.
  At its first active step, $r_D=1/1000$ and the effective loss coefficient is $0.1/1000$; after 1000 active steps, $r_D=1$ and the coefficient is $0.1$.
- Gradient reversal: $\gamma=1$.  The discriminator parameters receive the ordinary CE gradient and learn to identify the modality.  The shared branch receives that gradient multiplied by $-1$, so it learns to make the modality harder to identify.  $\gamma$ does not reverse the discriminator's own update.
- Exact gradient path: the recorder recomputes each $\Delta_s^{(l,a)}$ from a detached adapter input.  Therefore DANN updates the shared $A_s^{(l,a)},B_s^{(l,a)}$ parameters in every recorded layer and the discriminator, but it does not backpropagate through this DANN path into earlier Transformer activations, private LoRAs, shared biases, embeddings, or the output head.

### Final optimized objective

$$\boxed{\mathcal L_{\mathrm{step}}=\frac1{|\mathcal Q|}\sum_{q\in\mathcal Q}\left[\mathcal L_{\mathrm{diff}}^{(q)}+\lambda_D\cdot r_D\cdot\mathcal L_{\mathrm{DANN}}^{(q)}\right]}$$

$\mathcal Q$ is the set of active modality sub-batches: $q\in\{T,I\}$.  This outer average is the code's `len(sub_batches)` division before gradient accumulation.

## Available evaluated values

### Text only, 80% masking

Selected shared layer: **3** (highest attribute average among L2–L4).

| Count | Attribute avg | Attributes |
|---:|---:|---|
| 0.271 | 0.563 | gray=0.547; red=0.561; blue=0.500; green=0.516; brown=0.543; purple=0.504; cyan=0.516; yellow=0.567; cube=0.500; sphere=0.544; cylinder=0.500; metal=0.543; rubber=0.556; small=0.550; large=0.500; right=0.720; behind=0.671 |

### Image only, 80% masking

Selected shared layer: **4** (highest attribute average among L2–L4).

| Count | Attribute avg | Attributes |
|---:|---:|---|
| 0.322 | 0.564 | gray=0.498; red=0.598; blue=0.569; green=0.591; brown=0.500; purple=0.500; cyan=0.495; yellow=0.536; cube=0.533; sphere=0.502; cylinder=0.622; metal=0.500; rubber=0.500; small=0.499; large=0.952; right=0.500; behind=0.510 |

## Evaluation pointers

- [Experiment catalog menu](../README.md)
- [Multimodal separate-modality attribute table](/home/zd25e122/clevr_discrete_diffusion/outputs/LORA_MULTIMODAL_SEPARATE_MODALITY_ATTRIBUTE_REPORT.md)
- [Multimodal LoRA experiment notebook](/home/zd25e122/clevr_discrete_diffusion/notebooks/multimodal_lora_experiment_record.ipynb)
