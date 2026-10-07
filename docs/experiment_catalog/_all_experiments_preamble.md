# Every experiment, what it is, and what it cost

> **Type:** master index · **Status:** {total} trained runs in {families} families · **Updated:** 2026-09-24
> **Menu:** [experiment catalog](README.md) · **Regenerate:** `python3 scripts/build_all_experiments.py`

Grouped by **method family**, simplest first. Modality is a column, so a family
that exists in both text and images is read in one place.

**This page is generated.** Every number comes from a file on disk — the run's
config, its `train.log`, and the evaluation JSONs under `outputs/`. Edit
`docs/experiment_catalog/_all_experiments_preamble.md` for the prose and re-run
the generator for the tables; do not hand-edit the tables.

## Contents

{contents}

## Reading the columns

| Column | Meaning |
|---|---|
| **Mod** | text, image, or both (paired or unpaired multimodal) |
| **What it is** | the JEPA objective in full: target design, **whether an MLP predictor is used** and how many heads, EMA teacher or SIGReg, the loss, the adapter ranks and the masking |
| **From** | initialization; `scratch` means random |
| **Ep** | epochs with a saved checkpoint |
| **Own** | tokens this run consumed |
| **Total** | tokens including every stage it was built on |
| **Val** | final validation loss — *meaningless for any `no diffusion` run* |
| **d_sem / d_bind full** | [semantic effect sizes](evaluations/semantic_dprime.md) of the model as it runs; `d_sem` at `L7.residual`, `d_bind` at the model's best feature; bag-of-words control is −2.53 / 0.000 |
| **… shared** | the same, with every modality-private adapter suppressed, so only the shared route runs |
| **… private** | the mirror: the shared write zeroed, only the private branch |
| **text probe / img probe** | linear scene-fact readout at L7, [cross-modal structure](evaluations/cross_modal_structure.md) |
| **img bind** | [image binding](evaluations/image_binding.md) on content-matched pairs, chance 0.500 |

Tokens per epoch: text 2M captions × 91.7 content tokens = **183M**; images
1.2M × 384 = **461M**; multimodal sees both = **571M**; the old 90k paired set
42.8M; MS-COCO 118k captions ≈ 1.5M.

## Three traps when comparing rows

**1. Match the token budget.** The 4-epoch dense baseline (0.73B) is
undertrained as a reference: its binding effect size rises 0.244 → 0.443 between
4 and 6 epochs. A from-dense JEPA run costs 1.10B total, so it must be compared
against dense at **6** epochs. Read the **Total** column, not **Own**. An earlier
version of this catalog compared against the 4-epoch number and reported that
JEPA "more than doubles" binding; the correct figure is **+19%**.

**2. Ignore Val for `no diffusion` runs.** Nothing trains the output head, so
~8.8 is uniform guessing and 1.4–2.0 only measures how far a JEPA phase drifted
the trunk away from a frozen head. It says nothing about representation quality.

**3. Route isolation means different things in different families.** The shared
and private columns come from an actual forward pass with the other route's
write zeroed by route id, not from decomposing a mixed pass, so later blocks
also receive a branch-free input.

This is meaningful for **dense shared + private LoRA** (family 5), where the
trunk alone is a complete network: the frozen-trunk run read trunk-only
reproduces its stage-1 parent to three decimals (−0.763 / +0.293 against
−0.76 / +0.293), which is the strongest end-to-end check the evaluation code
has.

It is **not** a clean read-out for **Tri-LoRA** (families 1–2), where each
adapted linear is `shared_delta + private_delta` with no base weight: removing
either branch does not isolate a route, it breaks the layer, because the
remaining half was never trained to work alone. Plain Tri-LoRA scores +0.009
private-only and +0.019 shared-only against +0.293 intact — binding exists only
in the sum. For that family the per-module `*_shared` / `*_private` **write**
features in [semantic effect sizes](evaluations/semantic_dprime.md) remain the
meaningful comparison.

## The families

**1. Baselines** — mask tokens, predict them. Dense, or Tri-LoRA where every
adapted linear is `shared_delta + private_delta` with no frozen base (rank 384
= shared 256 + private 128, one private branch per modality).

**2. Tri-LoRA + JEPA on adapter writes** — an EMA-teacher JEPA applied to the
*adapter writes* of selected modules, averaged or layerwise across blocks,
optionally with the gradient routed end-to-end into the shared A/B tensors
("gated"), optionally with HSIC penalizing shared/private dependence. Trained
together with the diffusion loss.

**3–4. data2vec on hidden states** — predict the EMA teacher's block outputs at
masked positions, either **averaged** (one target, `t = mean_k LN(h_k^EMA)`, one
predictor on the final block) or **layerwise** (each block predicts its own
target through its own MLP head; 8 heads = 4.72M parameters). The split between
families 3 and 4 is the single most decisive result in the study: **from a
pretrained trunk it produces the best representations measured; from scratch it
collapses to a surface representation, in both modalities, at every masking
rate and both anti-collapse mechanisms tried.**

**5. SIGReg / LeJEPA** — the same layerwise hidden-state objective with the EMA
teacher, the stop-gradient and the target LayerNorm all removed, replaced by the
sliced Epps–Pulley isotropic-Gaussian test of
[LeJEPA](https://arxiv.org/abs/2511.08544) (1024 slices, 17 evaluation points),
combined as `(1 − λ)·L_pred + λ·SIGReg` at λ = 0.05. Four runs — text from
scratch, text from the stage-1 dense trunk, multimodal from the unpaired dense
trunk, multimodal from scratch. **All four collapse**, and the from-dense run
ends up *below* the bag-of-words control on `d_semantic` (−2.68). λ was not
swept; that is the obvious open question.

**6. Stage 2** — keep the dense weight as the shared route
(`y = Wx + (α/ρ_p)·B_p A_p x + b`) and rank-limit only the private branch
(128 = d/3), training diffusion + HSIC with the trunk frozen or free. Read
trunk-only, this family contains the best text model in the study.

**7. Multimodal** — one model, both modalities. **Paired** carries a caption
with its own image; **unpaired** never does (a derangement guarantees it).
Paired dense is the reference target on every cross-modal measure.

**8–9. Other corpora and pilots** — MS-COCO transfer, and short exploratory runs
(EMA decay, target fidelity, masking style). Not evaluated.

