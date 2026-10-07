# Evaluations: what each one measures, how, and how to read it

This file covers every evaluation used to compare the dense diffusion, I-JEPA and
LeJEPA models in this repo. That is nine representation evaluations plus the
diagnostics printed in training logs. Each section gives the question, the exact
computation (with `file:line` where useful), the output format, how to read the
numbers, a worked example with **real inputs and real results from `outputs/`**,
the caveats, and the command.

Model labels used in the examples: `dense13` is the dense image diffusion model
after 13 epochs; `ijepa13` and `lejepa13` are I-JEPA and token-level LeJEPA
branched from dense epoch 12 and trained one more epoch; `lejepaP21` is the
paper-style LeJEPA (multi-crop + projector) text model at 21 epochs. Text labels
count epochs the same way, from the dense text epoch-20 checkpoint.

---

## 0. At a glance

| # | Evaluation (script) | Modality | Question | Headline numbers | Chance / floor | What good looks like | Unaffected by the cone? |
|---|---|---|---|---|---|---|---|
| 1 | Cross-modal structure (`evaluate_cross_modal_structure.py`) | text + image | How much binding content is in each layer? Is the geometry organised by the scene? Do text and image models agree? | `probe_accuracy`, `rsa_scene`, `mean_pairwise_cosine`, `effective_rank`, `cka`, `rsa_cross` | probe 0.50; RSA 0; CKA ≈0.14 at `embedding` | text probe 0.85–0.97, image 0.70–0.84; CKA 0.3–0.42 | probe, CKA, rank: **yes**. RSA, mean cos, rsa_cross: **no** |
| 2 | Cross-modal retrieval (`evaluate_cross_modal_retrieval.py`) | text × image | Can a linear map find an image's own caption among 1000? | ridge R@1 and MRR (i→t, t→i), Procrustes, shuffled control | R@1 0.001, MRR 0.0075; random encoders reach MRR 0.07–0.15 | R@1 0.5–0.7, MRR 0.63–0.79 | **yes** (centred) |
| 3 | Layer-wise probe suite (`evaluate_probe_suite.py`) | image or text | Can a linear readout recover counts and colour–shape conjunctions, and how fast does that fall with masking? | count R², conj mAP, per t | random-init conj mAP 0.39 (image), 0.61 (text) | image conj mAP 0.6–0.74; text 0.97–0.98 | **yes** |
| 4 | Scene-graph retrieval (`evaluate_scene_retrieval.py`) | image or text | Does representation similarity rank scene pairs like scene-graph similarity? | Spearman ρ_bag, ρ_conj; p@10 | ρ 0; p@10 0.004 | image ρ_conj 0.12–0.20; text ≈0.3 | **no** (uncentred cosine) |
| 5 | Image binding probe (`evaluate_image_binding.py`) | image | Two scenes with the same attributes bound differently: can a probe tell which facts belong to which? | binding accuracy % (pairwise) **and strict binding %**, clean-probe balanced acc. | 50% (pairwise); `embedding` floor ≈65% | 80–90% | **yes** (z-scored) |
| 6 | Image triples d′ (`evaluate_image_triples.py`) | image | From the same new camera, is the anchor closer to the correct scene than to the colour-swapped one? | d′ raw and **centred**, preference rate | 0 and 0.5; `embedding` ≈0.09 | centred d′ 0.3–0.5 | centred: **yes**; raw: partly |
| 7 | Image semantic sensitivity (`evaluate_semantic_sensitivity.py`) | image | Does the representation move more for a colour swap than for a camera/light change? | `S` = d_s / d_n with **d_s and d_n reported**, in **raw, centred and z-scored** geometry; median ratio, fraction | 1 is "equal", but it is not reachable here; random init 0.07 | higher | raw: partly; centred / z: **yes** |
| 8 | Text sensitivity (`evaluate_text_sensitivity.py`) | text | Does it move more for a binding swap (`S_bind`) or a new scene (`S_cont`) than for a rewording? | `S_bind`, `S_cont` with their distances, in **raw, centred and z-scored** geometry | bag-of-words: `S_bind` 0, `S_cont` 0.91 | `S_cont` > 1; higher `S_bind` | raw: partly; centred / z: **yes** |
| 9b | **JEPA prediction check** (`evaluate_jepa_prediction.py`) | JEPA checkpoints | Does the JEPA predict *scene-specific* targets, or only the shared direction? | C₊ (own target), C₋ (other scene, same position), Δ = C₊ − C₋, raw and centred, per layer | Δ ≈ 0 = shortcut | centred Δ 0.3–0.7 | centred: **yes** |
| 9 | Suite report (`scripts/summarize_eval_suite.py`) | — | Puts 3, 4, 7, 8 and 2 in one markdown view | inherits the source metrics | — | — | inherits |
| 10 | Training-log diagnostics (`train_multimodal.py`, `shared_jepa.py`, `sigreg.py`, `lejepa_views.py`) | — | Is training healthy, and is it collapsing or shortcutting? | val loss, `*_acc`, `cos`, `target_spread`, `sigreg`, `pooled_cos`, `inv`, `emb_cos`, `proj_cos` | see §3 | see §3 | mixed |

**Which eval answers which question**

| You want to know… | Use |
|---|---|
| Is scene content (binding) linearly decodable? | 1 (probe), 3 (conj mAP), 5 (binding %) |
| Is the geometry organised by scene content? | 1 (`rsa_scene`), 4 (ρ_conj) |
| Is the representation invariant to viewpoint/wording but sensitive to meaning? | 6, 7 (image), 8 (text) |
| Do text and image models converge on shared structure? | 1 (`cka`, `rsa_cross`), 2 (retrieval) |
| Is the representation collapsing or sitting in a cone? | 1 (`mean_pairwise_cosine`, `effective_rank`), §3 (`target_spread`, `pooled_cos`) |

**Known issues** (details in §4). Three of them change how existing results should be read:

1. **Relations are never tested** in evals 1, 2 and 5. The code reads `world["relations"]`, but the manifests store CLEVR's `world["relationships"]`.
2. The image I-JEPA **sweep ranking** looks for the wrong JSON key, so its phase 2 never ran.
3. **`best.pt` of JEPA runs is meaningless**: it is selected by a diffusion loss those runs do not train. Use `epoch_*.pt`.

---

## 1. Shared machinery (used by most evals)

### 1.1 Readouts

Every representation eval runs a frozen model on **clean** input, unless the eval
masks tokens on purpose, and records these readouts:

| Readout | Hook | What it is |
|---|---|---|
| `embedding` | forward pre-hook on `blocks[0]` | token + position (+ modality) embedding |
| `Lk` (k = 0…7) | output of block k | residual stream after block k; `L7` is taken **before** `norm_out` |
| `Lk.attn_out` | output of `blocks[k].attn.out_proj` | what attention writes into the residual |
| `Lk.mlp_out` | output of `blocks[k].mlp[3]` | what the FFN writes into the residual |

That gives 25 readouts. `evaluate_cross_modal_structure.py --token-readouts` additionally reads, for every block output, the hidden state at BOS (`@bos`), EOS (`@eos`), the `[TEXT]`/`[IMAGE]` token (`@mod`) and the mean over all tokens including specials (`@all`). Tested on dense, trunk and JEPA models (Oct 2026): special tokens carry less than the content mean (text L7 probe 93.0% mean vs 89.2% `[TEXT]`, 86.3% EOS, 81.5% BOS; image BOS/EOS ≈ 65%, the floor), so mean pooling stays the default. Retrieval (2) and the probe suite (3) drop `attn_out`
and use 17. Text sensitivity (8) uses its own decomposition, with 42 readouts:
`.residual`, `.writes`, `.attn_out`, `.mlp_out` and `.attn_plus_residual` per block.

**Pooling.** Each readout is mean-pooled over the **content** tokens only
(`eligible_mask`). A sequence is `[TEXT|IMAGE] BOS content… EOS`; the modality
token, BOS, EOS and padding are excluded. Pooling is done in float64, then
**L2-normalised** and stored as float32. Nothing is centred at this stage.

### 1.2 Random-init baseline

A checkpoint written as `RANDOM:<path>` builds the same architecture with fresh
weights (seed 0); see `evaluate_probe_suite.py:25-47`. Always compare against it.
Several metrics sit far above "chance" even for a random network.

### 1.3 Scene facts used as labels

`binding_swap_captions.all_conjunction_labels` defines 90 candidate facts:
- 72 object conjunctions `(object, attr A, value a, attr B, value b)`, e.g.
  `(object, color, cyan, material, metal)` = "there is a cyan metal object";
- 18 relation facts `(relation, subject, kind, target)`.

A fact is kept if it has at least 50 positive and 50 negative scenes.
**In practice all 72 kept facts are object conjunctions.** The relation facts are
empty because of issue §4.1. A typical scene has about 24 true facts, and each
fact is true in about 33% of scenes.

The probe suite (3) and scene retrieval (4) use their own targets: counts per
attribute value, and 24 (colour, shape) conjunctions.

### 1.4 The shared logistic probe (`evaluate_binding_swap.py:196-216`)

Used by evals 1 and 5. Each step below happens once per readout:
- **Features:** z-scored with the training-split statistics.
- **Model:** multi-label logistic regression.
- **Loss:** BCE with `pos_weight = (1−p)/p`, plus a 1e-4 · ΣW² penalty.
- **Fit:** full-batch Adam, lr 0.05, 300 steps.
- **Prediction:** logit > 0.
- **Score:** **balanced accuracy** ½(TPR + TNR), averaged over facts.

The regularisation is fixed, not tuned.

### 1.5 "Best readout" is optimistic

Most log lines print a "best" readout chosen **on the same data it is scored on**,
out of 17–42 readouts. Use it to see *where* information lives. When comparing
models, prefer a fixed readout (e.g. L7) or a paired test.

### 1.6 Why some metrics are affected by the cone and others are not

All models here have pooled features in a narrow cone: mean pairwise cosine is
0.90–1.00 for dense and I-JEPA. The cone is mostly a shared mean direction.

- **Not affected:** anything that subtracts the mean first. This covers centred
  linear CKA, z-scored probes, ridge with centring, effective rank (computed after
  centring) and the centred triples d′.
- **Affected:** anything using the raw cosine `1 − cos` of uncentred vectors. This
  covers `rsa_scene`, `rsa_cross`, scene retrieval, the sensitivity ratios `S`, and
  `mean_pairwise_cosine` itself. Deep in a cone, all distances shrink to about 1e-4,
  and only the off-mean components matter.

So a coned model (dense) and a spread model (LeJEPA) are being compared on different
geometries. Partial exception: features are L2-normalised *before* any centring.

---

## 2. Representation evaluations

### 2.1 Cross-modal structure (`evaluate_cross_modal_structure.py`)

**Question.** At each readout:
- How much binding-dependent scene content does a frozen model hold?
- Is its geometry organised by that content?
- Do a text model and an image model, trained separately, have similar geometry on the same scenes?

**Data.**
- **Scenes:** the first `--num-scenes` (4000) rows of `val_image_only.jsonl`, with the
  VQ grids from `--image-cache` `val_tokens.pt`, matched **by row**. There is no ID check.
- **Captions:** each scene gets one caption rendered by `binding_swap_captions`, with a
  per-scene rng `Random(seed·1000003 + pos)` and seed 20260922.
- **Probe split:** trains on scenes 0–2999 and tests on 3000–3999, in manifest order.
- **RSA pairs:** 200 000 random scene pairs; dropping self-pairs leaves 199 951.

**Computation.**
1. `probe_accuracy`: the shared probe (§1.4) on the 72 facts, balanced accuracy on the test scenes.
2. `scene_distance(i, j)` = |F_i Δ F_j|, the number of facts true in exactly one of the two scenes.
3. `rsa_scene` = Spearman(1 − cos(z_i, z_j), scene_distance) over the pairs. Ties get
   ordinal ranks (double argsort), which slightly attenuates ρ.
4. `mean_pairwise_cosine` = mean cos(z_i, z_j) over the same pairs. This is the cone meter.
5. `effective_rank` = exp(−Σ p_k log p_k), where p_k = σ_k²/Σσ² of the **centred** 4000×384 matrix.
6. For every text × image pair, at matching readout names:
   - `cka` = ‖Yᶜᵀ Xᶜ‖²_F / (‖Xᶜᵀ Xᶜ‖_F · ‖Yᶜᵀ Yᶜ‖_F): linear, **centred** CKA over the 4000 scenes;
   - `rsa_cross` = Spearman(1 − cos_text, 1 − cos_image) over the pairs;
   - each side's effective rank.

**Outputs.**
- `<out>/<label>.json`, of the form `{"protocol": {...}, "metrics": {readout: {probe_accuracy, rsa_scene, mean_pairwise_cosine, effective_rank}}}`.
- `<out>/cross_modal.json`, of the form `{"TEXT|IMAGE": {readout: {cka, rsa_cross, effective_rank_text, effective_rank_image}}}`. It is merged into an existing file.

Log lines:
```
scenes: 4000, labels kept: 72, rsa pairs: 199951
dense13_IMAGE [image]: L7 probe 81.4% rsa +0.111 | best rsa L5.mlp_out +0.163 (probe 79.4%)
cross dense22_TEXT vs dense13_IMAGE: L7 cka 0.215 rsa +0.223 | best cka L4 0.344
```

**How to read it.**
- **Probe:** 0.5 is chance and ≈0.55 is "nothing learned". Text models reach 0.85–0.97, image models 0.70–0.84.
- **`rsa_scene`:** 0.1–0.4 is typical.
- **`mean_pairwise_cosine`:** above 0.9 means a strong cone.
- **CKA:** about 0.14 already at `embedding`; the best trained pairs reach 0.3–0.42.

**Example.**
*Input (scene 0, 4 objects).* Caption: "The picture shows 4 objects: a green
matte-textured rubber sphere that looks large, a large cyan glossy metal tube, a green
glossy metal box that looks smallish, plus a smallish red matte-textured rubber tube."
It has 24 true facts, among them `(object,color,cyan,material,metal)`,
`(object,color,cyan,shape,cylinder)`, `(object,color,green,material,rubber)` and
`(object,color,green,shape,cube)`.

*Result* (`outputs/lejepa_early_eval/structure/`, L7):

| model | probe | rsa_scene | mean cos | eff. rank |
|---|---|---|---|---|
| dense13_IMAGE | 0.814 | 0.111 | 0.965 | 11.3 |
| ijepa13_IMAGE | 0.817 | 0.153 | 0.904 | 24.4 |
| lejepa13_IMAGE (token-level LeJEPA) | **0.554** | 0.039 | **0.017** | 21.7 |
| dense22_TEXT | 0.928 | 0.096 | 0.91 | 22 |
| lejepaP21_TEXT (paper LeJEPA) | 0.835 | 0.045 | 0.96 | 19 |

Cross-modal L7 CKA:

| pair | L7 CKA |
|---|---|
| dense22 \| dense13 | 0.215 (best L4: 0.344) |
| ijepa22 \| ijepa13 | 0.392 |
| lejepa22 \| lejepa13 | 0.017 |
| lejepaP21 \| dense13 | 0.249 |

*Reading:*
- **Token-level LeJEPA** removed the cone (cos ≈ 0) but also the content: the probe is near chance and CKA ≈ 0.
- **I-JEPA** keeps the content and raises cross-modal CKA.
- **Paper LeJEPA** keeps its cone, and its probe drops to the input-embedding level (≈84–85%).

**Caveats.**
- No relation facts or relation sentences (§4.1).
- CKA has no permutation baseline and is dominated by the top principal components.
- `rsa_scene`, `rsa_cross` and `mean_pairwise_cosine` are affected by the cone (§1.6).
- `protocol.train_mode` says "dense" even for JEPA checkpoints.

**Command** (`scripts/run_cka_matched.sh`):
```bash
python3 evaluate_cross_modal_structure.py \
  --text-checkpoint  dense26_TEXT=outputs/text_dense_diffusion_2m_40e_continued/epoch_025.pt \
  --text-checkpoint  ijepa26_TEXT=outputs/text_ijepa_w4_8_from_dense_ep20_2m_6e/epoch_005.pt \
  --image-checkpoint dense16_IMAGE=outputs/image_dense_diffusion_2_5m_40e/epoch_015.pt \
  --image-checkpoint ijepa16_IMAGE=outputs/image_ijepa_sweep_blk_s15_t45/epoch_003.pt \
  --image-cache outputs/image_only_2_5m_token_cache/val_tokens.pt \
  --num-scenes 4000 --seed 20260922 --output-dir outputs/cka_matched
```

---

### 2.2 Cross-modal retrieval (`evaluate_cross_modal_retrieval.py`)

**Question.** Can one linear map, fitted between two frozen encoders, put an image and
its own caption close enough to retrieve one from the other among 1000 candidates?

**Data.**
- **Scenes:** the same loader and captions as 2.1, but 8000 scenes; the first 4000 are the 2.1 scenes.
- **Split:** `default_rng(seed).permutation(8000)` gives 6000 train, 1000 val and 1000 test.
- **Readouts:** 17 per side (`embedding`, `L0–L7`, `L0–L7.mlp_out`), so 289 text × image cells.

**Computation.**
1. Centre each modality with the mean of the fit rows, and apply that same mean to val and test.
2. **Ridge, image→text:** W = (XᵀX + α·s̄·I)⁻¹XᵀY, where s̄ is the mean eigenvalue of XᵀX and α ∈ {1e-3, 1e-2, 0.1, 1, 10}.
3. **Scoring:** s_ij = cos(x_iW, y_j). The rank of the true pair = 1 + #{j : s_ij > s_ii}.
   - i→t reads rows and t→i reads columns.
   - R@k = mean(rank ≤ k); MRR = mean(1/rank).
4. **Selection:** for every cell, fit on train and score on val by the mean of the
   i→t and t→i MRR. Pick the best α per cell, then the best cell.
5. Refit the chosen cell on train+val (7000) and report on **test**, with a 1000× bootstrap 95% CI over queries.
6. **Extras:**
   - **Procrustes** (an orthogonal map from SVD of XᵀY), selected separately;
   - the **diagonal** Lk\|Lk cells, with CKA;
   - a **sample-efficiency** curve (100 … 6000 training pairs);
   - a **shuffled control** (training targets permuted);
   - an **identity map**, which is only meaningful for a shared-trunk model.

**Outputs.**
- `<out>/features/<label>.npz`: cached features.
- `<out>/results/<TEXT>__<IMAGE>.json`, with keys `grid`, `ridge` (readouts, α, `cka_test`, both directions with `*_ci95`), `procrustes`, `diagonal`, `sample_efficiency`, `shuffled_control`, `identity_map` and `protocol.chance`.

Log line:
```
dense22_TEXT x dense13_IMAGE: ridge L5|L7 i->t R@1 0.537 MRR 0.644 | t->i R@1 0.664 MRR 0.758 | procrustes i->t MRR 0.192 | shuffled MRR 0.0096
```

**How to read it.**
- The shuffled control must sit at chance (MRR ≈ 0.0075).
- Compare against the random-encoder pairs, not against chance:
  - random × random: MRR 0.067;
  - random text × dense image: MRR 0.149.
- Good pairs reach R@1 0.5–0.7.
- If ridge is far above Procrustes (0.64 vs 0.19), the two spaces are linearly related but not by a rotation.

**Example.**
*Input.* Test scene 4, a five-object scene. The query is its VQ grid, encoded by
dense13 at L7, pooled, centred and multiplied by W. It is compared by cosine against
all 1000 centred dense22 L5 caption vectors, and the target is its own caption:
"You can see five objects here: a cube that is sizable, red, and rubber; a tube that
is smallish, grey, and metal; …".

*Result* (`outputs/lejepa_early_eval/xret*.log`):

| text × image | chosen cell | i→t R@1 / MRR | t→i R@1 |
|---|---|---|---|
| dense22 × dense13 | L5\|L7 | 0.537 / 0.644 [CI 0.617–0.668] | 0.664 |
| ijepa22 × ijepa13 | L6\|L7 | 0.525 / 0.635 | 0.707 |
| lejepa22 × lejepa13 | embedding\|embedding | **0.013** / 0.043 | 0.018 |
| lejepaP21 × dense13 | embedding\|L7 | 0.087 / 0.168 | 0.144 |

*Reading:* for both LeJEPA variants, the best cell is the **input embedding**: the
transformer layers carry no content that can be aligned. The paper-LeJEPA text
model is at the level of a random-init text encoder.

**Caveats.**
- α = 1e-3, the edge of the grid, is chosen for dense pairs, so the grid may be too narrow.
- A 384×384 map gives random encoders R@1 of 0.03–0.08.
- Captions contain no relations (§4.1), so retrieval can be solved by matching object inventories.
- Features and results are cached **by label** and reused whenever the file exists. Reusing a label for a different checkpoint silently returns stale numbers.
- The code comment says ties favour the distractor; the strict `>` actually favours the true pair.

**Command** (`scripts/run_lejepa_early_eval.sh`):
```bash
python3 evaluate_cross_modal_retrieval.py --num-scenes 8000 --test 1000 --val 1000 \
  --output-dir outputs/lejepa_early_eval/xret \
  --text-checkpoint dense22_TEXT=$TD22 --image-checkpoint dense13_IMAGE=$ID \
  --pair "dense22_TEXT|dense13_IMAGE"
```

---

### 2.3 Layer-wise probe suite (`evaluate_probe_suite.py`)

**Question.** How well can a linear readout recover ground-truth scene structure from
each frozen layer, and how fast does that fall off as more of the input is masked?

**Data.**
- **Scenes:** the first 6000 val scenes.
  - Image: `val_image_only.jsonl` + `image_only_2_5m_token_cache/val_tokens.pt`.
  - Text: `val_text_only_human.jsonl`, field `caption_human`.
- **Split:** one permutation (seed 20261005) gives 3750 fit, 750 val (to choose the penalty) and 1500 test. It is the same for every model and every t.
- **Corruption:** each content token is replaced by `[MASK]` i.i.d. with probability t ∈ {0, 0.25, 0.5, 0.75}. The masks are re-seeded, so every model sees identical masks.

**Targets** (lines 56-79):
- `count`: number of objects.
- `color_count` (8 values), `shape_count` (3), `size_count` (2), `material_count` (2): objects per attribute value.
- `conj` (24): presence of each (colour, shape) pair. This is the binding test.
- `conj_count` (24): count of each pair.

**Computation.**
1. Ridge regression in the eigenbasis of the centred fit design, with penalty α·mean(eigenvalue) and α ∈ {1e-4 … 10}.
2. Choose α on val: by R² for counts, by mAP for `conj`.
3. Refit on fit+val and score on test:
   - **R²** per column, averaged (MAE also stored);
   - **AP** per label, dropping labels with no positive test scene, then averaged into **mAP**.

**Outputs.**
- `<out>/<label>.json`, e.g. `metrics["t=0.0"]["L7.mlp_out"]["conj"] = {mAP, per_label, alpha}`. Count tasks store `{r2, mae, per_label, alpha}`.
- Feature cache in `features/<label>_t<t>.npz`.

**How to read it.**
- Compare against random init, not chance. Chance mAP for `conj` ≈ label prevalence ≈ 0.185.
- In text, counts are near ceiling even at random init (count R² 0.986 at `embedding`). So `conj` and `conj_count` are the informative tasks.

**Example** (`outputs/probe_suite/`), conj mAP at t = 0:

| model | best conj mAP | at t = 0.25 / 0.5 / 0.75 |
|---|---|---|
| image random_init | 0.393 @L2 | — |
| image dense_ep25 | 0.609 @L7 | — |
| image I-JEPA s15_t45 | **0.735** @L7.mlp_out (α = 0.01) | 0.701 / 0.634 / 0.497 |
| text random_init | 0.613 | — |
| text dense_ep41 | 0.967 @L5.mlp_out | — / — / 0.377 |
| text I-JEPA from_ep20 | **0.982** @L5.mlp_out | — / — / 0.434 |

**Caveats.**
- The JEPA models were trained on block (image) or window (text) masks, not i.i.d. masks, so the noise curve mixes robustness with distribution shift.
- An existing JSON makes the script skip that label, and cached `.npz` files are reused without checking the checkpoint.
- The module docstring is stale: the attribute tasks are count R², not existence mAP.

**Command** (`scripts/run_probes_tuned.sh` / `run_probe_suite_all.sh`):
```bash
python3 evaluate_probe_suite.py --modality image --num-scenes 6000 \
  --noise-levels 0.0 0.25 0.5 0.75 --output-dir outputs/probe_suite/image \
  --checkpoint "random_init=RANDOM:outputs/image_dense_diffusion_2_5m_40e/epoch_011.pt" \
  --checkpoint "dense_ep12=outputs/image_dense_diffusion_2_5m_40e/epoch_011.pt"
# text: --modality text --manifest .../platonic_text_only_v1_1m/val_text_only_human.jsonl
```

---

### 2.4 Scene-graph retrieval (`evaluate_scene_retrieval.py`)

**Question.** Does representation similarity rank scene pairs the way ground-truth scene
similarity does? There are two versions: binding-blind (`bag`) and binding-aware (`conj`).

**Data.**
- **Scenes:** the first 2500 val scenes, clean input. There is no fitting and no split.
- **Pairs:** 200 000 random pairs, of which 199 930 have i ≠ j.

**Computation.**
1. Scene vectors, each L2-normalised; S is the cosine between them.
   - `bag`: 15 counts (8 colours, 3 shapes, 2 sizes, 2 materials).
   - `conj`: 24 (colour, shape) counts.
2. R = cosine between the representation vectors (uncentred).
3. ρ = Spearman between R[i, j] and S[i, j] over the pairs.
4. p@10 = for each scene, |top-10 by R ∩ top-10 by S| / 10, averaged over scenes.

**Outputs.** `<label>.json` with `metrics[readout] = {spearman_bag, spearman_conj, p@10_bag, p@10_conj}`.
The log prints the best conj ρ with its readout.

**How to read it.** If ρ_conj is much lower than ρ_bag, the geometry tracks *which
ingredients are present*, not *which attribute belongs to which object*.

**Example** (`outputs/scene_retrieval/`):

| model | best ρ_conj | ρ_bag |
|---|---|---|
| image random_init | 0.060 | — |
| image dense_ep25 | 0.121 | — |
| image I-JEPA s30_t60 | 0.198 @L7 | 0.352 |
| text dense_ep20 | 0.298 @L6 | 0.629 |
| text I-JEPA from_ep20 | 0.285 @L4.mlp_out | — |

**Caveats.**
- The uncentred cosine is affected by the cone (§1.6), and the scene vectors have their own cone (mean S_bag = 0.78).
- S_conj has many ties (11% of pairs are exactly 0), which attenuates ρ.
- The best readout is chosen on the reported data.
- Duplicate scenes push the scene itself off rank 0 in 229 of 2500 rows.

**Command** (`scripts/run_scene_retrieval.sh`):
```bash
python3 evaluate_scene_retrieval.py --modality image --num-scenes 2500 \
  --output-dir outputs/scene_retrieval/image --checkpoint "dense_ep12=..."
```

---

### 2.5 Image binding probe (`evaluate_image_binding.py`)

**Question.** Take two rendered scenes with **identical attribute multisets but
different attribute-to-object assignments**. Can a linear probe tell which conjunction
facts belong to which scene? A representation that only counts attributes scores 50%.

**Data.**
- **Scenes:** the first 20 000 val scenes.
- **Facts:** 72 kept (§1.3).
- **Pairs:**
  - Scenes are grouped by `content_signature` (object count + the multiset of each attribute), then sub-grouped by `binding_signature`.
  - The representatives of different sub-groups are paired, giving 2724 pairs, shuffled with `Random(20260924)`.
  - Only pairs with both scenes in the test range (index ≥ 12 000) are kept: **397 pairs**, with on average 11.8 "flipped" facts each.

**Computation.**
1. Train the shared probe (§1.4) on scenes 0–11 999, separately for each readout.
2. For each pair (A, B) and each flipped fact, score 1 if (logit_A − logit_B) has the correct sign. If |difference| ≤ 1e-9 the fact scores 0.5.
3. **Binding accuracy** = mean over facts, then over pairs, with a 1000× bootstrap CI over pairs.
   **Strict binding accuracy** (added Oct 2026) asks for more: the probe must classify **both** scenes correctly,
   i.e. 𝟙[sign·s_A > 0 ∧ sign·s_B < 0] (the true side above 0, the false side below 0). Example: s_A = 2.0,
   s_B = 0.5 counts for pairwise binding (2.0 > 0.5) but not for strict (the probe also calls the fact present
   in B). It has its own bootstrap stream, so the pairwise CIs are unchanged. JSON: `binding_strict_accuracy`,
   `per_pair_strict` (per layer too).
4. **Clean probe** = balanced accuracy on all 8000 test scenes. Check it to confirm the probe works at all.
5. The headline readout is fixed (L7, or `last`), so there is no best-of selection.

**Outputs.**
- `construction.json`, recording the scene, label and pair counts.
- `<label>.json`, with `binding_accuracy{value, ci95}`, `clean_probe_balanced_accuracy`, `per_pair` and `layers{readout: …}` (when run with `--per-layer`).

Log line:
```
imgdense_ep16: binding 79.7% [78.2, 81.3] | clean probe 82.5%  | per layer embedding=64.7 L0=64.3 ... L6.mlp_out=83.6 L7=79.7 ...
```

**How to read it.**
- 50% is the attribute-counting floor, and the `embedding` readout already gets about 65%.
- To compare two models, use the paired difference on `per_pair`, not overlapping CIs.

**Example.**
*Input pair (scenes 17460 vs 19870):*
- A: small yellow rubber sphere, small blue metal cylinder, small brown metal cylinder, small blue rubber cylinder, large yellow rubber cube.
- B: small yellow rubber cube, large yellow metal sphere, small blue rubber cylinder, small brown metal cylinder, small blue rubber cylinder.

There are 10 flipped facts. `(blue, metal)` and `(cube, large)` are true only in A;
`(yellow, metal)` and `(sphere, large)` are true only in B.

*Result* (`outputs/image_binding_sweep/`):

| model | L7 binding [95% CI] | clean probe | best layer |
|---|---|---|---|
| dense ep16 | 79.7% [78.2, 81.3] | 82.5% | 83.6% @L6.mlp_out |
| I-JEPA blk_s15_t45 (ep16) | 82.3% [80.9, 83.6] | 83.2% | **89.8%** @L7.mlp_out |

Dense binding peaks at L6.mlp_out and falls to 72% at L7.mlp_out.

**Caveats.**
- The pairs are completely different renders, not minimal edits.
- Scenes recur across pairs, so the bootstrap overstates independence.
- No relation facts (§4.1).
- This script writes `binding_accuracy`, but the sweep ranking reads `d_binding` (§4.2).

**Command** (`scripts/run_image_jepa_sweep.sh`):
```bash
python3 evaluate_image_binding.py --per-layer --checkpoint "LABEL=outputs/RUN/epoch_003.pt" \
  --seed 20260924 --image-cache outputs/image_only_2_5m_token_cache/val_tokens.pt \
  --output-dir outputs/image_binding_sweep
```

---

### 2.6 Image triples d′ (`evaluate_image_triples.py`)

**Question.** Seen from the same new camera, is the anchor's representation closer to
the correctly coloured scene (paraphrase) than to the colour-swapped one?

**Data: 1829 triples** in `outputs/image_eval_triples/triples_tokens.pt`. They were
rendered fresh, at 96×64, by `scripts/run_image_eval_triples.sh`:

| render | what it shows | seed |
|---|---|---|
| **anchor** | the scene from camera A | `SEED_CAM = 4242` |
| **paraphrase** | the same scene, with camera and lights jittered | `SEED_PAR = 7777` |
| **swap** | the same camera as the paraphrase, but two objects that differ in colour **and** shape exchange colours | `SEED_PAR = 7777` |

- Para and swap therefore differ from the anchor by the *same* viewpoint change; only the binding separates them.
- In code space, anchor→para changes 118 of 384 VQ codes on average, and para→swap changes 32.
- Scenes with no valid swap pair are skipped (7).

**Computation.**
1. pos_i = cos(a_i, p_i); neg_i = cos(a_i, s_i).
2. **Centred variant:** subtract the mean anchor feature μ from every split, re-normalise, then recompute the cosines. This removes the cone.
3. d′ = (mean(pos) − mean(neg)) / √((var(pos) + var(neg)) / 2). This is unpaired.
4. **Preference** = mean(pos_i > neg_i). This is paired.
5. Bootstrap: 1000 resamples with a fixed rng, identical for every readout and model.
6. `best` = argmax of the centred d′.

This script also defines `encode()`, the readout extractor shared with 2.7.

**Outputs.** `<label>.json` with `metrics[readout][d_binding | d_binding_centred] = {d: {value, ci95}, preference: {value, ci95}, mean_positive, mean_negative}`.

Log line:
```
imgdense_ep12: L7 raw +0.309 centred +0.290 | best centred L5.attn_out +0.372
```

**How to read it.**
- 0 means binding-blind. `embedding` already gets a centred d′ ≈ 0.09, because the swap changes local patch codes.
- Deep readouts reach 0.3–0.5.
- Raw cosines are about 0.9998, which is why the centred version exists.

**Example.**
*Input (triple 0, `CLEVR_gen_000000`).* The anchor has a large red rubber cube, a
large yellow metal sphere, a small red rubber cylinder and a small red metal cube.
- Paraphrase: the same objects from a jittered camera (84 codes differ from the anchor).
- Swap: from the paraphrase camera, the sphere becomes red and the cylinder yellow (31 codes differ from the paraphrase).

*Result* (`logs/triples_centred.log`):

| model | L7 raw d′ | L7 centred d′ [CI] | best centred |
|---|---|---|---|
| dense ep12 | 0.309 | 0.290 [0.265, 0.317] (preference 0.72) | 0.372 @L5.attn_out |
| dense ep16 | — | 0.262 | 0.380 @L6.mlp_out |
| I-JEPA blk_s15_t45 (ep16) | — | 0.190 | 0.325 @L1.mlp_out |

**Caveats.**
- **This is not proof of binding.** The swap is detectable from local appearance: the raw positional code grid alone gets d′ 0.175 and preference 0.795. That preference is *higher* than every model's L7 preference (0.66–0.72).
- The d′ formula ignores the pairing.
- The best readout is chosen on the eval set.

**Command** (`scripts/run_triples_eval.sh`):
```bash
python3 evaluate_image_triples.py --triples outputs/image_eval_triples/triples_tokens.pt \
  --output-dir outputs/image_triples_eval --checkpoint LABEL=PATH
```

---

### 2.7 Image semantic sensitivity (`evaluate_semantic_sensitivity.py`)

**Question.** Does the pooled image representation move more for a colour swap (a change
of meaning) than for a camera and light re-jitter (a nuisance change)?

**Data.** The same 1829 triples as in 2.6.

**Computation.**
1. Nuisance distance dn_i = 1 − cos(anchor_i, para_i); semantic distance ds_i = 1 − cos(para_i, swap_i).
2. Three statistics:
   - **`S_ratio_of_means`** = mean(ds) / mean(dn): the headline;
   - **`S_median_paired`** = median(ds_i / dn_i);
   - **`fraction_semantic_larger`** = mean(ds_i > dn_i).
3. `best` = argmax of `S_ratio_of_means` over the 25 readouts. No CI is computed.
4. **Three geometries** (added Oct 2026, `sensitivity_geometry.py`): everything above is computed on the raw
   unit features (top-level JSON keys, as before), on **centred** features (minus the mean over all
   anchor / paraphrase / swap vectors, renormalised; key `centred`) and on **per-dimension z-scored** features
   (key `zscore`). The log prints d_s, d_n and S for all three at L7. Raw S compares models with very different
   cones (mean cosine 0.85–1.00), so read the centred / z-scored S, and always d_s and d_n separately: a falling
   S can mean semantic distance shrank or nuisance distance grew.

   *Example (L7):* trunk JEPA B1 has raw S 0.240 vs 0.444 for its start, but centred d_s is unchanged
   (0.109 vs 0.105) while d_n grew 60% (0.366 vs 0.229); z-scored S 0.317 equals the diffusion control (0.313).
   So JEPA made the trunk more viewpoint-sensitive, not less meaning-sensitive.

**Outputs.** `<label>.json` with `metrics[readout] = {nuisance_mean, semantic_mean, S_ratio_of_means, S_median_paired, fraction_semantic_larger}`.

Log line:
```
dense13: L7 S=0.331 | best L6.mlp_out S=0.392 (sem 0.00111 / nui 0.00282)
```

**How to read it.**
- S > 1 would mean meaning moves the representation more than viewpoint. **Every model is far below 1**, because the two edits are not matched in size (32 vs 118 codes).
- Read S against the model-free floors instead:
  - bag of VQ codes: S = 0.074;
  - raw positional code grid: S = 0.269;
  - random init: L7 S = 0.071.

**Example.** The triple is the same as in 2.6. *Result* (`outputs/lejepa_early_eval/sens_image.log` and the JSONs, L7):

| model | S (ratio of means) | S (median) | fraction semantic > nuisance |
|---|---|---|---|
| dense13 | 0.331 | 0.267 | 0.142 |
| ijepa13 | 0.237 | 0.185 | 0.102 |
| lejepa13 | **0.483** | **0.015** | 0.094 |

*Reading:* LeJEPA "wins" on the ratio of means only because a few items have large
distances. On a typical item it is the *least* sensitive to the swap. Always check
the median and the fraction.

**Caveats.**
- The ratio of means is dominated by heavy tails.
- The uncentred cosine is affected by the cone (§1.6): dense L7 distances are about 1e-4.
- The best readout is chosen on the eval set.
- There is no CI.

**Command** (`scripts/run_lejepa_early_eval.sh`):
```bash
python3 evaluate_semantic_sensitivity.py --output-dir outputs/lejepa_early_eval/sens_image \
  --checkpoint dense13=outputs/image_dense_diffusion_2_5m_40e/epoch_012.pt
# random baseline: --checkpoint "random_init=RANDOM:outputs/image_dense_diffusion_2_5m_40e/epoch_015.pt"
```

---

### 2.8 Text sensitivity (`evaluate_text_sensitivity.py`)

**Question.** For captions, does the representation move more when a **binding** is
swapped (`S_bind`) or when the **scene** is replaced (`S_cont`) than when the same scene
is merely **reworded**?

**Data.** Built by `evaluate_semantic_dprime.build`, with `lexicon_matched = True`.
- **Worlds:** the first 2000 worlds in `val_text_only_human.jsonl` that have ≥ 3 objects and no duplicate objects.
- **Four captions per world:**
  - **query:** phrasing plan A.
  - **paraphrase:** plan B. It keeps A's synonyms but re-samples phrase styles, "made of / built from" wording, openers, digit-vs-word counts, and object, relation and sentence order.
  - **swap:** one valid attribute exchange between two objects, or a relation subject/anchor flip, rendered with plan B. It is kept only if its **token multiset equals the paraphrase's**.
  - **twin:** the next world with the same object and relation counts but different content, rendered with plan A.
- **Drops:** worlds with no valid swap or twin, among other checks. Here 6 were dropped, leaving **1994 items**.

**Computation.**
1. With d = 1 − cos: dn = d(query, para), db = d(para, swap), dc = d(query, twin).
2. `S_binding` = mean(db) / mean(dn) and `S_content` = mean(dc) / mean(dn), plus the per-item fractions db > dn and dc > dn.
3. `best` = argmax of `S_binding` over 42 readouts. In the log, `L7` means `L7.residual`. No CI is computed.
4. **Three geometries** (added Oct 2026): as for images, `S_binding`, `S_content` and their distances are also
   computed on centred (`centred`) and per-dimension z-scored (`zscore`) features; the log prints all three at L7.

**Outputs.** `<label>.json` with `metrics[readout] = {nuisance_mean, binding_mean, content_mean, S_binding, S_content, fraction_*}`.

Log line:
```
dense22: L7 S_bind 0.497 S_cont 1.766 | best S_bind 1.214 @L6.mlp_out (bind 0.06018 / nui 0.04955)
```

**How to read it.**
- `S_cont` > 1 means the representation tracks the scene rather than the wording.
- `S_bind` > 1 means it registers the binding more than the rewording.
- Controls:
  - bag-of-words (`emb.token`): `S_bind` = 0 (same multiset) and `S_cont` ≈ 0.91;
  - random init at L7: `S_bind` 0.037, `S_cont` 0.905.

**Example.**
*Input (item 0: blue↔brown swapped between the large metal box and the small rubber tube).*
- **query:** "If you look closely, the yellow large sphere with a matte-textured rubber finish is off to the right of the blue glossy metal box that looks large. … The picture shows six objects: …"
- **paraphrase:** "Present in the scene: a blue glossy metal sphere that looks smallish; a tube that is smallish, brown, and matte-textured rubber; … a large blue glossy metal box. …"
- **swap:** "Present in the scene: …; a large **brown** glossy metal box; and a smallish **blue** matte-textured rubber tube. …"
- **twin:** "If you look closely, the yellow large sphere … is off to the right of the red matte-textured rubber box that looks smallish. …"

*Result* (`outputs/lejepa_early_eval/sens_text*.log`, L7):

| model | S_bind | S_cont | best S_bind |
|---|---|---|---|
| dense22 | 0.497 | 1.766 | 1.214 @L6.mlp_out |
| ijepa22 | 0.297 | **2.229** | 0.992 @L5.mlp_out |
| lejepa22 (token-level) | 0.352 | **0.123** | 0.408 @L3.mlp_out |
| lejepaP21 (paper) | 0.619 | **0.342** | 0.728 @L5.attn_out |

*Reading:*
- Dense and I-JEPA follow scene content, and I-JEPA gives up some binding.
- **Both LeJEPA variants fall below the bag-of-words control on `S_cont`**: a rewording moves them more than a different scene does.
- Over training, dense L7 `S_bind` grows: 0.32 (ep4) → 0.60 (ep20) → 0.71 (ep41).

**Caveats.**
- Para and swap share a token multiset but **not** word order: about 62% of positions differ. So the "binding" change includes a reordering, and contextual readouts get a non-zero floor (random init up to 0.22).
- Adjacent epochs vary (dense21 0.608 vs dense22 0.497), and there is no CI.
- The best readout is chosen over 42 readouts.

**Command:**
```bash
python3 evaluate_text_sensitivity.py --num-worlds 2000 --output-dir outputs/lejepa_early_eval/sens_text \
  --checkpoint dense22=outputs/text_dense_diffusion_2m_40e_continued/epoch_021.pt
```

---

### 2.9 Suite report (`scripts/summarize_eval_suite.py`)

**What it does.** Reads every finished JSON under `outputs/` and prints markdown tables.
It must be run from the repo root. Sources:
- `probe_suite/{image,text}` for the probes and the noise curve;
- `semantic_sensitivity(_text)`;
- `scene_retrieval`;
- `cross_modal_retrieval/results` and `cka_matched`.

**"Best" columns** take the maximum over all readouts, each column choosing independently.
- For probes, α is chosen on val but the layer is effectively chosen on test.
- Retrieval has no split at all.
- The noise table and the p@10 columns re-pick the readout without printing it, despite the docstring.

So treat this report as optimistic. `outputs/EVAL_SUITE_REPORT.md` is a saved copy of its output.

```bash
cd clevr_discrete_diffusion && python3 scripts/summarize_eval_suite.py > outputs/EVAL_SUITE_REPORT.md
```

---

### 2.10 JEPA prediction check (`evaluate_jepa_prediction.py`)

**Question.** A JEPA's logged cosine between prediction and target can be close to 1 just because every vector
shares one dominant direction. Does the predictor actually predict *this scene's* target?

**Data.** 512 validation scenes per modality the checkpoint was trained on. Masks are the same as in training:
per-modality 2D blocks or spans at the run's mask rate, trunk-only routing for `jepa_trunk_only` runs. The
masks are seeded, so two passes see identical masks.

**Computation.**
1. **Rebuild the networks.** The student comes from the checkpoint, and the EMA teacher from
   `shared_jepa_ema_teacher`. For teacherless LeJEPA, the model itself gives the clean targets without LayerNorm.
2. **Predictions and targets.** At every masked token and supervised layer: the prediction is
   `predict_data2vec(student hidden)`, and the target is the LayerNorm'd teacher state.
3. **C₊** = mean cos(prediction, own target).
4. **C₋** = mean cos(prediction, the target of **another scene at the same token position**). Using the same
   position keeps positional structure from inflating it.
5. **Δ** = C₊ − C₋.
6. **Centred version:** repeat steps 3–5 after subtracting, from predictions and from targets, their mean over
   all evaluated masked tokens of that layer and modality (two passes).

**How to read it.** A large centred Δ means the predictor carries instance-specific information. Raw C₊ ≈ C₋
(Δ ≈ 0) means it mostly learned the shared direction.

**Example** (mean over the 8 supervised layers):

| checkpoint | raw C₊ / C₋ / Δ | centred C₊ / C₋ / Δ |
|---|---|---|
| trunk JEPA B, text, 4 JEPA epochs | 0.97 / 0.64 / 0.34 | 0.94 / 0.23 / **0.72** |
| trunk JEPA B, image, 4 JEPA epochs | 0.93 / 0.76 / 0.18 | 0.87 / 0.51 / **0.36** |
| image I-JEPA (single-modality) | 0.94 / 0.76 / 0.18 | 0.89 / 0.52 / **0.37** |
| token-level LeJEPA, image (collapsed run) | 0.997 / 0.950 / 0.05 | 0.996 / 0.927 / **0.07** |

**Caveat.** Δ measures token-level prediction. The text trunk has a large Δ (it predicts which word belongs at a
masked position) while its pooled scene features stay at random-init level on binding probes. Read it together
with §2.1 and §2.3.

**Command:**
```bash
python3 evaluate_jepa_prediction.py --output-dir outputs/eval_all/jepa_prediction \
  --checkpoint B4=outputs/mm_trunk_ijepa_private_diff_from_ep6_4e/epoch_003.pt
```

---

## 3. Training-log diagnostics

All of these come from the **last micro-batch of the logging step only**. They are not
averages, so single values are noisy.

### 3.1 Validation loss at fixed t = 0.75

- **What:** the 1/t-weighted masked cross-entropy of the diffusion head, with 75% of content tokens masked i.i.d. It is computed over the whole val loader at the end of each epoch.
- **Log:** `validation epoch=000 fixed-t=0.75 loss=2.5185`.
- **How to read:** lower is better, and it is only comparable across epochs of the same run. Plain CE = loss × 0.75.
- **Caveat (important):** with `diffusion.weight: 0` (all JEPA/LeJEPA runs), the head gets no gradient while the backbone moves, so this number is meaningless. For example, I-JEPA s15_t45 goes 4.70 → 5.30 while being the best model on conj probes. **It still selects `best.pt`** (§4.3).

### 3.2 Train losses and masked accuracy

- **Fields:**
  - `text_loss` / `image_loss`: per-modality 1/t-weighted CE;
  - `loss`: their mean;
  - `unweighted_loss`: plain CE;
  - `optimized_loss`: the objective actually optimised (diffusion weight × task + JEPA term);
  - `*_acc`: argmax accuracy over masked positions.
- **Example** (unpaired diffusion trunk): `loss=3.7181 unweighted_loss=2.1551 text_loss=1.8907 image_loss=5.5455 text_acc=0.604 image_acc=0.228`.
- **Caveat:** these depend on each batch's t, so they are noisy. With diffusion weight 0 they only track the frozen head.

### 3.3 I-JEPA / data2vec: `d2v_*/cos`, `global_cos`, `target_spread` (`shared_jepa.py:328-525`)

- **`cos`:** mean cosine between the predictor output and the target over masked tokens, averaged over the supervised blocks. A healthy I-JEPA run is around 0.93. A value near 1 can also mean the targets became trivial.
- **`global_cos`:** **0.000 means "not computed"** (it only runs when the global or variance loss weight > 0), not "orthogonal".
- **`target_spread`:** the **minimum over target blocks** of the mean per-dimension std of the L2-normalised targets across masked tokens. Taking the minimum means a single collapsing block cannot hide behind the others.
  - Maximum ≈ 1/√384 ≈ 0.051; 0 means collapse.
  - Healthy I-JEPA runs sit at 0.030–0.033.
- **Example:** I-JEPA s15_t45, `cos=0.382 … target_spread=0.0297` at step 10 → `cos=0.934 … 0.0327` at step 39 500. In the token-level LeJEPA image run, cos rose to 0.994 while spread fell from 0.027 to 0.012: a warning sign, later confirmed by the probe collapse.
- **Caveat:** spread pools tokens across positions and scenes, so a representation that only encodes position still looks healthy.

### 3.4 Token-level LeJEPA: `lejepa_*/pred`, `sigreg`, `pooled_cos`

These are logged for `data2vec_teacherless`.
- **`pred`:** the layerwise token-level prediction loss.
- **`sigreg`:** SIGReg on the pooled clean states, averaged over the 8 blocks, N = 64.
- **`pooled_cos`:** mean off-diagonal cosine of the pooled clean states. This is the cone meter; 0 means isotropic.
- The logged `jepa_*` = 0.95·pred + 0.05·sigreg.
- **Example** (image): step 10 `pred=2.0163 / sigreg=84.90 / pooled_cos=0.729` → step 11 100 `pred=0.0323 / sigreg=1.78 / pooled_cos=-0.001`.
- **Caveat:** pooled_cos ≈ 0 together with a tiny pred did **not** mean good features. The content was lost (structure probe 58.9% vs 92.8%; CKA 0.017 vs 0.392).

### 3.5 Paper LeJEPA: `inv`, `sigreg`, `emb_cos`, `proj_cos`, `kept_g`, `kept_l` (`lejepa_views.py`)

**Fields:**
- **`inv`:** mean squared distance between each view's projection and the mean of the 2 global views' projections, over 8 views, the batch and 128 dims. If inv drops to about 0 within a few hundred steps, that signals a shortcut.
- **`sigreg`:** per-view SIGReg on the projector output, averaged over views (N = 128).
- **`emb_cos`** / **`proj_cos`:** mean off-diagonal cosine of view 0's **backbone** embedding / **projector** output. proj_cos ≈ 0 says nothing about the backbone.
- **`kept_g`** / **`kept_l`:** fraction of content tokens kept by global / local crops. Expect ≈ 0.65 and ≈ 0.175, the means of the 0.3–1.0 and 0.05–0.3 scale ranges.

**Examples:**
- Fixed text run, step 900: `inv=0.3518/sigreg=3.904/emb_cos=0.766/proj_cos=-0.004/kept_g=0.67/kept_l=0.17`.
- **Aborted run** (`outputs/lejepa_paper/aborted_eos_leak/`): `inv=0.0045` already at step 900. The EOS position leaked the caption length into every crop, a shortcut; EOS is now dropped from views.

### 3.6 The SIGReg statistic itself (`sigreg.py:14-51`)

**Computation:**
1. Project the [N, D] batch onto S = 1024 random unit directions.
2. On each 1-D projection, compare the empirical characteristic function with that of N(0, 1) at 17 points t ∈ [0, 3].
3. Weight by trapezoid-rule weights × e^{−t²/2}.
4. Multiply by N, then average over slices.

It tests against N(0, I) on an **absolute** scale: no centring, no standardisation.

**Reference values** (CPU check, 1024 slices):

| batch | SIGReg |
|---|---|
| **truly Gaussian**, N = 64, D = 384 | **1.06 ± 0.05** (the theoretical floor is 1.0525 for any N, D) |
| Gaussian, N = 128, D = 128 | 1.09 |
| constant batch | 25.7 |
| Gaussian + 0.3 shift on every coordinate | 3.74 |
| Gaussian × 0.5 | 10.1 |
| Gaussian × 2 | 16.8 |

Runs end at 1.78 (token-level image, N = 64) and 1.59 (paper image, N = 128).
Values from runs with different N are not directly comparable, because the excess
above the floor scales with N.

---

## 4. Known issues found while writing this

| # | Issue | Effect | Fix |
|---|---|---|---|
| 4.1 | **Relations never read.** `evaluate_cross_modal_structure.py:79` and `binding_swap_captions.py` read `world["relations"]`, but the manifests store CLEVR's `world["relationships"]` (a dict of behind/front/left/right lists). This was verified on `val_image_only.jsonl`. | In evals 2.1, 2.2 and 2.5, the eval captions have **no spatial sentences** and all 18 relation facts are dropped. So these evals only test attribute binding, and the captions differ from the training captions, which do contain relations. | Convert `relationships` into the generator's relation list before captioning and labelling. |
| 4.2 | **Sweep ranking reads the wrong key.** `scripts/rank_image_jepa_sweep.py:14,18` reads `d_binding`, but `evaluate_image_binding.py` writes `binding_accuracy`. | `outputs/image_jepa_sweep/RESULTS.md` is all "–", so "no winner parsed" and **phase 2 of the sweep never ran**. | Read `binding_accuracy.value` (per layer: `layers[readout].binding_accuracy`). |
| 4.3 | **`best.pt` chosen by diffusion val loss** (`selection_metric: marginal_t0.75`), even when `diffusion.weight: 0`. | For JEPA/LeJEPA runs `best.pt` is just epoch 0. For example, `image_ijepa_sweep_blk_s15_t45/best.pt` has the same timestamp as `epoch_000.pt`. | Always evaluate `epoch_*.pt` for JEPA runs, or add a representation-based selection metric. |
| 4.4 | **Caching by label.** Retrieval (`features/*.npz`, `results/*.json`), the probe suite and the sensitivity scripts skip any label whose output file exists. | Reusing a label for a different checkpoint silently returns stale numbers. | Use unique labels, or delete the old files. |
| 4.5 | **Image triples d′ is solvable from local appearance.** | The raw positional code grid alone has preference 0.795, higher than any model's L7 preference. | Report it next to the code-grid baseline; don't read it as binding. |
| 4.6 | **Ratio-of-means sensitivity is dominated by tails** (2.7). | LeJEPA "wins" S while losing on the median and the fraction. | Report all three statistics. |
| 4.7 | **Text binding swaps also reorder words** (2.8). | Contextual readouts get a non-zero `S_bind` floor. | Compare against random init and `emb.token`. |
| 4.8 | **Cosine metrics are uncentred** (1.6). | The cone shrinks distances, so coned and spread models are not compared fairly on RSA, S and scene retrieval. | Also report centred (or per-dimension z-scored) versions. |
| 4.9 | **Retrieval ridge grid edge.** | α = 1e-3, the smallest value, is often chosen. | Extend the grid downwards. |
| 4.10 | **Stale docs.** | Misleading descriptions in the probe-suite docstring (says existence mAP), `build_image_eval_triples.py` (says the swap uses the anchor camera), the retrieval tie comment, and the teacherless startup line ("LayerNorm-ed EMA-teacher"). | Update the text. |

### Reporting checklist

Following the literature review on anisotropy and representation-similarity metrics:

- **Cone diagnostics:** report mean cosine, effective rank and the top-dimension share of cosine.
- **Cosine metrics, three ways:** report each raw, centred and z-scored.
- **Baselines:** always show the random-init and input-embedding baselines.
- **Paired comparisons:** prefer paired differences, e.g. `per_pair` in 2.5.
- **No best-on-test:** use a fixed readout, or choose the readout on a held-out split.
- **CKA:** add a permutation null, since chance CKA grows with dimension / samples.
