# Running these runs on another cluster

> **Type:** runbook · **Updated:** 2026-09-24 · **Menu:** [experiment catalog](README.md)

Everything here is single-GPU, text-only, and needs no image data. One run uses
about 6 GB of VRAM and, on an RTX 3090 / A5000, about 30 minutes per epoch of
2M captions at ~1,150 samples/s.

## 1. What to copy

| What | Path here | Size | Needed by |
|---|---|---:|---|
| The repo | `/home/zd25e122/clevr_discrete_diffusion` (excluding `outputs/`, `wandb/`, `.git/`) | 6.5 MB | everything |
| VQ-VAE checkpoint | `outputs/vqvae_training_bs128/best.pt` | 22 MB | everything (it is loaded unconditionally, even for text-only runs) |
| Train captions | `.../platonic_text_only_v1_2m/train_text_only_human.jsonl` | 1.0 GB | everything |
| Val captions | `.../platonic_text_only_v1_1m/val_text_only_human.jsonl` | 11 MB | everything |
| Dense diffusion checkpoint | `outputs/text_dense_diffusion_2m_4e_matched/epoch_003.pt` | 169 MB | only the *from-dense* variants |
| Stage-1 data2vec checkpoint | `outputs/text_data2vec_from_dense_2m_2e/epoch_001.pt` | 269 MB | only the *stage-2* dense+private runs |

The **from-scratch window runs need no checkpoint at all** beyond the VQ-VAE —
that is the point of them.

## 2. Environment

Python 3.10 or newer (3.10 and 3.13 both tested), and:

```bash
pip install torch numpy pyyaml pillow      # torch 2.x with CUDA; 2.9.1+cu128 here
pip install wandb                          # only if you keep logging enabled
```

No other third-party packages are imported. There is no `requirements.txt` in
the repo; the list above is the complete set.

## 3. Point the configs at your paths

Every YAML key is also a command-line flag, so nothing has to be edited in
place — override on the command line instead:

```bash
DATA=/your/path/to/captions
python3 train_multimodal.py --config configs/<name>.yaml \
  --train-manifest $DATA/train_text_only_human.jsonl \
  --val-manifest   $DATA/val_text_only_human.jsonl \
  --train-dir      $DATA --val-dir $DATA \
  --vqvae-checkpoint /your/path/vqvae_best.pt \
  --output-dir /your/scratch/outputs/<name> \
  --no-wandb
```

`--train-dir` / `--val-dir` are only used to resolve relative manifest entries;
for text-only runs they can point at the directory holding the manifests.
Keep `--no-wandb` unless you set `--wandb-entity` to your own account.

## 4. The runs

From-scratch JEPA with window masking — the two queued runs:

```bash
python3 train_multimodal.py --config configs/text_data2vec_scratch_window4_8_15pct_2m_4e.yaml
python3 train_multimodal.py --config configs/text_data2vec_scratch_window12_24_30pct_2m_4e.yaml
```

From-scratch JEPA with data2vec's token masking (the control, not yet run):

```bash
python3 train_multimodal.py --config configs/text_data2vec_scratch_avg_all_2m_4e.yaml
```

The from-scratch control that isolates initialization (same masking as the
from-dense runs):

```bash
python3 train_multimodal.py --config configs/text_data2vec_scratch_layerwise_all_randt_2m_4e.yaml
```

Image data2vec with 2D block masking (needs the image token cache):

```bash
python3 train_multimodal.py --config configs/image_data2vec_scratch_block2d_30pct_avg_l4to7_1_2m_4e.yaml
python3 train_multimodal.py --config configs/image_data2vec_scratch_block2d_30pct_layerwise_l4to7_1_2m_4e.yaml
```

From-dense data2vec variants (need the dense checkpoint, 2 epochs each):

```bash
python3 train_multimodal.py --config configs/text_data2vec_from_dense_2m_2e.yaml                 # average, all 8 blocks
python3 train_multimodal.py --config configs/text_data2vec_from_dense_avg_l4to7_2m_2e.yaml       # average, blocks 4-7
python3 train_multimodal.py --config configs/text_data2vec_from_dense_layerwise_l4to7_2m_2e.yaml # layerwise, blocks 4-7
python3 train_multimodal.py --config configs/text_data2vec_from_dense_layerwise_all_2m_2e.yaml   # layerwise, all 8
```

Stage 2, dense trunk as the shared route plus rank-128 private LoRA (needs the
stage-1 checkpoint; see [the plan](plans/dense_shared_private_lora_plan.md)).
All four cells of the 2×2 — two stage-1 trunks × frozen/trainable:

```bash
python3 train_multimodal.py --config configs/text_dense_private_frozen_hsic_2m_2e.yaml
python3 train_multimodal.py --config configs/text_dense_private_trainable_hsic_2m_2e.yaml
python3 train_multimodal.py --config configs/text_dense_private_frozen_hsic_from_d2v_lw_all_2m_2e.yaml
python3 train_multimodal.py --config configs/text_dense_private_trainable_hsic_from_d2v_lw_all_2m_2e.yaml
```

SIGReg / LeJEPA in place of the EMA teacher (`--data2vec-teacherless` plus
`--data2vec-sigreg-weight`; the configs set both):

```bash
python3 train_multimodal.py --config configs/text_lejepa_scratch_layerwise_all_2m_4e.yaml
python3 train_multimodal.py --config configs/text_lejepa_from_dense_layerwise_all_2m_2e.yaml
```

Harder masking from scratch — ~61% realized on text, ~57% on images, testing
whether the from-scratch collapse is an objective that converges too quickly:

```bash
python3 train_multimodal.py --config configs/text_data2vec_scratch_window12_24_60pct_avg_l4to7_2m_4e.yaml
python3 train_multimodal.py --config configs/text_data2vec_scratch_window12_24_60pct_layerwise_l4to7_2m_4e.yaml
python3 train_multimodal.py --config configs/image_data2vec_scratch_block2d_65pct_avg_l4to7_1_2m_4e.yaml
python3 train_multimodal.py --config configs/image_data2vec_scratch_block2d_65pct_layerwise_l4to7_1_2m_4e.yaml
```

Multimodal, one model over both modalities (needs the image token cache **and**
the paired manifest). `data.mode: unpaired` deranges the batch so a caption is
never carried with its own image:

```bash
python3 train_multimodal.py --config configs/multimodal_paired_dense_1_2m_4e.yaml
python3 train_multimodal.py --config configs/multimodal_unpaired_dense_1_2m_4e.yaml
python3 train_multimodal.py --config configs/multimodal_unpaired_d2v_from_text_dense_1_2m_2e.yaml
python3 train_multimodal.py --config configs/multimodal_unpaired_d2v_from_image_dense_1_2m_2e.yaml
python3 train_multimodal.py --config configs/multimodal_unpaired_d2v_from_unpaired_dense_ema_1_2m_2e.yaml
python3 train_multimodal.py --config configs/multimodal_unpaired_d2v_from_unpaired_dense_sigreg_1_2m_2e.yaml
python3 train_multimodal.py --config configs/multimodal_unpaired_lejepa_scratch_1_2m_4e.yaml
```

> **Duplicate-run hazard.** The launchers guard against relaunching a run with
> `tmux has-session` and an existing-checkpoint check. **Neither guard crosses
> machines.** If the filesystem is shared — as it is between these clusters —
> two hosts will happily write the same `outputs/<run>/` directory and corrupt
> each other's checkpoints. Before launching here, confirm nothing is running
> there for the same config.

Pin the GPU with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n>`, and
use `PYTHONUNBUFFERED=1` if you redirect the log to a file.

## 5. Slurm

```bash
#!/usr/bin/env bash
#SBATCH --job-name=d2v_window
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8          # matches train.num_workers: 8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
srun python3 train_multimodal.py \
  --config configs/text_data2vec_scratch_window12_24_30pct_2m_4e.yaml \
  --output-dir "$SCRATCH/outputs/text_data2vec_scratch_window12_24_30pct_2m_4e" \
  --no-wandb
```

Four epochs of 2M captions takes about two hours on one 3090, so a 4-hour limit
is comfortable. Lower `--batch-size` (64 here) if the GPU has under 8 GB;
`--gradient-accumulation-steps` (4) keeps the effective batch at 256.

## 5b. Scoring the model while it trains

Hard retrieval can run on the live model every N optimizer steps, without
blocking training:

```yaml
train:
  probe_every_steps: 1000
  probe_num_worlds: 500          # about 45 s per probe
  probe_sublayer_layers: [4, 5, 6, 7]
  probe_gpu: 3                   # optional; defaults to the training GPU
```

A slim checkpoint (weights, args and vocabulary — no optimizer, no EMA teacher)
is written and handed to `evaluate_hard_retrieval.py` in a **detached** process
that deletes it when done. If the previous probe is still running the interval
is skipped, so probes cannot pile up. Results land in
`outputs/<run>/probes/step_NNNNNNN.json`, with one summary line per probe in
`outputs/<run>/probes/probe.log`.

For per-epoch scoring of finished checkpoints instead, without touching the
trainer:

```bash
scripts/watch_and_evaluate_epoch_checkpoints.sh <gpu> outputs/<run_dir> [more dirs...]
```

It scores each `epoch_00N.pt` as it appears with both the hard-retrieval and
binding-swap probes, skips anything already scored, and exits when the runs'
tmux sessions are gone.

## 6. Check it is doing what you think

The first lines of the log state the objective and the masking. For the window
runs expect all three of:

```
training masks a fixed 30% of eligible tokens per sequence (t is not sampled)
window masking: contiguous windows of 12-24 tokens, sizes drawn uniformly per window ...
faithful data2vec: target=mean of parameter-free LayerNorm-ed EMA-teacher hidden states over blocks [0..7] of 8; student=final block through one prediction head; ... diffusion_weight=0.0
```

Then, per step, `jepa_text` should fall and `d2v_text/cos` should rise toward
~0.9 while `target_spread` stays well above zero — a spread collapsing to 0 is
the signature of a degenerate teacher.

**The validation loss of any `diffusion_weight=0` run is meaningless**: nothing
trains the output head, so masked-token loss stays near random and `best.pt` is
arbitrary. Evaluate `epoch_00N.pt` instead.

## 7. What each run writes

Into its own `--output-dir`: `epoch_000.pt … epoch_00N.pt`, `best.pt`,
`last.pt`, `text_tokenizer.json`, `train.log`, and for LoRA runs an extra
`*_adapter.pt` per checkpoint. Nothing is shared between runs.

## 8. Evaluating the checkpoints

Both probes are single-GPU and self-contained; they rebuild the captions from a
fixed seed, so results are comparable across machines as long as the caption
generator and manifest are the same:

```bash
python3 evaluate_hard_retrieval.py --num-worlds 2000 --sublayer-layers 0 1 2 3 4 5 6 7 \
  --output-dir outputs/hard_retrieval_eval_all_sublayers \
  --checkpoint <label>=<path/to/epoch_00N.pt>

python3 evaluate_binding_swap.py --output-dir outputs/binding_swap_eval \
  --checkpoint <label>=<path/to/epoch_00N.pt>
```

The two newer evaluations, which are the ones to rank models by:

```bash
# scene-vs-wording and binding as signed effect sizes
python3 evaluate_semantic_dprime.py --num-worlds 2000 \
  --output-dir outputs/semantic_dprime_eval --checkpoint LABEL=path/to/epoch_00N.pt \
  --untrained-from path/to/any/dense_checkpoint.pt

# image-side semantics, and agreement between a text and an image model
python3 evaluate_cross_modal_structure.py --num-scenes 4000 \
  --output-dir outputs/cross_modal_structure \
  --text-checkpoint text=path/to/text.pt --image-checkpoint image=path/to/image.pt
```

Route isolation, for any model with a shared and a private route — the flags
suppress one route at every layer by route id, in an actual forward pass:

```bash
python3 evaluate_semantic_dprime.py --ablate-private --num-worlds 2000 \
  --output-dir outputs/semantic_dprime_trunk_only --checkpoint LABEL_trunkonly=<path>

python3 evaluate_semantic_dprime.py --ablate-shared --num-worlds 2000 \
  --output-dir outputs/semantic_dprime_private_only --checkpoint LABEL=<path>
```

Image-side binding, the image analogue of `d_bind` (needs the image token cache
and the image-only validation manifest):

```bash
python3 evaluate_image_binding.py --output-dir outputs/image_binding_eval \
  --checkpoint LABEL=path/to/epoch_00N.pt
```

One model encoding both modalities — gap, AUC, paired R@1 and per-layer CKA
(needs the **paired** validation manifest):

```bash
python3 evaluate_paired_modality_alignment.py --num-scenes 2000 \
  --manifest .../val_pairs_human.jsonl \
  --image-cache outputs/image_only_1_2m_token_cache/val_tokens.pt \
  --output-dir outputs/unpaired_modality_alignment --model LABEL=<path>
```

All of these **merge** into their output directory rather than overwriting it,
so a run that scores a subset of the models does not erase earlier results.
That was a real bug in `evaluate_paired_modality_alignment.py` and
`evaluate_cross_modal_structure.py`, fixed on 2026-09-24; if you are running
from an older copy, check before you trust a summary file that names fewer
models than you expect.

Both caption-based evaluations need `--caption-generator` to point at
`clevr-dataset-gen_clone/image_generation/generate_human_captions.py`, so copy
that too if you evaluate on the other cluster. Add `--untrained-from <path>` to
`evaluate_hard_retrieval.py` for the chance-level control.

### Killing only your own GPU processes

On a shared node, never `pkill -f python`. Filter by owner:

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read pid; do
  [ "$(ps -o user= -p "$pid")" = "$USER" ] && kill "$pid"
done
```

For runs started under tmux, `tmux kill-session -t <name>` is cleaner still: it
stops the launcher as well as the trainer.
