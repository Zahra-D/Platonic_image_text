# Running this project on a cluster

This file is the hand-off for a new machine, or for a new Claude Code session that
has none of the development history. It covers setup, data, multi-GPU training
(DDP), evaluation, and where the research stands. The evaluation protocols are
in [EVALUATIONS.md](EVALUATIONS.md).

---

## 1. Code and environment

```bash
git clone git@github.com:Zahra-D/Platonic_image_text.git && cd Platonic_image_text
python3 -m venv ~/venvs/clevr && source ~/venvs/clevr/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match the cluster CUDA
pip install -r requirements.txt
wandb login                                                              # or pass --no-wandb
```

GitHub deploy keys are per repository and per machine. On the cluster, generate a
key (`ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_clevr`) and add its `.pub` as a
deploy key with write access on the GitHub repo, or clone over HTTPS with a token.

The development machine used Python 3.13 and torch 2.12.

## 2. Data (not in git, ~7.5 GB)

On the development machine, pack everything into one tarball:

```bash
bash scripts/cluster/pack_data.sh /path/to/clevr_cluster_data.tar
```

Copy it to the cluster, unpack, and point the environment at it:

```bash
tar -xf clevr_cluster_data.tar -C /scratch/$USER/clevr
export CLEVR_DATA=/scratch/$USER/clevr/clevr_data          # manifests
export CLEVR_GEN=/scratch/$USER/clevr/image_generation     # caption generator (evals)
mkdir -p outputs && cp -r /scratch/$USER/clevr/repo_outputs/* outputs/
```

| Where | File | Size |
|---|---|---|
| `$CLEVR_DATA/platonic_clevr_v1_5M_train_gpu_visible/` | `train_image_only_2_5m.jsonl`, `val_image_only.jsonl` | 2.26 GB |
| `$CLEVR_DATA/platonic_text_only_v1_2m/` | `train_text_only_human.jsonl` | 3.21 GB |
| `$CLEVR_DATA/platonic_text_only_v1_1m/` | `val_text_only_human.jsonl` | 0.03 GB |
| `outputs/image_only_2_5m_token_cache/` | `train_tokens.pt`, `val_tokens.pt` (VQ codes) | 1.96 GB |
| `outputs/vqvae_training_bs128/` | `best.pt` (VQ-VAE) | 0.02 GB |
| `outputs/image_eval_triples/` | `triples_tokens.pt` (image triples eval) | small |
| `$CLEVR_GEN/` | `generate_human_captions.py` and friends | 6 MB |

All configs reference `${CLEVR_DATA}`, and the eval scripts read `CLEVR_DATA` /
`CLEVR_GEN` (`clevr_paths.py`). With these unset, they fall back to the
development-machine paths.

## 3. Training with DDP

`train_multimodal.py` runs unchanged on one GPU, or under `torchrun` across GPUs and nodes:
- **Gradients:** averaged over ranks once per optimizer step (`all_reduce_gradients`).
- **Data:** each rank reads a disjoint shard of every epoch (`DistributedSampler`).
- **Rank 0 only:** logging, wandb, checkpoints and samples.
- **Validation:** loss averaged over ranks, so best-checkpoint and early-stopping decisions are identical everywhere.
- **Sync check:** at every validation the log reports `distributed: all N ranks hold identical weights`. A `WARNING ... DIFFERENT weights` means something is wrong.

**Single node:**

```bash
NGPU=4 bash scripts/cluster/train_single_node.sh configs/cluster/mm_unpaired_dense_2m_40e_ddp.yaml
```

**Multi-node (SLURM).** First edit the `#SBATCH` lines (account, partition, GPUs per node, time) and the environment block:

```bash
CONFIG=configs/cluster/mm_unpaired_dense_2m_40e_ddp.yaml sbatch scripts/cluster/train_ddp.slurm
# resume after a time limit:
CONFIG=... sbatch scripts/cluster/train_ddp.slurm --resume outputs/mm_unpaired_dense_2m_40e_ddp/last.pt
```

**Batch and learning rate.** `train.batch_size` is per rank:
- For unpaired runs, each rank processes `batch_size / 2` captions plus `batch_size / 2` images per micro-step.
- Global batch = `batch_size × gradient_accumulation_steps × #GPUs`, printed at start-up.
- The single-GPU runs used a global batch of 256 with constant lr 3e-4.
- `configs/cluster/mm_unpaired_dense_2m_40e_ddp.yaml` uses a per-rank batch of 128. On 16 GPUs that is a global batch of 2048, so it uses lr 8.5e-4 (square-root scaling) with a 2% warmup and a final 10% linear decay.
- For exact comparability with the single-GPU runs, keep the global batch at 256 and use a constant lr of 3e-4.
- Step-based settings (JEPA warmup 1000 steps, EMA ramp 5000 steps) cover more data at a larger global batch. Rescale them if you port JEPA configs.

**Resources per rank:**
- About 14–20 GB host RAM (manifests and token cache), plus `num_workers` data-loader processes.
- On an RTX 3090, the unpaired dense model ran at about 700–800 samples/s.
- On an H200, expect roughly 3–6× that per GPU. The model is small (27M parameters), so speed is bound by memory bandwidth and the data loader. Raise `num_workers` if GPU utilization is low.
- A 40-epoch run is about 2.5 days on one 3090. Expect about 3–5 h on 16 H200s, assuming near-linear scaling; benchmark with `--max-steps 200` first.

## 4. Evaluation

Evals are single-GPU scripts. Checkpoints from DDP runs are ordinary checkpoints.
- **Full suite** for a set of checkpoints: `scripts/run_eval_trunk_jepa.sh LABEL=CKPT ...`. It is memory-guarded, runs one eval at a time, and takes `GPU`, `OUT` and the dense-baseline settings `DE`/`DT`/`DI` from the environment.
- **Shared-trunk models** (`train_mode: dense_private`): always evaluate the **shared trunk only** by prefixing the checkpoint with `TRUNK:`. This rebuilds a plain dense model from the trunk weights and drops every private-LoRA tensor.
- **JEPA / LeJEPA runs:** use `epoch_*.pt`. `best.pt` is chosen by a diffusion loss those runs do not train.
- **Pitfalls:** read EVALUATIONS.md §4 for the known issues, in particular that spatial relations are not tested in structure / retrieval / binding, and that cosine metrics are uncentred.

## 5. Where the research stands (Oct 2026)

- **Goal:** do independently trained / unpaired text and image models converge on shared structure (the "platonic" question)? Tested on CLEVR, with small 8-layer, d=384 masked-diffusion transformers trained on captions and on VQ image tokens.
- **Cone.** Pooled representations of every model sit in a narrow cone: mean cosine 0.9–1.0, effective rank 10–30 of 384. It is mostly a shared mean offset. Linear probes and centred CKA ignore it.
  - Forcing it away with SIGReg / LeJEPA destroyed content: probes fell to chance (`lejepa_*` runs).
  - Paper-style multi-crop LeJEPA also lost binding information on CLEVR.
  - Do not target the cone directly.
- **Unpaired shared trunk + private LoRA** (`mm_unpaired_dense_private_lora_r128_2m_40e`): trunk-only, the text side carries almost no content and collapses (cosine → 1). The rank-128 private LoRA does most of the modality-specific work.
- **Trunk JEPA** (`mm_trunk_ijepa_*_from_ep6_4e`, variants A/B/C):
  - I-JEPA on the trunk-only forward quickly makes the trunk a strong *image* encoder, beating dense image models on probes and binding.
  - The *text* side stays at random-init level.
  - The SIGReg-projector variant (C) collapses text↔image CKA.
- **Text↔image linear CKA:**
  - At 2–4 epochs the shared trunk's text and image features are more similar (0.40) than two separately trained dense models (0.29–0.32).
  - By 7 epochs the separate models catch up (0.40).
  - JEPA on the trunk lowers CKA.
  - Cross-modal retrieval is far better for the separate dense models.
- **Next:** a DDP run of the dense unpaired model (no private LoRA, `configs/cluster/`), and ways to make the shared trunk carry text content (private-branch dropout, lower private rank).
