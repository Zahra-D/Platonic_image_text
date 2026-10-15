# PixMo-Cap: the CLEVR three-stage pipeline on real photos

Branch `pixmo` of `Zahra-D/Platonic_image_text`. Same trainer (`train_multimodal.py`) and the same three stages as
the CLEVR shared-trunk lineage, each trained until it converges, on PixMo-Cap (real photos with long, dense
captions), strictly unpaired:

| Stage | Config | What trains | Starts from | Converged when |
|---|---|---|---|---|
| 1 | `configs/pixmo/stage1_dense_private_diffusion.yaml` | shared trunk + private LoRA (r128), masked diffusion on images and captions, private dropout 0.5, gradient balancing | scratch | validation loss: no gain > 0.01 for 3 epochs (trainer early stopping) |
| 2 | `configs/pixmo/stage2_xview_xpara_trunk_jepa.yaml` | trunk only, data2vec/JEPA with EMA teacher: images across two photometric views, captions with a pooled target from their paraphrase | stage-1 `best.pt` | trunk's oracle-W score (`pixmo/eval_oracle.py`) not improved by ≥ 0.5 R@1 points for 3 epochs (≥ 4 epochs) |
| 3 | `configs/pixmo/stage3_frozen_trunk_private_lora.yaml` | fresh private LoRA (r128) on the frozen stage-2 trunk, masked diffusion | stage-2 best epoch, private LoRA removed (`init_trunk.pt`) | validation loss: no gain > 0.02 for 2 epochs |

`pixmo/pipeline.py` runs all of it: every call continues the current stage from `last.pt`, evaluates each new epoch,
applies the rule above, moves on to the next stage, and exits before its time budget. It is safe to kill and rerun.

## Model and data

* Model: the CLEVR architecture scaled for photos: 12 layers × 768, 12 heads (~110M incl. embeddings), rank-128
  private LoRA on qkv / out_proj / mlp per modality; global batch 256 (128 images + 128 captions per step, as on CLEVR),
  constant lr 3e-4, bf16.
* Images: MaskGIT's ImageNet VQGAN (taming f16, 1024 codes), whole photo squashed to 256×256 → 16×16 tokens (the view
  the DINOv2 teachers saw). Captions: the CLEVR word tokenizer, words seen < 5 times → `<unk>`, up to 320 words.
* Strictly unpaired: every training photo goes either to the image half or to the caption half (hash of its PixMo
  id), so no image is trained together with its own caption. Paraphrase of a caption = the PixMo audio transcript it
  was written from. Image views = the CLEVR photometric jitter (brightness, contrast, gamma, blur, noise).
* Evaluation (`pixmo/eval_oracle.py`): trunk only (private LoRA dropped, like `TRUNK:` in the CLEVR evals). Every
  layer's output and every layer's MLP output, mean-pooled over content tokens, for the 5,000 PixMo test images and
  their captions. **oracle**: orthogonal Procrustes W fitted on 10,000 probe pairs (training images + their own,
  never-trained captions); **identity**: no W. FOSCTTM and R@1/R@10 both ways on the 5k × 5k test pool.
* Reference on the same 5k test set (frozen teachers, oracle W fitted on 94k pairs, `~/platonic_align/RESULTS.md`
  §5.2): DINOv2-g ↔ Qwen3-Emb-8B R@1 57.5 / 63.0, DINOv2-B ↔ MPNet 34.3 / 36.9.

The data is prepared on the development machine (`pixmo/prepare_pixmo.py`, needs the downloaded PixMo images and
the VQGAN) and shipped as one tarball (`pixmo/pack_data.sh`): manifests, token caches, view cache, paraphrases and
evaluation pairs.

## Run on a cluster

```bash
git clone git@github.com:Zahra-D/Platonic_image_text.git platonic_pixmo && cd platonic_pixmo && git checkout pixmo
python3 -m venv ~/venvs/clevr && source ~/venvs/clevr/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match the cluster CUDA
pip install -r requirements.txt
tar -xf platonic_pixmo_data.tar -C /scratch/$USER/                      # -> /scratch/$USER/platonic_pixmo_data

# SLURM, 4 GPUs, 1.5 h slots; the job resubmits itself until stage 3 has converged
PIXMO_DATA=/scratch/$USER/platonic_pixmo_data PIXMO_RUNS=/scratch/$USER/platonic_pixmo_runs \
PIXMO_VENV=~/venvs/clevr sbatch --account=<account> pixmo/slurm_pipeline.sh

# or without SLURM, one long call (e.g. a dedicated node)
PIXMO_DATA=... PIXMO_RUNS=... python pixmo/pipeline.py --ngpu 4 --budget-min 100000
```

Add `PIXMO_FLAGS="--wandb"` for W&B (project `platonic_pixmo`; every slot is its own run in group
`pixmo_three_stage`). Check progress with `cat $PIXMO_RUNS/pipeline.log` and `python pixmo/pipeline.py --status`.

## Outputs (`$PIXMO_RUNS`)

* `pipeline.log`, `pipeline_state.json` (current stage, seconds per epoch, every evaluation score).
* `<stage>/train.log` (trainer), `trainer_stdout.log`, `last.pt`, `best.pt`; stage 2 keeps its best epoch and writes
  `init_trunk.pt`.
* `<stage>/eval/epoch_XXX.json`: per epoch, every readout's oracle and identity scores (stages 1 and 2);
  `stage3.../eval/final_trunk.json` and `final_full.json` at the end.
* Epoch checkpoints are deleted once evaluated (except the ones still needed), so a stage keeps ~3 checkpoints.

## Smoke test (any single GPU, ~15 min)

```bash
PIXMO_MAX_CHUNKS=3 PIXMO_DATA=/tmp/pixmo_smoke python pixmo/prepare_pixmo.py --tokenize --gpu 0   # dev machine only
PIXMO_MAX_CHUNKS=3 PIXMO_DATA=/tmp/pixmo_smoke python pixmo/prepare_pixmo.py --merge
PIXMO_DATA=/tmp/pixmo_smoke PIXMO_RUNS=/tmp/pixmo_smoke_runs python pixmo/pipeline.py --ngpu 1 --budget-min 50 \
  --margin-min 1 --first-epoch-min 1 --global-batch 32 --eval-batch 32 --stage2-min-epochs 2 --stage2-patience 1 \
  --stage2-max-epochs 3 --max-epochs '{"stage1": 2, "stage3": 2}' \
  --override "--d-model 128 --n-layers 8 --n-heads 4 --lora-private-rank 8 --num-workers 2 --val-max-samples 256"
```
