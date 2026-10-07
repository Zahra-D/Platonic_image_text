"""Train a joint CLEVR text/image masked-diffusion model from scratch."""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import hashlib
import json
import logging
import math
import sys
import time
from pathlib import Path

import clevr_paths  # noqa: F401  (sets CLEVR_DATA / CLEVR_GEN defaults for ${CLEVR_DATA} in configs)
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import BalancedUnpairedDataset, ClevrMultimodalDataset, ClevrTextTokenizer, MultimodalCollator
from models import (
    MultimodalMaskedTransformer,
    condition_target_route_ids,
    inject_tri_lora,
    iter_tri_lora,
    shared_route_parameters,
    lora_state_dict,
    parameter_counts,
)
from models.vqvae import VQVAE
from multimodal_diffusion import corrupt_batch, generate_conditioned_images, masked_loss
from unpaired_backtranslation import unpaired_backtranslation_losses
from image_utils import save_image_grid
from alignment_evaluation import evaluate_paired_conditioning
from sigreg import gaussianity_diagnostics, sigreg_loss
from shared_jepa import (
    data2vec_hidden_loss,
    modulewise_data2vec_loss,
    modulewise_private_hsic_loss,
    modulewise_shared_jepa_loss,
    shared_latent_jepa_loss,
)
from lejepa_views import lejepa_multiview_loss, mean_offdiagonal_cosine
from cross_modal_moments import CrossModalMoments
from objective_gradient_diagnostics import objective_shared_gradient_metrics


DEFAULT_CONFIG = str(Path(__file__).resolve().parent / "configs" / "multimodal_dense.yaml")


# Distributed data parallel (torchrun). Single-process runs keep RANK 0 / WORLD 1
# and behave exactly as before.
_RANK, _WORLD, _LOCAL_RANK = 0, 1, 0


def init_distributed() -> None:
    """Join the torchrun process group when WORLD_SIZE > 1 and pin this rank's GPU."""
    global _RANK, _WORLD, _LOCAL_RANK
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return
    _RANK, _WORLD = int(os.environ["RANK"]), world
    _LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(_LOCAL_RANK)
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")


def all_reduce_gradients(model) -> None:
    """Average accumulated gradients over ranks once per optimizer step.

    Explicit instead of a DistributedDataParallel wrapper: the trainer runs
    several forwards per step (EMA teacher, trunk-only JEPA, split diffusion /
    representation backward passes) and some parameters are unused in some
    steps, which DDP's hooks handle poorly. A parameter with a gradient on any
    rank is reduced on every rank (zeros where absent) so all ranks apply the
    same update; one with no gradient anywhere stays None, as in a single run.
    """
    if _WORLD <= 1:
        return
    params = [q for q in model.parameters() if q.requires_grad]
    device = next(model.parameters()).device
    present = torch.tensor([q.grad is not None for q in params], dtype=torch.int32, device=device)
    dist.all_reduce(present)
    grads = []
    for q, n in zip(params, present.tolist()):
        if n == 0:
            continue
        if q.grad is None:
            q.grad = torch.zeros_like(q)
        grads.append(q.grad)
    for dtype in {g.dtype for g in grads}:
        bucket = [g for g in grads if g.dtype == dtype]
        flat = torch._utils._flatten_dense_tensors(bucket)
        dist.all_reduce(flat)
        flat /= _WORLD
        for g, synced in zip(bucket, torch._utils._unflatten_dense_tensors(flat, bucket)):
            g.copy_(synced)


def broadcast_module(module) -> None:
    """Copy rank 0's parameters and buffers to every rank."""
    if _WORLD > 1 and module is not None:
        for tensor in module.state_dict().values():
            dist.broadcast(tensor, 0)


def parameters_in_sync(model) -> bool:
    """Cheap cross-rank check that every rank holds the same weights."""
    if _WORLD <= 1:
        return True
    total = torch.stack([q.detach().double().sum() for q in model.parameters()]).sum().reshape(1)
    gathered = [torch.zeros_like(total) for _ in range(_WORLD)]
    dist.all_gather(gathered, total)
    return all(torch.allclose(g, gathered[0], rtol=1e-9, atol=1e-6) for g in gathered)


def all_reduce_mean(value: float, device) -> float:
    if _WORLD <= 1:
        return value
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    dist.all_reduce(tensor)
    return float(tensor.item() / _WORLD)


def _expand_env(value):
    """Expand $VAR / ${VAR} in config strings so paths can point at a cluster's data root."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=DEFAULT_CONFIG)
    known, _ = pre.parse_known_args()
    with open(known.config) as handle:
        cfg = _expand_env(yaml.safe_load(handle) or {})
    data = cfg.get("data", {})
    tokenizer = cfg.get("tokenizer", {})
    model = cfg.get("model", {})
    lora = cfg.get("lora", {})
    diffusion = cfg.get("diffusion", {})
    train = cfg.get("train", {})
    alignment = cfg.get("alignment", {})
    backtranslation = cfg.get("backtranslation", {})
    translation = cfg.get("translation", {})
    generate = cfg.get("generate", {})
    evaluation = cfg.get("evaluation", {})
    wandb = cfg.get("wandb", {})

    parser = argparse.ArgumentParser(parents=[pre])
    parser.add_argument("--train-dir", default=data.get("train_dir"))
    parser.add_argument("--val-dir", default=data.get("val_dir"))
    parser.add_argument("--train-manifest", default=data.get("train_manifest"))
    parser.add_argument("--val-manifest", default=data.get("val_manifest"))
    parser.add_argument("--train-text-manifest", default=data.get("train_text_manifest"),
                        help="unpaired only: captions from this separate corpus; images from train_manifest.")
    parser.add_argument("--val-text-manifest", default=data.get("val_text_manifest"))
    parser.add_argument("--caption-field", default=data.get("caption_field", "text"),
                        help="Text field in paired manifests, e.g. caption or caption_human.")
    parser.add_argument("--token-cache-dir", default=data.get("token_cache_dir"))
    parser.add_argument("--data-mode", choices=["paired", "unpaired", "text_only", "image_only"], default=data.get("mode", "paired"))
    parser.add_argument("--balanced-modalities", action=argparse.BooleanOptionalAction, default=data.get("balanced_modalities", True))
    parser.add_argument("--max-text-length", type=int, default=data.get("max_text_length", 128))
    parser.add_argument("--num-image-codes", type=int, default=tokenizer.get("num_codes", 512))
    parser.add_argument("--vqvae-checkpoint", default=tokenizer.get("checkpoint"))
    parser.add_argument("--vqvae-embed-dim", type=int, default=tokenizer.get("embed_dim", 64))
    parser.add_argument("--grid-size", type=int, nargs=2, default=model.get("grid_size", [16, 24]))
    parser.add_argument("--d-model", type=int, default=model.get("d_model", 384))
    parser.add_argument("--n-layers", type=int, default=model.get("n_layers", 8))
    parser.add_argument("--n-heads", type=int, default=model.get("n_heads", 6))
    parser.add_argument("--mlp-ratio", type=int, default=model.get("mlp_ratio", 4))
    parser.add_argument("--dropout", type=float, default=model.get("dropout", 0.1))
    parser.add_argument(
        "--use-modality-embeddings", action=argparse.BooleanOptionalAction,
        default=model.get("use_modality_embeddings", True),
        help="Add learned text/image/padding embeddings to every input token.",
    )
    parser.add_argument(
        "--train-mode", choices=["dense", "lora", "dense_private"],
        default=lora.get("train_mode", "dense"),
        help=(
            "dense: ordinary linear layers. lora: shared+private adapters replacing the "
            "base weight. dense_private: keep the dense weight as the shared route and add "
            "only modality-private adapters of rank lora.private_rank."
        ),
    )
    parser.add_argument("--lora-rank", type=int, default=lora.get("rank", 16))
    parser.add_argument(
        "--lora-private-rank", type=int, default=lora.get("private_rank", None),
        help="Private adapter rank in dense_private mode (default d_model/3).",
    )
    parser.add_argument(
        "--freeze-base", action=argparse.BooleanOptionalAction,
        default=lora.get("freeze_base", False),
        help="dense_private only: hold the pretrained dense trunk fixed and train private adapters alone.",
    )
    parser.add_argument(
        "--diffusion-private-only", action=argparse.BooleanOptionalAction,
        default=lora.get("diffusion_private_only", False),
        help="dense_private only: send the diffusion gradient to the private adapters alone and "
             "the JEPA gradient to everything else, so the shared trunk is shaped by the "
             "representation objective while the private branch carries reconstruction. "
             "Requires a JEPA objective and an unfrozen trunk.",
    )
    parser.add_argument("--lora-alpha", type=float, default=lora.get("alpha", 32))
    parser.add_argument("--lora-dropout", type=float, default=lora.get("dropout", 0.05))
    parser.add_argument("--lora-targets", nargs="+", default=lora.get("target_modules", ["qkv", "out_proj", "mlp.0", "mlp.3"]))
    parser.add_argument(
        "--asymmetric-condition-target", action=argparse.BooleanOptionalAction,
        default=lora.get("asymmetric_condition_target", False),
        help="Keep the conditioning modality full-routed and route target tokens through private LoRA only.",
    )
    parser.add_argument("--objective", choices=["both", "image", "text"], default=diffusion.get("objective", "image"))
    parser.add_argument("--eps", type=float, default=diffusion.get("eps", 1e-3))
    # ``unweighting`` is deliberately worded from the user's decision: the
    # default objective is the 1/t-weighted masked loss; turn this on only to
    # request a plain per-masked-token average.  Old YAMLs remain reproducible.
    legacy_weight_by_t = diffusion.get("weight_by_t")
    default_unweighting = diffusion.get(
        "unweighting",
        (not legacy_weight_by_t) if legacy_weight_by_t is not None else False,
    )
    parser.add_argument(
        "--unweighting", action=argparse.BooleanOptionalAction,
        default=default_unweighting,
        help="Use a plain masked-token average; default applies 1/t weighting.",
    )
    parser.add_argument("--full-mask-probability", type=float, default=diffusion.get("full_mask_probability", 0.0))
    parser.add_argument(
        "--mask-block-2d", action=argparse.BooleanOptionalAction,
        default=diffusion.get("mask_block_2d", False),
        help="Mask rectangular blocks on the image token grid (I-JEPA style) instead of "
             "independent tokens; image_only runs, where the eligible tokens are the grid.",
    )
    parser.add_argument(
        "--mask-block-scale", type=float, nargs=2,
        default=diffusion.get("mask_block_scale", [0.10, 0.25]),
        help="Per-block area, as a fraction of the grid, drawn uniformly in this range.",
    )
    parser.add_argument(
        "--mask-block-aspect", type=float, nargs=2,
        default=diffusion.get("mask_block_aspect", [0.75, 1.5]),
        help="Per-block aspect ratio drawn uniformly in this range.",
    )
    parser.add_argument(
        "--mask-span-min", type=int, default=diffusion.get("mask_span_min"),
        help="Mask contiguous windows instead of independent tokens: smallest window size.",
    )
    parser.add_argument(
        "--mask-span-max", type=int, default=diffusion.get("mask_span_max"),
        help="Largest window size; each window's size is drawn uniformly in [min, max].",
    )
    parser.add_argument(
        "--bert-replacement", action=argparse.BooleanOptionalAction,
        default=diffusion.get("bert_replacement", False),
        help="Corrupt selected tokens as BERT and data2vec do: 80%% [MASK], 10%% a random "
             "token of the same modality, 10%% unchanged (the loss still covers all of them).",
    )
    parser.add_argument(
        "--train-fixed-t", type=float, default=diffusion.get("train_fixed_t"),
        help="Mask a constant fraction of eligible tokens during training instead of "
             "sampling t per sample (data2vec-style fixed masking; 0.15 is BERT's rate).",
    )
    parser.add_argument("--train-fixed-t-text", type=float, default=diffusion.get("train_fixed_t_text"),
                        help="Per-modality override of train_fixed_t for text sub-batches.")
    parser.add_argument("--train-fixed-t-image", type=float, default=diffusion.get("train_fixed_t_image"),
                        help="Per-modality override of train_fixed_t for image sub-batches.")
    parser.add_argument("--batch-size", type=int, default=train.get("batch_size", 32))
    parser.add_argument("--epochs", type=int, default=train.get("epochs", 100))
    parser.add_argument("--lr", type=float, default=train.get("lr", 3e-4))
    parser.add_argument(
        "--lr-schedule", choices=["constant", "data2vec_tristage"],
        default=train.get("lr_schedule", "constant"),
        help="data2vec_tristage uses linear warmup, hold, then linear decay.",
    )
    parser.add_argument(
        "--lr-warmup-fraction", type=float,
        default=train.get("lr_warmup_fraction", 0.05),
    )
    parser.add_argument(
        "--lr-hold-fraction", type=float,
        default=train.get("lr_hold_fraction", 0.80),
    )
    parser.add_argument("--shared-lora-lr", type=float, default=train.get("shared_lora_lr"))
    parser.add_argument("--private-lora-lr", type=float, default=train.get("private_lora_lr"))
    parser.add_argument("--embedding-lr", type=float, default=train.get("embedding_lr"))
    parser.add_argument(
        "--shared-only-epochs", type=int, default=train.get("shared_only_epochs", 0),
        help="For Tri-LoRA, freeze text/image-private adapters for the first N epochs.",
    )
    parser.add_argument("--weight-decay", type=float, default=train.get("weight_decay", 0.01))
    parser.add_argument("--num-workers", type=int, default=train.get("num_workers", 4))
    parser.add_argument("--amp", choices=["none", "bf16"], default=train.get("amp", "bf16"))
    parser.add_argument("--output-dir", default=train.get("output_dir", "outputs/multimodal_dense"))
    parser.add_argument(
        "--probe-every-steps", type=int, default=train.get("probe_every_steps", 0),
        help="Score the selected representation probe every N optimizer steps (0 disables). "
             "The probe runs as a detached process, so training never waits for it.",
    )
    parser.add_argument(
        "--probe-type", choices=["hard_retrieval", "semantic_dprime"],
        default=train.get("probe_type", "hard_retrieval"),
    )
    parser.add_argument(
        "--probe-num-worlds", type=int, default=train.get("probe_num_worlds", 500),
        help="Scenes per periodic probe; 500 takes about 45 s.",
    )
    parser.add_argument(
        "--probe-sublayer-layers", type=int, nargs="+",
        default=train.get("probe_sublayer_layers", [4, 5, 6, 7]),
        help="Blocks whose attention/MLP writes the periodic probe scores in addition "
             "to every block's residual stream.",
    )
    parser.add_argument(
        "--probe-gpu", default=train.get("probe_gpu"),
        help="GPU for the periodic probe; defaults to the training GPU (it needs a few GB).",
    )
    parser.add_argument(
        "--probe-lexicon-matched", action=argparse.BooleanOptionalAction,
        default=train.get("probe_lexicon_matched", True),
        help="For --probe-type semantic_dprime: give the paraphrase the query's attribute "
             "vocabulary, so d_semantic measures invariance to phrasing rather than synonym "
             "knowledge. On by default; d_binding is unaffected either way.")
    parser.add_argument(
        "--probe-bootstrap", type=int, default=train.get("probe_bootstrap", 500),
        help="Bootstrap repetitions for semantic-dprime probes.",
    )
    parser.add_argument(
        "--probe-keep-checkpoints", action=argparse.BooleanOptionalAction,
        default=train.get("probe_keep_checkpoints", False),
        help="Retain the slim probe-time model checkpoint after evaluation.",
    )
    parser.add_argument("--resume", default=train.get("resume"))
    parser.add_argument("--init-checkpoint", default=train.get("init_checkpoint"),
                        help="Initialize model weights only; unlike --resume, starts a fresh optimizer/run")
    parser.add_argument("--log-every", type=int, default=train.get("log_every", 20))
    parser.add_argument("--eval-every", type=int, default=train.get("eval_every", 1))
    parser.add_argument(
        "--early-stopping-patience", type=int,
        default=train.get("early_stopping_patience", 0),
        help="Stop after this many epoch evaluations without selection-loss improvement; 0 disables it.",
    )
    parser.add_argument(
        "--early-stopping-min-delta", type=float,
        default=train.get("early_stopping_min_delta", 0.0),
        help="Minimum decrease in validation selection loss that resets early-stopping patience.",
    )
    parser.add_argument("--eval-only", action="store_true", help="Evaluate --resume and exit without an optimizer step")
    parser.add_argument("--save-every", type=int, default=train.get("save_every", 1))
    parser.add_argument("--max-train-samples", type=int, default=train.get("max_train_samples"))
    parser.add_argument(
        "--train-subset-seed", type=int, default=train.get("subset_seed"),
        help="Select max_train_samples with an independent deterministic permutation and record the indices.",
    )
    parser.add_argument("--val-max-samples", type=int, default=train.get("val_max_samples", 1024))
    parser.add_argument(
        "--val-diffusion-nll-draws", type=int,
        default=train.get("val_diffusion_nll_draws", 0),
        help=(
            "Deterministic Monte Carlo draws per held-out example for the full "
            "random-t diffusion objective; 0 disables this additional validation."
        ),
    )
    parser.add_argument("--paired-val", action=argparse.BooleanOptionalAction, default=evaluation.get("paired_enabled", True))
    parser.add_argument("--paired-val-max-samples", type=int, default=evaluation.get("paired_max_samples", 512))
    parser.add_argument("--paired-val-mask-ratios", type=float, nargs="+", default=evaluation.get("mask_ratios", [0.5, 0.75, 1.0]))
    parser.add_argument("--paired-val-control-ratios", type=float, nargs="+", default=evaluation.get("control_ratios", [0.75, 1.0]))
    parser.add_argument(
        "--paired-val-directions", nargs="+",
        choices=["text_to_image", "image_to_text"],
        default=evaluation.get("directions", ["text_to_image", "image_to_text"]),
    )
    parser.add_argument("--step-eval-every", type=int, default=evaluation.get("step_every", 0))
    parser.add_argument("--step-eval-max-samples", type=int, default=evaluation.get("step_max_samples", 128))
    parser.add_argument("--step-eval-mask-ratios", type=float, nargs="+", default=evaluation.get("step_mask_ratios", [1.0]))
    parser.add_argument("--step-eval-control-ratios", type=float, nargs="+", default=evaluation.get("step_control_ratios", [1.0]))
    parser.add_argument("--validation-selection", choices=["marginal_t0.75", "paired_t1_matched", "paired_t1_text_to_image_matched"],
                        default=evaluation.get("selection_metric", "marginal_t0.75"))
    parser.add_argument("--max-steps", type=int, default=train.get("max_steps"))
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                        default=train.get("gradient_checkpointing", False),
                        help="Recompute Transformer block activations in backward to save memory.")
    parser.add_argument("--seed", type=int, default=train.get("seed", 13))
    parser.add_argument("--gradient-accumulation-steps", type=int, default=train.get("gradient_accumulation_steps", 1))
    parser.add_argument("--max-grad-norm", type=float, default=train.get("max_grad_norm", 1.0))
    parser.add_argument("--gradient-balance", action=argparse.BooleanOptionalAction, default=alignment.get("gradient_balance_enabled", False))
    parser.add_argument("--gradient-balance-ema-decay", type=float, default=alignment.get("gradient_balance_ema_decay", 0.9))
    parser.add_argument("--gradient-balance-min-weight", type=float, default=alignment.get("gradient_balance_min_weight", 0.25))
    parser.add_argument("--gradient-balance-max-weight", type=float, default=alignment.get("gradient_balance_max_weight", 4.0))
    parser.add_argument("--shared-gradient-diagnostics", action=argparse.BooleanOptionalAction, default=alignment.get("shared_gradient_diagnostics", False))
    parser.add_argument("--gradient-diagnostics-samples", type=int, default=alignment.get("gradient_diagnostics_samples_per_tensor", 64))
    parser.add_argument(
        "--objective-gradient-diagnostics-every", type=int,
        default=alignment.get("objective_gradient_diagnostics_every", 0),
        help=(
            "Every N optimizer steps, decompose shared A/B gradients into diffusion, "
            "weighted JEPA, and weighted SIGReg components; 0 disables it"
        ),
    )
    parser.add_argument(
        "--objective-gradient-diagnostic-layers", type=int, nargs="+",
        default=alignment.get("objective_gradient_diagnostic_layers", [2, 3, 4]),
        help="Transformer layers included in objective-gradient decomposition",
    )
    parser.add_argument(
        "--cross-modal-moment-weight", type=float,
        default=alignment.get("cross_modal_moment_weight", 0.0),
        help="Pull each modality's pooled mean and covariance toward the other modality's "
             "EMA statistics. Unlike SIGReg the target is the other modality's own "
             "distribution, not a fixed isotropic prior, so shared structure is what the "
             "two are aligned onto. 0 disables it.")
    parser.add_argument("--cross-modal-moment-layer", type=int,
                        default=alignment.get("cross_modal_moment_layer", -1),
                        help="Block whose pooled output is aligned; -1 is the last block.")
    parser.add_argument("--cross-modal-moment-decay", type=float,
                        default=alignment.get("cross_modal_moment_decay", 0.95))
    parser.add_argument("--modality-adversarial", action=argparse.BooleanOptionalAction, default=alignment.get("modality_adversarial_enabled", False))
    parser.add_argument("--modality-adversarial-weight", type=float, default=alignment.get("modality_adversarial_weight", 0.1))
    parser.add_argument("--modality-adversarial-grl-lambda", type=float, default=alignment.get("modality_adversarial_grl_lambda", 1.0))
    parser.add_argument(
        "--modality-adversarial-representation-normalization",
        choices=["none", "l2"],
        default=alignment.get("modality_adversarial_representation_normalization", "none"),
        help="Normalize the pooled shared activation before gradient reversal and DANN.",
    )
    parser.add_argument("--modality-adversarial-warmup-steps", type=int, default=alignment.get("modality_adversarial_warmup_steps", 0))
    parser.add_argument(
        "--modality-adversarial-start-epoch", type=int,
        default=alignment.get("modality_adversarial_start_epoch", 0),
        help="Keep DANN inactive before this epoch; its warmup begins when it becomes active.",
    )
    parser.add_argument("--modality-discriminator-hidden", type=int, default=alignment.get("modality_discriminator_hidden"))
    parser.add_argument("--modality-discriminator-lr", type=float, default=alignment.get("modality_discriminator_lr"))
    parser.add_argument("--sigreg", action=argparse.BooleanOptionalAction,
                        default=alignment.get("sigreg_enabled", False))
    parser.add_argument(
        "--sigreg-per-layer", action=argparse.BooleanOptionalAction,
        default=alignment.get("sigreg_per_layer", False),
        help="Apply SIGReg separately to each Transformer block's pooled shared adapters, then average the 8 statistics.",
    )
    parser.add_argument(
        "--sigreg-layers", type=int, nargs="+", default=alignment.get("sigreg_layers"),
        help="Apply per-layer SIGReg only to these layers; default is every layer",
    )
    parser.add_argument("--sigreg-start-epoch", type=int,
                        default=alignment.get("sigreg_start_epoch", 0))
    parser.add_argument("--sigreg-weight", type=float,
                        default=alignment.get("sigreg_weight", 0.01))
    parser.add_argument("--sigreg-warmup-steps", type=int,
                        default=alignment.get("sigreg_warmup_steps", 1000))
    parser.add_argument("--sigreg-num-slices", type=int,
                        default=alignment.get("sigreg_num_slices", 256))
    parser.add_argument("--sigreg-num-points", type=int,
                        default=alignment.get("sigreg_num_points", 17))
    parser.add_argument("--sigreg-t-max", type=float,
                        default=alignment.get("sigreg_t_max", 3.0))
    parser.add_argument(
        "--shared-jepa", action=argparse.BooleanOptionalAction,
        default=alignment.get("shared_jepa_enabled", False),
        help="Predict clean per-token shared-LoRA latents from the masked forward pass.",
    )
    parser.add_argument("--shared-jepa-start-epoch", type=int,
                        default=alignment.get("shared_jepa_start_epoch", 0))
    parser.add_argument("--shared-jepa-weight", type=float,
                        default=alignment.get("shared_jepa_weight", 0.1))
    parser.add_argument(
        "--shared-jepa-text-weight", type=float,
        default=alignment.get("shared_jepa_text_weight"),
        help="Optional text-specific JEPA coefficient; defaults to shared_jepa_weight",
    )
    parser.add_argument(
        "--shared-jepa-image-weight", type=float,
        default=alignment.get("shared_jepa_image_weight"),
        help="Optional image-specific JEPA coefficient; defaults to shared_jepa_weight",
    )
    parser.add_argument(
        "--shared-jepa-layers", type=int, nargs="+",
        default=alignment.get("shared_jepa_layers"),
        help="Apply shared JEPA only to these layers; default is every layer",
    )
    parser.add_argument(
        "--shared-jepa-loss", choices=["mse", "normalized_mse", "cosine"],
        default=alignment.get("shared_jepa_loss", "mse"),
    )
    parser.add_argument("--shared-jepa-warmup-steps", type=int,
                        default=alignment.get("shared_jepa_warmup_steps", 1000))
    parser.add_argument(
        "--shared-jepa-predictor-hidden-multiplier", type=int,
        default=alignment.get("shared_jepa_predictor_hidden_multiplier", 2),
    )
    parser.add_argument("--shared-jepa-predictor-lr", type=float,
                        default=alignment.get("shared_jepa_predictor_lr"))
    parser.add_argument(
        "--shared-jepa-ema", action=argparse.BooleanOptionalAction,
        default=alignment.get("shared_jepa_ema_enabled", False),
        help="Use a clean EMA-teacher shared representation instead of an online stop-gradient target.",
    )
    parser.add_argument(
        "--shared-jepa-ema-decay", type=float,
        default=alignment.get("shared_jepa_ema_decay", 0.999),
        help="Per-update EMA teacher decay.",
    )
    parser.add_argument(
        "--shared-jepa-dynamic-weight", action=argparse.BooleanOptionalAction,
        default=alignment.get("shared_jepa_dynamic_weight", False),
        help="Set the JEPA coefficient from the shared diffusion/JEPA gradient-norm ratio.",
    )
    parser.add_argument(
        "--shared-jepa-gradient-ratio", type=float,
        default=alignment.get("shared_jepa_gradient_ratio", 0.25),
        help="Desired norm of weighted JEPA gradient divided by diffusion gradient on shared A/B.",
    )
    parser.add_argument(
        "--shared-jepa-dynamic-ema-decay", type=float,
        default=alignment.get("shared_jepa_dynamic_ema_decay", 0.95),
    )
    parser.add_argument(
        "--shared-jepa-dynamic-min-weight", type=float,
        default=alignment.get("shared_jepa_dynamic_min_weight", 1e-4),
    )
    parser.add_argument(
        "--shared-jepa-dynamic-max-weight", type=float,
        default=alignment.get("shared_jepa_dynamic_max_weight", 10.0),
    )
    parser.add_argument(
        "--modulewise-jepa-mode", choices=[
            "none", "average", "layerwise", "data2vec_average", "data2vec_no_average"
        ],
        default=alignment.get("modulewise_jepa_mode", "none"),
        help="Use native modulewise JEPA instead of pooled shared-token JEPA.",
    )
    parser.add_argument(
        "--modulewise-jepa-modules", nargs="+",
        default=alignment.get("modulewise_jepa_modules", ["out_proj", "mlp.3"]),
        help="Native shared LoRA modules receiving modulewise JEPA/HSIC gradients.",
    )
    parser.add_argument(
        "--modulewise-jepa-gradient-modules", nargs="+",
        default=alignment.get("modulewise_jepa_gradient_modules"),
        help="Shared LoRA modules that receive the gated end-to-end data2vec JEPA gradient.",
    )
    parser.add_argument(
        "--data2vec-hidden", action=argparse.BooleanOptionalAction,
        default=alignment.get("data2vec_hidden", False),
        help="Faithful data2vec: regress the EMA teacher's averaged block hidden states "
             "from the student's final hidden state through one prediction head.",
    )
    parser.add_argument("--data2vec-top-k", type=int, default=alignment.get("data2vec_top_k", 8),
                        help="How many of the teacher's top blocks are averaged into the target.")
    parser.add_argument(
        "--data2vec-teacherless", action=argparse.BooleanOptionalAction,
        default=alignment.get("data2vec_teacherless", False),
        help="LeJEPA instead of data2vec: drop the EMA teacher, the stop-gradient and the "
             "parameter-free target LayerNorm, and prevent collapse with SIGReg on the "
             "embedding distribution. LeJEPA's lambda: the objective is "
             "(1 - lambda) * prediction + lambda * SIGReg (arXiv 2511.08544).",
    )
    parser.add_argument(
        "--data2vec-sigreg-weight", type=float,
        default=alignment.get("data2vec_sigreg_weight", 0.05),
        help="Weight of the SIGReg term in the teacherless objective.",
    )
    parser.add_argument(
        "--lejepa-views", action=argparse.BooleanOptionalAction,
        default=alignment.get("lejepa_views", False),
        help="Multi-view LeJEPA as in the paper: V_g global + V_l local crops through one "
             "network, pooled, projected; invariance to the global-view centre plus "
             "per-view SIGReg on the projector output (see lejepa_views.py).",
    )
    parser.add_argument("--lejepa-global-views", type=int, default=alignment.get("lejepa_global_views", 2))
    parser.add_argument("--lejepa-local-views", type=int, default=alignment.get("lejepa_local_views", 6))
    parser.add_argument("--lejepa-global-scale", type=float, nargs=2,
                        default=alignment.get("lejepa_global_scale", [0.3, 1.0]))
    parser.add_argument("--lejepa-local-scale", type=float, nargs=2,
                        default=alignment.get("lejepa_local_scale", [0.05, 0.3]))
    parser.add_argument("--lejepa-aspect", type=float, nargs=2,
                        default=alignment.get("lejepa_aspect", [0.75, 4.0 / 3.0]))
    parser.add_argument("--lejepa-projector-hidden", type=int,
                        default=alignment.get("lejepa_projector_hidden", 2048))
    parser.add_argument("--lejepa-projector-dim", type=int,
                        default=alignment.get("lejepa_projector_dim", 128))
    parser.add_argument("--lejepa-lambda", type=float, default=alignment.get("lejepa_lambda", 0.05))
    parser.add_argument(
        "--jepa-trunk-only", action=argparse.BooleanOptionalAction,
        default=alignment.get("jepa_trunk_only", False),
        help="dense_private: run the data2vec student and teacher through the shared trunk "
             "alone (private LoRA routes off). A diffusion loss, if any, still uses the full "
             "model through a separate forward.",
    )
    parser.add_argument(
        "--data2vec-projector-sigreg-weight", type=float,
        default=alignment.get("data2vec_projector_sigreg_weight", 0.0),
        help="LeJEPA hybrid: keep the EMA teacher and add SIGReg on a projector over the "
             "pooled last-block student embedding, per modality sub-batch; the JEPA term "
             "becomes (1 - w) * data2vec + w * SIGReg.",
    )
    parser.add_argument(
        "--data2vec-target-layer", type=int,
        default=alignment.get("data2vec_target_layer", None),
        help="fixed_target mode only: the single teacher block every student block predicts.",
    )
    parser.add_argument(
        "--data2vec-mode", choices=["average", "layerwise", "fixed_target"],
        default=alignment.get("data2vec_mode", "average"),
        help="average: one head on the final block against the mean of the selected teacher "
             "blocks. layerwise: one head per selected block against that same block's target.",
    )
    parser.add_argument(
        "--data2vec-layers", type=int, nargs="+", default=alignment.get("data2vec_layers"),
        help="Blocks the data2vec objective uses; defaults to the last data2vec_top_k blocks.",
    )
    parser.add_argument(
        "--data2vec-target-type", choices=["block_residual", "ffn_output"],
        default=alignment.get("data2vec_target_type", "block_residual"),
        help="Teacher feature used as the latent target. ffn_output is the data2vec paper recipe.",
    )
    parser.add_argument(
        "--data2vec-share-input-encoder", action=argparse.BooleanOptionalAction,
        default=alignment.get("data2vec_share_input_encoder", False),
        help="Build clean teacher token/position/modality embeddings with the online student encoder.",
    )
    parser.add_argument("--data2vec-beta", type=float, default=alignment.get("data2vec_beta", 2.0),
                        help="SmoothL1 beta for the data2vec regression.")
    parser.add_argument("--data2vec-weight", type=float, default=alignment.get("data2vec_weight", 1.0))
    parser.add_argument(
        "--data2vec-token-weight", type=float,
        default=alignment.get("data2vec_token_weight", 1.0),
        help="Coefficient on masked-token data2vec regression inside the JEPA loss. "
             "Lower values let the pooled scene-level target dominate without enabling diffusion.",
    )
    parser.add_argument(
        "--data2vec-global-weight", type=float,
        default=alignment.get("data2vec_global_weight", 0.0),
        help="Additional clean-scene JEPA loss on content-token pooled hidden states. "
             "This gives from-scratch runs an explicit sequence-level target.",
    )
    parser.add_argument(
        "--data2vec-variance-weight", type=float,
        default=alignment.get("data2vec_variance_weight", 0.0),
        help="VICReg-style variance-floor coefficient on the pooled student hidden state.",
    )
    parser.add_argument(
        "--data2vec-variance-target", type=float,
        default=alignment.get("data2vec_variance_target", 1.0),
        help="Minimum per-feature batch standard deviation after per-example LayerNorm.",
    )
    parser.add_argument("--diffusion-weight", type=float, default=diffusion.get("weight", 1.0),
                        help="Coefficient on the masked-token diffusion loss; 0 trains representation only.")
    parser.add_argument("--shared-jepa-ema-decay-final", type=float,
                        default=alignment.get("shared_jepa_ema_decay_final"),
                        help="Final EMA decay; ramps linearly from shared_jepa_ema_decay (data2vec schedule).")
    parser.add_argument("--shared-jepa-ema-ramp-steps", type=int,
                        default=alignment.get("shared_jepa_ema_ramp_steps", 0))
    parser.add_argument(
        "--modulewise-jepa-gated-predictor", action=argparse.BooleanOptionalAction,
        default=alignment.get("modulewise_jepa_gated_predictor", False),
        help="Keep the JEPA predictor and the scale-free loss, but route the predictor "
             "gradient end-to-end into the gated shared LoRA modules instead of "
             "confining it to each supervised layer.",
    )
    parser.add_argument(
        "--modulewise-jepa-smooth-l1-beta", type=float,
        default=alignment.get("modulewise_jepa_smooth_l1_beta", 4.0),
        help="SmoothL1 beta for direct data2vec-style native-update regression.",
    )
    parser.add_argument(
        "--modulewise-hsic", action=argparse.BooleanOptionalAction,
        default=alignment.get("modulewise_hsic_enabled", False),
        help="Decorrelate selected native shared and text-private adapter updates with RBF-HSIC.",
    )
    parser.add_argument(
        "--modulewise-hsic-gradient-ratio", type=float,
        default=alignment.get("modulewise_hsic_gradient_ratio", 0.05),
    )
    parser.add_argument(
        "--modulewise-hsic-update-every", type=int,
        default=alignment.get("modulewise_hsic_update_every", 100),
    )
    parser.add_argument(
        "--modulewise-hsic-ema-decay", type=float,
        default=alignment.get("modulewise_hsic_ema_decay", 0.95),
    )
    parser.add_argument(
        "--modulewise-hsic-min-weight", type=float,
        default=alignment.get("modulewise_hsic_min_weight", 1.0e-4),
    )
    parser.add_argument(
        "--modulewise-hsic-max-weight", type=float,
        default=alignment.get("modulewise_hsic_max_weight", 1.0),
    )
    parser.add_argument(
        "--modulewise-hsic-warmup-steps", type=int,
        default=alignment.get("modulewise_hsic_warmup_steps", 1000),
    )
    parser.add_argument(
        "--modulewise-hsic-max-tokens", type=int,
        default=alignment.get("modulewise_hsic_max_tokens", 512),
    )
    parser.add_argument(
        "--shared-translation", action=argparse.BooleanOptionalAction,
        default=translation.get("enabled", False),
        help="Use condition-shared K/V cross-attention for translation.",
    )
    parser.add_argument(
        "--shared-translation-layers", type=int, nargs="+",
        default=translation.get("layers", [2, 3, 4]),
    )
    parser.add_argument(
        "--shared-translation-heads", type=int,
        default=translation.get("heads"),
    )
    parser.add_argument(
        "--shared-translation-dropout", type=float,
        default=translation.get("dropout", 0.0),
    )
    parser.add_argument(
        "--shared-translation-mode",
        choices=["projected_cross_attention", "soft_permutation", "module_replacement"],
        default=translation.get("mode", "projected_cross_attention"),
        help=(
            "projected_cross_attention learns V/output projections; soft_permutation "
            "uses source shared vectors directly and only resamples their token axis; "
            "module_replacement performs this substitution inside every Tri-LoRA module"
        ),
    )
    parser.add_argument(
        "--shared-translation-lr", type=float,
        default=translation.get("lr"),
    )
    parser.add_argument("--backtranslation", action=argparse.BooleanOptionalAction, default=backtranslation.get("enabled", False))
    parser.add_argument("--backtranslation-start-epoch", type=int, default=backtranslation.get("start_epoch", 0))
    parser.add_argument("--backtranslation-every-n-microsteps", type=int, default=backtranslation.get("every_n_microsteps", 10))
    parser.add_argument("--backtranslation-batch-size", type=int, default=backtranslation.get("batch_size", 8))
    parser.add_argument("--backtranslation-generation-steps", type=int, default=backtranslation.get("generation_steps", 12))
    parser.add_argument("--backtranslation-temperature", type=float, default=backtranslation.get("temperature", 1.0))
    parser.add_argument("--backtranslation-reveal-order", choices=["confidence", "random"], default=backtranslation.get("reveal_order", "confidence"))
    parser.add_argument("--backtranslation-min-confidence", type=float, default=backtranslation.get("min_confidence", 0.0))
    parser.add_argument("--backtranslation-cycle-weight", type=float, default=backtranslation.get("cycle_weight", 0.1))
    parser.add_argument("--pseudo-align-weight", type=float, default=backtranslation.get("pseudo_align_weight", 0.05))
    parser.add_argument("--pseudo-align-loss", choices=["cosine", "contrastive"], default=backtranslation.get("pseudo_align_loss", "cosine"))
    parser.add_argument("--pseudo-align-temperature", type=float, default=backtranslation.get("pseudo_align_temperature", 0.07))
    parser.add_argument("--backtranslation-warmup-steps", type=int, default=backtranslation.get("warmup_steps", 1000))
    parser.add_argument("--gen-num-samples", type=int, default=generate.get("num_samples", 4))
    parser.add_argument("--gen-num-steps", type=int, default=generate.get("num_steps", 50))
    parser.add_argument("--gen-temperature", type=float, default=generate.get("temperature", 1.0))
    parser.add_argument("--gen-reveal-order", choices=["confidence", "random"], default=generate.get("reveal_order", "confidence"))
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=wandb.get("enabled", False))
    parser.add_argument("--wandb-project", default=wandb.get("project", "clevr-multimodal-diffusion"))
    parser.add_argument("--wandb-entity", default=wandb.get("entity"))
    parser.add_argument("--wandb-group", default=wandb.get("group", "multimodal"))
    parser.add_argument("--wandb-run-name", default=wandb.get("run_name"))
    parser.add_argument(
        "--wandb-resume-id", default=wandb.get("resume_id"),
        help="Append metrics to an existing W&B run ID when resuming a checkpoint.",
    )
    parser.add_argument(
        "--wandb-tags", nargs="*", default=wandb.get("tags", []),
        help="W&B tags; YAML accepts wandb.tags as a list of strings.",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path, resume: str | None) -> logging.Logger:
    logger = logging.getLogger("train_multimodal")
    logger.setLevel(logging.INFO if _RANK == 0 else logging.WARNING)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handlers = [logging.StreamHandler(sys.stdout)]
    if _RANK == 0:
        handlers.append(logging.FileHandler(output_dir / "train.log", mode="a" if resume else "w"))
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def deterministic_subset_indices(size: int, limit: int, seed: int | None) -> list[int]:
    """Select an auditable subset without consuming the global training RNG."""
    count = min(max(1, limit), size)
    if seed is None:
        return list(range(count))
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(size, generator=generator)[:count].tolist()


def build_model(args, vocabulary_size: int):
    max_positions = max(args.max_text_length + 3, args.grid_size[0] * args.grid_size[1] + 3)
    model = MultimodalMaskedTransformer(
        vocabulary_size + args.num_image_codes,
        max_positions,
        args.d_model,
        args.n_layers,
        args.n_heads,
        args.mlp_ratio,
        args.dropout,
        getattr(args, "modality_adversarial", False),
        getattr(args, "modality_discriminator_hidden", None),
        getattr(args, "asymmetric_condition_target", False),
        getattr(args, "modality_adversarial_representation_normalization", "none"),
        getattr(args, "use_modality_embeddings", True),
        getattr(args, "shared_jepa", False),
        getattr(args, "shared_jepa_predictor_hidden_multiplier", 2),
        (
            getattr(args, "shared_translation_layers", [2, 3, 4])
            if getattr(args, "shared_translation", False) else None
        ),
        getattr(args, "shared_translation_heads", None),
        getattr(args, "shared_translation_dropout", 0.0),
        getattr(args, "shared_translation_mode", "projected_cross_attention"),
        getattr(args, "modulewise_jepa_mode", "none"),
        getattr(args, "shared_jepa_layers", None),
        getattr(args, "modulewise_jepa_modules", None),
        data2vec_hidden=getattr(args, "data2vec_hidden", False),
        data2vec_layerwise_layers=(
            data2vec_selected_layers(args)
            if getattr(args, "data2vec_hidden", False)
            and getattr(args, "data2vec_mode", "average") in {"layerwise", "fixed_target"} else None
        ),
        lejepa_projector_dims=(
            [args.lejepa_projector_hidden, args.lejepa_projector_hidden, args.lejepa_projector_dim]
            if getattr(args, "lejepa_views", False)
            or getattr(args, "data2vec_projector_sigreg_weight", 0.0) > 0 else None
        ),
    )
    lora_modules = []
    if args.train_mode == "dense_private":
        # The dense weight stays and becomes the shared route; only private
        # adapters are added, so the shared subspace is full rank and the
        # private subspace is rank-limited by construction.
        torch.manual_seed(args.seed)
        lora_modules = inject_tri_lora(
            model, args.lora_targets, args.lora_rank, args.lora_alpha, args.lora_dropout,
            delete_base_weights=False, shared_branch=False,
            private_rank=args.lora_private_rank,
        )
        if args.freeze_base:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            for _, module in iter_tri_lora(model):
                for branch in ("text", "image"):
                    for suffix in ("A", "B"):
                        parameter = getattr(module, f"{branch}_{suffix}", None)
                        if parameter is not None:
                            parameter.requires_grad_(True)
            for module_name, module in model.named_modules():
                if (
                    module_name.startswith("modality_discriminator")
                    or module_name.startswith("shared_jepa_predictors")
                    or module_name.startswith("modulewise_jepa_predictors")
                    or module_name.startswith(("data2vec_head", "data2vec_heads"))
                ):
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
        else:
            # TriLoRALinear freezes the base it wraps, so the trunk has to be
            # re-enabled explicitly when this condition trains it.
            for _, module in iter_tri_lora(model):
                for parameter in module.base.parameters():
                    parameter.requires_grad_(True)
    elif args.train_mode == "lora":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        # The optional DANN head is constructed before adapter injection and
        # consumes RNG. Give LoRA its own deterministic initialization point
        # so DANN/non-DANN ablations start from identical task parameters.
        torch.manual_seed(args.seed)
        lora_modules = inject_tri_lora(
            model, args.lora_targets, args.lora_rank, args.lora_alpha, args.lora_dropout,
            delete_base_weights=True,
        )
        # A randomly initialized model needs learned input/output bases and normalization.
        for module_name, module in model.named_modules():
            if (
                isinstance(module, (torch.nn.Embedding, torch.nn.LayerNorm))
                or module_name == "head"
                or module_name.startswith("modality_discriminator")
                or module_name.startswith("shared_jepa_predictors")
                or module_name.startswith("modulewise_jepa_predictors")
                or module_name.startswith(("data2vec_head", "data2vec_heads"))
                or module_name.startswith("shared_translation_bridges")
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
    return model, lora_modules


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def modality_sub_batches(batch, args):
    """Return independent loss specifications with matched paired/unpaired exposure."""
    if "text_batch" in batch:
        if args.objective == "text":
            return [("text", batch["text_batch"], "text", 0)]
        if args.objective == "image":
            return [("image", batch["image_batch"], "image", 1)]
        return [
            ("text", batch["text_batch"], "text", 0),
            ("image", batch["image_batch"], "image", 1),
        ]
    if args.objective == "both":
        if getattr(args, "asymmetric_condition_target", False):
            # One optimizer update sees a total of batch_size pairs: half are
            # image<-text and half are text<-image, as opposed to duplicating
            # the entire batch in both directions.
            midpoint = batch["input_ids"].size(0) // 2
            if midpoint < 1:
                raise ValueError("Role-routed bidirectional training requires batch size >= 2")
            text_batch = {key: value[:midpoint] for key, value in batch.items()}
            image_batch = {key: value[midpoint:] for key, value in batch.items()}
            text_batch["route_ids"] = condition_target_route_ids(
                text_batch["route_ids"], text_batch["modality_ids"], 1
            )
            image_batch["route_ids"] = condition_target_route_ids(
                image_batch["route_ids"], image_batch["modality_ids"], 2
            )
            return [
                ("text", text_batch, "text", 0),
                ("image", image_batch, "image", 1),
            ]
        # Bidirectional paired pretraining: each modality is predicted once while
        # the paired other modality remains clean context.
        return [
            ("text", batch, "text", 0),
            ("image", batch, "image", 1),
        ]
    route = 1 if args.objective == "image" else 0
    if getattr(args, "asymmetric_condition_target", False):
        batch = dict(batch)
        batch["route_ids"] = condition_target_route_ids(
            batch["route_ids"], batch["modality_ids"], 2 if args.objective == "image" else 1
        )
    return [(args.objective, batch, args.objective, route)]


def optimizer_groups(model, args):
    groups = {
        "backbone": [], "shared_lora": [], "private_lora": [], "embeddings": [],
        "discriminator": [], "jepa_predictor": [], "translation_bridge": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("modality_discriminator"):
            groups["discriminator"].append(parameter)
        elif name.startswith(("shared_jepa_predictors", "modulewise_jepa_predictors", "data2vec_head", "data2vec_heads", "lejepa_projector")):
            groups["jepa_predictor"].append(parameter)
        elif name.startswith("shared_translation_bridges"):
            groups["translation_bridge"].append(parameter)
        elif any(key in name for key in ("token_embed", "position_embed", "modality_embed", "head")):
            groups["embeddings"].append(parameter)
        elif any(marker in name for marker in (".shared_A", ".shared_B", ".shared_bias")):
            groups["shared_lora"].append(parameter)
        elif any(marker in name for marker in (".text_A", ".text_B", ".image_A", ".image_B")):
            groups["private_lora"].append(parameter)
        else:
            groups["backbone"].append(parameter)
    learning_rates = {
        "backbone": args.lr,
        "shared_lora": args.shared_lora_lr if args.shared_lora_lr is not None else args.lr,
        "private_lora": args.private_lora_lr if args.private_lora_lr is not None else args.lr,
        "embeddings": args.embedding_lr if args.embedding_lr is not None else args.lr,
        "discriminator": args.modality_discriminator_lr if args.modality_discriminator_lr is not None else args.lr,
        "jepa_predictor": (
            getattr(args, "shared_jepa_predictor_lr", None)
            if getattr(args, "shared_jepa_predictor_lr", None) is not None
            else args.lr
        ),
        "translation_bridge": (
            getattr(args, "shared_translation_lr", None)
            if getattr(args, "shared_translation_lr", None) is not None
            else args.lr
        ),
    }
    return [
        {"params": parameters, "lr": learning_rates[name], "group_name": name}
        for name, parameters in groups.items() if parameters
    ]


def set_private_lora_trainable(model, enabled: bool) -> int:
    """Toggle both modality-private branches while leaving shared LoRA untouched."""
    tensors = 0
    for _, module in iter_tri_lora(model):
        for branch in ("text", "image"):
            for suffix in ("A", "B"):
                getattr(module, f"{branch}_{suffix}").requires_grad_(enabled)
                tensors += 1
    return tensors


class SharedGradientBalancer:
    """Omni-style EMA balancing applied only to shared Tri-LoRA gradients."""

    def __init__(self, model, args):
        self.balance_enabled = args.gradient_balance
        self.diagnostics = args.shared_gradient_diagnostics
        self.enabled = bool((self.balance_enabled or self.diagnostics) and list(iter_tri_lora(model)))
        self.decay = args.gradient_balance_ema_decay
        self.minimum = args.gradient_balance_min_weight
        self.maximum = args.gradient_balance_max_weight
        self.weights = {"text": 1.0, "image": 1.0}
        self.ema = {"text": None, "image": None}
        self.samples_per_tensor = args.gradient_diagnostics_samples
        self.active = None
        self.norm_squares = {"text": [], "image": []}
        self.samples = {"text": [], "image": []}
        self.handles = []
        if self.enabled:
            for _, module in iter_tri_lora(model):
                for parameter in shared_route_parameters(module):
                    self.handles.append(parameter.register_hook(self._hook))

    def _hook(self, gradient):
        if self.active is None:
            return gradient
        self.norm_squares[self.active].append(gradient.detach().float().square().sum())
        if self.diagnostics and self.samples_per_tensor > 0:
            flat = gradient.detach().float().flatten()
            count = min(self.samples_per_tensor, flat.numel())
            indices = torch.linspace(0, flat.numel() - 1, count, device=flat.device).long()
            self.samples[self.active].append(flat.index_select(0, indices))
        return gradient * self.weights[self.active]

    def begin(self, modality: str) -> None:
        self.active = modality if self.enabled else None

    def end(self) -> None:
        self.active = None

    def update(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        norms = {
            modality: sum(value.item() for value in values) ** 0.5
            for modality, values in self.norm_squares.items()
        }
        self.norm_squares = {"text": [], "image": []}
        text_sample = torch.cat(self.samples["text"]) if self.samples["text"] else None
        image_sample = torch.cat(self.samples["image"]) if self.samples["image"] else None
        self.samples = {"text": [], "image": []}
        if self.balance_enabled and all(value > 0 for value in norms.values()):
            for modality, value in norms.items():
                previous = self.ema[modality]
                self.ema[modality] = value if previous is None else self.decay * previous + (1 - self.decay) * value
            inverses = {modality: 1.0 / max(value, 1e-12) for modality, value in self.ema.items()}
            normalizer = 2.0 / sum(inverses.values())
            self.weights = {
                modality: min(self.maximum, max(self.minimum, inverse * normalizer))
                for modality, inverse in inverses.items()
            }
        metrics = {
            "shared_grad/text_norm": norms["text"],
            "shared_grad/image_norm": norms["image"],
            "shared_grad/text_weight": self.weights["text"],
            "shared_grad/image_weight": self.weights["image"],
        }
        if text_sample is not None and image_sample is not None and text_sample.shape == image_sample.shape:
            denominator = text_sample.norm() * image_sample.norm()
            if denominator.item() > 0:
                metrics["shared_grad/cosine_sampled"] = float((text_sample @ image_sample) / denominator)
        return metrics


@torch.no_grad()
def evaluate(model, loader, args, tokenizer, device, amp_dtype) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    for batch in loader:
        sub_batches = modality_sub_batches(batch, args)
        sub_losses = []
        for _, sub_batch, objective, route_id in sub_batches:
            sub_batch = move_batch(sub_batch, device)
            corrupted, masked, t = corrupt_batch(
                sub_batch["input_ids"], sub_batch["eligible_mask"], sub_batch["modality_ids"],
                tokenizer.mask_id, args.eps, objective, fixed_t=0.75,
            )
            if not masked.any():
                continue
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(
                    corrupted, sub_batch["attention_mask"], sub_batch["position_ids"],
                    sub_batch["modality_ids"], sub_batch["route_ids"],
                )
                loss = masked_loss(logits, sub_batch["input_ids"], masked, t, not args.unweighting)
            sub_losses.append(loss.item())
        if sub_losses:
            total_loss += sum(sub_losses) / len(sub_losses)
            batches += 1
    model.train()
    return total_loss / max(1, batches)


@torch.no_grad()
def evaluate_diffusion_nll(model, loader, args, tokenizer, device, amp_dtype) -> dict[str, float]:
    """Estimate the held-out random-t diffusion objective reproducibly.

    This matches training: t is uniform on [eps, 1], selected tokens are
    corrupted according to t, and their cross entropy is weighted by 1/t.
    Caption length and the special tokens are observed, so this is a
    conditional-length content-token NLL objective rather than a likelihood
    for the complete variable-length string.
    """
    if args.val_diffusion_nll_draws < 1:
        return {}
    model.eval()
    weighted_total = 0.0
    denoising_total = 0.0
    batches = 0
    cuda_devices = [torch.cuda.current_device()] if str(device).startswith("cuda") else []
    # Validation must not advance the training RNG, and every epoch must use
    # the same t/mask draws so its curve is directly comparable.
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(args.seed + 91_773)
        if cuda_devices:
            torch.cuda.manual_seed_all(args.seed + 91_773)
        for _ in range(args.val_diffusion_nll_draws):
            for batch in loader:
                sub_losses = []
                sub_denoising = []
                for _, sub_batch, objective, _ in modality_sub_batches(batch, args):
                    sub_batch = move_batch(sub_batch, device)
                    corrupted, masked, t = corrupt_batch(
                        sub_batch["input_ids"], sub_batch["eligible_mask"],
                        sub_batch["modality_ids"], tokenizer.mask_id, args.eps, objective,
                    )
                    if not masked.any():
                        continue
                    with torch.autocast(
                        device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None
                    ):
                        logits = model(
                            corrupted, sub_batch["attention_mask"], sub_batch["position_ids"],
                            sub_batch["modality_ids"], sub_batch["route_ids"],
                        )
                    sub_losses.append(
                        masked_loss(logits, sub_batch["input_ids"], masked, t, True).item()
                    )
                    sub_denoising.append(
                        masked_loss(logits, sub_batch["input_ids"], masked, t, False).item()
                    )
                if sub_losses:
                    weighted_total += sum(sub_losses) / len(sub_losses)
                    denoising_total += sum(sub_denoising) / len(sub_denoising)
                    batches += 1
    model.train()
    nll = weighted_total / max(1, batches)
    denoising = denoising_total / max(1, batches)
    return {
        "val/diffusion_nll_mc_nats": nll,
        "val/diffusion_nll_mc_bits": nll / math.log(2.0),
        "val/denoising_nll_mc_nats": denoising,
        "val/diffusion_nll_mc_draws": float(args.val_diffusion_nll_draws),
    }


def load_vqvae(args, device):
    checkpoint = torch.load(args.vqvae_checkpoint, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    model = VQVAE(
        embed_dim=checkpoint_args.get("embed_dim", args.vqvae_embed_dim),
        num_codes=checkpoint_args.get("num_codes", args.num_image_codes),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def save_samples(model, tokenizer, collator, dataset, vqvae, args, device, path):
    examples = []
    for index in range(min(args.gen_num_samples, len(dataset.text_records))):
        examples.append({
            "kind": "paired",
            "text": dataset.text_records[index]["text"],
            "image_tokens": torch.zeros(args.grid_size, dtype=torch.long),
            "pair_index": index,
        })
    batch = move_batch(collator(examples), device)
    tokens = generate_conditioned_images(
        model, batch, collator.image_offset, args.num_image_codes, tokenizer.mask_id,
        args.gen_num_steps, args.gen_temperature, args.gen_reveal_order,
    ).view(-1, *args.grid_size)
    images = vqvae.decode_from_indices(tokens)
    save_image_grid(images, path, nrow=len(examples), value_range=(-1, 1))


def save_checkpoint(
    path, model, optimizer, epoch, step, best_val, args, tokenizer, lora_modules,
    epochs_without_improvement=0, ema_teacher=None, lr_scheduler=None,
):
    if _RANK != 0:  # rank 0 writes; the others hold identical weights
        return
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val,
        "epochs_without_improvement": epochs_without_improvement,
        "args": vars(args),
        "text_vocabulary": tokenizer.vocabulary,
        "lora_modules": lora_modules,
    }
    if ema_teacher is not None:
        payload["shared_jepa_ema_teacher"] = ema_teacher.state_dict()
    if lr_scheduler is not None:
        payload["lr_scheduler"] = lr_scheduler.state_dict()
    torch.save(payload, path)
    if args.train_mode in {"lora", "dense_private"}:
        torch.save(lora_state_dict(model), path.with_name(path.stem + "_adapter.pt"))


def adapter_initialization_state(model, state: dict) -> tuple[dict, set[str]]:
    """Rename dense trunk weights onto the wrapped base and list adapter keys.

    A dense checkpoint stores ``blocks.0.attn.qkv.weight``, while the same layer
    wrapped by ``TriLoRALinear`` expects ``blocks.0.attn.qkv.base.weight``.  The
    freshly initialized adapter tensors are legitimately absent from such a
    checkpoint, so they are reported as permitted omissions rather than errors.
    """
    adapted = [name for name, _ in iter_tri_lora(model)]
    if not adapted:
        return state, set()
    renamed = dict(state)
    for name in adapted:
        for suffix in ("weight", "bias"):
            source = f"{name}.{suffix}"
            target = f"{name}.base.{suffix}"
            if source in renamed and target not in renamed:
                renamed[target] = renamed.pop(source)
    absent = {
        f"{name}.{key}"
        for name in adapted
        for key in (
            "shared_A", "shared_B", "text_A", "text_B", "image_A", "image_B", "shared_bias",
        )
    }
    return renamed, absent


def data2vec_selected_layers(args) -> list[int]:
    """Blocks the data2vec objective supervises, explicit list or last top_k."""
    if getattr(args, "data2vec_layers", None):
        return sorted(args.data2vec_layers)
    return list(range(args.n_layers - args.data2vec_top_k, args.n_layers))


def accumulate_split_gradients(*, diffusion, representation, private, rest) -> None:
    """Send two losses to two disjoint parameter sets, in one graph traversal each.

    ``diffusion`` updates ``private`` only and ``representation`` updates ``rest``
    only. Gradients are accumulated with ``+=`` so gradient accumulation and
    multi-modality sub-batches behave exactly as they do for a normal backward.
    """
    for loss, parameters, keep in ((diffusion, private, True), (representation, rest, False)):
        if not parameters or loss is None:
            continue
        gradients = torch.autograd.grad(loss, parameters, retain_graph=keep, allow_unused=True)
        for parameter, gradient in zip(parameters, gradients):
            if gradient is None:
                continue
            parameter.grad = gradient if parameter.grad is None else parameter.grad + gradient


def launch_periodic_probe(model, tokenizer, args, output_dir, step, state) -> None:
    """Score the configured representation probe without blocking training.

    A slim checkpoint (weights, args and vocabulary -- no optimizer or EMA
    teacher) is written and handed to ``evaluate_hard_retrieval.py`` in a
    detached process, which deletes it when it is done.  If the previous probe
    is still running this interval is skipped, so the probes can never pile up
    and never slow the run down beyond sharing the GPU.
    """
    previous = state.get("process")
    if previous is not None and previous.poll() is None:
        state["skipped"] = state.get("skipped", 0) + 1
        return
    probe_dir = output_dir / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = probe_dir / f"step_{step:07d}.pt"
    torch.save(
        {
            "model": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
            "args": vars(args),
            "text_vocabulary": tokenizer.vocabulary,
            "step": step,
        },
        checkpoint,
    )
    label = f"step_{step:07d}"
    if args.probe_type == "semantic_dprime":
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "evaluate_semantic_dprime.py"),
            "--num-worlds", str(args.probe_num_worlds),
            "--bootstrap", str(args.probe_bootstrap),
            "--sublayer-layers", *[str(layer) for layer in args.probe_sublayer_layers],
            "--output-dir", str(probe_dir),
            "--checkpoint", f"{label}={checkpoint}",
        ]
        if args.probe_lexicon_matched:
            command.append("--lexicon-matched")
        if not args.probe_keep_checkpoints:
            command.append("--delete-checkpoint-when-done")
    else:
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "evaluate_hard_retrieval.py"),
            "--num-worlds", str(args.probe_num_worlds),
            "--sublayer-layers", *[str(layer) for layer in args.probe_sublayer_layers],
            "--output-dir", str(probe_dir),
            "--checkpoint", f"{label}={checkpoint}",
        ]
        if not args.probe_keep_checkpoints:
            command.append("--delete-checkpoint-when-done")
    environment = os.environ.copy()
    if args.probe_gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(args.probe_gpu)
    with open(probe_dir / "probe.log", "a") as log_file:
        state["process"] = subprocess.Popen(
            command, env=environment, stdout=log_file, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state["launched"] = state.get("launched", 0) + 1


def load_initial_model_weights(model, payload) -> None:
    """Load task weights while allowing optional auxiliary heads to differ."""
    state, adapter_keys = adapter_initialization_state(model, payload["model"])
    incompatible = model.load_state_dict(state, strict=False)
    allowed = (
        "modality_discriminator.", "shared_jepa_predictors.",
        "modulewise_jepa_predictors.",
        "shared_translation_bridges.",
        "data2vec_head.",
        "data2vec_heads.",
        "lejepa_projector.",
    )
    unexpected = [name for name in incompatible.unexpected_keys if not name.startswith(allowed)]
    missing = [
        name for name in incompatible.missing_keys
        if not name.startswith(allowed) and name not in adapter_keys
    ]
    if unexpected or missing:
        raise RuntimeError(f"Incompatible initialization checkpoint: missing={missing}, unexpected={unexpected}")


@torch.no_grad()
def make_ema_teacher(model):
    """Snapshot a frozen teacher after all student initialization/resume loading."""
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_ema_teacher(
    teacher, student, decay: float, share_input_encoder: bool = False
) -> None:
    """Update teacher parameters and buffers without adding teacher gradients.

    When the input encoder is shared, its online token/position/modality
    embeddings are passed directly into the clean teacher forward.  The stale
    private copies in the EMA module are therefore intentionally not updated.
    """
    shared_prefixes = ("token_embed.", "position_embed.", "modality_embed.")
    student_parameters = dict(student.named_parameters())
    for name, teacher_parameter in teacher.named_parameters():
        if share_input_encoder and name.startswith(shared_prefixes):
            continue
        teacher_parameter.lerp_(student_parameters[name].detach(), 1.0 - decay)
    student_buffers = dict(student.named_buffers())
    for name, teacher_buffer in teacher.named_buffers():
        teacher_buffer.copy_(student_buffers[name])


def data2vec_tristage_multiplier(
    update: int, total_updates: int, warmup_fraction: float, hold_fraction: float
) -> float:
    """Paper-style 5% warmup, 80% hold, then linear decay by default."""
    if total_updates < 1:
        raise ValueError("total_updates must be positive")
    warmup = max(1, round(total_updates * warmup_fraction))
    hold = round(total_updates * hold_fraction)
    decay = max(1, total_updates - warmup - hold)
    if update < warmup:
        return (update + 1) / warmup
    if update < warmup + hold:
        return 1.0
    return max(0.0, (total_updates - update) / decay)


def build_lr_scheduler(optimizer, args, total_updates: int):
    if args.lr_schedule == "constant":
        return None
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda update: data2vec_tristage_multiplier(
            update, total_updates, args.lr_warmup_fraction, args.lr_hold_fraction
        ),
    )


def shared_ab_parameters(model, layers: list[int] | None) -> list[torch.nn.Parameter]:
    selected = None if layers is None else set(layers)
    parameters = []
    for _, module in iter_tri_lora(model):
        if selected is None or module.layer_index in selected:
            parameters.extend(shared_route_parameters(module))
    return parameters


def shared_ab_parameters_for_modules(
    model, layers: list[int] | None, modules: list[str] | None
) -> list[torch.nn.Parameter]:
    """Shared A/B tensors receiving gated end-to-end data2vec gradients."""
    selected_layers = None if layers is None else set(layers)
    selected_modules = set(modules or ("qkv", "out_proj", "mlp.0", "mlp.3"))
    suffixes = {
        "qkv": "attn.qkv", "out_proj": "attn.out_proj",
        "mlp.0": "mlp.0", "mlp.3": "mlp.3",
    }
    unknown = selected_modules.difference(suffixes)
    if unknown:
        raise ValueError(f"Unknown data2vec JEPA gradient modules: {sorted(unknown)}")
    parameters = []
    for name, module in iter_tri_lora(model):
        if (
            (selected_layers is None or module.layer_index in selected_layers)
            and any(name.endswith(suffixes[key]) for key in selected_modules)
        ):
            parameters.extend(shared_route_parameters(module))
    return parameters


@torch.no_grad()
def adapter_delta_effective_ranks(
    model, layers: list[int] | None, modules: list[str] | None,
    private_branch: str = "text",
) -> dict[str, float]:
    """Entropy effective rank of each selected shared/private LoRA delta map.

    ``B @ A`` is the whole linear map the branch can express, so its spectrum
    collapsing to one direction is exactly representational collapse.  This is
    a parameter-only diagnostic: it needs no forward pass and no data, and it
    is the cheapest early warning that a JEPA/HSIC term is degenerating the
    branch it supervises.
    """
    selected_layers = None if layers is None else set(layers)
    selected_modules = set(modules or ("out_proj", "mlp.3"))
    suffixes = {
        "qkv": "attn.qkv", "out_proj": "attn.out_proj",
        "mlp.0": "mlp.0", "mlp.3": "mlp.3",
    }
    result: dict[str, float] = {}
    for name, module in iter_tri_lora(model):
        if selected_layers is not None and module.layer_index not in selected_layers:
            continue
        if not any(name.endswith(suffixes[key]) for key in selected_modules if key in suffixes):
            continue
        if not module.shared_branch_enabled and module.base.weight is not None:
            spectrum = torch.linalg.svdvals(module.base.weight.float()).square()
            weights = spectrum / spectrum.sum().clamp_min(1e-12)
            result[f"{name}.shared"] = float(
                (-(weights * weights.clamp_min(1e-12).log()).sum()).exp()
            )
        for branch in ("shared", private_branch):
            a = getattr(module, f"{branch}_A", None)
            b = getattr(module, f"{branch}_B", None)
            if a is None or b is None:
                continue
            spectrum = torch.linalg.svdvals((b @ a).float()).square()
            total = spectrum.sum()
            if total <= 0:
                result[f"{name}.{branch}"] = 0.0
                continue
            weights = spectrum / total
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum()
            result[f"{name}.{branch}"] = float(entropy.exp())
    return result


def modulewise_hsic_parameters(
    model, layers: list[int] | None, modules: list[str] | None
) -> list[torch.nn.Parameter]:
    """Selected shared/text-private A/B tensors used to calibrate HSIC."""
    selected = None if layers is None else set(layers)
    selected_modules = set(modules or ("out_proj", "mlp.3"))
    suffixes = {
        "qkv": "attn.qkv", "out_proj": "attn.out_proj",
        "mlp.0": "mlp.0", "mlp.3": "mlp.3",
    }
    unknown = selected_modules.difference(suffixes)
    if unknown:
        raise ValueError(f"Unknown modulewise HSIC modules: {sorted(unknown)}")
    parameters = []
    for name, module in iter_tri_lora(model):
        if (
            (selected is None or module.layer_index in selected)
            and any(name.endswith(suffixes[module]) for module in selected_modules)
        ):
            # autograd.grad rejects tensors that do not require grad, so a
            # frozen shared route contributes nothing to the calibration.
            parameters.extend(shared_route_parameters(module))
            parameters.extend(
                parameter for parameter in (module.text_A, module.text_B)
                if parameter.requires_grad
            )
    return parameters


def shared_gradient_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    """Exact unscaled gradient norm used only to calibrate dynamic JEPA weight."""
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True, materialize_grads=False,
    )
    squares = [gradient.detach().float().square().sum() for gradient in gradients if gradient is not None]
    if not squares:
        return loss.new_zeros(())
    return torch.stack(squares).sum().sqrt()


def paired_t1_selection_loss(
    metrics: dict[str, float], direction: str | None = None
) -> float:
    directions = [direction] if direction else ["text_to_image", "image_to_text"]
    keys = [f"val/paired/{name}/t1/matched_loss" for name in directions]
    missing = [key for key in keys if key not in metrics]
    if missing:
        raise ValueError(f"Paired t=1 checkpoint selection requires metrics: {missing}")
    return sum(metrics[key] for key in keys) / len(keys)


def main():
    args = parse_args()
    if args.resume and args.init_checkpoint:
        raise ValueError("Use either --resume or --init-checkpoint, not both")
    if not 0 <= args.lr_warmup_fraction < 1 or not 0 <= args.lr_hold_fraction < 1:
        raise ValueError("LR warmup and hold fractions must be in [0, 1)")
    if args.lr_warmup_fraction + args.lr_hold_fraction >= 1:
        raise ValueError("LR warmup + hold fractions must leave a positive decay stage")
    if args.probe_bootstrap < 1:
        raise ValueError("probe_bootstrap must be positive")
    if args.data_mode == "unpaired" and not args.balanced_modalities:
        raise ValueError("Omni-style unpaired training requires data.balanced_modalities=true")
    if args.data_mode == "unpaired" and (args.batch_size < 2 or args.batch_size % 2):
        raise ValueError("Balanced unpaired training requires an even batch size >= 2")
    if args.data_mode == "text_only" and args.objective != "text":
        raise ValueError("data.mode=text_only requires diffusion.objective=text")
    if args.lejepa_views:
        if args.data2vec_hidden or args.shared_jepa or args.sigreg or args.modality_adversarial:
            raise ValueError("lejepa_views is a standalone objective; disable data2vec/shared JEPA/SIGReg/DANN")
        if args.diffusion_weight != 0:
            raise ValueError("lejepa_views trains the representation only; set diffusion.weight=0")
        if args.data_mode not in {"image_only", "text_only"}:
            raise ValueError("lejepa_views needs a single-modality data.mode")
        if args.lejepa_global_views < 1 or args.lejepa_local_views < 0:
            raise ValueError("lejepa_views needs >=1 global view and >=0 local views")
        if not 0 <= args.lejepa_lambda <= 1:
            raise ValueError("lejepa_lambda is a convex weight in [0, 1]")
        for name in ("lejepa_global_scale", "lejepa_local_scale"):
            low, high = getattr(args, name)
            if not 0 < low <= high <= 1:
                raise ValueError(f"{name} must satisfy 0 < low <= high <= 1")
        if not 0 < args.lejepa_aspect[0] <= args.lejepa_aspect[1]:
            raise ValueError("lejepa_aspect must be positive and ordered")
    if args.jepa_trunk_only and (args.train_mode != "dense_private" or not args.data2vec_hidden):
        raise ValueError("jepa_trunk_only needs train_mode=dense_private and data2vec_hidden")
    if args.data2vec_projector_sigreg_weight > 0 and (
        not args.data2vec_hidden or args.data2vec_teacherless
        or not 0 < args.data2vec_projector_sigreg_weight <= 1
    ):
        raise ValueError("data2vec_projector_sigreg_weight needs data2vec_hidden with an EMA "
                         "teacher and a convex weight in (0, 1]")
    if args.data2vec_teacherless and not args.data2vec_hidden:
        raise ValueError("data2vec_teacherless is a variant of the data2vec_hidden objective")
    if args.data2vec_teacherless and not 0 <= args.data2vec_sigreg_weight <= 1:
        raise ValueError("data2vec_sigreg_weight is LeJEPA's lambda, a convex weight in [0, 1]")
    if args.data2vec_hidden:
        if not args.shared_jepa:
            raise ValueError("data2vec_hidden needs alignment.shared_jepa_enabled")
        if not args.shared_jepa_ema and not args.data2vec_teacherless:
            raise ValueError("data2vec_hidden needs shared_jepa_ema_enabled unless "
                             "data2vec_teacherless replaces the teacher with SIGReg")
        if args.data2vec_teacherless and args.shared_jepa_ema:
            raise ValueError("data2vec_teacherless has no teacher; set shared_jepa_ema_enabled=false")
        if args.modulewise_jepa_mode != "none" or args.modulewise_hsic:
            raise ValueError("data2vec_hidden is the whole-hidden-state objective; it does not combine "
                             "with modulewise JEPA/HSIC in one run")
        if args.data2vec_top_k < 1 or args.data2vec_top_k > args.n_layers:
            raise ValueError(f"data2vec_top_k must be in [1, {args.n_layers}]")
        if args.data2vec_layers is not None and (
            not args.data2vec_layers
            or any(layer < 0 or layer >= args.n_layers for layer in args.data2vec_layers)
            or len(set(args.data2vec_layers)) != len(args.data2vec_layers)
        ):
            raise ValueError("data2vec_layers must be distinct valid Transformer indices")
        if args.diffusion_weight < 0:
            raise ValueError("diffusion.weight cannot be negative")
        if (
            args.data2vec_token_weight < 0
            or args.data2vec_global_weight < 0
            or args.data2vec_variance_weight < 0
        ):
            raise ValueError("data2vec token, global, and variance weights cannot be negative")
        if (
            args.data2vec_token_weight == 0
            and args.data2vec_global_weight == 0
            and args.data2vec_variance_weight == 0
        ):
            raise ValueError("data2vec needs at least one non-zero loss weight")
        if args.data2vec_variance_target <= 0:
            raise ValueError("data2vec variance target must be positive")
    if args.diffusion_weight == 0 and not (args.data2vec_hidden or args.lejepa_views):
        raise ValueError("diffusion.weight=0 is only supported for the data2vec_hidden and lejepa_views objectives")
    if args.shared_jepa_ema_decay_final is not None and not 0.0 < args.shared_jepa_ema_decay_final < 1.0:
        raise ValueError("shared_jepa_ema_decay_final must be in (0, 1)")
    if args.data_mode == "image_only" and args.objective != "image":
        raise ValueError("data.mode=image_only requires diffusion.objective=image")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    if args.val_diffusion_nll_draws < 0:
        raise ValueError("val_diffusion_nll_draws must be non-negative")
    if args.objective_gradient_diagnostics_every < 0:
        raise ValueError("objective_gradient_diagnostics_every cannot be negative")
    if any(layer < 0 or layer >= args.n_layers for layer in args.objective_gradient_diagnostic_layers):
        raise ValueError("objective_gradient_diagnostic_layers must be valid Transformer indices")
    if args.shared_only_epochs < 0 or args.shared_only_epochs > args.epochs:
        raise ValueError("train.shared_only_epochs must be between 0 and train.epochs")
    if args.early_stopping_patience < 0:
        raise ValueError("train.early_stopping_patience must be non-negative")
    if args.early_stopping_min_delta < 0:
        raise ValueError("train.early_stopping_min_delta must be non-negative")
    if args.shared_only_epochs and args.train_mode != "lora":
        raise ValueError("Shared-only staging requires train_mode=lora")
    if args.asymmetric_condition_target and not (
        args.data_mode == "paired"
        and args.objective == "both"
        and args.train_mode == "lora"
    ):
        raise ValueError(
            "Asymmetric condition/target routing requires paired data, objective=both, and Tri-LoRA"
        )
    if args.modality_adversarial and args.data_mode != "unpaired":
        raise ValueError("Modality-adversarial distribution alignment is intended for unpaired training")
    if args.modality_adversarial_weight < 0 or args.modality_adversarial_grl_lambda < 0:
        raise ValueError("Modality-adversarial weight and GRL lambda must be non-negative")
    if args.modality_adversarial_start_epoch < 0 or args.modality_adversarial_start_epoch >= args.epochs:
        raise ValueError("alignment.modality_adversarial_start_epoch must be in [0, epochs)")
    if args.sigreg:
        if not 0 <= args.sigreg_start_epoch < args.epochs:
            raise ValueError("alignment.sigreg_start_epoch must be in [0, epochs)")
        if args.sigreg_weight < 0 or args.sigreg_warmup_steps < 0:
            raise ValueError("SIGReg weight and warmup must be non-negative")
        if args.sigreg_num_slices < 1 or args.sigreg_num_points < 3 or args.sigreg_num_points % 2 != 1:
            raise ValueError("SIGReg needs slices>=1 and an odd number of points>=3")
        if args.sigreg_t_max <= 0:
            raise ValueError("SIGReg t_max must be positive")
        if args.sigreg_per_layer and args.train_mode != "lora":
            raise ValueError("Per-layer SIGReg requires Tri-LoRA shared adapters")
        if args.sigreg_layers is not None:
            if not args.sigreg_per_layer:
                raise ValueError("sigreg_layers requires sigreg_per_layer=true")
            if not args.sigreg_layers or any(
                layer < 0 or layer >= args.n_layers for layer in args.sigreg_layers
            ):
                raise ValueError("sigreg_layers must contain valid Transformer indices")
    if args.shared_jepa:
        if args.train_mode != "lora" and not args.data2vec_hidden:
            raise ValueError("Shared latent JEPA requires Tri-LoRA")
        if not 0 <= args.shared_jepa_start_epoch < args.epochs:
            raise ValueError("alignment.shared_jepa_start_epoch must be in [0, epochs)")
        if args.shared_jepa_weight < 0 or args.shared_jepa_warmup_steps < 0:
            raise ValueError("Shared JEPA weight and warmup must be non-negative")
        if any(
            weight is not None and weight < 0
            for weight in (args.shared_jepa_text_weight, args.shared_jepa_image_weight)
        ):
            raise ValueError("Modality-specific shared JEPA weights must be non-negative")
        if args.shared_jepa_predictor_hidden_multiplier < 1:
            raise ValueError("Shared JEPA predictor hidden multiplier must be positive")
        if args.shared_jepa_layers is not None and (
            not args.shared_jepa_layers
            or any(layer < 0 or layer >= args.n_layers for layer in args.shared_jepa_layers)
        ):
            raise ValueError("shared_jepa_layers must contain valid Transformer indices")
        if not 0.0 < args.shared_jepa_ema_decay < 1.0:
            raise ValueError("shared_jepa_ema_decay must be in (0, 1)")
        if args.shared_jepa_gradient_ratio < 0:
            raise ValueError("shared_jepa_gradient_ratio must be non-negative")
        if not 0.0 <= args.shared_jepa_dynamic_ema_decay < 1.0:
            raise ValueError("shared_jepa_dynamic_ema_decay must be in [0, 1)")
        if not 0 <= args.shared_jepa_dynamic_min_weight <= args.shared_jepa_dynamic_max_weight:
            raise ValueError("dynamic JEPA coefficient bounds must be ordered and non-negative")
        if (
            args.modulewise_jepa_mode != "none"
            and not args.modulewise_jepa_mode.startswith("data2vec_")
            and args.shared_jepa_dynamic_weight
            and not args.modulewise_jepa_gated_predictor
        ):
            raise ValueError(
                "adaptive JEPA gradient weighting needs either a data2vec modulewise mode "
                "or modulewise_jepa_gated_predictor=true"
            )
    elif args.modulewise_jepa_mode != "none":
        raise ValueError("modulewise JEPA requires alignment.shared_jepa_enabled=true")
    if args.modulewise_jepa_gated_predictor and (
        args.modulewise_jepa_mode not in {"average", "layerwise"}
    ):
        raise ValueError(
            "modulewise_jepa_gated_predictor requires target mode average or "
            "layerwise; the data2vec modes have no predictor to gate"
        )
    if args.modulewise_jepa_mode != "none":
        # Single-modality only: the shared/private decomposition that JEPA and
        # HSIC act on is defined per modality, and HSIC records the private
        # adapter of the modality being trained.
        if (args.data_mode, args.objective) not in {("text_only", "text"), ("image_only", "image")}:
            raise ValueError(
                "modulewise JEPA/HSIC requires data.mode=text_only with objective=text "
                "or data.mode=image_only with objective=image"
            )
        if args.shared_jepa_layers is None or not args.shared_jepa_layers:
            raise ValueError("modulewise JEPA requires explicit shared_jepa_layers")
    if args.modulewise_hsic:
        if args.train_mode not in {"lora", "dense_private"}:
            raise ValueError("modulewise HSIC requires shared/private adapters")
        if (args.data_mode, args.objective) not in {("text_only", "text"), ("image_only", "image")}:
            raise ValueError(
                "modulewise HSIC requires data.mode=text_only with objective=text "
                "or data.mode=image_only with objective=image"
            )
        if not args.shared_jepa_layers:
            raise ValueError("modulewise HSIC requires explicit shared_jepa_layers")
        if args.shared_jepa and args.modulewise_jepa_mode == "none":
            raise ValueError(
                "modulewise HSIC without a modulewise JEPA mode cannot be combined with "
                "the pooled shared-latent JEPA; use modulewise_jepa_mode or data2vec_hidden"
            )
        if args.modulewise_hsic_gradient_ratio < 0:
            raise ValueError("modulewise HSIC gradient ratio must be non-negative")
        if args.modulewise_hsic_update_every < 1:
            raise ValueError("modulewise HSIC update interval must be positive")
        if not 0 <= args.modulewise_hsic_ema_decay < 1:
            raise ValueError("modulewise HSIC EMA decay must be in [0, 1)")
        if not 0 <= args.modulewise_hsic_min_weight <= args.modulewise_hsic_max_weight:
            raise ValueError("modulewise HSIC coefficient bounds must be ordered and non-negative")
        if args.modulewise_hsic_warmup_steps < 0 or args.modulewise_hsic_max_tokens < 3:
            raise ValueError("modulewise HSIC warmup must be non-negative and max_tokens >= 3")
    if args.shared_translation:
        if args.train_mode != "lora":
            raise ValueError("Shared-token translation requires Tri-LoRA")
        if not args.shared_translation_layers or any(
            layer < 0 or layer >= args.n_layers for layer in args.shared_translation_layers
        ):
            raise ValueError("translation.layers must contain valid Transformer indices")
        if args.shared_translation_heads is not None and (
            args.shared_translation_heads < 1
            or args.d_model % args.shared_translation_heads
        ):
            raise ValueError("translation.heads must divide model.d_model")
        if not 0 <= args.shared_translation_dropout < 1:
            raise ValueError("translation.dropout must be in [0, 1)")
        if args.data_mode != "unpaired" or not args.backtranslation:
            raise ValueError(
                "Shared-token translation training currently requires unpaired "
                "cycle/back-translation to be enabled"
            )
    if not 0 <= args.full_mask_probability <= 1:
        raise ValueError("diffusion.full_mask_probability must be in [0, 1]")
    if args.mask_block_2d:
        # Unpaired two-modality runs may set both: image sub-batches then get 2D
        # blocks and text sub-batches spans (see the training corruption call).
        per_modality = (args.data_mode == "unpaired" and args.objective == "both"
                        and args.mask_span_min is not None)
        if not per_modality and (args.data_mode != "image_only" or args.objective != "image"):
            raise ValueError("2D block masking needs data.mode=image_only with objective=image, "
                             "or an unpaired both-modality run that also sets span masking")
        if not per_modality and (args.mask_span_min is not None or args.mask_span_max is not None):
            raise ValueError("2D block masking and 1D window masking are exclusive")
    for name in ("train_fixed_t_text", "train_fixed_t_image"):
        value = getattr(args, name)
        if value is not None and not 0 < value <= 1:
            raise ValueError(f"diffusion.{name} must be in (0, 1]")
    if (args.mask_span_min is None) != (args.mask_span_max is None):
        raise ValueError("diffusion.mask_span_min and mask_span_max must be set together")
    if args.mask_span_min is not None and not 1 <= args.mask_span_min <= args.mask_span_max:
        raise ValueError("diffusion mask window sizes must satisfy 1 <= min <= max")
    if args.train_fixed_t is not None:
        if not 0 < args.train_fixed_t <= 1:
            raise ValueError("diffusion.train_fixed_t must be in (0, 1]")
        if args.full_mask_probability:
            raise ValueError("diffusion.train_fixed_t and full_mask_probability are exclusive")
    if args.backtranslation:
        if args.data_mode != "unpaired":
            raise ValueError("Back-translation requires genuinely unpaired data")
        if args.train_mode != "lora":
            raise ValueError("Shared-only source routing requires train_mode=lora")
        if not 0 <= args.backtranslation_start_epoch < args.epochs:
            raise ValueError("backtranslation.start_epoch must be in [0, epochs)")
        if args.backtranslation_every_n_microsteps < 1 or args.backtranslation_batch_size < 1:
            raise ValueError("Back-translation frequency and batch size must be positive")
        if args.backtranslation_generation_steps < 1:
            raise ValueError("backtranslation.generation_steps must be positive")
        if not 0 <= args.backtranslation_min_confidence <= 1:
            raise ValueError("backtranslation.min_confidence must be in [0, 1]")
        if args.backtranslation_cycle_weight < 0 or args.pseudo_align_weight < 0:
            raise ValueError("Back-translation loss weights must be non-negative")
        if args.backtranslation_cycle_weight == 0 and args.pseudo_align_weight == 0:
            raise ValueError("At least one back-translation loss weight must be positive")
        if args.pseudo_align_temperature <= 0:
            raise ValueError("backtranslation.pseudo_align_temperature must be positive")
        if args.backtranslation_warmup_steps < 0:
            raise ValueError("backtranslation.warmup_steps must be non-negative")
    init_distributed()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else None
    output_dir = Path(args.output_dir)
    (output_dir / "samples").mkdir(parents=True, exist_ok=True)
    log = setup_logging(output_dir, args.resume)

    cache_dir = Path(args.token_cache_dir).expanduser()
    train_source = ClevrMultimodalDataset(
        args.train_dir, cache_dir / "train_tokens.pt", args.data_mode,
        pair_manifest=args.train_manifest, caption_field=args.caption_field,
        text_manifest=args.train_text_manifest,
    )
    val_source = ClevrMultimodalDataset(
        args.val_dir, cache_dir / "val_tokens.pt", args.data_mode,
        pair_manifest=args.val_manifest, caption_field=args.caption_field,
        text_manifest=args.val_text_manifest,
    )
    paired_val_source = None
    if args.paired_val:
        if not args.val_manifest:
            raise ValueError("Paired conditional validation requires data.val_manifest")
        paired_val_source = ClevrMultimodalDataset(
            args.val_dir, cache_dir / "val_tokens.pt", "paired",
            pair_manifest=args.val_manifest, caption_field=args.caption_field,
        )
    if args.data_mode != "text_only" and train_source.grid_size != tuple(args.grid_size):
        raise ValueError(f"Token grid {train_source.grid_size} != configured grid {tuple(args.grid_size)}")
    initialization_path = args.resume or args.init_checkpoint
    initialization_payload = None
    if initialization_path:
        initialization_payload = torch.load(initialization_path, map_location="cpu", weights_only=False)
        tokenizer = ClevrTextTokenizer(initialization_payload["text_vocabulary"])
    else:
        tokenizer = ClevrTextTokenizer.build(train_source.texts)
    if _RANK == 0:
        tokenizer.save(output_dir / "text_tokenizer.json")
    collator = MultimodalCollator(tokenizer, args.num_image_codes, args.max_text_length)

    if args.data_mode == "unpaired":
        train_dataset = BalancedUnpairedDataset(train_source, seed=args.seed)
        val_dataset = BalancedUnpairedDataset(val_source, seed=args.seed + 1000)
        loader_batch_size = args.batch_size if args.objective != "both" else args.batch_size // 2
    else:
        train_dataset, val_dataset = train_source, val_source
        loader_batch_size = args.batch_size
    train_subset_artifact = None
    train_subset_sha256 = None
    if args.max_train_samples is not None:
        carrier_limit = (
            args.max_train_samples // 2
            if args.data_mode == "unpaired" and args.objective == "both"
            else args.max_train_samples
        )
        train_subset_indices = deterministic_subset_indices(
            len(train_dataset), carrier_limit, args.train_subset_seed
        )
        train_dataset_for_loader = Subset(train_dataset, train_subset_indices)
        canonical_indices = json.dumps(train_subset_indices, separators=(",", ":"))
        train_subset_sha256 = hashlib.sha256(canonical_indices.encode("utf-8")).hexdigest()
        manifest_path = Path(args.train_manifest).resolve() if args.train_manifest else None
        train_subset_artifact = output_dir / "train_subset_indices.json"
        train_subset_artifact.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "data_mode": args.data_mode,
                    "source_dataset_size": len(train_dataset),
                    "requested_examples": args.max_train_samples,
                    "carrier_count": len(train_subset_indices),
                    "subset_seed": args.train_subset_seed,
                    "indices_sha256": train_subset_sha256,
                    "train_manifest": str(manifest_path) if manifest_path else None,
                    "train_manifest_sha256": (
                        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                        if manifest_path and manifest_path.is_file() else None
                    ),
                    "indices": train_subset_indices,
                },
                indent=2,
            ) + "\n"
        )
        log.info(
            f"training subset: examples={args.max_train_samples} carriers={len(train_subset_indices)} "
            f"seed={args.train_subset_seed} indices_sha256={train_subset_sha256}"
        )
    else:
        train_dataset_for_loader = train_dataset
    if args.val_max_samples is not None:
        carrier_limit = (
            args.val_max_samples // 2
            if args.data_mode == "unpaired" and args.objective == "both"
            else args.val_max_samples
        )
        val_dataset_for_loader = Subset(val_dataset, range(min(max(1, carrier_limit), len(val_dataset))))
    else:
        val_dataset_for_loader = val_dataset
    # Under torchrun each rank reads a disjoint shard of every epoch; a
    # single-process run keeps the original shuffled loader.
    train_sampler = (
        DistributedSampler(train_dataset_for_loader, num_replicas=_WORLD, rank=_RANK,
                           shuffle=True, seed=args.seed, drop_last=True)
        if _WORLD > 1 else None
    )
    train_loader = DataLoader(train_dataset_for_loader, loader_batch_size, shuffle=train_sampler is None,
                              sampler=train_sampler, num_workers=args.num_workers, collate_fn=collator,
                              pin_memory=True, drop_last=True)
    if _WORLD > 1:
        log.info(f"distributed: {_WORLD} ranks, per-rank batch {args.batch_size} x accumulation "
                 f"{args.gradient_accumulation_steps} -> global batch "
                 f"{args.batch_size * args.gradient_accumulation_steps * _WORLD}")
    val_loader = DataLoader(val_dataset_for_loader, loader_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator, pin_memory=True)
    paired_val_loader = None
    step_val_loader = None
    if paired_val_source is not None:
        paired_val_dataset = paired_val_source
        if args.paired_val_max_samples is not None:
            paired_val_dataset = Subset(
                paired_val_source, range(min(args.paired_val_max_samples, len(paired_val_source)))
            )
        paired_val_loader = DataLoader(
            paired_val_dataset, max(2, args.batch_size // 2), shuffle=False,
            num_workers=args.num_workers, collate_fn=collator, pin_memory=True,
        )
        if args.step_eval_every and args.step_eval_every > 0:
            step_val_dataset = Subset(
                paired_val_source, range(min(args.step_eval_max_samples, len(paired_val_source)))
            )
            step_val_loader = DataLoader(
                step_val_dataset, max(2, min(args.batch_size // 2, args.step_eval_max_samples)),
                shuffle=False, num_workers=args.num_workers, collate_fn=collator, pin_memory=True,
            )

    model, lora_modules = build_model(args, len(tokenizer))
    model = model.to(device)
    model.gradient_checkpointing = args.gradient_checkpointing
    if args.init_checkpoint:
        load_initial_model_weights(model, initialization_payload)
        log.info(f"initialized model weights from {args.init_checkpoint}; optimizer and counters are fresh")
    groups = optimizer_groups(model, args)
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    for group in groups:
        log.info(f"optimizer group {group['group_name']}: lr={group['lr']:.3g}, tensors={len(group['params'])}")
    gradient_balancer = SharedGradientBalancer(model, args)
    optimizer_steps_per_epoch = (
        len(train_loader) + args.gradient_accumulation_steps - 1
    ) // args.gradient_accumulation_steps
    adversarial_start_step = args.modality_adversarial_start_epoch * optimizer_steps_per_epoch
    sigreg_start_step = args.sigreg_start_epoch * optimizer_steps_per_epoch
    shared_jepa_start_step = args.shared_jepa_start_epoch * optimizer_steps_per_epoch
    start_epoch = step = 0
    best_val = float("inf")
    epochs_without_improvement = 0
    if args.resume:
        model.load_state_dict(initialization_payload["model"])
        optimizer.load_state_dict(initialization_payload["optimizer"])
        start_epoch = initialization_payload["epoch"] + 1
        step = initialization_payload["step"]
        best_val = initialization_payload["best_val_loss"]
        epochs_without_improvement = initialization_payload.get("epochs_without_improvement", 0)
        log.info(f"resumed {args.resume} at epoch={start_epoch}, step={step}")

    total_optimizer_steps = args.max_steps or (args.epochs * optimizer_steps_per_epoch)
    lr_scheduler = build_lr_scheduler(optimizer, args, total_optimizer_steps)
    if lr_scheduler is not None and args.resume:
        if "lr_scheduler" in initialization_payload:
            lr_scheduler.load_state_dict(initialization_payload["lr_scheduler"])
        else:
            # Backward-compatible resume from checkpoints written before
            # scheduler state was recorded.
            lr_scheduler.last_epoch = step
            multiplier = data2vec_tristage_multiplier(
                step, total_optimizer_steps,
                args.lr_warmup_fraction, args.lr_hold_fraction,
            )
            for base_lr, group in zip(lr_scheduler.base_lrs, optimizer.param_groups):
                group["lr"] = base_lr * multiplier
    log.info(
        f"lr_schedule={args.lr_schedule} total_updates={total_optimizer_steps} "
        f"warmup_fraction={args.lr_warmup_fraction} hold_fraction={args.lr_hold_fraction}"
    )

    ema_teacher = None
    if args.shared_jepa and args.shared_jepa_ema and not args.data2vec_teacherless:
        ema_teacher = make_ema_teacher(model)
        if args.resume and "shared_jepa_ema_teacher" in initialization_payload:
            ema_teacher.load_state_dict(initialization_payload["shared_jepa_ema_teacher"])
        log.info(
            f"shared JEPA target=EMA teacher; decay={args.shared_jepa_ema_decay:.6f}"
        )
    dynamic_jepa_state = {"diffusion": None, "jepa": None, "weight": None}
    modulewise_hsic_state = {"diffusion": None, "hsic": None, "weight": 0.0}
    periodic_probe_state: dict[str, object] = {}
    if args.probe_every_steps:
        log.info(
            f"periodic probe: {args.probe_type} every {args.probe_every_steps} steps on "
            f"{args.probe_num_worlds} scenes, sublayers {args.probe_sublayer_layers}, "
            f"gpu={args.probe_gpu or 'training gpu'}, keep_checkpoints={args.probe_keep_checkpoints}; results in "
            f"{Path(args.output_dir) / 'probes'}; runs detached so training never waits"
        )

    private_lora_enabled = not args.shared_only_epochs or start_epoch >= args.shared_only_epochs
    private_tensor_count = set_private_lora_trainable(model, private_lora_enabled) if lora_modules else 0
    total, trainable = parameter_counts(model)
    log.info(f"mode={args.train_mode} data={args.data_mode} objective={args.objective}")
    log.info(f"parameters: {trainable:,} trainable / {total:,} total ({100 * trainable / total:.2f}%)")
    if lora_modules:
        log.info(f"LoRA modules ({len(lora_modules)}): {', '.join(lora_modules)}")
    if args.asymmetric_condition_target:
        log.info(
            "routing=condition(shared+modality-private), target(modality-private-only); "
            "bidirectional minibatches are split evenly between text and image targets"
        )
    if args.shared_only_epochs:
        stage = "shared+private" if private_lora_enabled else "shared-only"
        log.info(
            f"no-base Tri-LoRA stage={stage}; shared_only_epochs={args.shared_only_epochs}; "
            f"private_tensors={private_tensor_count}"
        )
    if args.modality_adversarial:
        log.info(
            f"adversarial schedule: start_epoch={args.modality_adversarial_start_epoch} "
            f"start_step={adversarial_start_step} warmup_steps={args.modality_adversarial_warmup_steps} "
            f"representation_normalization={args.modality_adversarial_representation_normalization}"
        )
    if args.sigreg:
        log.info(
            f"SIGReg schedule: start_epoch={args.sigreg_start_epoch} "
            f"start_step={sigreg_start_step} weight={args.sigreg_weight} "
            f"warmup_steps={args.sigreg_warmup_steps} slices={args.sigreg_num_slices} "
            f"points={args.sigreg_num_points} t_max={args.sigreg_t_max} "
            f"per_layer={args.sigreg_per_layer} layers={args.sigreg_layers or 'all'}"
        )
    if args.train_fixed_t is not None:
        log.info(
            f"training masks a fixed {args.train_fixed_t:.0%} of eligible tokens per sequence "
            "(t is not sampled); validation still uses fixed t=0.75"
        )
    if args.mask_block_2d:
        log.info(
            f"2D block masking on the {args.grid_size[0]}x{args.grid_size[1]} token grid: "
            f"per-block area {args.mask_block_scale[0]:.0%}-{args.mask_block_scale[1]:.0%} of the "
            f"grid, aspect {args.mask_block_aspect[0]}-{args.mask_block_aspect[1]}, as many blocks "
            "as the target fraction needs; overlap makes the realized fraction approximate"
        )
    if args.mask_span_min is not None:
        log.info(
            f"window masking: contiguous windows of {args.mask_span_min}-{args.mask_span_max} "
            "tokens, sizes drawn uniformly per window, as many windows per sequence as the "
            "target fraction needs; overlapping windows and edge clipping make the realized "
            "fraction approximate"
        )
    # Random replacements are drawn per modality so a text position can never
    # receive an image code.  Special tokens are excluded from the draw.
    random_token_ranges = {
        1: (len(ClevrTextTokenizer.SPECIAL_TOKENS), len(tokenizer)),
        2: (len(tokenizer), len(tokenizer) + args.num_image_codes),
    }
    if args.bert_replacement:
        log.info(
            "BERT/data2vec corruption: of the selected positions 80% become [MASK], "
            "10% a random token of the same modality, 10% stay unchanged; the loss "
            f"covers all selected positions. text ids {random_token_ranges[1]}, "
            f"image ids {random_token_ranges[2]}"
        )
    if args.data2vec_teacherless:
        log.info(
            "LeJEPA mode: no EMA teacher, no stop-gradient, no target LayerNorm; the clean "
            "pass is a second view through the same network and collapse is prevented by "
            f"SIGReg (lambda {args.data2vec_sigreg_weight}, {args.sigreg_num_slices} slices, "
            f"{args.sigreg_num_points} points, t_max {args.sigreg_t_max}) on the pooled "
            "embedding distribution of every supervised block"
        )
    if args.data2vec_hidden:
        selected_data2vec_layers = data2vec_selected_layers(args)
        data2vec_mode_label = "fixed-target" if args.data2vec_mode == "fixed_target" else "layerwise"
        data2vec_target_label = (
            f"block {args.data2vec_target_layer} for every student"
            if args.data2vec_mode == "fixed_target" else "each block's own")
        log.info(
            ("faithful data2vec: target=mean of parameter-free LayerNorm-ed EMA-teacher "
             f"hidden states over blocks {selected_data2vec_layers} of {args.n_layers}; "
             "student=final block through one prediction head; "
             if args.data2vec_mode == "average" else
             f"{data2vec_mode_label} data2vec: target={data2vec_target_label} "
             "parameter-free LayerNorm-ed "
             f"EMA-teacher hidden state; students are blocks {selected_data2vec_layers} "
             f"of {args.n_layers}, each through its own prediction head; ")
            + "SmoothL1 beta="
            f"{args.data2vec_beta}; masked positions only; weight={args.data2vec_weight}; "
            f"target_type={args.data2vec_target_type}; "
            f"share_input_encoder={args.data2vec_share_input_encoder}; "
            f"token_weight={args.data2vec_token_weight}; "
            f"global_weight={args.data2vec_global_weight}; "
            f"variance_weight={args.data2vec_variance_weight}; "
            f"variance_target={args.data2vec_variance_target}; "
            f"warmup_steps={args.shared_jepa_warmup_steps}; diffusion_weight={args.diffusion_weight}; "
            f"ema_decay={args.shared_jepa_ema_decay}"
            + (f" ramping to {args.shared_jepa_ema_decay_final} over {args.shared_jepa_ema_ramp_steps} steps"
               if args.shared_jepa_ema_decay_final is not None and args.shared_jepa_ema_ramp_steps > 0 else "")
        )
    elif args.shared_jepa:
        log.info(
            f"shared JEPA schedule: start_epoch={args.shared_jepa_start_epoch} "
            f"start_step={shared_jepa_start_step} weight={args.shared_jepa_weight} "
            f"text_weight={args.shared_jepa_weight if args.shared_jepa_text_weight is None else args.shared_jepa_text_weight} "
            f"image_weight={args.shared_jepa_weight if args.shared_jepa_image_weight is None else args.shared_jepa_image_weight} "
            f"warmup_steps={args.shared_jepa_warmup_steps} "
            f"predictor_hidden_multiplier={args.shared_jepa_predictor_hidden_multiplier} "
            f"layers={args.shared_jepa_layers or 'all'} loss={args.shared_jepa_loss}; "
            f"target={'clean EMA-teacher' if ema_teacher is not None else 'stop-gradient online clean'} "
            "shared token latent; source=masked student shared token latent"
        )
        if args.shared_jepa_dynamic_weight:
            log.info(
                "shared JEPA dynamic coefficient: "
                f"target_ratio={args.shared_jepa_gradient_ratio:.3g} "
                f"ema_decay={args.shared_jepa_dynamic_ema_decay:.3g} "
                f"bounds=[{args.shared_jepa_dynamic_min_weight:.3g}, "
                f"{args.shared_jepa_dynamic_max_weight:.3g}]"
            )
        if args.modulewise_jepa_mode != "none":
            if args.modulewise_jepa_gated_predictor:
                log.info(
                    "modulewise JEPA gradient routing: gated end-to-end; "
                    f"target_modules={args.modulewise_jepa_modules}; "
                    f"gated_shared_gradient_modules={args.modulewise_jepa_gradient_modules or 'all'}; "
                    "predictors trained through the gated path"
                )
            if args.modulewise_jepa_mode.startswith("data2vec_"):
                log.info(
                    "modulewise data2vec JEPA: "
                    f"mode={args.modulewise_jepa_mode}; target_modules={args.modulewise_jepa_modules}; "
                    f"teacher_layers={args.shared_jepa_layers}; student_layer={max(args.shared_jepa_layers)}; "
                    f"smooth_l1_beta={args.modulewise_jepa_smooth_l1_beta}; no_predictor; "
                    f"gated_shared_gradient_modules={args.modulewise_jepa_gradient_modules or 'all'}"
                )
            else:
                log.info(
                    "modulewise JEPA: "
                    f"mode={args.modulewise_jepa_mode}; native_modules={args.modulewise_jepa_modules}; "
                    f"layers={args.shared_jepa_layers}; equal_module_layer_weighting; "
                    "teacher=clean EMA, student=masked"
                )
    if args.modulewise_hsic:
        log.info(
            f"modulewise HSIC: shared/{'image' if args.data_mode == 'image_only' else 'text'}-private "
            f"native updates at {args.modulewise_jepa_modules}; "
            f"layers={args.shared_jepa_layers}; max_tokens={args.modulewise_hsic_max_tokens}; "
            f"target_gradient_ratio={args.modulewise_hsic_gradient_ratio}; "
            f"update_every={args.modulewise_hsic_update_every}; "
            f"warmup_steps={args.modulewise_hsic_warmup_steps}; "
            f"bounds=[{args.modulewise_hsic_min_weight}, {args.modulewise_hsic_max_weight}]"
        )
    if args.backtranslation:
        log.info(
            "back-translation schedule: "
            f"start_epoch={args.backtranslation_start_epoch} "
            f"every_n_microsteps={args.backtranslation_every_n_microsteps} "
            f"pseudo_batch={args.backtranslation_batch_size} "
            f"generation_steps={args.backtranslation_generation_steps} "
            f"cycle_weight={args.backtranslation_cycle_weight} "
            f"pseudo_align_weight={args.pseudo_align_weight} "
            f"pseudo_align_loss={args.pseudo_align_loss} "
            f"pseudo_align_temperature={args.pseudo_align_temperature} "
            f"min_confidence={args.backtranslation_min_confidence} "
            f"warmup_steps={args.backtranslation_warmup_steps}; "
            "source-private routing disabled in both cycle directions"
        )
    if args.shared_translation:
        log.info(
            "shared-token translation: "
            f"layers={args.shared_translation_layers} "
            f"mode={args.shared_translation_mode} "
            f"heads={args.shared_translation_heads or args.n_heads} "
            f"dropout={args.shared_translation_dropout} "
            f"lr={args.shared_translation_lr or args.lr}; "
            "condition shared tokens are K/V, target residual tokens are Q, "
            "and target shared LoRA is suppressed only in bridge layers"
        )

    vqvae = load_vqvae(args, device)
    wandb_run = None
    if args.wandb and _RANK == 0:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name,
            tags=args.wandb_tags,
            config=vars(args),
            id=args.wandb_resume_id,
            resume="must" if args.wandb_resume_id else None,
            allow_val_change=bool(args.wandb_resume_id),
        )
        config_source = Path(args.config).resolve()
        if config_source.is_file():
            # Keep the exact human-authored YAML beside the expanded W&B config
            # so a run can be reproduced without reconstructing CLI defaults.
            wandb.save(str(config_source), base_path=str(Path.cwd()), policy="now")
            wandb_run.summary["config_yaml_path"] = str(config_source.relative_to(Path.cwd()))
            wandb_run.summary["config_yaml_sha256"] = hashlib.sha256(config_source.read_bytes()).hexdigest()
            wandb_run.summary["config_yaml_uploaded"] = True
        if train_subset_artifact is not None:
            wandb.save(str(train_subset_artifact.resolve()), base_path=str(Path.cwd()), policy="now")
            wandb_run.summary["train_subset_artifact"] = str(train_subset_artifact)
            wandb_run.summary["train_subset_seed"] = args.train_subset_seed
            wandb_run.summary["train_subset_carrier_count"] = len(train_dataset_for_loader)
            wandb_run.summary["train_subset_indices_sha256"] = train_subset_sha256

    if args.eval_only:
        if not args.resume:
            raise ValueError("--eval-only requires --resume CHECKPOINT")
        val_loss = evaluate(model, val_loader, args, tokenizer, device, amp_dtype)
        diffusion_nll_metrics = evaluate_diffusion_nll(
            model, val_loader, args, tokenizer, device, amp_dtype
        )
        paired_metrics = evaluate_paired_conditioning(
            model, paired_val_loader, tokenizer.mask_id, device, amp_dtype,
            args.paired_val_mask_ratios, args.paired_val_control_ratios, args.seed,
            args.paired_val_directions,
            args.asymmetric_condition_target,
        ) if paired_val_loader is not None else {}
        log.info(f"evaluation-only marginal_loss_t0.75={val_loss:.4f}")
        if diffusion_nll_metrics:
            log.info(
                "evaluation-only diffusion_nll_mc_nats="
                f"{diffusion_nll_metrics['val/diffusion_nll_mc_nats']:.4f} "
                f"draws={args.val_diffusion_nll_draws}"
            )
        for key, value in paired_metrics.items():
            log.info(f"evaluation-only {key}={value:.6f}")
        if wandb_run:
            wandb_run.log(
                {"val/loss_t0.75": val_loss, **diffusion_nll_metrics, **paired_metrics},
                step=step,
            )
            wandb_run.finish()
        return

    # Optional modules such as the DANN head consume random numbers during
    # construction. Reset the training RNG after all model/evaluator/W&B
    # setup so matched ablations receive the same loader shuffle, masks, and
    # dropout sequence rather than a configuration-dependent RNG offset.
    broadcast_module(model)
    broadcast_module(ema_teacher)
    rank_seed = args.seed + 1_000_003 * _RANK
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)
    log.info(f"training RNG reset after initialization: seed={args.seed}"
             + (f" (+ per-rank offset, {_WORLD} ranks)" if _WORLD > 1 else ""))

    stop = False
    window_start = time.time()
    micro_step = 0
    latest_backtranslation_metrics = {}
    # Parameter groups for --diffusion-private-only: the per-modality adapter
    # tensors, and everything else that still wants a gradient.
    cross_modal_moments = (
        CrossModalMoments(args.d_model, decay=args.cross_modal_moment_decay)
        if args.cross_modal_moment_weight > 0 else None
    )
    if cross_modal_moments is not None:
        print(f"cross-modal moment matching: weight={args.cross_modal_moment_weight}, "
              f"block={args.cross_modal_moment_layer}, ema decay={args.cross_modal_moment_decay}",
              flush=True)
    private_parameters, non_private_parameters = [], []
    if args.diffusion_private_only:
        private_ids = set()
        for _, module in iter_tri_lora(model):
            for branch in ("text", "image"):
                for suffix in ("A", "B"):
                    parameter = getattr(module, f"{branch}_{suffix}", None)
                    if parameter is not None and parameter.requires_grad:
                        private_parameters.append(parameter)
                        private_ids.add(id(parameter))
        non_private_parameters = [p for p in model.parameters()
                                  if p.requires_grad and id(p) not in private_ids]
        if not private_parameters:
            raise ValueError("--diffusion-private-only needs trainable private adapters; use "
                             "--train-mode dense_private (or lora) without --freeze-base")
        if not args.data2vec_hidden and args.modulewise_jepa_mode == "none":
            raise ValueError("--diffusion-private-only needs a representation objective to train "
                             "the trunk with; enable data2vec or modulewise JEPA")
        print(f"split gradients: diffusion -> {len(private_parameters)} private adapter tensors "
              f"({sum(p.numel() for p in private_parameters):,} parameters); representation -> "
              f"{len(non_private_parameters)} other tensors "
              f"({sum(p.numel() for p in non_private_parameters):,} parameters)", flush=True)

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if args.data_mode == "unpaired":
            # DataLoader workers are non-persistent, so each epoch's workers
            # observe this freshly drawn strict text/image derangement.
            train_dataset.set_epoch(epoch)
            log.info(f"unpaired carrier redraw for epoch={epoch:03d}")
        if not private_lora_enabled and epoch >= args.shared_only_epochs:
            private_tensor_count = set_private_lora_trainable(model, True)
            private_lora_enabled = True
            _, trainable = parameter_counts(model)
            log.info(
                f"Tri-LoRA stage transition at epoch={epoch:03d}: enabled "
                f"{private_tensor_count} private tensors; trainable_parameters={trainable:,}"
            )
            if wandb_run:
                wandb_run.log({"train/private_lora_enabled": 1, "epoch": epoch}, step=step)
        adversarial_active = args.modality_adversarial and epoch >= args.modality_adversarial_start_epoch
        sigreg_active = args.sigreg and epoch >= args.sigreg_start_epoch
        shared_jepa_active = args.shared_jepa and epoch >= args.shared_jepa_start_epoch
        modulewise_jepa_active = (
            shared_jepa_active and args.modulewise_jepa_mode != "none"
        )
        data2vec_modulewise_active = (
            modulewise_jepa_active and args.modulewise_jepa_mode.startswith("data2vec_")
        )
        gated_predictor_active = (
            modulewise_jepa_active and args.modulewise_jepa_gated_predictor
        )
        # HSIC compares the shared branch with the private branch of the
        # modality being trained.
        private_branch = "image" if args.data_mode == "image_only" else "text"
        data2vec_hidden_active = args.data2vec_hidden and epoch >= args.shared_jepa_start_epoch
        # Both paths hand their JEPA gradient to the trainer instead of adding
        # it to the backward loss, so they share the manual application below.
        jepa_gradient_gated = data2vec_modulewise_active or gated_predictor_active
        modulewise_hsic_active = args.modulewise_hsic and (
            modulewise_jepa_active or args.modulewise_jepa_mode == "none"
        )
        # Either objective needs the per-module shared/private writes.
        native_maps_active = modulewise_jepa_active or modulewise_hsic_active
        backtranslation_active = args.backtranslation and epoch >= args.backtranslation_start_epoch
        if args.modality_adversarial and epoch == args.modality_adversarial_start_epoch:
            log.info(
                f"adversarial transition at epoch={epoch:03d} step={step:07d}: "
                "enabled DANN and started its local warmup"
            )
            if wandb_run:
                wandb_run.log({"train/adversarial_enabled": 1, "epoch": epoch}, step=step)
        model.train()
        for batch_index, batch in enumerate(train_loader):
            sub_batches = modality_sub_batches(batch, args)
            losses = {}
            optimized_losses = {}
            unweighted_losses = {}
            adversarial_losses = {}
            sigreg_losses = {}
            sigreg_layer_losses = {}
            sigreg_diagnostics = {}
            shared_jepa_losses = {}
            shared_jepa_layer_losses = {}
            shared_jepa_layer_cosines = {}
            shared_jepa_weights = {}
            shared_jepa_gradient_ratios = {}
            cross_modal_moment_losses = {}
            cross_modal_moment_details = {}
            modulewise_hsic_losses = {}
            modulewise_hsic_details = {}
            modulewise_hsic_weights = {}
            modulewise_hsic_gradient_ratios = {}
            objective_gradient_diagnostics = {}
            adversarial_representation_norms = {}
            discriminator_accuracies = {}
            masked_fractions = {}
            masked_accuracies = {}
            jepa_progress = 0.0
            hsic_progress = 0.0
            for modality_name, sub_batch, objective, route_id in sub_batches:
                gated_jepa_gradients = None
                gated_jepa_parameters = None
                sub_batch = move_batch(sub_batch, device)
                if args.lejepa_views:
                    # Paper LeJEPA: crops in place of corruption, no diffusion loss.
                    with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
                        lejepa_loss, lejepa_details = lejepa_multiview_loss(
                            model, sub_batch,
                            global_views=args.lejepa_global_views,
                            local_views=args.lejepa_local_views,
                            global_scale=tuple(args.lejepa_global_scale),
                            local_scale=tuple(args.lejepa_local_scale),
                            aspect=tuple(args.lejepa_aspect),
                            grid=tuple(args.grid_size) if objective == "image" else None,
                            lam=args.lejepa_lambda,
                            num_slices=args.sigreg_num_slices,
                            num_points=args.sigreg_num_points,
                            t_max=args.sigreg_t_max, seed=args.seed + step,
                        )
                    gradient_balancer.begin(modality_name)
                    try:
                        (lejepa_loss / (len(sub_batches) * args.gradient_accumulation_steps)).backward()
                    finally:
                        gradient_balancer.end()
                    value = lejepa_loss.detach().item()
                    losses[modality_name] = optimized_losses[modality_name] = value
                    unweighted_losses[modality_name] = value
                    shared_jepa_losses[modality_name] = value
                    shared_jepa_weights[modality_name] = 1.0
                    shared_jepa_layer_losses[modality_name] = lejepa_details
                    masked_fractions[modality_name] = 1.0 - lejepa_details["global_kept_fraction"]
                    continue
                # With both block and span masking configured (an unpaired run), images
                # get 2D blocks and text gets spans; either alone applies as before.
                both_masks = args.mask_block_2d and args.mask_span_min is not None
                use_blocks = args.mask_block_2d and (not both_masks or objective == "image")
                use_spans = args.mask_span_min is not None and (not both_masks or objective == "text")
                modality_fixed_t = {"text": args.train_fixed_t_text,
                                    "image": args.train_fixed_t_image}.get(objective)
                corrupted, masked, t = corrupt_batch(
                    sub_batch["input_ids"], sub_batch["eligible_mask"], sub_batch["modality_ids"],
                    tokenizer.mask_id, args.eps, objective,
                    fixed_t=modality_fixed_t if modality_fixed_t is not None else args.train_fixed_t,
                    full_mask_probability=args.full_mask_probability,
                    bert_replacement=args.bert_replacement,
                    random_token_ranges=random_token_ranges,
                    mask_span_min=args.mask_span_min if use_spans else None,
                    mask_span_max=args.mask_span_max if use_spans else None,
                    mask_block_grid=tuple(args.grid_size) if use_blocks else None,
                    mask_block_scale=tuple(args.mask_block_scale),
                    mask_block_aspect=tuple(args.mask_block_aspect),
                )
                if not masked.any():
                    continue
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
                    if data2vec_hidden_active:
                        # Trunk-only JEPA: private LoRA routes off (-1) in student and
                        # teacher, so the objective trains exactly the shared trunk.
                        jepa_routes = (
                            torch.full_like(sub_batch["route_ids"], -1)
                            if args.jepa_trunk_only else sub_batch["route_ids"]
                        )
                        # Faithful data2vec: the student sees corrupted tokens and
                        # regresses the EMA teacher's averaged block outputs on the
                        # same, clean sequence.  No shared/private machinery.
                        if args.data2vec_target_type == "ffn_output":
                            logits, student_hidden, _ = model(
                                corrupted, sub_batch["attention_mask"], sub_batch["position_ids"],
                                sub_batch["modality_ids"], jepa_routes,
                                return_data2vec_by_layer=True,
                            )
                        else:
                            logits, student_hidden = model(
                                corrupted, sub_batch["attention_mask"], sub_batch["position_ids"],
                                sub_batch["modality_ids"], jepa_routes,
                                return_hidden_by_layer=True,
                            )
                        if args.data2vec_teacherless:
                            # LeJEPA: one network, no stop-gradient, no eval-mode
                            # switch -- the clean pass is a second view, not a
                            # teacher, and SIGReg below is what rules out collapse.
                            clean_input_embeddings = (
                                model.input_embeddings(
                                    sub_batch["input_ids"], sub_batch["position_ids"],
                                    sub_batch["modality_ids"],
                                )
                                if args.data2vec_share_input_encoder else None
                            )
                            if args.data2vec_target_type == "ffn_output":
                                _, _, teacher_hidden = model(
                                    sub_batch["input_ids"], sub_batch["attention_mask"],
                                    sub_batch["position_ids"], sub_batch["modality_ids"],
                                    jepa_routes, return_data2vec_by_layer=True,
                                    input_embeddings=clean_input_embeddings,
                                )
                            else:
                                _, teacher_hidden = model(
                                    sub_batch["input_ids"], sub_batch["attention_mask"],
                                    sub_batch["position_ids"], sub_batch["modality_ids"],
                                    jepa_routes, return_hidden_by_layer=True,
                                    input_embeddings=clean_input_embeddings,
                                )
                        else:
                            was_training = ema_teacher.training
                            ema_teacher.eval()
                            with torch.no_grad():
                                clean_input_embeddings = (
                                    model.input_embeddings(
                                        sub_batch["input_ids"], sub_batch["position_ids"],
                                        sub_batch["modality_ids"],
                                    )
                                    if args.data2vec_share_input_encoder else None
                                )
                                if args.data2vec_target_type == "ffn_output":
                                    _, _, teacher_hidden = ema_teacher(
                                        sub_batch["input_ids"], sub_batch["attention_mask"],
                                        sub_batch["position_ids"], sub_batch["modality_ids"],
                                        jepa_routes, return_data2vec_by_layer=True,
                                        input_embeddings=clean_input_embeddings,
                                    )
                                else:
                                    _, teacher_hidden = ema_teacher(
                                        sub_batch["input_ids"], sub_batch["attention_mask"],
                                        sub_batch["position_ids"], sub_batch["modality_ids"],
                                        jepa_routes, return_hidden_by_layer=True,
                                        input_embeddings=clean_input_embeddings,
                                    )
                            if was_training:
                                ema_teacher.train()
                        if args.jepa_trunk_only and args.diffusion_weight > 0:
                            # The diffusion loss still trains the full model (trunk +
                            # private LoRA) through its own forward.
                            logits = model(
                                corrupted, sub_batch["attention_mask"], sub_batch["position_ids"],
                                sub_batch["modality_ids"], sub_batch["route_ids"],
                            )
                        task_loss = masked_loss(
                            logits, sub_batch["input_ids"], masked, t, not args.unweighting
                        )
                        unweighted_task_loss = masked_loss(
                            logits, sub_batch["input_ids"], masked, t, False
                        )
                        raw_data2vec, data2vec_details = data2vec_hidden_loss(
                            model, student_hidden, teacher_hidden, masked,
                            top_k=args.data2vec_top_k, beta=args.data2vec_beta,
                            mode=args.data2vec_mode, layers=args.data2vec_layers,
                            target_layer=args.data2vec_target_layer,
                            pool_mask=sub_batch["eligible_mask"],
                            token_weight=args.data2vec_token_weight,
                            global_weight=args.data2vec_global_weight,
                            variance_weight=args.data2vec_variance_weight,
                            variance_target=args.data2vec_variance_target,
                            normalize_targets=not args.data2vec_teacherless,
                            stop_gradient=not args.data2vec_teacherless,
                        )
                        if args.data2vec_teacherless and args.data2vec_sigreg_weight > 0:
                            # SIGReg on the embedding distribution, one pooled
                            # vector per sample per supervised block, averaged
                            # over blocks. This is what replaces the teacher.
                            weights = sub_batch["eligible_mask"].unsqueeze(-1)
                            statistics, pooled_cosines = [], []
                            for layer in data2vec_selected_layers(args):
                                hidden = teacher_hidden[layer].float()
                                pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
                                statistics.append(sigreg_loss(
                                    pooled, num_slices=args.sigreg_num_slices,
                                    num_points=args.sigreg_num_points,
                                    t_max=args.sigreg_t_max, seed=args.seed + step,
                                ))
                                # Mean off-diagonal cosine of the pooled scenes: the
                                # cone SIGReg should open. ~0 when isotropic.
                                with torch.no_grad():
                                    unit = F.normalize(pooled.detach(), dim=1)
                                    gram = unit @ unit.T
                                    count = gram.size(0)
                                    pooled_cosines.append(
                                        ((gram.sum() - gram.diagonal().sum())
                                         / max(count * (count - 1), 1)).item()
                                    )
                            raw_sigreg = torch.stack(statistics).mean()
                            # LeJEPA combines the two convexly rather than adding a
                            # weighted penalty: (1 - lambda) * prediction + lambda * SIGReg.
                            weight = args.data2vec_sigreg_weight
                            data2vec_details["prediction"] = float(raw_data2vec.detach())
                            raw_data2vec = (1.0 - weight) * raw_data2vec + weight * raw_sigreg
                            data2vec_details["sigreg"] = float(raw_sigreg.detach())
                            data2vec_details["pooled_cosine"] = sum(pooled_cosines) / len(pooled_cosines)
                        if args.data2vec_projector_sigreg_weight > 0:
                            # LeJEPA hybrid: the EMA teacher stays; SIGReg acts on a
                            # projector over the pooled last-block student embedding of
                            # this modality's batch, so each modality is pushed to the
                            # same isotropic Gaussian in projector space.
                            last = student_hidden[max(student_hidden)]
                            pool_weights = sub_batch["eligible_mask"].to(last.dtype).unsqueeze(-1)
                            pooled_student = (last * pool_weights).sum(1) / pool_weights.sum(1).clamp_min(1)
                            projected = model.lejepa_projector(pooled_student.float())
                            raw_projector_sigreg = sigreg_loss(
                                projected, num_slices=args.sigreg_num_slices,
                                num_points=args.sigreg_num_points,
                                t_max=args.sigreg_t_max, seed=args.seed + step,
                            )
                            weight = args.data2vec_projector_sigreg_weight
                            data2vec_details["prediction"] = float(raw_data2vec.detach())
                            raw_data2vec = (1.0 - weight) * raw_data2vec + weight * raw_projector_sigreg
                            data2vec_details["sigreg"] = float(raw_projector_sigreg.detach())
                            data2vec_details["embedding_cosine"] = mean_offdiagonal_cosine(pooled_student)
                            data2vec_details["projection_cosine"] = mean_offdiagonal_cosine(projected)
                        if args.shared_jepa_warmup_steps > 0:
                            jepa_progress = min(1.0, (max(0, step - shared_jepa_start_step) + 1) / args.shared_jepa_warmup_steps)
                        else:
                            jepa_progress = 1.0
                        total_loss = (
                            args.diffusion_weight * task_loss
                            + args.data2vec_weight * jepa_progress * raw_data2vec
                        )
                        if cross_modal_moments is not None:
                            block = (args.cross_modal_moment_layer
                                     if args.cross_modal_moment_layer >= 0
                                     else max(student_hidden))
                            pooled = cross_modal_moments.pooled(
                                student_hidden[block], sub_batch["eligible_mask"])
                            moment_loss, moment_details = cross_modal_moments.loss(
                                modality_name, pooled)
                            if moment_loss is not None:
                                total_loss = total_loss + args.cross_modal_moment_weight * moment_loss
                                cross_modal_moment_losses[modality_name] = float(moment_loss.detach())
                                cross_modal_moment_details[modality_name] = moment_details
                        shared_jepa_losses[modality_name] = raw_data2vec.detach().item()
                        shared_jepa_weights[modality_name] = args.data2vec_weight * jepa_progress
                        shared_jepa_layer_losses[modality_name] = data2vec_details
                        shared_jepa_layer_cosines[modality_name] = {
                            "prediction_target_cosine": data2vec_details["prediction_target_cosine"],
                            "global_cosine": data2vec_details["global_cosine"],
                            "target_position_spread": data2vec_details["target_position_spread"],
                        }
                        optimized_losses[modality_name] = total_loss.detach().item()
                        denominator = len(sub_batches) * args.gradient_accumulation_steps
                        backward_loss = total_loss / denominator
                        gradient_balancer.begin(modality_name)
                        try:
                            if args.diffusion_private_only:
                                # Two exact backward passes over one graph instead of one over
                                # their sum: the diffusion loss reaches the private adapters
                                # only, the JEPA loss reaches everything else. Summing first
                                # and masking afterwards is not equivalent, because the two
                                # objectives would already have been added together.
                                accumulate_split_gradients(
                                    diffusion=args.diffusion_weight * task_loss / denominator,
                                    representation=(args.data2vec_weight * jepa_progress
                                                    * raw_data2vec / denominator),
                                    private=private_parameters,
                                    rest=non_private_parameters,
                                )
                            else:
                                backward_loss.backward()
                        finally:
                            gradient_balancer.end()
                        losses[modality_name] = task_loss.item()
                        unweighted_losses[modality_name] = unweighted_task_loss.item()
                        masked_fractions[modality_name] = masked.float().mean().item()
                        masked_accuracies[modality_name] = (
                            logits.detach().argmax(dim=-1)[masked].eq(sub_batch["input_ids"][masked]).float().mean().item()
                        )
                        continue
                    model_output = model(
                        corrupted, sub_batch["attention_mask"], sub_batch["position_ids"],
                        sub_batch["modality_ids"], sub_batch["route_ids"],
                        return_shared=(
                            adversarial_active or sigreg_active
                            or shared_jepa_active or modulewise_hsic_active
                        ),
                        return_shared_by_layer=sigreg_active and args.sigreg_per_layer,
                        return_shared_tokens_by_layer=shared_jepa_active and not native_maps_active,
                        return_shared_private_native_by_module=native_maps_active,
                        return_shared_differentiable_native_by_module=jepa_gradient_gated,
                        private_branch=private_branch,
                    )
                    masked_shared_differentiable_native = None
                    if native_maps_active:
                        if jepa_gradient_gated:
                            (
                                logits,
                                shared_representation,
                                masked_shared_native,
                                masked_private_native,
                                masked_shared_differentiable_native,
                            ) = model_output
                        else:
                            (
                                logits,
                                shared_representation,
                                masked_shared_native,
                                masked_private_native,
                            ) = model_output
                        shared_by_layer = None
                        masked_shared_tokens_by_layer = None
                    elif shared_jepa_active:
                        (
                            logits,
                            shared_representation,
                            shared_by_layer,
                            masked_shared_tokens_by_layer,
                        ) = model_output
                    elif adversarial_active or sigreg_active:
                        masked_shared_tokens_by_layer = None
                        if sigreg_active and args.sigreg_per_layer:
                            logits, shared_representation, shared_by_layer = model_output
                        else:
                            logits, shared_representation = model_output
                            shared_by_layer = None
                    else:
                        logits, shared_representation, shared_by_layer = model_output, None, None
                        masked_shared_tokens_by_layer = None
                    task_loss = masked_loss(
                        logits, sub_batch["input_ids"], masked, t, not args.unweighting
                    )
                    # Always log the ordinary masked-token CE as a comparison
                    # metric. It uses the exact same logits, targets, and mask
                    # as the optimized objective, but never divides by t.
                    unweighted_task_loss = masked_loss(
                        logits, sub_batch["input_ids"], masked, t, False
                    )
                    total_loss = task_loss
                    weighted_jepa_term = None
                    weighted_sigreg_term = None
                    weighted_hsic_term = None
                    if shared_jepa_active:
                        # The clean branch is a stop-gradient target.  It uses
                        # the same token positions and route as the corrupted
                        # branch, but receives the original unmasked tokens.
                        # Evaluation mode makes this moving target deterministic
                        # by disabling Transformer dropout; training mode is
                        # restored immediately afterward.
                        target_model = ema_teacher if ema_teacher is not None else model
                        was_training = target_model.training
                        target_model.eval()
                        try:
                            with torch.no_grad():
                                clean_output = target_model(
                                    sub_batch["input_ids"], sub_batch["attention_mask"],
                                    sub_batch["position_ids"], sub_batch["modality_ids"],
                                    sub_batch["route_ids"],
                                    return_shared=True,
                                    return_shared_tokens_by_layer=not modulewise_jepa_active,
                                    return_shared_private_native_by_module=modulewise_jepa_active,
                                    private_branch=private_branch,
                                )
                                if modulewise_jepa_active:
                                    clean_shared_native = clean_output[2]
                                else:
                                    clean_shared_tokens_by_layer = clean_output[3]
                        finally:
                            if target_model is model and was_training:
                                model.train()
                        if modulewise_jepa_active:
                            raw_jepa, per_layer_jepa, per_layer_cosine = modulewise_shared_jepa_loss(
                                model,
                                masked_shared_differentiable_native
                                if gated_predictor_active else masked_shared_native,
                                clean_shared_native, masked,
                                layers=args.shared_jepa_layers, modules=args.modulewise_jepa_modules,
                                mode=args.modulewise_jepa_mode, loss_type=args.shared_jepa_loss,
                            ) if not data2vec_modulewise_active else (*modulewise_data2vec_loss(
                                masked_shared_differentiable_native, clean_shared_native, masked,
                                layers=args.shared_jepa_layers, modules=args.modulewise_jepa_modules,
                                mode=args.modulewise_jepa_mode,
                                beta=args.modulewise_jepa_smooth_l1_beta,
                            ), {})
                        else:
                            raw_jepa, per_layer_jepa, per_layer_cosine = shared_latent_jepa_loss(
                                model,
                                masked_shared_tokens_by_layer,
                                clean_shared_tokens_by_layer,
                                masked,
                                layers=args.shared_jepa_layers,
                                loss_type=args.shared_jepa_loss,
                            )
                        if args.shared_jepa_warmup_steps > 0:
                            local_jepa_step = max(0, step - shared_jepa_start_step)
                            jepa_progress = min(
                                1.0, (local_jepa_step + 1) / args.shared_jepa_warmup_steps
                            )
                        else:
                            jepa_progress = 1.0
                        modality_jepa_weight = (
                            args.shared_jepa_text_weight
                            if modality_name == "text" else args.shared_jepa_image_weight
                        )
                        if modality_jepa_weight is None:
                            modality_jepa_weight = args.shared_jepa_weight
                        if args.shared_jepa_dynamic_weight:
                            parameters = shared_ab_parameters(model, args.shared_jepa_layers)
                            diffusion_norm = shared_gradient_norm(task_loss, parameters)
                            jepa_norm = shared_gradient_norm(raw_jepa, parameters)
                            decay = args.shared_jepa_dynamic_ema_decay
                            previous_diffusion = dynamic_jepa_state["diffusion"]
                            previous_jepa = dynamic_jepa_state["jepa"]
                            dynamic_jepa_state["diffusion"] = (
                                diffusion_norm.item() if previous_diffusion is None else
                                decay * previous_diffusion + (1.0 - decay) * diffusion_norm.item()
                            )
                            dynamic_jepa_state["jepa"] = (
                                jepa_norm.item() if previous_jepa is None else
                                decay * previous_jepa + (1.0 - decay) * jepa_norm.item()
                            )
                            modality_jepa_weight = min(
                                args.shared_jepa_dynamic_max_weight,
                                max(
                                    args.shared_jepa_dynamic_min_weight,
                                    args.shared_jepa_gradient_ratio
                                    * dynamic_jepa_state["diffusion"]
                                    / max(dynamic_jepa_state["jepa"], 1e-12),
                                ),
                            )
                            dynamic_jepa_state["weight"] = modality_jepa_weight
                            shared_jepa_gradient_ratios[modality_name] = (
                                modality_jepa_weight * jepa_norm.item()
                                / max(diffusion_norm.item(), 1e-12)
                            )
                        weighted_jepa_term = modality_jepa_weight * jepa_progress * raw_jepa
                        if jepa_gradient_gated:
                            gated_jepa_parameters = shared_ab_parameters_for_modules(
                                model, args.shared_jepa_layers,
                                args.modulewise_jepa_gradient_modules,
                            )
                            if gated_predictor_active:
                                # This term never reaches backward(), so the
                                # predictors have to be trained through the gated
                                # path or they stay at initialization forever.
                                gated_jepa_parameters = gated_jepa_parameters + [
                                    parameter
                                    for parameter in model.modulewise_jepa_predictors.parameters()
                                    if parameter.requires_grad
                                ]
                            gated_jepa_gradients = torch.autograd.grad(
                                weighted_jepa_term,
                                gated_jepa_parameters,
                                retain_graph=True,
                                allow_unused=True,
                                materialize_grads=False,
                            )
                        else:
                            total_loss = total_loss + weighted_jepa_term
                        shared_jepa_losses[modality_name] = raw_jepa.detach().item()
                        shared_jepa_weights[modality_name] = modality_jepa_weight
                        shared_jepa_layer_losses[modality_name] = per_layer_jepa
                        shared_jepa_layer_cosines[modality_name] = per_layer_cosine
                    if modulewise_hsic_active:
                        raw_hsic, hsic_details = modulewise_private_hsic_loss(
                            masked_shared_native,
                            masked_private_native,
                            masked,
                            layers=args.shared_jepa_layers,
                            modules=args.modulewise_jepa_modules,
                            max_tokens=args.modulewise_hsic_max_tokens,
                            seed=args.seed + step,
                        )
                        if args.modulewise_hsic_warmup_steps > 0:
                            hsic_progress = min(
                                1.0, (step + 1) / args.modulewise_hsic_warmup_steps
                            )
                        else:
                            hsic_progress = 1.0
                        # Calibrate once at the first microbatch of the chosen
                        # optimizer update, then use that same coefficient for
                        # every microbatch accumulated into the update.
                        if (
                            (step + 1) % args.modulewise_hsic_update_every == 0
                            and micro_step % args.gradient_accumulation_steps == 0
                        ):
                            parameters = modulewise_hsic_parameters(
                                model, args.shared_jepa_layers, args.modulewise_jepa_modules
                            )
                            diffusion_norm = shared_gradient_norm(task_loss, parameters)
                            hsic_norm = shared_gradient_norm(raw_hsic, parameters)
                            decay = args.modulewise_hsic_ema_decay
                            previous_diffusion = modulewise_hsic_state["diffusion"]
                            previous_hsic = modulewise_hsic_state["hsic"]
                            modulewise_hsic_state["diffusion"] = (
                                diffusion_norm.item() if previous_diffusion is None else
                                decay * previous_diffusion + (1.0 - decay) * diffusion_norm.item()
                            )
                            modulewise_hsic_state["hsic"] = (
                                hsic_norm.item() if previous_hsic is None else
                                decay * previous_hsic + (1.0 - decay) * hsic_norm.item()
                            )
                            if modulewise_hsic_state["hsic"] > 1e-12:
                                modulewise_hsic_state["weight"] = min(
                                    args.modulewise_hsic_max_weight,
                                    max(
                                        args.modulewise_hsic_min_weight,
                                        args.modulewise_hsic_gradient_ratio
                                        * modulewise_hsic_state["diffusion"]
                                        / modulewise_hsic_state["hsic"],
                                    ),
                                )
                                modulewise_hsic_gradient_ratios[modality_name] = (
                                    modulewise_hsic_state["weight"] * hsic_norm.item()
                                    / max(diffusion_norm.item(), 1e-12)
                                )
                        weighted_hsic_term = (
                            modulewise_hsic_state["weight"] * hsic_progress * raw_hsic
                        )
                        total_loss = total_loss + weighted_hsic_term
                        modulewise_hsic_losses[modality_name] = raw_hsic.detach().item()
                        modulewise_hsic_details[modality_name] = hsic_details
                        modulewise_hsic_weights[modality_name] = modulewise_hsic_state["weight"]
                    if sigreg_active:
                        if args.sigreg_per_layer:
                            available_layers = set(range(len(model.blocks)))
                            if set(shared_by_layer) != available_layers:
                                raise RuntimeError(
                                    "Per-layer SIGReg expected one shared representation for every "
                                    f"Transformer block, got {sorted(shared_by_layer)}"
                                )
                            expected_layers = (
                                available_layers
                                if args.sigreg_layers is None else set(args.sigreg_layers)
                            )
                            per_layer = {
                                layer: sigreg_loss(
                                    representation,
                                    num_slices=args.sigreg_num_slices,
                                    num_points=args.sigreg_num_points,
                                    t_max=args.sigreg_t_max,
                                    seed=args.seed + step + layer,
                                )
                                for layer, representation in shared_by_layer.items()
                                if layer in expected_layers
                            }
                            raw_sigreg = torch.stack(list(per_layer.values())).mean()
                            sigreg_layer_losses[modality_name] = {
                                layer: value.item() for layer, value in per_layer.items()
                            }
                        else:
                            raw_sigreg = sigreg_loss(
                                shared_representation,
                                num_slices=args.sigreg_num_slices,
                                num_points=args.sigreg_num_points,
                                t_max=args.sigreg_t_max,
                                seed=args.seed + step,
                            )
                        if args.sigreg_warmup_steps > 0:
                            local_sigreg_step = max(0, step - sigreg_start_step)
                            sigreg_progress = min(
                                1.0, (local_sigreg_step + 1) / args.sigreg_warmup_steps
                            )
                        else:
                            sigreg_progress = 1.0
                        weighted_sigreg_term = args.sigreg_weight * sigreg_progress * raw_sigreg
                        total_loss = total_loss + weighted_sigreg_term
                        sigreg_losses[modality_name] = raw_sigreg.item()
                        sigreg_diagnostics[modality_name] = gaussianity_diagnostics(
                            shared_representation
                        )
                    if adversarial_active:
                        discriminator_logits = model.modality_logits(
                            shared_representation, args.modality_adversarial_grl_lambda
                        )
                        # Adversarial pretraining uses homogeneous unpaired rows;
                        # the discriminator predicts one modality label per row.
                        route_targets = torch.full(
                            (sub_batch["input_ids"].size(0),), route_id,
                            dtype=torch.long, device=sub_batch["input_ids"].device,
                        )
                        adversarial_loss = F.cross_entropy(discriminator_logits.float(), route_targets)
                        if args.modality_adversarial_warmup_steps > 0:
                            local_adversarial_step = max(0, step - adversarial_start_step)
                            progress = min(
                                1.0,
                                (local_adversarial_step + 1) / args.modality_adversarial_warmup_steps,
                            )
                        else:
                            progress = 1.0
                        total_loss = total_loss + args.modality_adversarial_weight * progress * adversarial_loss
                        adversarial_losses[modality_name] = adversarial_loss.item()
                        adversarial_representation_norms[f"{modality_name}_raw_shared_l2"] = (
                            shared_representation.detach().float().norm(dim=-1).mean().item()
                        )
                        adversarial_representation_norms[f"{modality_name}_dann_input_l2"] = (
                            model.modality_discriminator_input(shared_representation.detach())
                            .float().norm(dim=-1).mean().item()
                        )
                        discriminator_accuracies[modality_name] = (
                            discriminator_logits.detach().argmax(dim=-1).eq(route_targets).float().mean().item()
                        )
                    diagnostic_due = (
                        args.objective_gradient_diagnostics_every > 0
                        and (step + 1) % args.objective_gradient_diagnostics_every == 0
                        and (micro_step + 1) % args.gradient_accumulation_steps == 0
                    )
                    if diagnostic_due:
                        objective_gradient_diagnostics[modality_name] = (
                            objective_shared_gradient_metrics(
                                model,
                                {
                                    "diffusion": task_loss,
                                    "jepa": weighted_jepa_term,
                                    "sigreg": weighted_sigreg_term,
                                    "hsic": weighted_hsic_term,
                                },
                                selected_layers=set(args.objective_gradient_diagnostic_layers),
                            )
                        )
                    backward_loss = total_loss / (len(sub_batches) * args.gradient_accumulation_steps)
                    optimized_losses[modality_name] = (
                        total_loss.detach().item()
                        + (weighted_jepa_term.detach().item() if jepa_gradient_gated else 0.0)
                    )
                gradient_balancer.begin(modality_name)
                try:
                    backward_loss.backward()
                finally:
                    gradient_balancer.end()
                if gated_jepa_gradients is not None:
                    scale = 1.0 / (len(sub_batches) * args.gradient_accumulation_steps)
                    for parameter, gradient in zip(gated_jepa_parameters, gated_jepa_gradients):
                        if gradient is None:
                            continue
                        gated_gradient = gradient.detach().mul(scale)
                        if parameter.grad is None:
                            parameter.grad = gated_gradient
                        else:
                            parameter.grad.add_(gated_gradient)
                losses[modality_name] = task_loss.item()
                unweighted_losses[modality_name] = unweighted_task_loss.item()
                masked_fractions[modality_name] = masked.float().mean().item()
                masked_accuracies[modality_name] = (
                    logits.detach().argmax(dim=-1)[masked].eq(sub_batch["input_ids"][masked]).float().mean().item()
                )
            if (
                backtranslation_active
                and (micro_step + 1) % args.backtranslation_every_n_microsteps == 0
            ):
                text_batch = move_batch(batch["text_batch"], device)
                image_batch = move_batch(batch["image_batch"], device)
                backtranslation_results = unpaired_backtranslation_losses(
                    model, text_batch, image_batch, tokenizer, collator.image_offset,
                    args.num_image_codes, args.backtranslation_batch_size,
                    args.backtranslation_generation_steps, args.backtranslation_temperature,
                    args.backtranslation_reveal_order, args.backtranslation_min_confidence,
                    args.backtranslation_cycle_weight, args.pseudo_align_weight,
                    args.eps, not args.unweighting, amp_dtype,
                    args.pseudo_align_loss, args.pseudo_align_temperature,
                )
                valid_directions = [
                    (name, result) for name, result in backtranslation_results.items()
                    if result["loss"] is not None
                ]
                if args.backtranslation_warmup_steps > 0:
                    backtranslation_progress = min(
                        1.0, (step + 1) / args.backtranslation_warmup_steps
                    )
                else:
                    backtranslation_progress = 1.0
                for direction, result in valid_directions:
                    backward_loss = (
                        result["loss"] * backtranslation_progress
                        / (max(1, len(valid_directions)) * args.gradient_accumulation_steps)
                    )
                    reconstruction_modality = "text" if direction == "text_to_image" else "image"
                    gradient_balancer.begin(reconstruction_modality)
                    try:
                        backward_loss.backward()
                    finally:
                        gradient_balancer.end()
                latest_backtranslation_metrics = {
                    "backtranslation/progress": backtranslation_progress,
                    "backtranslation/active_directions": len(valid_directions),
                }
                for direction, result in backtranslation_results.items():
                    latest_backtranslation_metrics.update({
                        f"backtranslation/{direction}/{key}": value
                        for key, value in result.items() if key != "loss"
                    })
                    if result["loss"] is not None:
                        latest_backtranslation_metrics[
                            f"backtranslation/{direction}/weighted_loss"
                        ] = result["loss"].detach().item() * backtranslation_progress
            micro_step += 1
            is_last_batch = batch_index + 1 == len(train_loader)
            if micro_step % args.gradient_accumulation_steps != 0 and not is_last_batch:
                continue
            all_reduce_gradients(model)
            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            alignment_metrics = gradient_balancer.update()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            if ema_teacher is not None:
                decay = args.shared_jepa_ema_decay
                if args.shared_jepa_ema_decay_final is not None and args.shared_jepa_ema_ramp_steps > 0:
                    # data2vec ramps the teacher decay linearly: the teacher follows
                    # the student early and stabilizes later.
                    progress = min(1.0, step / args.shared_jepa_ema_ramp_steps)
                    decay = args.shared_jepa_ema_decay + progress * (
                        args.shared_jepa_ema_decay_final - args.shared_jepa_ema_decay
                    )
                update_ema_teacher(
                    ema_teacher, model, decay,
                    share_input_encoder=args.data2vec_share_input_encoder,
                )
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if args.probe_every_steps and step % args.probe_every_steps == 0 and _RANK == 0:
                launch_periodic_probe(
                    model, tokenizer, args, output_dir, step, periodic_probe_state
                )
            loss_value = sum(losses.values()) / max(1, len(losses))
            optimized_loss_value = (
                sum(optimized_losses.values()) / max(1, len(optimized_losses))
            )
            unweighted_loss_value = (
                sum(unweighted_losses.values()) / max(1, len(unweighted_losses))
            )
            if step % args.log_every == 0:
                effective_samples = args.log_every * args.batch_size * args.gradient_accumulation_steps
                rate = effective_samples / max(time.time() - window_start, 1e-6)
                modality_text = " ".join(f"{name}_loss={value:.4f}" for name, value in losses.items())
                accuracy_text = " ".join(f"{name}_acc={value:.3f}" for name, value in masked_accuracies.items())
                adversarial_text = " ".join(
                    f"adv_{name}={adversarial_losses[name]:.4f}/acc={discriminator_accuracies[name]:.3f}"
                    for name in adversarial_losses
                )
                sigreg_text = " ".join(
                    f"sigreg_{name}={value:.4f}" for name, value in sigreg_losses.items()
                )
                jepa_text = " ".join(
                    f"jepa_{name}={value:.4f}" for name, value in shared_jepa_losses.items()
                )
                if args.data2vec_hidden:
                    # target_spread near 0 means the teacher has collapsed to one
                    # vector; cosine says how well the head tracks the target.
                    jepa_text += " " + " ".join(
                        f"d2v_{name}/cos={values.get('prediction_target_cosine', float('nan')):.3f}"
                        f"/global_cos={values.get('global_cosine', float('nan')):.3f}"
                        f"/target_spread={values.get('target_position_spread', float('nan')):.4f}"
                        for name, values in shared_jepa_layer_cosines.items()
                    )
                if args.data2vec_teacherless:
                    jepa_text += " " + " ".join(
                        f"lejepa_{name}/pred={values.get('prediction', float('nan')):.4f}"
                        f"/sigreg={values.get('sigreg', float('nan')):.4f}"
                        f"/pooled_cos={values.get('pooled_cosine', float('nan')):.3f}"
                        for name, values in shared_jepa_layer_losses.items()
                    )
                if args.data2vec_projector_sigreg_weight > 0:
                    jepa_text += " " + " ".join(
                        f"hybrid_{name}/pred={values.get('prediction', float('nan')):.4f}"
                        f"/sigreg={values.get('sigreg', float('nan')):.3f}"
                        f"/emb_cos={values.get('embedding_cosine', float('nan')):.3f}"
                        f"/proj_cos={values.get('projection_cosine', float('nan')):.3f}"
                        for name, values in shared_jepa_layer_losses.items()
                    )
                if args.lejepa_views:
                    # embedding_cos is the backbone cone; projection_cos should sit near 0.
                    jepa_text += " " + " ".join(
                        f"lejepa_{name}/inv={values['invariance']:.4f}/sigreg={values['sigreg']:.3f}"
                        f"/emb_cos={values['embedding_cosine']:.3f}/proj_cos={values['projection_cosine']:.3f}"
                        f"/kept_g={values['global_kept_fraction']:.2f}"
                        f"/kept_l={values.get('local_kept_fraction', float('nan')):.2f}"
                        for name, values in shared_jepa_layer_losses.items()
                    )
                hsic_text = " ".join(
                    f"hsic_{name}={value:.5f}/w={modulewise_hsic_weights.get(name, 0.0):.3g}"
                    for name, value in modulewise_hsic_losses.items()
                )
                balance_text = " ".join(f"{key}={value:.3g}" for key, value in alignment_metrics.items())
                backtranslation_text = " ".join(
                    f"{key.removeprefix('backtranslation/')}={value:.3g}"
                    for key, value in latest_backtranslation_metrics.items()
                )
                stage_text = "shared+private" if private_lora_enabled else "shared-only"
                if (
                    (modulewise_jepa_active or modulewise_hsic_active)
                    and args.objective_gradient_diagnostics_every > 0
                    and step % args.objective_gradient_diagnostics_every == 0
                ):
                    ranks = adapter_delta_effective_ranks(
                        model, args.shared_jepa_layers, args.modulewise_jepa_modules,
                        "image" if args.data_mode == "image_only" else "text",
                    )
                    log.info(
                        f"epoch={epoch:03d} step={step:07d} delta_effective_rank "
                        + " ".join(
                            f"{key.removeprefix('blocks.')}={value:.1f}"
                            for key, value in sorted(ranks.items())
                        )
                    )
                    if wandb_run:
                        wandb_run.log(
                            {f"delta_effective_rank/{key}": value for key, value in ranks.items()},
                            step=step,
                        )
                log.info(f"epoch={epoch:03d} step={step:07d} stage={stage_text} lr={optimizer.param_groups[0]['lr']:.3g} loss={loss_value:.4f} optimized_loss={optimized_loss_value:.4f} unweighted_loss={unweighted_loss_value:.4f} {modality_text} {accuracy_text} {adversarial_text} {sigreg_text} {jepa_text} {hsic_text} {backtranslation_text} {balance_text} samples/s={rate:.1f}" + (f" global_samples/s={rate * _WORLD:.1f}" if _WORLD > 1 else ""))
                if wandb_run:
                    payload = {
                        "train/loss": loss_value,
                        "train/optimized_loss": optimized_loss_value,
                        "train/loss_1_over_t": loss_value,
                        "train/loss_unweighted": unweighted_loss_value,
                        "train/samples_per_second": rate,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/private_lora_enabled": int(private_lora_enabled),
                        "train/adversarial_enabled": int(adversarial_active),
                        "train/sigreg_enabled": int(sigreg_active),
                        "train/shared_jepa_enabled": int(shared_jepa_active),
                        "train/modulewise_jepa_enabled": int(modulewise_jepa_active),
                        "train/modulewise_hsic_enabled": int(modulewise_hsic_active),
                        "jepa/progress": jepa_progress,
                        "hsic/progress": hsic_progress,
                        "train/backtranslation_enabled": int(backtranslation_active),
                        "train/epoch_progress": epoch + (batch_index + 1) / len(train_loader),
                        "epoch": epoch,
                        **alignment_metrics,
                        **latest_backtranslation_metrics,
                    }
                    payload.update({f"train/{name}_loss": value for name, value in losses.items()})
                    payload.update({
                        f"train/{name}_optimized_loss": value
                        for name, value in optimized_losses.items()
                    })
                    payload.update({
                        f"train/{name}_loss_1_over_t": value
                        for name, value in losses.items()
                    })
                    payload.update({
                        f"train/{name}_loss_unweighted": value
                        for name, value in unweighted_losses.items()
                    })
                    payload.update({f"train/{name}_masked_fraction": value for name, value in masked_fractions.items()})
                    payload.update({f"train/{name}_masked_accuracy": value for name, value in masked_accuracies.items()})
                    payload.update({f"adversarial/{name}_loss": value for name, value in adversarial_losses.items()})
                    payload.update({f"adversarial/{name}_accuracy": value for name, value in discriminator_accuracies.items()})
                    payload.update({f"sigreg/{name}_loss": value for name, value in sigreg_losses.items()})
                    for name, layer_losses in sigreg_layer_losses.items():
                        payload.update({
                            f"sigreg/{name}/layer_{layer:02d}_loss": value
                            for layer, value in layer_losses.items()
                        })
                    payload.update({
                        f"jepa/{name}_loss": value
                        for name, value in shared_jepa_losses.items()
                    })
                    payload.update({
                        f"jepa/{name}_weight": value
                        for name, value in shared_jepa_weights.items()
                    })
                    payload.update({
                        f"jepa/{name}_weighted_to_diffusion_grad_ratio": value
                        for name, value in shared_jepa_gradient_ratios.items()
                    })
                    for name, layer_losses in shared_jepa_layer_losses.items():
                        payload.update({
                            (
                                f"jepa/{name}/layer_{layer:02d}_loss"
                                if isinstance(layer, int)
                                else f"jepa/{name}/module_{layer}_loss"
                            ): value
                            for layer, value in layer_losses.items()
                        })
                    for name, layer_cosines in shared_jepa_layer_cosines.items():
                        payload.update({
                            (
                                f"jepa/{name}/layer_{layer:02d}_cosine"
                                if isinstance(layer, int)
                                else f"jepa/{name}/module_{layer}_cosine"
                            ): value
                            for layer, value in layer_cosines.items()
                        })
                    payload.update({
                        f"hsic/{name}_loss": value
                        for name, value in modulewise_hsic_losses.items()
                    })
                    payload.update({
                        f"hsic/{name}_weight": value
                        for name, value in modulewise_hsic_weights.items()
                    })
                    payload.update({
                        f"hsic/{name}_weighted_to_diffusion_grad_ratio": value
                        for name, value in modulewise_hsic_gradient_ratios.items()
                    })
                    for name, details in modulewise_hsic_details.items():
                        payload.update({
                            f"hsic/{name}/{module}_loss": value
                            for module, value in details.items()
                        })
                    for name, diagnostics in objective_gradient_diagnostics.items():
                        payload.update({
                            f"objective_grad/{name}/{key}": value
                            for key, value in diagnostics.items()
                        })
                    for name, diagnostics in sigreg_diagnostics.items():
                        payload.update({
                            f"sigreg/{name}_{key}": value
                            for key, value in diagnostics.items()
                        })
                    payload.update({f"adversarial/{name}": value for name, value in adversarial_representation_norms.items()})
                    wandb_run.log(payload, step=step)
                window_start = time.time()
            if step_val_loader is not None and step % args.step_eval_every == 0:
                step_metrics = evaluate_paired_conditioning(
                    model, step_val_loader, tokenizer.mask_id, device, amp_dtype,
                    args.step_eval_mask_ratios, args.step_eval_control_ratios, args.seed,
                    args.paired_val_directions,
                    args.asymmetric_condition_target,
                )
                selection_loss = paired_t1_selection_loss(
                    step_metrics,
                    "text_to_image" if args.validation_selection == "paired_t1_text_to_image_matched" else None,
                )
                log.info(
                    f"step validation epoch={epoch:03d} step={step:07d} "
                    f"paired_t1_matched_loss={selection_loss:.4f}"
                )
                if wandb_run:
                    wandb_run.log({**step_metrics, "val/fast_selection_loss": selection_loss, "epoch": epoch}, step=step)
            if args.max_steps is not None and step >= args.max_steps:
                stop = True
                break

        if (epoch + 1) % args.eval_every == 0 or stop:
            val_loss = all_reduce_mean(evaluate(model, val_loader, args, tokenizer, device, amp_dtype), device)
            log.info(f"validation epoch={epoch:03d} fixed-t=0.75 loss={val_loss:.4f}")
            if not parameters_in_sync(model):
                log.warning(f"distributed: ranks hold DIFFERENT weights at epoch={epoch:03d}")
            elif _WORLD > 1:
                log.info(f"distributed: all {_WORLD} ranks hold identical weights at epoch={epoch:03d}")
            diffusion_nll_metrics = evaluate_diffusion_nll(
                model, val_loader, args, tokenizer, device, amp_dtype
            )
            if diffusion_nll_metrics:
                log.info(
                    f"validation epoch={epoch:03d} diffusion_nll_mc_nats="
                    f"{diffusion_nll_metrics['val/diffusion_nll_mc_nats']:.4f} "
                    f"denoising_nll_mc_nats="
                    f"{diffusion_nll_metrics['val/denoising_nll_mc_nats']:.4f} "
                    f"draws={args.val_diffusion_nll_draws}"
                )
            paired_metrics = {}
            if paired_val_loader is not None:
                paired_metrics = evaluate_paired_conditioning(
                    model, paired_val_loader, tokenizer.mask_id, device, amp_dtype,
                    args.paired_val_mask_ratios, args.paired_val_control_ratios, args.seed,
                    args.paired_val_directions,
                    args.asymmetric_condition_target,
                )
                summary = " ".join(
                    f"{key.removeprefix('val/paired/')}={value:.4f}"
                    for key, value in paired_metrics.items() if key.endswith(("context_gain", "shuffle_gap"))
                )
                log.info(f"paired validation epoch={epoch:03d} {summary}")
            sample_path = None
            if args.gen_num_samples > 0 and args.objective != "text" and _RANK == 0:
                sample_path = output_dir / "samples" / f"epoch_{epoch:03d}.png"
                save_samples(model, tokenizer, collator, val_source, vqvae, args, device, sample_path)
            if wandb_run:
                payload = {
                    "val/loss_t0.75": val_loss,
                    **diffusion_nll_metrics,
                    "epoch": epoch,
                    **paired_metrics,
                }
                if sample_path is not None:
                    import wandb
                    payload["val/samples"] = wandb.Image(str(sample_path))
                wandb_run.log(payload, step=step)
            selection_loss = (
                paired_t1_selection_loss(
                    paired_metrics,
                    "text_to_image"
                    if args.validation_selection == "paired_t1_text_to_image_matched"
                    else None,
                )
                if args.validation_selection.startswith("paired_t1_") else val_loss
            )
            if wandb_run:
                wandb_run.log({"val/selection_loss": selection_loss, "epoch": epoch}, step=step)
            # Best-checkpoint selection always uses the broader, fixed epoch
            # validation set; the 50-step subset is for tracking only.
            if selection_loss < best_val - args.early_stopping_min_delta:
                best_val = selection_loss
                epochs_without_improvement = 0
                save_checkpoint(
                    output_dir / "best.pt", model, optimizer, epoch, step, best_val,
                    args, tokenizer, lora_modules, epochs_without_improvement, ema_teacher,
                    lr_scheduler,
                )
            else:
                epochs_without_improvement += 1
            if wandb_run:
                wandb_run.log({
                    "val/best_selection_loss": best_val,
                    "val/epochs_without_improvement": epochs_without_improvement,
                    "epoch": epoch,
                }, step=step)
            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                log.info(
                    f"early stopping at epoch={epoch:03d}: selection_loss={selection_loss:.4f} "
                    f"best={best_val:.4f} patience={args.early_stopping_patience}"
                )
                stop = True
        if (epoch + 1) % args.save_every == 0 or stop:
            save_checkpoint(
                output_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, step, best_val,
                args, tokenizer, lora_modules, epochs_without_improvement, ema_teacher,
                lr_scheduler,
            )
            save_checkpoint(
                output_dir / "last.pt", model, optimizer, epoch, step, best_val,
                args, tokenizer, lora_modules, epochs_without_improvement, ema_teacher,
                lr_scheduler,
            )
        if stop:
            break
    if wandb_run:
        wandb_run.finish()
    log.info("training complete")
    if _WORLD > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
