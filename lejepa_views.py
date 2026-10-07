"""Multi-view LeJEPA, as in the paper and its reference code (arXiv 2511.08544).

One network, no teacher, no stop-gradient. Every sample is seen through
``V = V_g + V_l`` crops; each crop is encoded, mean-pooled to one vector, and
passed through a projector MLP. On the projector output ``z[v, n]``

    centre[n] = mean over the V_g global views of z[v, n]
    invariance = mean over v, n, k of (centre[n] - z[v, n])^2
    SIGReg     = mean over v of SIGReg({z[v, n]}_n)          (per view)
    loss       = (1 - lambda) * invariance + lambda * SIGReg

The backbone embedding (pooled last block, before the projector) is what is
evaluated, exactly as the paper probes the backbone and not the projector.

A crop here is the token-space analogue of RandomResizedCrop: a rectangle on
the VQ token grid for images, a contiguous span of the caption for text. Only
the kept tokens are fed to the network, at their original position ids, so a
crop costs what its length costs. The leading special tokens (modality, BOS)
are always kept; EOS is not, since its position would leak caption length. There is no resize -- tokens cannot be rescaled -- and no
photometric or flip augmentation, since colour and left/right are semantic in
CLEVR.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from sigreg import sigreg_loss


def crop_keep_mask(
    eligible: torch.Tensor,
    attention: torch.Tensor,
    num_views: int,
    scale: tuple[float, float],
    aspect: tuple[float, float],
    grid: tuple[int, int] | None,
) -> torch.Tensor:
    """Sample ``num_views`` crops per row. Returns keep masks [V, B, L]."""
    batch, length = eligible.shape
    device = eligible.device
    count = num_views * batch
    eligible = eligible.unsqueeze(0).expand(num_views, -1, -1).reshape(count, length)
    attention = attention.unsqueeze(0).expand(num_views, -1, -1).reshape(count, length)
    # Rank of each content token within its row: 0, 1, 2, ... (raster order for images).
    rank = eligible.long().cumsum(dim=1) - 1
    fraction = torch.empty(count, device=device).uniform_(*scale)
    if grid is None:
        content = eligible.sum(dim=1)
        span = (fraction * content).round().long().clamp(min=1)
        span = torch.minimum(span, content.clamp(min=1))
        start = (torch.rand(count, device=device) * (content - span + 1)).floor().long()
        inside = (rank >= start[:, None]) & (rank < (start + span)[:, None])
    else:
        rows, cols = grid
        if not eligible.sum(dim=1).eq(rows * cols).all():
            raise ValueError("2D crops need exactly grid rows * cols content tokens per row")
        log_ratio = torch.empty(count, device=device).uniform_(math.log(aspect[0]), math.log(aspect[1]))
        # Aspect is relative to the grid's own shape, so scale 1 can be the whole
        # image (the 16x24 grid is itself 1.5:1).
        ratio = log_ratio.exp() * cols / rows
        area = fraction * rows * cols
        height = (area / ratio).sqrt().round().long().clamp(1, rows)
        width = (area * ratio).sqrt().round().long().clamp(1, cols)
        top = (torch.rand(count, device=device) * (rows - height + 1)).floor().long()
        left = (torch.rand(count, device=device) * (cols - width + 1)).floor().long()
        row, col = rank // cols, rank % cols
        inside = (
            (row >= top[:, None]) & (row < (top + height)[:, None])
            & (col >= left[:, None]) & (col < (left + width)[:, None])
        )
    # Keep only the special tokens before the content (modality, BOS). EOS is
    # dropped: its position id is the caption length, which every crop of a
    # caption shares -- an invariance shortcut a resized image crop cannot have.
    keep = (eligible & inside) | (attention & ~eligible & (rank < 0))
    return keep.reshape(num_views, batch, length)


def pack_views(batch: dict[str, torch.Tensor], keep: torch.Tensor) -> dict[str, torch.Tensor]:
    """Drop non-kept tokens. ``keep`` is [V, B, L]; outputs are [V * B, L_kept]."""
    views = keep.size(0)
    keep = keep.flatten(0, 1) & batch["attention_mask"].repeat(views, 1)
    counts = keep.sum(dim=1)
    width = int(counts.max())
    # Stable sort puts kept tokens first in their original order.
    order = torch.sort((~keep).to(torch.int8), dim=1, stable=True).indices[:, :width]
    valid = torch.arange(width, device=keep.device)[None] < counts[:, None]
    packed = {
        name: batch[name].repeat(views, 1).gather(1, order)
        for name in ("input_ids", "position_ids", "modality_ids", "route_ids", "eligible_mask")
    }
    packed["attention_mask"] = valid
    packed["eligible_mask"] = packed["eligible_mask"] & valid
    packed["route_ids"] = packed["route_ids"].masked_fill(~valid, -1)
    packed["kept_fraction"] = (
        (packed["eligible_mask"].sum(1).float()
         / batch["eligible_mask"].sum(1).repeat(views).clamp_min(1).float()).mean()
    )
    return packed


def encode_views(model, packed: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean-pooled last-block hidden state over the kept content tokens: [N, D]."""
    _, hidden = model(
        packed["input_ids"], packed["attention_mask"], packed["position_ids"],
        packed["modality_ids"], packed["route_ids"], return_hidden_by_layer=True,
    )
    last = hidden[max(hidden)]
    weights = packed["eligible_mask"].to(last.dtype).unsqueeze(-1)
    return (last * weights).sum(1) / weights.sum(1).clamp_min(1)


def mean_offdiagonal_cosine(x: torch.Tensor) -> float:
    unit = F.normalize(x.detach().float(), dim=1)
    gram = unit @ unit.T
    n = gram.size(0)
    return float((gram.sum() - gram.diagonal().sum()) / max(n * (n - 1), 1))


def lejepa_multiview_loss(
    model, batch: dict[str, torch.Tensor], *, global_views: int, local_views: int,
    global_scale: tuple[float, float], local_scale: tuple[float, float],
    aspect: tuple[float, float], grid: tuple[int, int] | None, lam: float,
    num_slices: int, num_points: int, t_max: float, seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    eligible, attention = batch["eligible_mask"], batch["attention_mask"]
    batch_size = eligible.size(0)
    embeddings, details = [], {}
    # Globals and locals are packed separately so short local crops are not
    # padded to the length of the global ones.
    for name, views, scale in (("global", global_views, global_scale),
                               ("local", local_views, local_scale)):
        if views == 0:
            continue
        packed = pack_views(batch, crop_keep_mask(eligible, attention, views, scale, aspect, grid))
        embeddings.append(encode_views(model, packed))
        details[f"{name}_kept_fraction"] = float(packed["kept_fraction"])
    embedding = torch.cat(embeddings, dim=0)              # [V * B, D], globals first
    # One projector call over every view, as in the reference code, so its
    # BatchNorm sees all views together.
    projected = model.lejepa_projector(embedding.float())
    projected = projected.reshape(global_views + local_views, batch_size, -1)
    centre = projected[:global_views].mean(dim=0)
    invariance = (centre - projected).square().mean()
    sigreg = torch.stack([
        sigreg_loss(view, num_slices=num_slices, num_points=num_points, t_max=t_max, seed=seed)
        for view in projected
    ]).mean()
    loss = (1.0 - lam) * invariance + lam * sigreg
    details.update({
        "invariance": float(invariance.detach()),
        "sigreg": float(sigreg.detach()),
        "embedding_cosine": mean_offdiagonal_cosine(embedding[:batch_size]),
        "projection_cosine": mean_offdiagonal_cosine(projected[0]),
    })
    return loss, details
