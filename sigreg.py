"""Sketched Isotropic Gaussian Regularization for shared representations.

This follows LeJEPA's sliced Epps--Pulley construction: project a batch onto
random unit directions, compare each empirical characteristic function with
that of N(0, 1), then average the test statistic across slices.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sigreg_loss(
    embeddings: torch.Tensor,
    *,
    num_slices: int = 256,
    num_points: int = 17,
    t_max: float = 3.0,
    seed: int = 0,
) -> torch.Tensor:
    """Return a differentiable sliced Gaussianity statistic for [N, D]."""
    if embeddings.ndim != 2:
        raise ValueError("SIGReg embeddings must have shape [samples, dimensions]")
    if embeddings.size(0) < 2:
        raise ValueError("SIGReg requires at least two samples")
    if num_slices < 1 or num_points < 3 or num_points % 2 != 1 or t_max <= 0:
        raise ValueError("SIGReg requires slices>=1, odd points>=3, and t_max>0")

    x = embeddings.float()
    generator = torch.Generator(device=x.device).manual_seed(int(seed))
    directions = torch.randn(
        x.size(1), num_slices, device=x.device, dtype=x.dtype, generator=generator
    )
    directions = F.normalize(directions, p=2, dim=0, eps=1e-12)
    projected = x @ directions

    t = torch.linspace(0.0, t_max, num_points, device=x.device, dtype=x.dtype)
    dt = t_max / (num_points - 1)
    weights = torch.full_like(t, 2.0 * dt)
    weights[0] = dt
    weights[-1] = dt
    normal_cf = torch.exp(-0.5 * t.square())
    weights = weights * normal_cf

    phases = projected.unsqueeze(-1) * t
    empirical_real = phases.cos().mean(dim=0)
    empirical_imag = phases.sin().mean(dim=0)
    error = (empirical_real - normal_cf).square() + empirical_imag.square()
    per_slice = (error * weights).sum(dim=-1) * x.size(0)
    return per_slice.mean()


@torch.no_grad()
def gaussianity_diagnostics(embeddings: torch.Tensor) -> dict[str, float]:
    """Cheap batch diagnostics accompanying the full SIGReg statistic."""
    x = embeddings.detach().float()
    mean = x.mean(dim=0)
    centered = x - mean
    variance = centered.square().mean(dim=0)
    return {
        "mean_abs": mean.abs().mean().item(),
        "variance_mean": variance.mean().item(),
        "variance_error": (variance - 1.0).abs().mean().item(),
    }
