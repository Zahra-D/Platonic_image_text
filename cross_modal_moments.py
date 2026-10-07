#!/usr/bin/env python3
"""Pull two modalities' representation distributions toward each other.

CKA between a text and an image representation is a comparison of their
second-moment structure.  When one model encodes both modalities they share a
basis, so that structure can be aligned *directly*: match the mean and the
covariance of the pooled representations.

The two modalities are processed in separate sub-batches and each graph is
freed before the other runs, so the coupling is computed against an EMA buffer
of the *other* modality's statistics rather than against a live batch.  The
gradient therefore flows only through the modality currently in hand.

This is deliberately not SIGReg.  SIGReg pushes every modality toward one fixed
isotropic Gaussian, which closes the modality gap by flattening both
representations -- measured at CKA 0.023 with scene probes at 55%.  Here the
target is the *other modality's own* distribution, so the shared structure is
what the two are pulled onto, and nothing rewards isotropy.
"""
from __future__ import annotations

import torch


class CrossModalMoments:
    """EMA buffers of each modality's pooled mean and covariance."""

    def __init__(self, width: int, decay: float = 0.95, device=None):
        self.decay = decay
        self.width = width
        self.mean: dict[str, torch.Tensor] = {}
        self.covariance: dict[str, torch.Tensor] = {}
        self.device = device

    @staticmethod
    def pooled(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mask-weighted mean over positions, one vector per example."""
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)

    def statistics(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        centred = pooled - pooled.mean(0, keepdim=True)
        covariance = centred.T @ centred / max(pooled.shape[0] - 1, 1)
        return pooled.mean(0), covariance

    def update(self, modality: str, mean: torch.Tensor, covariance: torch.Tensor) -> None:
        mean, covariance = mean.detach(), covariance.detach()
        if modality not in self.mean:
            self.mean[modality], self.covariance[modality] = mean, covariance
            return
        d = self.decay
        self.mean[modality] = d * self.mean[modality] + (1 - d) * mean
        self.covariance[modality] = d * self.covariance[modality] + (1 - d) * covariance

    def loss(self, modality: str, pooled: torch.Tensor) -> tuple[torch.Tensor | None, dict]:
        """Distance from this batch's moments to the other modality's EMA moments."""
        mean, covariance = self.statistics(pooled.float())
        others = [m for m in self.mean if m != modality]
        self.update(modality, mean, covariance)
        if not others:
            return None, {}
        other = others[0]
        target_mean = self.mean[other].to(mean.device)
        target_covariance = self.covariance[other].to(covariance.device)
        mean_term = (mean - target_mean).pow(2).sum()
        # Normalized by width so the two terms are on comparable scales and the
        # weight means the same thing at any model size.
        covariance_term = (covariance - target_covariance).pow(2).sum() / self.width
        return mean_term + covariance_term, {
            "mean_distance": float(mean_term.detach()),
            "covariance_distance": float(covariance_term.detach()),
        }
