"""Raw / centred / per-dimension z-scored versions of pooled features.

Cosine distances on uncentred pooled features are dominated by the shared mean
direction (the "cone"), which differs a lot between models (mean pairwise cosine
0.85-1.00), so sensitivity ratios are reported in all three geometries. Statistics
come from every vector being evaluated; each version is L2-normalised again.
"""
from __future__ import annotations
import numpy as np


def geometry_variants(x: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    unit = lambda v: v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    mean = x.mean(0, keepdims=True)
    std = x.std(0, keepdims=True)
    return {"raw": unit(x), "centred": unit(x - mean), "zscore": unit((x - mean) / np.clip(std, 1e-8, None))}
