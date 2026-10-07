"""Minimal image-grid writing without a torchvision dependency."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def save_image_grid(
    images: torch.Tensor,
    path: str | Path,
    nrow: int | None = None,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> None:
    if images.ndim != 4 or images.size(1) not in {1, 3}:
        raise ValueError(f"Expected images [N,C,H,W], got {tuple(images.shape)}")
    images = images.detach().float().cpu()
    low, high = value_range
    images = ((images - low) / (high - low)).clamp(0, 1)
    if images.size(1) == 1:
        images = images.expand(-1, 3, -1, -1)
    count, _, height, width = images.shape
    columns = nrow or min(8, count)
    rows = math.ceil(count / columns)
    canvas = torch.zeros(3, rows * height, columns * width)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        canvas[:, row * height : (row + 1) * height, column * width : (column + 1) * width] = image
    array = (canvas.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(output)
