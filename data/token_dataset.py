"""On-disk token-cache support for Stage-2 diffusion training."""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class TokenGridDataset(Dataset):
    """A memory-resident collection of VQ token grids saved as uint16.

    Keeping codebook indices on CPU in uint16 makes the complete 70k CLEVR
    training split about 54 MB.  They are converted back to int64 after being
    transferred to the accelerator, immediately before the Transformer.
    """

    def __init__(self, cache_path: str):
        self.cache_path = Path(cache_path).expanduser()
        if not self.cache_path.is_file():
            raise FileNotFoundError(
                f"Token cache not found: {self.cache_path}. "
                "Create it with pretokenize_clevr.py first."
            )
        payload = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "tokens" not in payload:
            raise ValueError(f"Invalid token cache: {self.cache_path}")
        self.tokens = payload["tokens"].contiguous()
        self.metadata = payload.get("metadata", {})
        if self.tokens.ndim != 3:
            raise ValueError(
                f"Expected [N, grid_h, grid_w] tokens in {self.cache_path}, "
                f"got {tuple(self.tokens.shape)}"
            )

    @property
    def grid_size(self):
        return tuple(self.tokens.shape[1:])

    def __len__(self):
        return self.tokens.size(0)

    def __getitem__(self, idx):
        return self.tokens[idx]
