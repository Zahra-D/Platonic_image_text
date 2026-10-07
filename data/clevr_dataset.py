import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ClevrImageFolder(Dataset):
    """Loads CLEVR PNGs from a split directory (e.g. .../clevr/train) into
    [-1, 1]-normalized RGB tensors of shape [3, H, W].

    Expects `images.jsonl` (relative `image_path` per line) next to an
    `images/` subfolder, matching the layout already staged under
    ~/Omni/data/clevr/{train,val,test}.
    """

    def __init__(self, split_dir: str, image_size=(64, 96), manifest_path: str | None = None):
        self.split_dir = Path(split_dir).expanduser()
        self.height, self.width = image_size

        manifest = Path(manifest_path).expanduser() if manifest_path else self.split_dir / "images.jsonl"
        if manifest.exists():
            paths = []
            with open(manifest) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    paths.append(self.split_dir / rec["image_path"])
        else:
            paths = sorted((self.split_dir / "images").glob("*.png"))

        if not paths:
            raise FileNotFoundError(f"No CLEVR images found under {self.split_dir}")
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        img = img.resize((self.width, self.height), Image.Resampling.BILINEAR)
        array = np.asarray(img, dtype=np.float32).copy()
        x = torch.from_numpy(array).permute(2, 0, 1) / 255.0
        return x * 2.0 - 1.0  # [-1, 1]
