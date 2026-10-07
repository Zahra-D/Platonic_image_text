"""Paired and unpaired CLEVR text/image-token datasets."""

from __future__ import annotations

import json
import hashlib
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .text_tokenizer import ClevrTextTokenizer


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path).expanduser()
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def image_manifest_fingerprint(records: list[dict]) -> str:
    """Hash ordered image paths so a same-length but misordered cache is detectable."""
    digest = hashlib.sha256()
    for record in records:
        path = record.get("image_path")
        if not isinstance(path, str):
            raise ValueError("Every images.jsonl record must contain a string image_path")
        digest.update(path.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class ClevrMultimodalDataset(Dataset):
    """Load aligned CLEVR manifests and a matching VQ-token cache.

    ``mode='paired'`` returns a text/image pair at the same manifest row.
    ``mode='unpaired'`` returns the two modality datasets independently (all
    text examples followed by all image examples). It deliberately does not
    construct random pairs, since those would teach false conditioning.
    ``mode='text_only'`` and ``mode='image_only'`` are single-modality
    corpora: the other modality's records are empty and never sampled.
    """

    def __init__(
        self,
        split_dir: str | Path,
        token_cache: str | Path,
        mode: str = "paired",
        pair_manifest: str | Path | None = None,
        caption_field: str = "text",
        text_manifest: str | Path | None = None,
    ) -> None:
        if mode not in {"paired", "unpaired", "text_only", "image_only"}:
            raise ValueError("mode must be 'paired', 'unpaired', 'text_only', or 'image_only'")
        self.mode = mode
        self.split_dir = Path(split_dir).expanduser()
        if pair_manifest is not None:
            paired_records = read_jsonl(pair_manifest)
            # An image-only corpus has no captions, so the caption field is
            # required for every other mode only.
            missing = [] if mode == "image_only" or text_manifest is not None else [
                index for index, record in enumerate(paired_records) if caption_field not in record
            ]
            if missing:
                raise ValueError(
                    f"{pair_manifest} is missing caption field {caption_field!r} at row {missing[0]}"
                )
            self.image_records = paired_records
            self.text_records = [] if mode == "image_only" or text_manifest is not None else [
                {"text": record[caption_field]} for record in paired_records
            ]
        else:
            self.image_records = read_jsonl(self.split_dir / "images.jsonl")
            self.text_records = read_jsonl(self.split_dir / "text.jsonl")
        # Unpaired training from two separate corpora: captions from their own
        # manifest, images (and the token cache) from ``pair_manifest``. No row
        # of one corpus corresponds to a row of the other.
        self.independent_sources = text_manifest is not None
        if text_manifest is not None:
            if mode != "unpaired":
                raise ValueError("text_manifest is only meaningful with mode='unpaired'")
            text_rows = read_jsonl(text_manifest)
            missing = [index for index, record in enumerate(text_rows) if caption_field not in record]
            if missing:
                raise ValueError(f"{text_manifest} is missing caption field {caption_field!r} at row {missing[0]}")
            self.text_records = [{"text": record[caption_field]} for record in text_rows]

        # A pure text corpus intentionally has neither images nor a VQ-token
        # cache.  Keeping it as an explicit mode prevents accidental use as
        # paired multimodal supervision.
        if mode == "text_only":
            self.image_records = []
            self.image_tokens = None
            return

        payload = torch.load(Path(token_cache).expanduser(), map_location="cpu", weights_only=False)
        self.image_tokens = payload["tokens"] if isinstance(payload, dict) else payload
        if self.image_tokens.ndim != 3:
            raise ValueError(f"Expected [N,H,W] image tokens, got {tuple(self.image_tokens.shape)}")
        if len(self.image_records) != len(self.image_tokens):
            raise ValueError(
                f"Image manifest/cache mismatch: {len(self.image_records)} records versus "
                f"{len(self.image_tokens)} token grids"
            )
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        cached_fingerprint = metadata.get("image_manifest_sha256")
        if cached_fingerprint and cached_fingerprint != image_manifest_fingerprint(self.image_records):
            raise ValueError(
                "Image token cache ordering does not match images.jsonl; rebuild it with "
                "pretokenize_clevr.py --overwrite"
            )
        if self.independent_sources:
            # Images come from the token cache; the records were only needed for
            # the fingerprint above. Dropping the 2.5M scene-graph dicts saves
            # ~15 GB of host RAM per process (and per forked worker).
            self.image_records = range(len(self.image_records))
        if mode == "paired" and len(self.text_records) != len(self.image_records):
            raise ValueError(
                "Paired mode requires equal image/text manifest lengths; pairing is by row index "
                f"({len(self.image_records)} images, {len(self.text_records)} texts)"
            )

    @property
    def grid_size(self) -> tuple[int, int]:
        if self.image_tokens is None:
            raise RuntimeError("A text-only dataset has no image-token grid")
        return tuple(self.image_tokens.shape[1:])

    @property
    def texts(self) -> list[str]:
        return [record["text"] for record in self.text_records]

    def __len__(self) -> int:
        if self.mode == "text_only":
            return len(self.text_records)
        if self.mode == "image_only":
            return len(self.image_records)
        if self.mode == "paired":
            return len(self.image_records)
        return len(self.text_records) + len(self.image_records)

    def __getitem__(self, index: int) -> dict:
        if self.mode == "text_only":
            return {"kind": "text", "text": self.text_records[index]["text"], "pair_index": index}
        if self.mode == "image_only":
            return {"kind": "image", "image_tokens": self.image_tokens[index].long(), "pair_index": index}
        if self.mode == "paired":
            return {
                "kind": "paired",
                "text": self.text_records[index]["text"],
                "image_tokens": self.image_tokens[index].long(),
                "pair_index": index,
            }
        if index < len(self.text_records):
            return {"kind": "text", "text": self.text_records[index]["text"], "pair_index": index}
        image_index = index - len(self.text_records)
        return {
            "kind": "image",
            "image_tokens": self.image_tokens[image_index].long(),
            "pair_index": image_index,
        }

    def get_modality(self, index: int, modality: str) -> dict:
        if modality == "text":
            return {"kind": "text", "text": self.text_records[index]["text"], "pair_index": index}
        if modality == "image":
            if self.image_tokens is None:
                raise RuntimeError("A text-only dataset has no image examples")
            return {"kind": "image", "image_tokens": self.image_tokens[index].long(), "pair_index": index}
        raise ValueError("modality must be text or image")


class BalancedUnpairedDataset(Dataset):
    """Return one independently shuffled text and image carrier per item.

    The collator separates the carrier into two batches and the trainer executes
    two independent forwards. A derangement guarantees that a caption is never
    carried alongside its source image, matching Omni's strict unpaired path.
    ``set_epoch`` redraws the two permutations, so the artificial carrier is
    fresh in every epoch rather than becoming a fixed pseudo-pair.
    """

    def __init__(self, dataset: ClevrMultimodalDataset, seed: int = 13) -> None:
        self.dataset = dataset
        size = min(len(dataset.text_records), len(dataset.image_records))
        if size < 2:
            raise ValueError("Balanced strictly-unpaired training requires at least two examples")
        self.size = size
        self.seed = seed
        self.epoch = -1
        self.text_indices: list[int] = []
        self.image_indices: list[int] = []
        self._previous_image_by_text: dict[int, int] | None = None
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        """Draw a new strict derangement for this epoch.

        For datasets with at least three examples, we also reject a draw that
        reuses any text->image carrier from the immediately preceding epoch.
        This makes the translation/pseudo-pair exposure genuinely change at
        every epoch, rather than merely permuting the same carrier rows.
        With exactly two examples there is only one strict derangement, so that
        stronger condition is mathematically impossible.
        """
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if getattr(self.dataset, "independent_sources", False):
            # Two separate corpora: row indices carry no pairing, so there is no
            # derangement to enforce. Each epoch uses a fresh random ``size``
            # subset of each corpus, so the larger one is fully covered over epochs.
            text_pool = list(range(len(self.dataset.text_records)))
            image_pool = list(range(len(self.dataset.image_records)))
            random.Random(self.seed + 2 * epoch).shuffle(text_pool)
            random.Random(self.seed + 2 * epoch + 1).shuffle(image_pool)
            self.epoch = epoch
            self.text_indices = text_pool[:self.size]
            self.image_indices = image_pool[:self.size]
            return
        pool = list(range(self.size))
        text_indices = pool.copy()
        random.Random(self.seed + 2 * epoch).shuffle(text_indices)
        previous = self._previous_image_by_text
        attempt = 0
        while True:
            image_indices = pool.copy()
            random.Random(self.seed + 2 * epoch + 1 + attempt).shuffle(image_indices)
            strict_derangement = all(
                text_index != image_index
                for text_index, image_index in zip(text_indices, image_indices)
            )
            fresh_against_previous = (
                self.size == 2
                or previous is None
                or all(previous[text_index] != image_index for text_index, image_index in zip(text_indices, image_indices))
            )
            if strict_derangement and fresh_against_previous:
                break
            attempt += 1
        self.epoch = epoch
        self.text_indices = text_indices
        self.image_indices = image_indices
        self._previous_image_by_text = dict(zip(text_indices, image_indices))

    def __len__(self) -> int:
        return len(self.text_indices)

    def __getitem__(self, index: int) -> dict:
        return {
            "text_example": self.dataset.get_modality(self.text_indices[index], "text"),
            "image_example": self.dataset.get_modality(self.image_indices[index], "image"),
        }


class MultimodalCollator:
    TEXT_MODALITY = 1
    IMAGE_MODALITY = 2

    def __init__(
        self,
        tokenizer: ClevrTextTokenizer,
        num_image_codes: int,
        max_text_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_image_codes = num_image_codes
        self.max_text_length = max_text_length
        self.image_offset = len(tokenizer)
        self.vocab_size = self.image_offset + num_image_codes

    def _text_segment(self, text: str) -> tuple[list[int], list[int], list[int], list[bool]]:
        content = self.tokenizer.encode(text, self.max_text_length)
        ids = [self.tokenizer.text_id, self.tokenizer.bos_id, *content, self.tokenizer.eos_id]
        positions = list(range(len(ids)))
        modalities = [self.TEXT_MODALITY] * len(ids)
        eligible = [False, False, *([True] * len(content)), False]
        return ids, positions, modalities, eligible

    def _image_segment(self, image_tokens: torch.Tensor) -> tuple[list[int], list[int], list[int], list[bool]]:
        content = (image_tokens.flatten().long() + self.image_offset).tolist()
        ids = [self.tokenizer.image_id, self.tokenizer.bos_id, *content, self.tokenizer.eos_id]
        positions = list(range(len(ids)))
        modalities = [self.IMAGE_MODALITY] * len(ids)
        eligible = [False, False, *([True] * len(content)), False]
        return ids, positions, modalities, eligible

    def __call__(self, examples: list[dict]) -> dict[str, Any]:
        if examples and "text_example" in examples[0]:
            return {
                "text_batch": self([example["text_example"] for example in examples]),
                "image_batch": self([example["image_example"] for example in examples]),
            }
        rows = []
        for example in examples:
            parts = []
            if example["kind"] in {"paired", "text"}:
                parts.append(self._text_segment(example["text"]))
            if example["kind"] in {"paired", "image"}:
                parts.append(self._image_segment(example["image_tokens"]))
            rows.append(tuple(sum((list(part[field]) for part in parts), []) for field in range(4)))

        max_length = max(len(row[0]) for row in rows)
        batch = len(rows)
        input_ids = torch.full((batch, max_length), self.tokenizer.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch, max_length), dtype=torch.bool)
        position_ids = torch.zeros((batch, max_length), dtype=torch.long)
        modality_ids = torch.zeros((batch, max_length), dtype=torch.long)
        eligible_mask = torch.zeros((batch, max_length), dtype=torch.bool)
        for index, (ids, positions, modalities, eligible) in enumerate(rows):
            length = len(ids)
            input_ids[index, :length] = torch.tensor(ids)
            attention_mask[index, :length] = True
            position_ids[index, :length] = torch.tensor(positions)
            modality_ids[index, :length] = torch.tensor(modalities)
            eligible_mask[index, :length] = torch.tensor(eligible)
        # Route every token through the private adapter for its own modality.
        # Padding receives -1 and therefore uses only the shared/base path.
        route_ids = torch.full_like(modality_ids, -1)
        route_ids[modality_ids.eq(self.TEXT_MODALITY)] = 0
        route_ids[modality_ids.eq(self.IMAGE_MODALITY)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "modality_ids": modality_ids,
            "eligible_mask": eligible_mask,
            "pair_indices": torch.tensor([example["pair_index"] for example in examples]),
            "route_ids": route_ids,
        }
