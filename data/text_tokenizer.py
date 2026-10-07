"""A tiny, dependency-free word tokenizer for training CLEVR from scratch."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


TOKEN_PATTERN = re.compile(r"\w+(?:'\w+)?|[^\w\s]", re.UNICODE)


class ClevrTextTokenizer:
    """Lower-case word/punctuation tokenizer with a serializable vocabulary.

    CLEVR captions use a small controlled language, so a vocabulary learned from
    the training manifest is both cheaper and more appropriate than downloading a
    pretrained tokenizer for a model that is otherwise trained from scratch.
    """

    SPECIAL_TOKENS = ("<pad>", "<mask>", "<bos>", "<eos>", "<unk>", "<text>", "<image>")

    def __init__(self, vocabulary: list[str]):
        if list(vocabulary[: len(self.SPECIAL_TOKENS)]) != list(self.SPECIAL_TOKENS):
            raise ValueError("Vocabulary must begin with ClevrTextTokenizer.SPECIAL_TOKENS")
        if len(vocabulary) != len(set(vocabulary)):
            raise ValueError("Vocabulary contains duplicate tokens")
        self.vocabulary = list(vocabulary)
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocabulary)}

    @classmethod
    def build(cls, texts: list[str], min_frequency: int = 1) -> "ClevrTextTokenizer":
        counts = Counter(token for text in texts for token in cls.tokenize(text))
        lexical = sorted(token for token, count in counts.items() if count >= min_frequency)
        lexical = [token for token in lexical if token not in cls.SPECIAL_TOKENS]
        return cls(list(cls.SPECIAL_TOKENS) + lexical)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        tokens = self.tokenize(text)
        if max_length is not None:
            tokens = tokens[:max_length]
        unk = self.unk_id
        return [self.token_to_id.get(token, unk) for token in tokens]

    def decode(self, ids: list[int]) -> str:
        tokens = [self.vocabulary[idx] for idx in ids if 0 <= idx < len(self.vocabulary)]
        text = " ".join(token for token in tokens if token not in self.SPECIAL_TOKENS)
        return re.sub(r"\s+([.,!?;:])", r"\1", text)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"vocabulary": self.vocabulary}, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ClevrTextTokenizer":
        payload = json.loads(Path(path).read_text())
        return cls(payload["vocabulary"])

    def __len__(self) -> int:
        return len(self.vocabulary)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def mask_id(self) -> int:
        return self.token_to_id["<mask>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def text_id(self) -> int:
        return self.token_to_id["<text>"]

    @property
    def image_id(self) -> int:
        return self.token_to_id["<image>"]
