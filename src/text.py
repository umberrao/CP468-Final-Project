from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = (
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
)

TOKEN_PATTERN = re.compile(
    r"\w+(?:['’-]\w+)*|[^\w\s]",
    flags=re.UNICODE,
)


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into words and punctuation."""
    return TOKEN_PATTERN.findall(text.casefold().strip())


def detokenize(tokens: Sequence[str]) -> str:
    """Join tokens into readable text."""
    text = " ".join(tokens)

    # WikiAuto uses <SEP> to separate split sentences.
    text = re.sub(
        r"([.!?])\s*<\s*sep\s*>\s*",
        r"\1 ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*<\s*sep\s*>\s*",
        ". ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\s+([,.;:!?%…)\]}])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)

    return text.strip()


class Vocabulary:
    def __init__(self, tokens: Sequence[str]) -> None:
        self.id_to_token = list(tokens)

        if self.id_to_token[: len(SPECIAL_TOKENS)] != list(
            SPECIAL_TOKENS
        ):
            raise ValueError("Vocabulary must begin with special tokens.")

        if len(set(self.id_to_token)) != len(self.id_to_token):
            raise ValueError("Vocabulary contains duplicate tokens.")

        self.token_to_id = {
            token: index
            for index, token in enumerate(self.id_to_token)
        }

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        min_frequency: int = 2,
        max_size: int | None = 20_000,
    ) -> "Vocabulary":
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1.")

        if max_size is not None and max_size < len(SPECIAL_TOKENS):
            raise ValueError("max_size is smaller than special tokens.")

        counts: Counter[str] = Counter()

        for text in texts:
            counts.update(tokenize(text))

        candidates = [
            token
            for token, frequency in counts.items()
            if frequency >= min_frequency
            and token not in SPECIAL_TOKENS
        ]

        candidates.sort(
            key=lambda token: (-counts[token], token)
        )

        if max_size is not None:
            available = max_size - len(SPECIAL_TOKENS)
            candidates = candidates[:available]

        return cls([*SPECIAL_TOKENS, *candidates])

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    def __len__(self) -> int:
        return len(self.id_to_token)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: int | None = None,
    ) -> list[int]:
        token_ids = [
            self.token_to_id.get(token, self.unk_id)
            for token in tokenize(text)
        ]

        special_count = 2 if add_special_tokens else 0

        if max_length is not None:
            if max_length < special_count:
                raise ValueError(
                    "max_length cannot fit the special tokens."
                )

            token_ids = token_ids[: max_length - special_count]

        if add_special_tokens:
            token_ids = [
                self.bos_id,
                *token_ids,
                self.eos_id,
            ]

        return token_ids

    def decode(
        self,
        token_ids: Iterable[int],
        skip_special_tokens: bool = True,
    ) -> str:
        tokens = []

        for token_id in token_ids:
            if token_id == self.eos_id:
                break

            if 0 <= token_id < len(self):
                token = self.id_to_token[token_id]
            else:
                token = UNK_TOKEN

            if (
                skip_special_tokens
                and token in {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}
            ):
                continue

            tokens.append(token)

        return detokenize(tokens)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"tokens": self.id_to_token},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        data = json.loads(
            Path(path).read_text(encoding="utf-8")
        )
        return cls(data["tokens"])