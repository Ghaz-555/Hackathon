"""A fixed, inspectable Unicode code-point vocabulary (not subword tokenization)."""

import numpy as np


class CharacterTokenizer:
    def __init__(self, characters):
        characters = tuple(characters)
        if (len(characters) < 2 or len(set(characters)) != len(characters)
                or any(not isinstance(c, str) or len(c) != 1 for c in characters)):
            raise ValueError("Vocabulary needs at least two unique single characters")
        self.characters = characters
        self._ids = {c: i for i, c in enumerate(characters)}

    @classmethod
    def from_text(cls, text: str):
        return cls(sorted(set(text)))

    def __len__(self):
        return len(self.characters)

    def encode(self, text: str) -> np.ndarray:
        # Reject unfamiliar characters instead of silently inventing a token ID.
        unknown = set(text) - self._ids.keys()
        if unknown:
            raise ValueError(f"Characters outside vocabulary: {sorted(unknown)!r}")
        return np.array([self._ids[c] for c in text], dtype=np.int64)

    def decode(self, ids) -> str:
        ids = np.asarray(ids)
        if ids.ndim != 1 or (ids.size and (ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= len(self)))):
            raise ValueError("decode expects a one-dimensional array of valid token IDs")
        return "".join(self.characters[int(i)] for i in ids)
