from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class EmbeddingBackend(ABC):
    """Turns images into unit-length embedding vectors.

    Contract: ``embed_batch(paths)`` returns exactly one row per input path, in order. A row
    for an image that could not be embedded is all-NaN. Callers drop those rows together with
    their ids; a backend that silently skipped failures would shift every later id onto the
    wrong vector, and that mislabelling would be saved into the index and checkpoint.
    """

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_batch(self, image_paths: list[Path]) -> np.ndarray: ...


def aligned_unit_rows(rows: list[np.ndarray | None], dimension: int) -> np.ndarray:
    """Stack per-image vectors into an (n, dimension) float32 array, one row per input.

    ``None`` (a failed image) becomes an all-NaN row; every other row is scaled to unit
    length so inner product equals cosine similarity.
    """
    out = np.full((len(rows), dimension), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        if row is None:
            continue
        vec = np.asarray(row, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            out[i] = vec / norm
    return out


def embedded_mask(embeddings: np.ndarray) -> np.ndarray:
    """True for rows that hold a real embedding (not a failed image)."""
    if embeddings.size == 0:
        return np.zeros(embeddings.shape[0], dtype=bool)
    return ~np.isnan(embeddings).any(axis=1)
