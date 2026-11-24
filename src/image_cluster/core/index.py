import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger("image_cluster")


class SimilarityIndex:
    def __init__(self, dimension: int = 1280, threshold: float = 0.91):
        self.dimension = dimension
        self.threshold = threshold
        self.index = faiss.IndexFlatIP(dimension)
        self._ids: list[str] = []

    @property
    def size(self) -> int:
        return self.index.ntotal

    def add(self, ids: list[str], embeddings: np.ndarray) -> None:
        self.index.add(embeddings)
        self._ids.extend(ids)

    def find_similar_pairs(
        self, embeddings: np.ndarray, ids: list[str], k: int = 10
    ) -> list[tuple[str, str, float]]:
        if self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(embeddings, k)
        pairs = []

        for i, (score_row, idx_row) in enumerate(zip(scores, indices)):
            for score, j in zip(score_row, idx_row):
                if j == -1 or score < self.threshold:
                    continue
                target_id = self._ids[j]
                source_id = ids[i]
                if source_id != target_id:
                    pairs.append((source_id, target_id, float(score)))

        return pairs

    def find_pairs_within(
        self, embeddings: np.ndarray, ids: list[str]
    ) -> list[tuple[str, str, float]]:
        """Similar pairs inside one batch.

        find_similar_pairs only searches vectors that are already indexed, and callers add a
        batch after searching it, so without this two near-duplicates that arrive in the same
        batch are never compared. Each pair is reported once.
        """
        n = len(ids)
        if n < 2:
            return []
        scores = embeddings @ embeddings.T
        rows, cols = np.triu_indices(n, k=1)
        hits = scores[rows, cols] >= self.threshold
        return [
            (ids[i], ids[j], float(scores[i, j]))
            for i, j in zip(rows[hits], cols[hits], strict=True)
            if ids[i] != ids[j]
        ]

    def save(self, path: Path) -> None:
        faiss.write_index(self.index, str(path / "index.faiss"))
        np.save(path / "ids.npy", np.array(self._ids))
        logger.info(f"Index saved: {self.size} vectors")

    def load(self, path: Path) -> bool:
        index_path = path / "index.faiss"
        ids_path = path / "ids.npy"
        if not index_path.exists() or not ids_path.exists():
            return False
        self.index = faiss.read_index(str(index_path))
        self._ids = list(np.load(ids_path, allow_pickle=True))
        logger.info(f"Index loaded: {self.size} vectors")
        return True
