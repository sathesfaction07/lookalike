from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class EmbeddingBackend(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_batch(self, image_paths: list[Path]) -> np.ndarray: ...
