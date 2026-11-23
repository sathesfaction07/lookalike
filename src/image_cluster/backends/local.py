import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from .base import EmbeddingBackend

logger = logging.getLogger("image_cluster")

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class LocalEmbeddingBackend(EmbeddingBackend):
    def __init__(self, **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model()
        self._dimension = 1280

    @property
    def dimension(self) -> int:
        return self._dimension

    def _load_model(self) -> torch.nn.Module:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier = torch.nn.Identity()
        model = model.to(self.device)
        model.eval()
        logger.info(f"EfficientNet-B0 loaded on {self.device}")
        return model

    def embed_batch(self, image_paths: list[Path]) -> np.ndarray:
        tensors = []
        for path in image_paths:
            tensor = self._load_image(path)
            if tensor is not None:
                tensors.append(tensor)

        if not tensors:
            return np.zeros((0, self._dimension), dtype=np.float32)

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            embeddings = self.model(batch).cpu().numpy()

        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings.astype(np.float32)

    def _load_image(self, path: Path) -> torch.Tensor | None:
        try:
            img = Image.open(path).convert("RGB")
            return TRANSFORM(img)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return None
