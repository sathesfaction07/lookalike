import logging
from pathlib import Path

import numpy as np

from .base import EmbeddingBackend, aligned_unit_rows

logger = logging.getLogger("image_cluster")


class VertexEmbeddingBackend(EmbeddingBackend):
    def __init__(self, project_id: str = "", location: str = "us-central1", **kwargs):
        self.project_id = project_id
        self.location = location
        self._dimension = 1408

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_batch(self, image_paths: list[Path]) -> np.ndarray:
        from google.cloud import aiplatform
        from vertexai.vision_models import Image as VertexImage
        from vertexai.vision_models import MultiModalEmbeddingModel

        aiplatform.init(project=self.project_id, location=self.location)
        model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")

        rows: list[np.ndarray | None] = []
        for path in image_paths:
            try:
                image = VertexImage.load_from_file(str(path))
                result = model.get_embeddings(image=image)
                rows.append(np.array(result.image_embedding, dtype=np.float32))
            except Exception as e:
                logger.warning(f"Vertex embed failed for {path.name}: {e}")
                rows.append(None)
        return aligned_unit_rows(rows, self._dimension)
