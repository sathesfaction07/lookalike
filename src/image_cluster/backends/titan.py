import base64
import json
import logging
from pathlib import Path

import boto3
import numpy as np

from .base import EmbeddingBackend

logger = logging.getLogger("image_cluster")


class TitanEmbeddingBackend(EmbeddingBackend):
    def __init__(self, region: str = "us-east-1", **kwargs):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = "amazon.titan-embed-image-v1"
        self._dimension = 1024

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_batch(self, image_paths: list[Path]) -> np.ndarray:
        embeddings = []
        for path in image_paths:
            embedding = self._embed_one(path)
            if embedding is not None:
                embeddings.append(embedding)

        if not embeddings:
            return np.zeros((0, self._dimension), dtype=np.float32)

        result = np.array(embeddings, dtype=np.float32)
        result = result / np.linalg.norm(result, axis=1, keepdims=True)
        return result

    def _embed_one(self, path: Path) -> np.ndarray | None:
        try:
            image_bytes = path.read_bytes()
            body = json.dumps(
                {
                    "inputImage": base64.b64encode(image_bytes).decode("utf-8"),
                    "embeddingConfig": {"outputEmbeddingLength": self._dimension},
                }
            )

            response = self.client.invoke_model(
                modelId=self.model_id, body=body, contentType="application/json"
            )
            result = json.loads(response["body"].read())
            return np.array(result["embedding"], dtype=np.float32)
        except Exception as e:
            logger.warning(f"Titan embed failed for {path.name}: {e}")
            return None
