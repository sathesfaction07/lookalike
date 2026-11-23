from .base import EmbeddingBackend
from .local import LocalEmbeddingBackend
from .titan import TitanEmbeddingBackend
from .vertex import VertexEmbeddingBackend

BACKENDS = {
    "local": LocalEmbeddingBackend,
    "titan": TitanEmbeddingBackend,
    "vertex": VertexEmbeddingBackend,
}


def get_embedding_backend(name: str, **kwargs) -> EmbeddingBackend:
    cls = BACKENDS.get(name)
    if not cls:
        available = ", ".join(BACKENDS.keys())
        raise ValueError(f"Unknown embedding backend: {name}. Available: {available}")
    return cls(**kwargs)
