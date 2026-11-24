from importlib import import_module

from .base import EmbeddingBackend

# Backends are imported only when requested. Titan needs boto3 and Vertex needs the Google SDK,
# both optional extras (`pip install lookalike[titan]` / `[vertex]`); importing every backend
# here made boto3 mandatory in practice, so a default install failed on the first import.
BACKENDS = {
    "local": "image_cluster.backends.local:LocalEmbeddingBackend",
    "titan": "image_cluster.backends.titan:TitanEmbeddingBackend",
    "vertex": "image_cluster.backends.vertex:VertexEmbeddingBackend",
}

EXTRAS = {"titan": "titan", "vertex": "vertex"}


def get_embedding_backend(name: str, **kwargs) -> EmbeddingBackend:
    target = BACKENDS.get(name)
    if not target:
        available = ", ".join(BACKENDS.keys())
        raise ValueError(f"Unknown embedding backend: {name}. Available: {available}")
    module_name, class_name = target.split(":")
    try:
        module = import_module(module_name)
    except ImportError as e:
        extra = EXTRAS.get(name)
        hint = f" Install it with: pip install 'lookalike[{extra}]'" if extra else ""
        raise ModuleNotFoundError(f"The '{name}' backend needs {e.name}.{hint}") from e
    return getattr(module, class_name)(**kwargs)


__all__ = ["BACKENDS", "EmbeddingBackend", "get_embedding_backend"]
