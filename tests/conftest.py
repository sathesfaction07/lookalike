import numpy as np
import pytest


@pytest.fixture
def tmp_checkpoint(tmp_path):
    return tmp_path / "checkpoints"


@pytest.fixture
def sample_embeddings():
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((10, 1280)).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


@pytest.fixture
def sample_ids():
    return [f"IMG-{i:04d}" for i in range(10)]


@pytest.fixture
def sample_csv(tmp_path):
    csv_file = tmp_path / "input.csv"
    csv_file.write_text(
        "id,image_url,local_path\n"
        "IMG-0001,https://example.com/1.jpg,\n"
        "IMG-0002,https://example.com/2.jpg,\n"
        "IMG-0003,,\n"
    )
    return csv_file
