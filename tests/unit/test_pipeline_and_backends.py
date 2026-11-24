"""The clustering pipeline end to end, the CLI, the HTML report, the downloader and the
cloud embedding backends -- with no model weights and no network.

Images are real files (written with Pillow) fetched through file:// URLs, so the real
downloader, cache, FAISS index, union-find clustering, checkpointing and report all run.
Only the embedding model is replaced: AverageColour embeds an image as its normalised mean
RGB, so near-identical images genuinely score near 1.0 and different colours do not.
"""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from image_cluster import cli
from image_cluster.backends import BACKENDS, get_embedding_backend
from image_cluster.backends.base import EmbeddingBackend
from image_cluster.backends.titan import TitanEmbeddingBackend
from image_cluster.config import PipelineConfig
from image_cluster.core import pipeline as pipeline_mod
from image_cluster.core.downloader import ImageDownloader
from image_cluster.core.pipeline import ClusterPipeline
from image_cluster.core.report import generate_report


class AverageColour(EmbeddingBackend):
    def __init__(self, **kwargs):
        self.calls = 0

    @property
    def dimension(self):
        return 3

    def embed_batch(self, image_paths):
        self.calls += 1
        rows = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
                for p in image_paths]
        if not rows:
            return np.zeros((0, 3), dtype=np.float32)
        arr = np.array(rows, dtype=np.float32) + 1e-3
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def image(path: Path, rgb) -> Path:
    Image.new("RGB", (16, 16), rgb).save(path, format="JPEG")
    return path


@pytest.fixture
def catalogue(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    rows = [("red-1", (250, 10, 10)), ("red-2", (245, 12, 8)), ("blue-1", (10, 10, 250)),
            ("green-1", (10, 240, 10))]
    lines = ["id,image_url"]
    for item_id, rgb in rows:
        lines.append(f"{item_id},{image(src / f'{item_id}.jpg', rgb).as_uri()}")
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


@pytest.fixture
def stand_ins(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "get_embedding_backend", lambda name, **kw: AverageColour())

    class Memory:
        percent = 10.0

    monkeypatch.setattr(pipeline_mod.psutil, "virtual_memory", lambda: Memory)
    return Memory


def config(tmp_path, csv_path, **overrides):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    values = dict(input_path=csv_path, output_path=out, batch_size=1,
                  checkpoint_dir=out / ".checkpoints", download_workers=2, report=True)
    values.update(overrides)
    return PipelineConfig(**values)


class TestPipeline:
    def test_near_duplicates_cluster_and_everything_is_written(self, tmp_path, catalogue,
                                                               stand_ins):
        cfg = config(tmp_path, catalogue)
        ClusterPipeline(cfg).run()

        result = json.loads((cfg.output_path / "clusters.json").read_text(encoding="utf-8"))
        assert result["total_clusters"] == 1
        assert sorted(result["clusters"][0]["members"]) == ["red-1", "red-2"]
        assert (cfg.output_path / ".cache" / "red-1.jpg").exists()  # downloaded and cached
        assert (cfg.checkpoint_dir / "index.faiss").exists()
        report = (cfg.output_path / "report.html").read_text(encoding="utf-8")
        assert "1 clusters, 2 items total" in report
        assert report.count("data:image/jpeg;base64,") == 2

    def test_a_resumed_run_skips_items_already_processed(self, tmp_path, catalogue,
                                                         stand_ins):
        ClusterPipeline(config(tmp_path, catalogue)).run()
        second = ClusterPipeline(config(tmp_path, catalogue))
        second.run()
        assert second.embedder.calls == 0
        # the saved index still carries the earlier vectors
        assert second.index.size == 4

    def test_input_columns_have_fallbacks_and_rows_without_an_id_are_dropped(self, tmp_path,
                                                                             stand_ins):
        csv_path = tmp_path / "in.csv"
        csv_path.write_text("sku,url,local_path\nA,http://x/a.jpg,\n,http://x/b.jpg,\nC,,/p.jpg\n",
                            encoding="utf-8")
        items = ClusterPipeline(config(tmp_path, csv_path))._load_input()
        assert items == [{"id": "A", "url": "http://x/a.jpg", "local_path": ""},
                         {"id": "C", "url": "", "local_path": "/p.jpg"}]

    def test_high_memory_halves_the_batch_size_but_not_below_32(self, tmp_path, catalogue,
                                                                stand_ins):
        stand_ins.percent = 90.0
        pipe = ClusterPipeline(config(tmp_path, catalogue, batch_size=256))
        pipe._check_memory()
        assert pipe.config.batch_size == 128
        pipe.config.batch_size = 40
        pipe._check_memory()
        assert pipe.config.batch_size == 32

    def test_near_duplicates_in_the_same_batch_are_found(self, tmp_path, catalogue,
                                                         stand_ins):
        cfg = config(tmp_path, catalogue, batch_size=64, report=False)
        ClusterPipeline(cfg).run()
        result = json.loads((cfg.output_path / "clusters.json").read_text(encoding="utf-8"))
        assert result["total_clusters"] == 1


def test_the_cli_builds_the_config_and_runs_the_pipeline(monkeypatch, tmp_path, catalogue):
    seen = {}

    class StandInPipeline:
        def __init__(self, cfg):
            seen["cfg"] = cfg

        def run(self):
            seen["ran"] = True

    monkeypatch.setattr(cli, "ClusterPipeline", StandInPipeline)
    out = tmp_path / "results"
    result = CliRunner().invoke(cli.main, [
        "--input", str(catalogue), "--output", str(out), "--threshold", "0.8",
        "--batch-size", "16", "--workers", "3", "--backend", "titan", "--report",
        "--no-resume", "--aws-region", "eu-west-1",
    ])
    assert result.exit_code == 0, result.output
    cfg = seen["cfg"]
    assert seen["ran"]
    assert (cfg.similarity_threshold, cfg.batch_size, cfg.download_workers) == (0.8, 16, 3)
    assert (cfg.backend, cfg.report, cfg.resume, cfg.aws_region) == ("titan", True, False,
                                                                     "eu-west-1")
    assert cfg.checkpoint_dir == out / ".checkpoints"


def test_the_report_caps_large_clusters_and_lists_members_without_images(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    image(cache / "a0.png", (200, 0, 0))
    members = [f"a{i}" for i in range(15)]
    generate_report({"a0": members, "b0": ["b0", "b1"]}, cache, tmp_path)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "2 clusters, 17 items total" in html
    assert "Cluster 1 (15 items)" in html  # largest first
    assert "+3 more" in html
    assert html.count("data:image/jpeg;base64,") == 1  # only a0 has an image


def test_the_downloader_prefers_local_files_then_the_cache(tmp_path):
    local = image(tmp_path / "local.jpg", (1, 2, 3))
    remote = image(tmp_path / "remote.jpg", (4, 5, 6))
    cache = tmp_path / "cache"
    downloader = ImageDownloader(cache, max_workers=2)
    paths = downloader.download_batch([
        {"id": "L", "local_path": str(local), "url": ""},
        {"id": "R", "url": remote.as_uri()},
        {"id": "BROKEN", "url": (tmp_path / "missing.jpg").as_uri()},
        {"id": "NOTHING"},
    ])
    assert paths == [local, cache / "R.jpg"]
    # second time round, R comes from the cache rather than the URL
    remote.unlink()
    assert downloader.download_batch([{"id": "R", "url": remote.as_uri()}]) == [cache / "R.jpg"]


class TestBackends:
    def test_the_registry_names_its_options(self):
        assert set(BACKENDS) == {"local", "titan", "vertex"}
        with pytest.raises(ValueError, match="Available: local, titan, vertex"):
            get_embedding_backend("nope")

    def test_titan_normalises_embeddings_and_skips_failures(self, monkeypatch, tmp_path):
        good = image(tmp_path / "good.jpg", (9, 9, 9))
        bad = image(tmp_path / "bad.jpg", (9, 9, 9))
        sent = []

        class Bedrock:
            def invoke_model(self, **kwargs):  # boto3's own keyword names (modelId, ...)
                payload = json.loads(kwargs["body"])
                sent.append((kwargs["modelId"], payload["embeddingConfig"]))
                if len(sent) == 2:
                    raise RuntimeError("throttled")
                return {"body": io.BytesIO(json.dumps({"embedding": [3.0, 4.0]}).encode())}

        monkeypatch.setattr("image_cluster.backends.titan.boto3.client",
                            lambda *a, **k: Bedrock())
        backend = TitanEmbeddingBackend(region="us-east-1")
        out = backend.embed_batch([good, bad])
        assert backend.dimension == 1024
        assert out.shape == (1, 2)
        assert np.allclose(out[0], [0.6, 0.8])  # unit length
        assert sent[0] == ("amazon.titan-embed-image-v1", {"outputEmbeddingLength": 1024})

    def test_titan_with_nothing_embedded_returns_an_empty_matrix(self, monkeypatch, tmp_path):
        class Down:
            def invoke_model(self, **kw):
                raise RuntimeError("down")

        monkeypatch.setattr("image_cluster.backends.titan.boto3.client", lambda *a, **k: Down())
        out = TitanEmbeddingBackend().embed_batch([image(tmp_path / "x.jpg", (1, 1, 1))])
        assert out.shape == (0, 1024)


def test_a_default_install_without_boto3_still_loads_and_explains_the_extra(monkeypatch):
    """boto3 is the optional [titan] extra. The registry used to import every backend
    eagerly, so a default install failed on the first import of the pipeline, CLI or API."""
    import sys

    monkeypatch.setitem(sys.modules, "boto3", None)  # behave as if boto3 is not installed
    monkeypatch.delitem(sys.modules, "image_cluster.backends.titan", raising=False)
    import importlib

    importlib.import_module("image_cluster.core.pipeline")  # must not need boto3
    with pytest.raises(ModuleNotFoundError, match=r"pip install 'lookalike\[titan\]'"):
        get_embedding_backend("titan")
