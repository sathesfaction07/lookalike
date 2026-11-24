"""The HTTP API: upload, cluster (as a background task), progress stream, results,
re-clustering at a new threshold, and image serving. Only the embedding model is replaced.
"""

import importlib
import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from image_cluster.backends.base import EmbeddingBackend


class AverageColour(EmbeddingBackend):
    @property
    def dimension(self):
        return 3

    def embed_batch(self, image_paths):
        rows = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
                for p in image_paths]
        arr = np.array(rows, dtype=np.float32) + 1e-3
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def jpeg(rgb) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), rgb).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def api(tmp_path, monkeypatch):
    # Importing the module creates ./uploads, so import it from a scratch directory.
    monkeypatch.chdir(tmp_path)
    mod = importlib.import_module("image_cluster.api.app")
    monkeypatch.setattr(mod, "UPLOAD_DIR", tmp_path / "uploads")
    (tmp_path / "uploads").mkdir(exist_ok=True)
    monkeypatch.setattr(mod, "get_embedding_backend", lambda name, **kw: AverageColour())
    mod.jobs.clear()
    with TestClient(mod.app) as client:
        client.mod = mod
        yield client


def upload(client, files):
    resp = client.post("/api/upload", files=[("files", f) for f in files])
    assert resp.status_code == 200
    return resp.json()


def wait_for(client, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/results/{job_id}").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("clustering did not finish")


def test_health_counts_jobs_in_progress(api):
    assert api.get("/health").json() == {"status": "healthy", "active_jobs": 0}


def test_upload_keeps_images_and_csvs_only(api):
    body = upload(api, [("a.jpg", jpeg((200, 0, 0)), "image/jpeg"),
                        ("notes.txt", b"x", "text/plain"),
                        ("list.csv", b"id,url\n", "text/csv")])
    assert body["files_saved"] == 2
    assert api.mod.jobs[body["job_id"]]["files"] == ["a.jpg", "list.csv"]


def test_unknown_jobs_are_404_everywhere(api):
    assert api.post("/api/cluster/nope", json={}).status_code == 404
    assert api.get("/api/progress/nope").status_code == 404
    assert api.get("/api/results/nope").status_code == 404
    assert api.post("/api/recluster", json={"job_id": "nope", "threshold": 0.5}).status_code == 404
    assert api.get("/api/images/nope/a.jpg").status_code == 404


def test_a_job_runs_to_completion_and_reports_progress(api):
    job_id = upload(api, [("red1.jpg", jpeg((250, 10, 10)), "image/jpeg"),
                          ("blue1.jpg", jpeg((10, 10, 250)), "image/jpeg")])["job_id"]
    assert api.get(f"/api/results/{job_id}").json() == {"status": "uploaded"}
    assert api.post(f"/api/cluster/{job_id}", json={"threshold": 0.95}).json() == {
        "status": "started"}
    body = wait_for(api, job_id)
    assert body["status"] == "completed"
    assert body["threshold"] == 0.95
    progress = api.get(f"/api/progress/{job_id}").text
    assert '"status": "completed"' in progress and '"done": 2' in progress


def test_a_job_with_no_images_fails_cleanly(api):
    job_id = upload(api, [("empty.csv", b"id,image_url\n", "text/csv")])["job_id"]
    api.post(f"/api/cluster/{job_id}", json={})
    body = wait_for(api, job_id)
    assert body["status"] == "failed"
    assert api.mod.jobs[job_id]["error"] == "No images found"


def test_recluster_needs_pairs_and_then_applies_the_new_threshold(api):
    job_id = upload(api, [("a.jpg", jpeg((250, 10, 10)), "image/jpeg"),
                          ("b.jpg", jpeg((245, 12, 8)), "image/jpeg"),
                          ("c.jpg", jpeg((240, 20, 20)), "image/jpeg")])["job_id"]
    resp = api.post("/api/recluster", json={"job_id": job_id, "threshold": 0.5})
    assert resp.status_code == 400

    api.mod.jobs[job_id]["pairs"] = [("a", "b", 0.99), ("b", "c", 0.80)]
    loose = api.post("/api/recluster", json={"job_id": job_id, "threshold": 0.75}).json()
    assert [c["size"] for c in loose["clusters"]] == [3]
    strict = api.post("/api/recluster", json={"job_id": job_id, "threshold": 0.95}).json()
    [cluster] = strict["clusters"]
    assert sorted(m["id"] for m in cluster["members"]) == ["a", "b"]
    assert all(m["filename"].endswith(".jpg") for m in cluster["members"])


def test_images_are_served_with_their_media_type(api):
    job_id = upload(api, [("a.jpg", jpeg((1, 2, 3)), "image/jpeg"),
                          ("b.png", b"\x89PNG fake", "image/png")])["job_id"]
    jpg = api.get(f"/api/images/{job_id}/a.jpg")
    assert jpg.status_code == 200 and jpg.headers["content-type"] == "image/jpeg"
    assert api.get(f"/api/images/{job_id}/b.png").headers["content-type"] == "image/png"
    assert api.get(f"/api/images/{job_id}/missing.jpg").status_code == 404


def test_csv_rows_need_both_an_id_and_a_url(api, tmp_path):
    path = tmp_path / "items.csv"
    path.write_text("product_id,url\nA,http://x/a.jpg\nB,\n,http://x/c.jpg\n", encoding="utf-8")
    assert api.mod._parse_csv(path) == [{"id": "A", "url": "http://x/a.jpg"}]


@pytest.mark.xfail(strict=True, reason=(
    "Known gap shared with the CLI pipeline: images are searched against the index before "
    "their batch (32) is added, so near-duplicates uploaded together are never compared."
))
def test_uploaded_near_duplicates_are_clustered(api):
    job_id = upload(api, [("red1.jpg", jpeg((250, 10, 10)), "image/jpeg"),
                          ("red2.jpg", jpeg((245, 12, 8)), "image/jpeg")])["job_id"]
    api.post(f"/api/cluster/{job_id}", json={"threshold": 0.95})
    assert len(wait_for(api, job_id)["clusters"]) == 1
