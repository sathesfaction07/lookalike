import asyncio
import csv
import json
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..backends import get_embedding_backend
from ..core.clusters import build_clusters
from ..core.downloader import ImageDownloader
from ..core.index import SimilarityIndex

logging.basicConfig(
    level="INFO",
    format='{"time":"%(asctime)s","level":"%(levelname)s","module":"%(module)s","msg":"%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("image_cluster")

app = FastAPI(title="ImageCluster API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

jobs: dict[str, dict] = {}


class ClusterRequest(BaseModel):
    backend: str = "local"
    threshold: float = 0.91


class ReclusterRequest(BaseModel):
    job_id: str
    threshold: float


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "active_jobs": sum(1 for j in jobs.values() if j["status"] == "processing"),
    }


@app.post("/api/upload")
async def upload_images(files: list[UploadFile] = File(...)):
    job_id = str(uuid.uuid4())[:8]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True)

    saved = []
    for f in files:
        if f.filename and f.filename.lower().endswith((".jpg", ".jpeg", ".png", ".csv")):
            dest = job_dir / f.filename
            content = await f.read()
            dest.write_bytes(content)
            saved.append(f.filename)

    jobs[job_id] = {
        "status": "uploaded",
        "dir": str(job_dir),
        "files": saved,
        "results": None,
        "pairs": [],
    }
    return {"job_id": job_id, "files_saved": len(saved)}


@app.post("/api/cluster/{job_id}")
async def start_clustering(job_id: str, req: ClusterRequest):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    job["status"] = "processing"
    job["threshold"] = req.threshold
    job["backend"] = req.backend
    job["progress"] = {"done": 0, "total": 0}

    asyncio.create_task(_run_clustering(job_id, req))
    return {"status": "started"}


@app.get("/api/progress/{job_id}")
async def stream_progress(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    async def event_stream():
        while True:
            job = jobs[job_id]
            progress = job.get("progress", {})
            yield f"data: {json.dumps({'status': job['status'], **progress})}\n\n"
            if job["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/results/{job_id}")
def get_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "completed":
        return {"status": job["status"]}
    return {
        "status": "completed",
        "clusters": job["results"],
        "threshold": job.get("threshold", 0.91),
    }


@app.post("/api/recluster")
def recluster(req: ReclusterRequest):
    if req.job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[req.job_id]
    if not job.get("pairs"):
        raise HTTPException(400, "No embedding pairs available. Run clustering first.")

    filtered_pairs = [(s, t, score) for s, t, score in job["pairs"] if score >= req.threshold]
    clusters = build_clusters(filtered_pairs)

    result = _format_clusters(clusters, Path(job["dir"]))
    job["results"] = result
    job["threshold"] = req.threshold
    return {"clusters": result, "threshold": req.threshold}


@app.get("/api/images/{job_id}/{filename}")
def serve_image(job_id: str, filename: str):
    if job_id not in jobs:
        raise HTTPException(404)
    path = Path(jobs[job_id]["dir"]) / filename
    if not path.exists():
        raise HTTPException(404)
    media = "image/jpeg" if path.suffix in (".jpg", ".jpeg") else "image/png"
    return StreamingResponse(open(path, "rb"), media_type=media)


async def _run_clustering(job_id: str, req: ClusterRequest):
    job = jobs[job_id]
    job_dir = Path(job["dir"])

    try:
        image_files = [
            f for f in job_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]

        csv_files = [f for f in job_dir.iterdir() if f.suffix.lower() == ".csv"]
        if csv_files and not image_files:
            downloader = ImageDownloader(job_dir / ".cache", max_workers=8)
            items = _parse_csv(csv_files[0])
            image_files = downloader.download_batch(items)
            job["progress"]["total"] = len(items)

        if not image_files:
            job["status"] = "failed"
            job["error"] = "No images found"
            return

        embedder = get_embedding_backend(req.backend)
        index = SimilarityIndex(dimension=embedder.dimension, threshold=req.threshold)

        job["progress"]["total"] = len(image_files)
        all_pairs = []
        batch_size = 32
        ids = [f.stem for f in image_files]

        for i in range(0, len(image_files), batch_size):
            batch_paths = image_files[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            embeddings = embedder.embed_batch(batch_paths)
            if embeddings.shape[0] > 0:
                pairs = index.find_similar_pairs(embeddings, batch_ids)
                index.add(batch_ids, embeddings)
                all_pairs.extend(pairs)

            job["progress"]["done"] = min(i + batch_size, len(image_files))

        job["pairs"] = all_pairs
        clusters = build_clusters(all_pairs)
        job["results"] = _format_clusters(clusters, job_dir)
        job["status"] = "completed"
        logger.info(f"Job {job_id}: {len(clusters)} clusters from {len(image_files)} images")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job["status"] = "failed"
        job["error"] = str(e)


def _parse_csv(csv_path: Path) -> list[dict]:
    items = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row.get("id") or row.get("product_id") or ""
            url = row.get("image_url") or row.get("url") or ""
            if item_id and url:
                items.append({"id": item_id, "url": url})
    return items


def _format_clusters(clusters: dict[str, list[str]], job_dir: Path) -> list[dict]:
    result = []
    for i, (rep, members) in enumerate(sorted(clusters.items(), key=lambda x: -len(x[1]))):
        member_data = []
        for m in members:
            img_file = _find_image(m, job_dir)
            member_data.append({"id": m, "filename": img_file.name if img_file else None})
        result.append(
            {"cluster_id": i, "size": len(members), "representative": rep, "members": member_data}
        )
    return result


def _find_image(item_id: str, directory: Path) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        path = directory / f"{item_id}{ext}"
        if path.exists():
            return path
    for f in directory.iterdir():
        if f.stem == item_id:
            return f
    return None
