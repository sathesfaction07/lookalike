# ImageCluster — Technical Architecture

## Overview

ImageCluster groups product images by visual similarity using deep learning embeddings (EfficientNet-B0, AWS Titan Multimodal, or Google Vertex AI) combined with FAISS vector similarity search and Union-Find transitive clustering. Designed for large catalogs (500K+ images), it detects duplicates, near-duplicates, and visually similar listings through a crash-tolerant, checkpoint-based pipeline with optional interactive web UI for threshold tuning.

**Design principle:** Incremental by default. All processing state checkpoints to disk every 500 items. Stop anytime, add new images to the CSV, re-run — only new items are processed and searched against the full existing index.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLIENT (Browser / Terminal)                       │
│                                                                         │
│  ┌──────────────────────┐         ┌───────────────────────────────────┐  │
│  │  CLI Mode             │         │  Web Mode                          │  │
│  │  - Batch CSV input    │         │  - Upload images / CSV             │  │
│  │  - Progress bar       │         │  - Backend selector                │  │
│  │  - Checkpoint resume  │         │  - Threshold slider (live reclust) │  │
│  │  - HTML report output │         │  - SSE progress stream             │  │
│  └──────────┬───────────┘         │  - Visual cluster viewer           │  │
│             │                      └────────────────┬──────────────────┘  │
│             │                                       │ POST /api/cluster   │
│             │                                       │ GET /api/progress   │
│             ▼                                       ▼                     │
└─────────────┼───────────────────────────────────────┼─────────────────────┘
              │                                       │
              ▼                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI SERVER (Port 8000)                        │
│                                                                         │
│  Endpoints:                                                              │
│    POST /api/upload           → Save uploaded files, return job_id       │
│    POST /api/cluster/<id>     → Start clustering with backend + threshold│
│    GET  /api/progress/<id>    → SSE stream (status, done/total)          │
│    GET  /api/results/<id>     → Return completed clusters                │
│    POST /api/recluster        → Re-run Union-Find with new threshold     │
│    GET  /api/images/<id>/<fn> → Serve cached images                      │
│    GET  /health               → Active jobs count                        │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │ shared core logic
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       CORE PIPELINE (src/image_cluster/core/)             │
│                                                                         │
│  ClusterPipeline orchestrates:                                           │
│                                                                         │
│  1. ImageDownloader (downloader.py)                                      │
│     - ThreadPoolExecutor (N=8) parallel download                         │
│     - Local cache by item ID (skips re-download)                         │
│     - Supports CSV URLs + local paths                                    │
│                                                                         │
│  2. EmbeddingBackend (backends/) — pluggable                             │
│     - embed_batch(paths) → L2-normalized np.ndarray                      │
│     - Returns fixed dimension per backend                                │
│                                                                         │
│  3. SimilarityIndex (index.py) — FAISS wrapper                           │
│     - IndexFlatIP (flat inner-product, exact search)                     │
│     - find_similar_pairs(new_embeddings, k=10) → pairs above threshold   │
│     - add(ids, embeddings) → append to index                             │
│     - save/load index.faiss + ids.npy                                    │
│                                                                         │
│  4. CheckpointManager (checkpoint.py)                                    │
│     - Tracks processed item IDs + all similarity pairs                   │
│     - Saves state.json every 500 items                                   │
│     - resume=True skips already-processed IDs                            │
│                                                                         │
│  5. Union-Find (clusters.py)                                             │
│     - Transitive merge: A↔B + B↔C → {A,B,C}                              │
│     - Output: clusters.json with representative + members                │
│                                                                         │
│  6. ReportGenerator (report.py) — optional HTML                          │
│     - Top 50 clusters × 12 thumbnails each                               │
│     - Base64-encoded inline images                                       │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   EMBEDDING BACKENDS (configurable at runtime)            │
│                                                                         │
│  ┌───────────────────┐   ┌────────────────────┐   ┌──────────────────┐  │
│  │ local              │   │ titan               │   │ vertex            │  │
│  │ EfficientNet-B0    │   │ Amazon Titan        │   │ Google Vertex AI  │  │
│  │ PyTorch + torchvision│   │ Bedrock Runtime  │   │ aiplatform SDK    │  │
│  │ Dimension: 1280    │   │ Dimension: 1024     │   │ Dimension: 1408   │  │
│  │ Runs: CPU/GPU      │   │ Runs: API call      │   │ Runs: API call    │  │
│  │ Cost: $0           │   │ Cost: ~$0.0001/img  │   │ Cost: ~$0.0001/img│  │
│  │ Offline: Yes       │   │ Offline: No         │   │ Offline: No       │  │
│  └───────────────────┘   └────────────────────┘   └──────────────────┘  │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL SERVICES (optional)                      │
│                                                                         │
│  ┌─────────────────────────┐        ┌──────────────────────────────────┐│
│  │ AWS Bedrock              │        │ Google Cloud Vertex AI            ││
│  │ Model: amazon.titan-     │        │ Model: multimodalembedding        ││
│  │   embed-image-v1         │        │ Region: us-central1               ││
│  │ Region: us-east-1        │        │ Requires: project_id              ││
│  │ Requires: boto3 + creds  │        │ Requires: google-cloud-aiplatform ││
│  └─────────────────────────┘        └──────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. CLI (`src/image_cluster/cli.py`)

Entry point: `image-cluster` (console script).

| Option | Type | Default | Purpose |
|:--|:--|:--|:--|
| `--input` | Path | — | CSV with image URLs or local paths (required) |
| `--output` | Path | — | Output directory for clusters.json + report.html (required) |
| `--threshold` | float | 0.91 | Cosine similarity threshold (0.70-0.99) |
| `--batch-size` | int | 0 | Batch size (0 = auto-tune from RAM) |
| `--checkpoint-dir` | Path | `<output>/.checkpoints` | Checkpoint state + FAISS index |
| `--workers` | int | 8 | Parallel download threads |
| `--backend` | str | `local` | Embedding backend: `local`, `titan`, `vertex` |
| `--report` / `--no-report` | bool | False | Generate HTML visual report |
| `--resume` / `--no-resume` | bool | True | Resume from checkpoint |
| `--aws-region` | str | `us-east-1` | AWS region (titan backend) |
| `--gcp-project` | str | `""` | GCP project ID (vertex backend) |

Orchestrates: `ClusterPipeline.run()` → loads CSV, downloads images in batches, embeds, searches, clusters, saves results. Progress displayed via Rich progress bar.

### 2. Core Pipeline (`src/image_cluster/core/`)

**ClusterPipeline (`pipeline.py`)** — Orchestration coordinator

- Initializes: `CheckpointManager`, `ImageDownloader`, `EmbeddingBackend`, `SimilarityIndex`
- Loads input CSV, filters remaining items via checkpoint
- Processes in batches: download → embed → find_similar_pairs → add to index → checkpoint
- Memory monitor: if RAM >85%, halves batch size (never increases back)
- Final: `build_clusters()` + `save_clusters()` + optional `generate_report()`

**SimilarityIndex (`index.py`)** — FAISS wrapper

- `__init__(dimension, threshold)` → creates `IndexFlatIP(dimension)`
- `add(ids, embeddings)` → appends vectors to index
- `find_similar_pairs(embeddings, ids, k=10)` → searches existing index, returns `(source_id, target_id, score)` tuples where `score >= threshold`
- `save(path)` / `load(path)` → persists `index.faiss` + `ids.npy`

**UnionFind + Clustering (`clusters.py`)**

- `UnionFind.union(x, y)` → merges two items' sets
- `UnionFind.find(x)` → returns set representative (with path compression)
- `build_clusters(pairs)` → applies union for all similarity pairs, returns `{representative: [members]}` dict
- `save_clusters(clusters, path)` → writes `clusters.json` with sorted clusters (largest first)

**CheckpointManager (`checkpoint.py`)**

- Tracks: `processed: set[str]`, `all_pairs: list[tuple]`
- Loads `state.json` on init
- `mark_batch(ids, pairs)` → updates sets, auto-saves every 500 items
- `remaining(all_ids)` → filters unprocessed IDs for resume

**ImageDownloader (`downloader.py`)**

- `download_batch(items)` → parallel ThreadPoolExecutor download
- Checks: local_path exists? → use directly. Cached? → skip download. Otherwise fetch from URL
- Returns list of `Path` for successfully retrieved images

**ReportGenerator (`report.py`)**

- `generate_report(clusters, cache_dir, output_path)` → HTML with inline base64 images
- Renders top 50 clusters × 12 thumbnails each (150px max dimension, JPEG quality 70)
- Dark theme, responsive grid layout

### 3. Embedding Backends (`src/image_cluster/backends/`)

Common interface (`base.py`):

```python
class EmbeddingBackend(ABC):
    @property
    def dimension(self) -> int: ...
    def embed_batch(self, image_paths: list[Path]) -> np.ndarray: ...
```

All backends return L2-normalized embeddings (cosine similarity = dot product).

| Backend | Dimension | Model | Cost | Offline | When to use |
|:--|:--|:--|:--|:--|:--|
| `local` | 1280 | EfficientNet-B0 (PyTorch) | Free | Yes | Dev, testing, <1M images, no cloud access |
| `titan` | 1024 | Amazon Titan Multimodal via Bedrock | ~$0.0001/img | No | Production, commerce-trained embeddings, AWS infra |
| `vertex` | 1408 | Google Vertex AI Multimodal | ~$0.0001/img | No | GCP infra, multimodal tasks |

**LocalEmbeddingBackend (`local.py`)**

- Loads EfficientNet-B0 with pretrained weights, replaces classifier with Identity
- Uses torchvision transforms (224×224 resize, ImageNet normalization)
- Runs on CUDA if available, falls back to CPU
- Batch inference via `torch.no_grad()`

**TitanEmbeddingBackend (`titan.py`)**

- boto3 bedrock-runtime client
- Model ID: `amazon.titan-embed-image-v1`
- Encodes image as base64, sends via `invoke_model()`, extracts `embedding` from JSON response
- Retries handled by boto3 defaults

**VertexEmbeddingBackend (`vertex.py`)**

- google-cloud-aiplatform SDK
- Model: `multimodalembedding` (pretrained)
- Requires GCP project ID + location (default `us-central1`)
- Loads image via `Image.load_from_file()`, calls `model.get_embeddings()`

### 4. API Layer (`src/image_cluster/api/app.py`)

FastAPI server with SSE-based progress streaming + recluster without re-embedding.

| Endpoint | Method | Purpose |
|:--|:--|:--|
| `/health` | GET | Returns active jobs count |
| `/api/upload` | POST | Saves uploaded files, returns `{job_id, files_saved}` |
| `/api/cluster/<job_id>` | POST | Starts async clustering task with backend + threshold |
| `/api/progress/<job_id>` | GET | SSE stream emitting `{status, done, total}` every 0.5s |
| `/api/results/<job_id>` | GET | Returns `{status, clusters, threshold}` for completed job |
| `/api/recluster` | POST | Re-runs Union-Find on cached pairs with new threshold (instant) |
| `/api/images/<job_id>/<filename>` | GET | Serves cached images as JPEG/PNG |

**Recluster Feature**: After embedding once, all similarity pairs are stored in memory (`job["pairs"]`). Changing threshold filters pairs and re-runs Union-Find in milliseconds — no re-embedding. This enables the frontend's live threshold slider.

### 5. Frontend (`frontend/src/App.jsx`)

React 18 SPA with Tailwind CSS dark theme.

**Workflow:**
1. Drag-drop images or CSV → POST `/api/upload`
2. Select backend (`local` / `titan` / `vertex`) + adjust threshold slider
3. Click "Start Clustering" → POST `/api/cluster/<job_id>`
4. SSE stream updates progress bar in real-time
5. Display clusters as grid of thumbnails (max 8 visible + overflow count)
6. **Threshold slider** → POST `/api/recluster` → instant re-clustering without re-embedding

---

## Data Flow

```
1. User provides CSV (columns: id/product_id, image_url/url, local_path)
2. CLI parses CSV, loads checkpoint state (processed IDs + existing pairs)
3. Filter: remaining_items = all_items - processed_items
4. Download batch (N=batch_size) via ThreadPoolExecutor (8 workers)
   - Check local_path → check cache → fetch URL → save to .cache/<id>.jpg
5. Pass paths to EmbeddingBackend.embed_batch()
   - local: Load images → EfficientNet forward pass → L2 normalize
   - titan: Base64 encode → invoke_model(bedrock) → extract embedding
   - vertex: Load → model.get_embeddings() → extract image_embedding
   - All: return np.ndarray shape (batch_size, dimension), dtype float32, L2-normed
6. SimilarityIndex.find_similar_pairs(embeddings, ids)
   - FAISS search existing index for k=10 nearest neighbors per embedding
   - Filter pairs where score >= threshold (default 0.91)
   - Return list[(source_id, target_id, score)]
7. SimilarityIndex.add(ids, embeddings) → append to FAISS index
8. CheckpointManager.mark_batch(ids, pairs) → update processed set + all_pairs list
   - Auto-save state.json every 500 items
9. Repeat steps 4-8 for all batches
10. Build clusters: UnionFind iterates all_pairs, merges connected components
11. Save clusters.json (sorted by cluster size descending)
12. Optional: generate_report() → HTML with top 50 clusters × 12 thumbnails
13. CLI displays: total time, item count, cluster count, items in clusters
```

---

## Clustering Design

### FAISS Over Annoy (ADR-001)

ADR-001 records a benchmark on 100K vectors (1280-dim): FAISS flat searched in 1.2s with zero build time; Annoy with 100 trees searched in 0.8s but needed 45s to build its trees first. Annoy is the faster searcher, and it still lost.

It lost on two things this workload actually cares about. First, exactness — the clustering step compares scores against a threshold (0.91), and Annoy's approximation means pairs sitting near that boundary get silently included or dropped. Top-K retrieval tolerates approximation; threshold filtering does not. Second, build time — the pipeline adds vectors incrementally batch by batch, so a structure that must be rebuilt to accept new vectors fights the design. `IndexFlatIP` just appends.

FAISS flat is brute-force: every query computes dot products against every stored vector, so search cost grows linearly with catalog size. ADR-001 notes this becomes slow (~12s) somewhere around 10M vectors. The escape hatch is `IndexIVFFlat` with `nprobe=10`, which partitions the space and searches only nearby partitions — back under 1s with under 5% recall loss, and it is a one-line change within the same library. That upgrade path is the reason the simple index was acceptable to start with.

### Similarity Threshold Selection

Default threshold: **0.91 cosine similarity**. This is a starting default chosen by inspection of clustering behaviour, not a value derived from a labelled evaluation set — this repo ships no such set. The bands below describe what the threshold does qualitatively, and are the reason 0.91 was picked as a reasonable middle:
- **≥0.95**: Only near-identical images (same product, different lighting)
- **0.91-0.94**: Same product family (color variants, angle variations)
- **0.85-0.90**: Similar products (same category, different brands)
- **<0.85**: Broad visual similarity (noise, overclustering)

Threshold is user-configurable (`--threshold`) and **adjustable post-embedding** in web mode. The web UI stores all similarity pairs in memory and re-runs Union-Find on threshold change — no re-embedding required. This enables instant exploration: "What if I make it stricter? Looser?"

### Transitive Clustering and Chain-Linking

Union-Find merges all pairs transitively. If A is 0.92 similar to B, and B is 0.92 similar to C, then {A, B, C} form one cluster — even if A and C are only 0.78 similar (below threshold). This is **intentional for product deduplication** (a blue shirt from 3 angles should cluster together even if front/back views don't match directly), but it creates **cluster drift risk** at scale.

**Failure mode**: A long chain A→B→C→D→...→Z can form where endpoints A and Z are visually unrelated. This occurs when:
- Threshold too loose (captures weak similarities that chain together)
- Dataset has gradual transitions (e.g., color spectrum: red → orange → yellow)

**Mitigation strategies:**
1. Use stricter threshold (0.93+) to reduce weak links
2. Add max-cluster-size cutoff in post-processing (split clusters >100 items)
3. Compute cluster cohesion (average pairwise similarity within cluster) and flag low-cohesion clusters for manual review
4. Use hierarchical clustering instead of Union-Find (not implemented — Union-Find chosen for O(N⍺(N)) speed at scale)

The current implementation prioritizes **recall over precision** — better to group a few false positives (manually reviewable via HTML report) than miss true duplicates.

### Embedding Caching Strategy

Embeddings are never recomputed for the same image. The flow:
1. **Download cache** (`.cache/<id>.jpg`) — images fetched once
2. **Checkpoint state** (`processed` set) — item IDs embedded once
3. **FAISS index persistence** (`index.faiss`) — vectors saved on checkpoint
4. **API job memory** (`job["pairs"]`) — pairs cached for recluster

On resume, `checkpoint.remaining()` filters out processed IDs. Only new items are embedded. The FAISS index is reloaded and searched against new embeddings. This enables **incremental updates**: add 1000 new images to a 500K catalog, re-run, only 1000 embeddings computed.

For the Titan/Vertex backends the saving is direct API spend, since embedding is billed per image. For the local backend it is CPU time. Either way the point is the same: embedding is the expensive step and it is perfectly cacheable, because the same bytes always produce the same vector.

### Near-Duplicate vs. Same-Product Semantics

The system does not distinguish between:
- **Near-duplicate** (same image, cropped/resized)
- **Same product** (different angles, lighting, backgrounds)
- **Similar product** (different brand, same category)

All are treated as "similar above threshold." The threshold controls granularity, but semantic distinction requires additional metadata (product title, brand, category) or a multi-stage classifier. This is a **clustering tool**, not a product-matching pipeline. Downstream systems must interpret clusters based on business rules (e.g., flag clusters with different merchant IDs as potential policy violations).

### Why L2 Normalization?

FAISS `IndexFlatIP` computes dot products. For unit vectors (L2-normalized), dot product equals **cosine similarity**: `cos(θ) = (A · B) / (||A|| ||B||) = A · B` when `||A|| = ||B|| = 1`. This means:
- Threshold of 0.91 = "91% cosine similar"
- FAISS scores are directly interpretable without post-processing
- Normalization happens once during embedding, not per search

All three backends normalize embeddings before returning. EfficientNet outputs are divided by their L2 norm. Titan/Vertex embeddings are normalized by the cloud provider but re-normalized client-side for consistency (defensive programming against API changes).

---

## Configuration

All configuration via `PipelineConfig` dataclass with CLI argument mapping:

```python
@dataclass
class PipelineConfig:
    input_path: Path
    output_path: Path
    similarity_threshold: float = 0.91
    batch_size: int = 0
    checkpoint_dir: Path = Path(".checkpoints")
    download_workers: int = 8
    resume: bool = True
    backend: str = "local"  # "local" | "titan" | "vertex"
    report: bool = False
    aws_region: str = "us-east-1"
    gcp_project: str = ""

    def __post_init__(self):
        if self.batch_size == 0:
            self.batch_size = self._auto_batch_size()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _auto_batch_size(self) -> int:
        available_gb = psutil.virtual_memory().available / (1024**3)
        if available_gb > 16:
            return 512
        elif available_gb > 8:
            return 256
        elif available_gb > 4:
            return 128
        return 64
```

Auto-tuning uses `psutil` to read available RAM at startup. During runtime, `ClusterPipeline._check_memory()` monitors RAM usage — if it exceeds 85%, batch size halves and never increases back (conservative strategy to prevent OOM crashes on long-running jobs).

---

## Running

```bash
# Install (development mode with optional backends)
pip install -e ".[dev,titan,vertex]"

# CLI batch processing (local backend, default threshold)
image-cluster --input data/catalog.csv --output results/

# CLI with AWS Titan backend + HTML report
image-cluster --input data.csv --output results/ --backend titan --aws-region us-east-1 --report

# CLI with Google Vertex backend
image-cluster --input data.csv --output results/ --backend vertex --gcp-project my-project-id

# CLI with custom threshold (stricter = fewer, tighter clusters)
image-cluster --input data.csv --output results/ --threshold 0.95

# Start FastAPI server (web mode)
uvicorn image_cluster.api.app:app --port 8000 --reload

# Serve frontend (in separate terminal)
python -m http.server 3000 --directory frontend
# Open http://localhost:3000 → upload images → select backend → adjust threshold slider → view clusters
```

CSV format (any of these column name combinations work):
- `id` or `product_id` or `sku` (required)
- `image_url` or `url` (optional if `local_path` provided)
- `local_path` (optional if `image_url` provided)

---

## Testing

```bash
pytest tests/ -v                # All 23 tests
pytest tests/unit/ -v           # Fast unit tests (no I/O)
pytest tests/ --cov=src         # With coverage report
```

Test coverage:
- `test_index.py`: FAISS behavior (exact matches = 1.0, orthogonal vectors = 0.0, threshold filtering, save/load persistence)
- `test_clusters.py`: Union-Find transitive merge (A↔B + B↔C → {A,B,C}) and cluster output format (JSON structure, sorting by size, member lists)
- `test_checkpoint.py`: State persistence (processed set, pairs list, resume skips existing IDs)
- `test_config.py`: Auto batch-size tuning (16GB → 512, 8GB → 256, 4GB → 128, <4GB → 64)

Not yet covered by tests: the downloader (network I/O), the embedding backends (each needs either torch weights or cloud credentials), the FastAPI layer, and report generation. Those are the honest gaps in this suite.
