# Lookalike

Groups product images by visual similarity using deep learning embeddings and vector search. Designed for large catalogs (500K+ images) where you need to find duplicates, near-duplicates, or suspiciously similar listings. Supports three embedding backends — local EfficientNet, AWS Titan Multimodal, and Google Vertex — selectable at runtime.

---

## The Problem

A product catalog with half a million images accumulates duplicates — same product relisted under different IDs, sellers reusing competitor images, slight crops or color shifts that evade hash-based deduplication. Perceptual hashing catches exact duplicates but misses crops, overlays, or re-renders. You need something that understands image *content*, not just pixels.

---

## How It Works

1. **Load and download images** — reads CSV, downloads missing images in parallel (thread pool), caches locally
2. **Extract embeddings** — each image passes through the configured backend (EfficientNet locally, or Titan/Vertex via API), producing a normalized vector
3. **Build similarity index** — FAISS flat inner-product index. Each new batch is searched against existing vectors, pairs above threshold are collected
4. **Form clusters** — Union-Find merges similar pairs transitively. A→B + B→C means A,B,C are one cluster
5. **Output** — `clusters.json` + optional HTML visual report showing cluster members side-by-side

---

## Embedding Backends

| Backend | How | Dimension | Best for | Cost |
|:--|:--|:--|:--|:--|
| `local` (default) | EfficientNet-B0, runs on CPU/GPU | 1280 | Development, <1M images, offline | Free |
| `titan` | Amazon Titan Multimodal Embeddings via Bedrock | 1024 | Product images at scale (commerce-trained) | ~$0.0001/image |
| `vertex` | Google Vertex AI Multimodal Embeddings | 1408 | General multimodal, comparable to Titan | ~$0.0001/image |

Switch at runtime:
```bash
lookalike --input data.csv --output results/ --backend titan --aws-region us-east-1
lookalike --input data.csv --output results/ --backend vertex --gcp-project my-project
```

---

## Architecture

```
CSV Input
   │
   ▼ (parallel download, N threads)
Local Image Cache
   │
   ▼ (batched)
┌─────────────────────────────────────────────┐
│  Embedding Backend (configurable)            │
│  local (EfficientNet) │ titan │ vertex       │
└──────────────────────────┬──────────────────┘
                           │ normalized vectors
                           ▼
┌─────────────────────────────────────────────┐
│  FAISS Index (search existing → add new)     │
│  Pairs above threshold collected             │
└──────────────────────────┬──────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────┐
│  Union-Find → Cluster formation              │
└──────────────────────────┬──────────────────┘
                           │
                           ▼
           clusters.json + report.html (optional)

Checkpoint saves every 500 items.
On crash: reload → skip processed → continue.
```

---

## Running

Two modes: CLI (batch processing) and Web (interactive exploration).

**CLI mode** — for large batches, overnight runs, automation:
```bash
pip install -e ".[dev]"

lookalike --input data/sample_input.csv --output results/
lookalike --input data.csv --output results/ --backend titan --report
lookalike --input data.csv --output results/ --threshold 0.85
```

**Web mode** — for ad-hoc exploration, visual review, threshold tuning:
```bash
# Start the API
uvicorn image_cluster.api.app:app --port 8000

# Open frontend/index.html in your browser (or serve it)
python -m http.server 3000 --directory frontend
```

Then open `http://localhost:3000`. Upload images, pick a backend, adjust threshold, see clusters form in real-time. The threshold slider re-clusters *instantly* without re-embedding — it just re-runs Union-Find on existing similarity pairs.

---

## Project Structure

```
lookalike/
├── src/image_cluster/
│   ├── cli.py                  CLI entry point (batch mode)
│   ├── config.py               PipelineConfig with auto batch-size
│   ├── api/app.py              FastAPI (web mode — upload, cluster, recluster)
│   ├── backends/
│   │   ├── base.py            EmbeddingBackend interface
│   │   ├── local.py           EfficientNet-B0 (PyTorch)
│   │   ├── titan.py           Amazon Titan Multimodal (Bedrock)
│   │   └── vertex.py          Google Vertex AI Multimodal
│   └── core/
│       ├── pipeline.py         Orchestrates full CLI flow
│       ├── index.py            FAISS similarity search + persistence
│       ├── clusters.py         Union-Find and cluster output
│       ├── checkpoint.py       Save/resume across crashes
│       ├── downloader.py       Parallel image download with caching
│       └── report.py           HTML visual cluster report
├── frontend/
│   ├── index.html              Entry point (Tailwind dark theme)
│   └── src/App.jsx             React 18 — upload, progress, cluster viewer, threshold slider
├── tests/unit/                 23 tests
├── data/sample_input.csv
├── docs/adr/
└── pyproject.toml
```

---

## Testing

```bash
pytest tests/ -v              # 23 tests
pytest tests/ --cov=src       # coverage
```

Tests cover Union-Find correctness, FAISS index behavior (exact matches, orthogonal vectors, threshold filtering, save/load), checkpoint persistence, and config auto-tuning.

---

## Tradeoffs and Decisions I Made Along the Way

**Three backends instead of just using the best one**

Titan embeddings are better for product images (trained on commerce data). But they require AWS credentials, network access, and cost money per image. For development and testing against 10 images, spinning up a Bedrock call each time is overkill. EfficientNet runs locally in milliseconds. The backend swap is one CLI flag — the FAISS index, clustering, and output are identical regardless of which backend produced the vectors.

**FAISS stays local even with cloud embeddings**

I considered using OpenSearch Serverless (managed vector search) with the Titan backend. But FAISS flat index handles 500K vectors in <2 seconds for search, needs no infrastructure, and the index file is just a few hundred MB. OpenSearch makes sense at 10M+ vectors or when multiple services need to query the same index. For a single-pipeline CLI tool, local FAISS is simpler.

**HTML report limited to 50 clusters × 12 images each**

Rendering all clusters with all images would produce a multi-GB HTML file for large runs. 50 clusters × 12 thumbnails gives a reviewable snapshot. The full data is always in `clusters.json`. The report is for the human reviewer who wants to visually spot-check whether the clustering makes sense — not for exhaustive review of every cluster.

**Incremental by default (checkpoint + resume)**

The pipeline is designed to be stopped and restarted at any time. Checkpoint tracks which items are processed, FAISS index is saved to disk. On restart, already-processed items are skipped and the index is reloaded. This means you can add new images to the input CSV, re-run, and only the new ones get processed — they're searched against the full existing index.

**L2 normalization before indexing**

FAISS IndexFlatIP computes dot products. Dot product of two L2-normalized vectors equals cosine similarity. By normalizing once during extraction, every FAISS search is effectively cosine similarity without extra computation. The threshold (0.91) is directly interpretable as "91% cosine similar."

**CLI + Web instead of picking one**

Batch processing 500K images needs a CLI that runs unattended. But investigating 20 suspicious listings needs instant feedback and visual confirmation. Both are real workflows. The core library (backends, FAISS index, Union-Find) is shared — the CLI wraps it with checkpoint/resume and progress bars, the web app wraps it with upload/SSE/interactive threshold. The web app's killer feature: the threshold slider re-clusters *without re-embedding*. It re-runs Union-Find on the stored similarity pairs, which takes milliseconds even for thousands of pairs. This lets a reviewer explore "what if I make it stricter/looser" interactively.

**Auto batch-size based on RAM, reduces under pressure**

Starts by reading available memory and picking a conservative batch size. If memory usage crosses 85% mid-run (other processes started, OS cache pressure), the batch size halves. It never increases back — once in memory pressure territory, staying conservative is safer than oscillating.

---

## Stack

Python 3.11, PyTorch + torchvision (local backend), boto3/Bedrock (Titan), google-cloud-aiplatform (Vertex), FAISS, NumPy, Pillow, psutil, Rich, Click.
