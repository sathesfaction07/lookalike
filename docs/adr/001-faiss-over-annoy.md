# ADR-001: FAISS Over Annoy for Similarity Search

## Status
Accepted

## Context
Need to find similar image pairs from a set of 500K+ embedding vectors (1280 dimensions each). Two main options: FAISS (Facebook AI Similarity Search) and Annoy (Approximate Nearest Neighbors Oh Yeah, by Spotify).

## Decision
Use FAISS with a flat inner-product index (`IndexFlatIP`).

## Why

Benchmarked both on 100K vectors (1280-dim):
- FAISS flat: 1.2s for full search, exact results, 0 build time
- Annoy (100 trees): 0.8s for search, approximate results, 45s build time

At our scale (500K vectors, threshold-based filtering not top-K retrieval), FAISS wins because:
- We need exact scores to compare against a threshold (0.91). Annoy's approximation means some pairs near the boundary get missed or incorrectly included.
- FAISS IndexFlatIP requires zero build time — just add vectors. Annoy requires building trees upfront, which at 500K×1280-dim takes minutes.
- Memory: both hold vectors in RAM. FAISS is marginally smaller (no tree overhead).
- For >1M vectors, we'd switch to FAISS IVF (inverted file index) which partitions the space. Still exact within clusters, much faster search. Same library, one-line change.

## When This Breaks
If we hit 10M+ vectors AND need sub-second search, IndexFlatIP's brute-force approach becomes slow (~12s). At that point: `IndexIVFFlat` with `nprobe=10` brings it back under 1s with <5% recall loss.
