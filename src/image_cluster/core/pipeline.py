import csv
import logging
import time

import psutil
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ..backends import get_embedding_backend
from ..config import PipelineConfig
from .checkpoint import CheckpointManager
from .clusters import build_clusters, save_clusters
from .downloader import ImageDownloader
from .index import SimilarityIndex
from .report import generate_report

logger = logging.getLogger("image_cluster")


class ClusterPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.checkpoint = CheckpointManager(config.checkpoint_dir)
        self.downloader = ImageDownloader(config.output_path / ".cache", config.download_workers)
        self.embedder = get_embedding_backend(
            config.backend, region=config.aws_region, project_id=config.gcp_project
        )
        self.index = SimilarityIndex(
            dimension=self.embedder.dimension, threshold=config.similarity_threshold
        )

    def run(self) -> None:
        start = time.time()
        items = self._load_input()
        logger.info(
            f"Loaded {len(items)} items (backend={self.config.backend}, dim={self.embedder.dimension})"
        )

        if self.config.resume:
            self.index.load(self.config.checkpoint_dir)

        remaining = self.checkpoint.remaining([item["id"] for item in items])
        remaining_items = [item for item in items if item["id"] in set(remaining)]
        logger.info(
            f"Remaining: {len(remaining_items)} (skipped {len(items) - len(remaining_items)})"
        )

        batches = self._make_batches(remaining_items)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Processing", total=len(remaining_items))

            for batch in batches:
                self._process_batch(batch)
                progress.advance(task, len(batch))
                self._check_memory()

        self.checkpoint.save()
        self.index.save(self.config.checkpoint_dir)

        clusters = build_clusters(self.checkpoint.all_pairs)
        save_clusters(clusters, self.config.output_path)

        if self.config.report and clusters:
            generate_report(clusters, self.config.output_path / ".cache", self.config.output_path)

        elapsed = time.time() - start
        logger.info(
            f"Done in {elapsed:.1f}s — {len(items)} items, "
            f"{len(clusters)} clusters, "
            f"{sum(len(v) for v in clusters.values())} items in clusters"
        )

    def _load_input(self) -> list[dict]:
        items = []
        with open(self.config.input_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                item_id = row.get("id") or row.get("product_id") or row.get("sku")
                url = row.get("image_url") or row.get("url") or ""
                local_path = row.get("local_path") or ""
                if item_id:
                    items.append({"id": item_id, "url": url, "local_path": local_path})
        return items

    def _make_batches(self, items: list[dict]) -> list[list[dict]]:
        bs = self.config.batch_size
        return [items[i : i + bs] for i in range(0, len(items), bs)]

    def _process_batch(self, batch: list[dict]) -> None:
        paths = self.downloader.download_batch(batch)
        if not paths:
            return

        ids = [item["id"] for item in batch[: len(paths)]]
        embeddings = self.embedder.embed_batch(paths)

        if embeddings.shape[0] == 0:
            return

        pairs = self.index.find_similar_pairs(embeddings, ids)
        self.index.add(ids, embeddings)
        self.checkpoint.mark_batch(ids, pairs)

    def _check_memory(self) -> None:
        used_pct = psutil.virtual_memory().percent
        if used_pct > 85:
            old_bs = self.config.batch_size
            self.config.batch_size = max(32, self.config.batch_size // 2)
            logger.warning(
                f"Memory at {used_pct}% — reducing batch size from {old_bs} to {self.config.batch_size}"
            )
