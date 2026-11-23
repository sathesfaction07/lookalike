import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

logger = logging.getLogger("image_cluster")


class ImageDownloader:
    def __init__(self, cache_dir: Path, max_workers: int = 8):
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_batch(self, items: list[dict]) -> list[Path]:
        results: list[Path | None] = [None] * len(items)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}
            for i, item in enumerate(items):
                url = item.get("url", "")
                local_path = item.get("local_path", "")

                if local_path and Path(local_path).exists():
                    results[i] = Path(local_path)
                    continue

                if url:
                    cached = self.cache_dir / f"{item['id']}.jpg"
                    if cached.exists():
                        results[i] = cached
                        continue
                    futures[pool.submit(self._download_one, url, cached)] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(f"Download failed for item {items[idx]['id']}: {e}")

        return [p for p in results if p is not None]

    def _download_one(self, url: str, dest: Path) -> Path:
        urlretrieve(url, dest)
        return dest
