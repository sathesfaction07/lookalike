import json
import logging
from pathlib import Path

logger = logging.getLogger("image_cluster")


class CheckpointManager:
    def __init__(self, checkpoint_dir: Path, save_interval: int = 500):
        self.dir = checkpoint_dir
        self.save_interval = save_interval
        self.processed: set[str] = set()
        self.all_pairs: list[tuple[str, str, float]] = []
        self._load()

    def _state_file(self) -> Path:
        return self.dir / "state.json"

    def _load(self):
        state_file = self._state_file()
        if not state_file.exists():
            return
        with open(state_file) as f:
            state = json.load(f)
        self.processed = set(state.get("processed", []))
        self.all_pairs = [tuple(p) for p in state.get("pairs", [])]
        logger.info(
            f"Checkpoint loaded: {len(self.processed)} processed, {len(self.all_pairs)} pairs"
        )

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        state = {
            "processed": list(self.processed),
            "pairs": self.all_pairs,
        }
        with open(self._state_file(), "w") as f:
            json.dump(state, f)

    def mark_batch(self, ids: list[str], pairs: list[tuple[str, str, float]]) -> None:
        self.processed.update(ids)
        self.all_pairs.extend(pairs)
        if len(self.processed) % self.save_interval < len(ids):
            self.save()
            logger.info(f"Checkpoint saved at {len(self.processed)} items")

    def remaining(self, all_ids: list[str]) -> list[str]:
        return [i for i in all_ids if i not in self.processed]

    @property
    def progress(self) -> int:
        return len(self.processed)
