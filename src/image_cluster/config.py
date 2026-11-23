from dataclasses import dataclass
from pathlib import Path

import psutil


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
