from unittest.mock import patch

from image_cluster.config import PipelineConfig


class TestPipelineConfig:
    def test_auto_batch_size_high_ram(self, tmp_path):
        with patch("psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.available = 20 * (1024**3)
            config = PipelineConfig(input_path=tmp_path / "x.csv", output_path=tmp_path / "out")
            assert config.batch_size == 512

    def test_auto_batch_size_low_ram(self, tmp_path):
        with patch("psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.available = 3 * (1024**3)
            config = PipelineConfig(input_path=tmp_path / "x.csv", output_path=tmp_path / "out")
            assert config.batch_size == 64

    def test_explicit_batch_size_overrides(self, tmp_path):
        config = PipelineConfig(
            input_path=tmp_path / "x.csv", output_path=tmp_path / "out", batch_size=200
        )
        assert config.batch_size == 200

    def test_checkpoint_dir_created(self, tmp_path):
        config = PipelineConfig(
            input_path=tmp_path / "x.csv",
            output_path=tmp_path / "out",
            checkpoint_dir=tmp_path / "ckpt",
        )
        assert config.checkpoint_dir.exists()
