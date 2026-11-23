from pathlib import Path

import click

from .config import PipelineConfig
from .core.pipeline import ClusterPipeline


@click.command()
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="CSV with image URLs or paths",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(),
    help="Output directory for cluster results",
)
@click.option("--threshold", default=0.91, type=float, help="Similarity threshold (0-1)")
@click.option("--batch-size", default=0, type=int, help="Batch size (0 = auto-tune based on RAM)")
@click.option("--checkpoint-dir", default=None, type=click.Path(), help="Checkpoint directory")
@click.option("--workers", default=8, type=int, help="Parallel download workers")
@click.option(
    "--backend",
    default="local",
    type=click.Choice(["local", "titan", "vertex"]),
    help="Embedding backend",
)
@click.option("--report/--no-report", default=False, help="Generate HTML visual report")
@click.option("--resume/--no-resume", default=True, help="Resume from checkpoint")
@click.option("--aws-region", default="us-east-1", help="AWS region (for titan backend)")
@click.option("--gcp-project", default="", help="GCP project ID (for vertex backend)")
def main(
    input_path,
    output_path,
    threshold,
    batch_size,
    checkpoint_dir,
    workers,
    backend,
    report,
    resume,
    aws_region,
    gcp_project,
):
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        input_path=Path(input_path),
        output_path=output,
        similarity_threshold=threshold,
        batch_size=batch_size,
        checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else output / ".checkpoints",
        download_workers=workers,
        resume=resume,
        backend=backend,
        report=report,
        aws_region=aws_region,
        gcp_project=gcp_project,
    )

    pipeline = ClusterPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
