import base64
import logging
from pathlib import Path

logger = logging.getLogger("image_cluster")

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Image Cluster Report</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 24px; }}
h1 {{ font-size: 20px; margin-bottom: 8px; }}
.summary {{ font-size: 13px; color: #888; margin-bottom: 24px; }}
.cluster {{ background: #16213e; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
.cluster-header {{ font-size: 14px; font-weight: bold; margin-bottom: 10px; }}
.cluster-images {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.cluster-images img {{ width: 120px; height: 120px; object-fit: cover; border-radius: 4px; border: 1px solid #333; }}
.cluster-id {{ font-size: 11px; color: #666; margin-top: 4px; text-align: center; }}
</style>
</head>
<body>
<h1>Image Cluster Report</h1>
<p class="summary">{summary}</p>
{clusters_html}
</body>
</html>"""


def generate_report(clusters: dict[str, list[str]], cache_dir: Path, output_path: Path) -> None:
    sorted_clusters = sorted(clusters.items(), key=lambda x: -len(x[1]))

    summary = f"{len(clusters)} clusters, {sum(len(v) for v in clusters.values())} items total"
    clusters_html_parts = []

    for i, (representative, members) in enumerate(sorted_clusters[:50]):
        images_html = []
        for member_id in members[:12]:
            img_path = _find_image(member_id, cache_dir)
            if img_path:
                img_data = _encode_thumbnail(img_path)
                images_html.append(
                    f'<div><img src="data:image/jpeg;base64,{img_data}"><div class="cluster-id">{member_id}</div></div>'
                )
            else:
                images_html.append(f'<div class="cluster-id">{member_id}</div>')

        overflow = ""
        if len(members) > 12:
            overflow = f'<div class="cluster-id">+{len(members) - 12} more</div>'

        clusters_html_parts.append(
            f'<div class="cluster">'
            f'<div class="cluster-header">Cluster {i + 1} ({len(members)} items)</div>'
            f'<div class="cluster-images">{"".join(images_html)}{overflow}</div>'
            f"</div>"
        )

    html = HTML_TEMPLATE.format(summary=summary, clusters_html="\n".join(clusters_html_parts))
    report_path = output_path / "report.html"
    report_path.write_text(html, encoding="utf-8")
    logger.info(f"Report written to {report_path}")


def _find_image(item_id: str, cache_dir: Path) -> Path | None:
    for ext in (".jpg", ".png", ".jpeg"):
        path = cache_dir / f"{item_id}{ext}"
        if path.exists():
            return path
    return None


def _encode_thumbnail(path: Path, max_size: int = 150) -> str:
    import io

    from PIL import Image

    img = Image.open(path)
    img.thumbnail((max_size, max_size))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=70)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
