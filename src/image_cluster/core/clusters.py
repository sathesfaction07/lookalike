import json
import logging
from pathlib import Path

logger = logging.getLogger("image_cluster")


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def get_clusters(self) -> dict[str, list[str]]:
        clusters: dict[str, list[str]] = {}
        for item in self.parent:
            root = self.find(item)
            clusters.setdefault(root, []).append(item)
        return {k: v for k, v in clusters.items() if len(v) > 1}


def build_clusters(pairs: list[tuple[str, str, float]]) -> dict[str, list[str]]:
    uf = UnionFind()
    for source, target, _ in pairs:
        uf.union(source, target)
    return uf.get_clusters()


def save_clusters(clusters: dict[str, list[str]], output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    result = {
        "total_clusters": len(clusters),
        "total_items_in_clusters": sum(len(v) for v in clusters.values()),
        "clusters": [
            {"id": i, "representative": root, "members": members, "size": len(members)}
            for i, (root, members) in enumerate(sorted(clusters.items(), key=lambda x: -len(x[1])))
        ],
    }
    out_file = output_path / "clusters.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Wrote {len(clusters)} clusters to {out_file}")
