from image_cluster.core.clusters import UnionFind, build_clusters


class TestUnionFind:
    def test_single_union(self):
        uf = UnionFind()
        uf.union("a", "b")
        assert uf.find("a") == uf.find("b")

    def test_transitive_union(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        assert uf.find("a") == uf.find("c")

    def test_disjoint_sets(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("c", "d")
        assert uf.find("a") != uf.find("c")

    def test_get_clusters_excludes_singletons(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.find("x")
        clusters = uf.get_clusters()
        assert "x" not in str(clusters)
        assert any("a" in members and "b" in members for members in clusters.values())

    def test_large_cluster(self):
        uf = UnionFind()
        for i in range(100):
            uf.union(f"item-{i}", f"item-{i + 1}")
        clusters = uf.get_clusters()
        assert len(clusters) == 1
        assert len(list(clusters.values())[0]) == 101


class TestBuildClusters:
    def test_from_pairs(self):
        pairs = [("a", "b", 0.95), ("b", "c", 0.92), ("d", "e", 0.93)]
        clusters = build_clusters(pairs)
        assert len(clusters) == 2

    def test_empty_pairs(self):
        assert build_clusters([]) == {}

    def test_duplicate_pairs_handled(self):
        pairs = [("a", "b", 0.95), ("a", "b", 0.94), ("b", "a", 0.95)]
        clusters = build_clusters(pairs)
        assert len(clusters) == 1
