from image_cluster.core.checkpoint import CheckpointManager


class TestCheckpointManager:
    def test_fresh_start(self, tmp_checkpoint):
        cm = CheckpointManager(tmp_checkpoint)
        assert cm.progress == 0
        assert cm.remaining(["a", "b", "c"]) == ["a", "b", "c"]

    def test_mark_batch_tracks_progress(self, tmp_checkpoint):
        cm = CheckpointManager(tmp_checkpoint)
        cm.mark_batch(["a", "b"], [("a", "b", 0.95)])
        assert cm.progress == 2
        assert cm.remaining(["a", "b", "c"]) == ["c"]

    def test_save_and_reload(self, tmp_checkpoint):
        cm = CheckpointManager(tmp_checkpoint, save_interval=1)
        cm.mark_batch(["a", "b"], [("a", "b", 0.95)])
        cm.save()

        cm2 = CheckpointManager(tmp_checkpoint)
        assert cm2.progress == 2
        assert len(cm2.all_pairs) == 1
        assert cm2.remaining(["a", "b", "c"]) == ["c"]

    def test_pairs_accumulate(self, tmp_checkpoint):
        cm = CheckpointManager(tmp_checkpoint)
        cm.mark_batch(["a", "b"], [("a", "b", 0.95)])
        cm.mark_batch(["c", "d"], [("c", "d", 0.92)])
        assert len(cm.all_pairs) == 2

    def test_idempotent_marking(self, tmp_checkpoint):
        cm = CheckpointManager(tmp_checkpoint)
        cm.mark_batch(["a"], [])
        cm.mark_batch(["a"], [])
        assert cm.progress == 1
