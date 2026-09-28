import hashlib
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from training_ops.runtime import broadcast_retention as retention


class BroadcastRetentionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "run_trinity"
        self.broadcasts = self.root / "broadcasts"
        self.broadcasts.mkdir(parents=True)
        self.run_name = "trinity"
        self.loaded = None
        self.acks = []

    def tearDown(self):
        self.temp.cleanup()

    def _step(self, step, *, ack=True, model=None, writer=False):
        path = self.broadcasts / f"step_{step}"
        path.mkdir()
        if model is None:
            model = f"model-{step}".encode()
        config = b'{"r":32}\n'
        model_sha = hashlib.sha256(model).hexdigest()
        config_sha = hashlib.sha256(config).hexdigest()
        (path / "adapter_model.safetensors").write_bytes(model)
        (path / "adapter_config.json").write_bytes(config)
        (path / "adapter_model.safetensors.sha256").write_text(model_sha + "\n")
        (path / "adapter_manifest.json").write_text(json.dumps({
            "version": 1,
            "kind": "lora-adapter",
            "mode": "full",
            "target_step": step,
            "target_sha256": model_sha,
            "target_size": len(model),
            "config_sha256": config_sha,
            "config_size": len(config),
            "files": [
                {"name": "adapter_config.json", "sha256": config_sha, "size": len(config)},
                {"name": "adapter_model.safetensors", "sha256": model_sha, "size": len(model)},
            ],
        }))
        if writer:
            (path / "WRITING").write_text("1")
        else:
            (path / "STABLE").write_bytes(b"")
        source = f"http://127.0.0.1/trinity/run_trinity/broadcasts/step_{step}"
        identity = {
            "step": step,
            "source_url": source,
            "slot": f"trinity-slot{step % 2}",
            "target_sha256": model_sha,
            "target_size": len(model),
            "config_sha256": config_sha,
            "config_size": len(config),
            "server_instance": "a" * 32,
        }
        if ack:
            (path / "receivers-ready.json").write_text(json.dumps({
                "version": 1,
                "run_name": self.run_name,
                **{key: identity[key] for key in (
                    "step", "source_url", "target_sha256", "target_size",
                    "config_sha256", "config_size", "slot",
                )},
                "receivers": [{
                    "endpoint": "http://h200",
                    "receipt": identity,
                }],
            }))
        return identity

    def _evidence(self, loaded, *acks):
        return {
            "version": 1,
            "run_name": self.run_name,
            "h200_loaded": loaded,
            "acknowledged": list(acks),
        }

    def _age(self, path, seconds=3600):
        old = time.time() - seconds
        for item in [path, *path.rglob("*")]:
            os.utime(item, (old, old), follow_symlinks=False)

    def test_inventory_selects_only_acknowledged_old_stable_and_old_partial(self):
        old = self._step(0)
        active = self._step(2)
        partial = self.broadcasts / "step_1"
        partial.mkdir()
        (partial / "partial.bin").write_bytes(b"partial")
        self._age(partial)
        report = retention.inventory(
            self.root,
            self._evidence(active, old, active),
            keep_latest=1,
            partial_age_seconds=60,
            max_entries=8,
            max_bytes=100000,
            now=time.time(),
        )
        selected = {item["path"] for item in report["selected"]}
        self.assertIn(str(self.broadcasts / "step_0"), selected)
        self.assertIn(str(partial), selected)
        self.assertNotIn(str(self.broadcasts / "step_2"), selected)
        self.assertEqual(report["loaded"]["step"], 2)

    def test_dry_run_preserves_everything_and_keeps_unacknowledged_stable(self):
        old = self._step(0, ack=False)
        active = self._step(2)
        result = retention.cleanup(
            self.root,
            self._evidence(active, active),
            keep_latest=1,
            partial_age_seconds=0,
            max_entries=8,
            max_bytes=100000,
            dry_run=True,
        )
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["removed"], [])
        self.assertTrue((old and self.broadcasts / "step_0").exists())
        self.assertTrue((self.broadcasts / "step_2").exists())

    def test_apply_removes_selected_atomically_and_preserves_active(self):
        old = self._step(0)
        active = self._step(2)
        partial = self.broadcasts / "step_1"
        partial.mkdir()
        (partial / "partial.bin").write_bytes(b"partial")
        self._age(partial)
        result = retention.cleanup(
            self.root,
            self._evidence(active, old, active),
            keep_latest=1,
            partial_age_seconds=60,
            max_entries=8,
            max_bytes=100000,
            dry_run=False,
        )
        self.assertFalse(result["dry_run"])
        self.assertEqual({Path(item["path"]).name for item in result["removed"]}, {"step_0", "step_1"})
        self.assertFalse((self.broadcasts / "step_0").exists())
        self.assertFalse((self.broadcasts / "step_1").exists())
        self.assertTrue((self.broadcasts / "step_2" / "STABLE").is_file())
        self.assertFalse(list(self.broadcasts.glob(".retention-quarantine-*")))

    def test_active_identity_is_required_and_future_ack_is_rejected(self):
        active = self._step(2)
        with self.assertRaisesRegex(retention.RetentionError, "no matching local stable"):
            retention.inventory(self.root, self._evidence(dict(active, target_sha256="0" * 64), active))
        with self.assertRaisesRegex(retention.RetentionError, "future step"):
            retention.inventory(self.root, self._evidence(active, dict(active, step=3)))

    def test_writer_partial_and_path_safety_are_fail_closed(self):
        active = self._step(2)
        self._step(1, ack=False, writer=True)
        self._age(self.broadcasts / "step_1", seconds=3600)
        report = retention.inventory(self.root, self._evidence(active, active), partial_age_seconds=0)
        self.assertEqual(report["selected"], [])
        with self.assertRaisesRegex(retention.RetentionError, "n79"):
            retention.inventory("/data/n79/run_trinity", self._evidence(active, active))


if __name__ == "__main__":
    unittest.main()
