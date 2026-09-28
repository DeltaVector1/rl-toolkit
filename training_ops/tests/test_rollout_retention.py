import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training_ops.runtime import rollout_retention as retention


class RolloutRetentionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp.name) / "run_demo"
        (self.run_dir / "rollouts").mkdir(parents=True)
        for step in range(5):
            directory = self.run_dir / "rollouts" / f"step_{step}"
            directory.mkdir()
            payload = (f"batch-{step}-".encode() * 31)[:100 + step]
            digest = hashlib.sha256(payload).hexdigest()
            (directory / "train_rollouts.bin").write_bytes(payload)
            (directory / "train_rollouts.bin.delivered").write_text(digest + "\n")
        self.evidence = retention.build_node_evidence(
            self.run_dir, node="n76", run_name="demo", active_step=4, required_steps=[3, 4]
        )

    def tearDown(self):
        self.temp.cleanup()

    def _evidence(self, *, nodes=None):
        return {
            "version": 1,
            "run_name": "demo",
            "active_step": 4,
            "required_steps": [3, 4],
            "nodes": nodes or {"n76": self.evidence},
        }

    def test_dry_run_reports_exact_bytes_and_latest_protection(self):
        result = retention.cleanup(self.run_dir, self._evidence(), node="n76")
        expected = sum(
            (self.run_dir / "rollouts" / f"step_{step}" / name).stat().st_size
            for step in (0, 1, 2)
            for name in ("train_rollouts.bin", "train_rollouts.bin.delivered")
        )
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["selected_steps"], [0, 1, 2])
        self.assertEqual(result["selected_bytes"], expected)
        self.assertEqual(result["exact_bytes"], expected)
        self.assertTrue((self.run_dir / "rollouts" / "step_2").exists())

    def test_apply_quarantines_then_removes_and_reports_bytes(self):
        result = retention.cleanup(self.run_dir, self._evidence(), node="n76", dry_run=False)
        self.assertFalse(result["dry_run"])
        self.assertEqual(result["selected_steps"], [0, 1, 2])
        self.assertEqual(result["freed_bytes"], result["exact_bytes"])
        self.assertFalse((self.run_dir / "rollouts" / "step_0").exists())
        self.assertTrue((self.run_dir / "rollouts" / "step_3").exists())
        self.assertFalse(list((self.run_dir / "rollouts").glob(".retention-quarantine-*")))

    def test_stale_apply_fails_and_refreshed_no_candidates_is_verified_noop(self):
        retention.cleanup(self.run_dir, self._evidence(), node="n76", dry_run=False)
        with self.assertRaisesRegex(retention.RetentionError, "inventory disagrees|absent from authoritative"):
            retention.cleanup(self.run_dir, self._evidence(), node="n76", dry_run=False)
        refreshed = retention.build_node_evidence(
            self.run_dir, node="n76", run_name="demo", active_step=4, required_steps=[3, 4]
        )
        result = retention.cleanup(self.run_dir, self._evidence(nodes={"n76": refreshed}), node="n76", dry_run=False)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["freed_bytes"], 0)
        self.assertEqual(result["selected_steps"], [])

    def test_changed_identity_symlink_unknown_open_and_partial_fail_closed(self):
        batch = self.run_dir / "rollouts" / "step_0" / "train_rollouts.bin"
        batch.write_bytes(b"changed")
        (batch.parent / "train_rollouts.bin.delivered").write_text(
            hashlib.sha256(b"changed").hexdigest() + "\n"
        )
        with self.assertRaisesRegex(retention.RetentionError, "identity changed"):
            retention.inventory(self.run_dir, self._evidence(), node="n76")
        self.evidence = retention.build_node_evidence(
            self.run_dir, node="n76", run_name="demo", active_step=4, required_steps=[3, 4]
        )

        shutil_path = self.run_dir / "rollouts" / "step_1" / "unknown"
        shutil_path.write_bytes(b"x")
        with self.assertRaisesRegex(retention.RetentionError, "unknown"):
            retention.inventory(self.run_dir, self._evidence(), node="n76")
        shutil_path.unlink()
        self.evidence = retention.build_node_evidence(
            self.run_dir, node="n76", run_name="demo", active_step=4, required_steps=[3, 4]
        )

        symlink_batch = self.run_dir / "rollouts" / "step_2" / "train_rollouts.bin"
        symlink_target = self.run_dir / "rollouts" / "step_1" / "train_rollouts.bin"
        os.unlink(symlink_batch)
        os.symlink(symlink_target, symlink_batch)
        with self.assertRaisesRegex(retention.RetentionError, "regular file"):
            retention.inventory(self.run_dir, self._evidence(), node="n76")

        os.unlink(symlink_batch)
        symlink_batch.write_bytes(b"restored")
        marker = self.run_dir / "rollouts" / "step_0" / "train_rollouts.bin.delivered"
        marker.unlink()
        with self.assertRaisesRegex(retention.RetentionError, "partial or unknown"):
            retention.inventory(self.run_dir, self._evidence(), node="n76")

    def test_open_file_and_lock_refuse(self):
        batch = self.run_dir / "rollouts" / "step_0" / "train_rollouts.bin"
        with batch.open("rb"):
            with self.assertRaisesRegex(retention.RetentionError, "open .* descriptor"):
                retention.inventory(self.run_dir, self._evidence(), node="n76")
        lock = self.run_dir / ".batch-sync.lock"
        with lock.open("w") as stream:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            with self.assertRaisesRegex(retention.RetentionError, "writer or consumer lock"):
                retention.inventory(self.run_dir, self._evidence(), node="n76")
        lock.unlink()
        consumer_lock = self.run_dir / "rollouts" / "step_0" / ".microbatch-sync.lock"
        with consumer_lock.open("w") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            with self.assertRaisesRegex(retention.RetentionError, "writer or consumer lock"):
                retention.inventory(self.run_dir, self._evidence(), node="n76")

    def test_apply_race_after_stat_keeps_uncertain_quarantine(self):
        original_rename = retention.os.rename
        changed = {"done": False}

        def race(source, destination):
            if not changed["done"]:
                changed["done"] = True
                (Path(source) / "train_rollouts.bin").write_bytes(b"race")
            return original_rename(source, destination)

        with mock.patch.object(retention.os, "rename", side_effect=race):
            with self.assertRaisesRegex(retention.RetentionError, "delivery evidence|identity changed|content identity"):
                retention.cleanup(self.run_dir, self._evidence(), node="n76", dry_run=False)
        self.assertTrue(list((self.run_dir / "rollouts").glob(".retention-quarantine-*")))

    def test_apply_race_after_final_fd_check_is_detected(self):
        original_remove = retention._remove_tree_at

        def race(directory_fd, opened_files, expected):
            directory = Path(os.readlink(f"/proc/self/fd/{directory_fd}"))
            (directory / "train_rollouts.bin").write_bytes(b"late-race")
            return original_remove(directory_fd, opened_files, expected)

        with mock.patch.object(retention, "_remove_tree_at", side_effect=race):
            with self.assertRaisesRegex(retention.RetentionError, "content or delivery|identity changed"):
                retention.cleanup(self.run_dir, self._evidence(), node="n76", dry_run=False)
        self.assertTrue(list((self.run_dir / "rollouts").glob(".retention-quarantine-*")))

    def test_apply_race_after_content_check_before_first_unlink_fails(self):
        original_validate = retention._validate_held_contents

        def race(opened, expected):
            result = original_validate(opened, expected)
            directory = Path(os.readlink(f"/proc/self/fd/{opened[0]}"))
            (directory / "train_rollouts.bin").write_bytes(b"between-check-and-unlink")
            return result

        with mock.patch.object(retention, "_validate_held_contents", side_effect=race):
            with self.assertRaisesRegex(retention.RetentionError, "contents changed|identity changed"):
                retention.cleanup(self.run_dir, self._evidence(), node="n76", dry_run=False)
        self.assertTrue(list((self.run_dir / "rollouts").glob(".retention-quarantine-*")))

    def test_cross_node_disagreement_and_evidence_coverage_fail(self):
        other = json.loads(json.dumps(self.evidence))
        other["batches"][0]["sha256"] = "0" * 64
        other["batches"][0]["delivered_sha256"] = "0" * 64
        with self.assertRaisesRegex(retention.RetentionError, "disagree"):
            retention.validate_evidence(self._evidence(nodes={"n76": self.evidence, "n77": other}))
        other = json.loads(json.dumps(self.evidence))
        other["batches"] = other["batches"][1:]
        with self.assertRaisesRegex(retention.RetentionError, "complete step inventory|resume-required"):
            retention.validate_evidence(self._evidence(nodes={"n76": self.evidence, "n77": other}))
        with self.assertRaisesRegex(retention.RetentionError, "exactly the configured"):
            retention.validate_evidence(self._evidence(), nodes=["n76", "n77"])

    def test_step_directory_ownership_is_checked(self):
        evidence = json.loads(json.dumps(self.evidence))
        evidence["batches"][0]["dir_uid"] += 1
        with self.assertRaisesRegex(retention.RetentionError, "ownership disagrees"):
            retention.validate_evidence(self._evidence(nodes={"n76": evidence}))


if __name__ == "__main__":
    unittest.main()
