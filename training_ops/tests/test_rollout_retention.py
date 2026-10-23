import hashlib
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
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

    def test_peer_without_publisher_lock_does_not_get_a_fabricated_lock(self):
        lock = self.run_dir / ".microbatch-sync.lock"
        with retention._writer_namespace_locks(self.run_dir) as held:
            self.assertNotIn(lock, held)
            self.assertFalse(lock.exists())

    def test_rank_zero_existing_publisher_lock_is_held_and_released(self):
        lock = self.run_dir / ".microbatch-sync.lock"
        lock.touch()
        with retention._writer_namespace_locks(self.run_dir) as held:
            self.assertIn(lock, held)
            descriptor = os.open(lock, os.O_RDWR)
            try:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(descriptor)
        descriptor = os.open(lock, os.O_RDWR)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def test_existing_writer_lock_owner_group_and_mode_must_match_run(self):
        lock = self.run_dir / ".microbatch-sync.lock"
        lock.touch()
        lock.chmod(0o400)
        with self.assertRaisesRegex(retention.RetentionError, "ownership or mode"):
            with retention._writer_namespace_locks(self.run_dir):
                self.fail("unsafe writer lock was accepted")

    def test_active_puller_blocks_nonmutating_preview(self):
        lock = self.run_dir / ".rollout-pull.lock"
        lock.touch()
        namespace = os.open(self.run_dir, os.O_RDONLY | os.O_DIRECTORY)
        fcntl.flock(namespace, fcntl.LOCK_SH)
        with lock.open("r+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            paths_before = sorted(path.name for path in self.run_dir.iterdir())
            steps_before = sorted(path.name for path in (self.run_dir / "rollouts").iterdir())
            with self.assertRaisesRegex(retention.RetentionError, "namespace is busy"):
                retention.cleanup(self.run_dir, self._evidence(), node="n76")
            self.assertEqual(
                sorted(path.name for path in (self.run_dir / "rollouts").iterdir()),
                steps_before,
            )
            self.assertEqual(sorted(path.name for path in self.run_dir.iterdir()), paths_before)
            descriptor = os.open(lock, os.O_RDWR)
            try:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(descriptor)
        os.close(namespace)

    def test_writer_alive_refuses_before_inventory(self):
        (self.run_dir / ".microbatch-sync.lock").touch()
        script = (
            "import fcntl, os, pathlib, time; "
            f"p=pathlib.Path({str(self.run_dir / '.microbatch-sync.lock')!r}); "
            "fd=os.open(p, os.O_RDWR); fcntl.flock(fd, fcntl.LOCK_EX); "
            f"pathlib.Path({str(self.run_dir / '.microbatch-sync.pid')!r}).write_text(str(os.getpid())+'\\n'); "
            "time.sleep(5)"
        )
        process = subprocess.Popen([sys.executable, "-c", script])
        try:
            time.sleep(0.15)
            with self.assertRaisesRegex(retention.RetentionError, "writer is alive|lock is busy|writer or consumer lock"):
                retention.cleanup(self.run_dir, self._evidence(), node="n76")
        finally:
            process.terminate()
            process.wait(timeout=3)

    def test_process_identity_refuses_unrelated_pid(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
        try:
            (self.run_dir / ".microbatch-sync.pid").write_text(f"{process.pid}\n")
            with self.assertRaisesRegex(retention.RetentionError, "does not identify the exact run"):
                retention.cleanup(self.run_dir, self._evidence(), node="n76")
        finally:
            process.terminate()
            process.wait(timeout=3)

    def test_stale_process_identity_refuses_restart_ambiguity(self):
        (self.run_dir / ".microbatch-sync.pid").write_text("99999999\n")
        with self.assertRaisesRegex(retention.RetentionError, "stale.*restart responsibility"):
            retention.cleanup(self.run_dir, self._evidence(), node="n76")

    def test_nonregular_rollout_file_refuses(self):
        path = self.run_dir / "rollouts" / "step_0" / "train_rollouts.bin"
        path.unlink()
        os.mkfifo(path)
        with self.assertRaisesRegex(retention.RetentionError, "regular file"):
            retention.cleanup(self.run_dir, self._evidence(), node="n76")

    def test_candidate_path_confinement_refuses_forged_path(self):
        preview = retention.inventory(self.run_dir, self._evidence(), node="n76")
        preview["selected"][0]["path"] = str(self.run_dir / "outside" / "step_0")
        with mock.patch.object(retention, "inventory", return_value=preview):
            with self.assertRaisesRegex(retention.RetentionError, "escaped the rollout directory"):
                retention.cleanup(self.run_dir, self._evidence(), node="n76", dry_run=False)

    def test_directory_mtime_recheck_refuses_change_before_quarantine(self):
        original_rename = retention.os.rename
        changed = {"done": False}

        def race(source, destination):
            if not changed["done"]:
                changed["done"] = True
                info = source.stat()
                os.utime(source, ns=(info.st_atime_ns, info.st_mtime_ns + 1))
            return original_rename(source, destination)

        with mock.patch.object(retention.os, "rename", side_effect=race):
            with self.assertRaisesRegex(retention.RetentionError, "identity changed"):
                retention.cleanup(self.run_dir, self._evidence(), node="n76", dry_run=False)
        self.assertTrue(list((self.run_dir / "rollouts").glob(".retention-quarantine-*")))

    def test_concurrent_rename_cannot_acquire_namespace_lock(self):
        (self.run_dir / ".microbatch-sync.lock").touch()
        result_path = self.run_dir / "rename-result"
        source = self.run_dir / "rollouts" / "step_0"
        replacement = self.run_dir / "rollouts" / "step_0-renamed"
        original_inventory = retention.inventory
        child = {}

        def start_child(*args, **kwargs):
            result = original_inventory(*args, **kwargs)
            if "process" not in child:
                script = (
                    "import fcntl, os, pathlib\n"
                    f"lock=os.open({str(self.run_dir / '.microbatch-sync.lock')!r}, os.O_RDWR)\n"
                    "result='blocked'\n"
                    "try:\n"
                    "    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
                    "except BlockingIOError:\n"
                    "    pass\n"
                    "else:\n"
                    f"    os.rename({str(source)!r}, {str(replacement)!r})\n"
                    "    result='renamed'\n"
                    f"pathlib.Path({str(result_path)!r}).write_text(result)\n"
                )
                child["process"] = subprocess.Popen([sys.executable, "-c", script])
                child["process"].wait(timeout=3)
            return result

        try:
            with mock.patch.object(retention, "inventory", side_effect=start_child):
                result = retention.cleanup(self.run_dir, self._evidence(), node="n76")
            self.assertTrue(result["dry_run"])
            self.assertEqual(result_path.read_text(), "blocked")
            self.assertTrue(source.exists())
            self.assertFalse(replacement.exists())
        finally:
            process = child.get("process")
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=3)

    def test_cleanup_failure_keeps_quarantine_for_recovery(self):
        with mock.patch.object(retention, "_delete_quarantine", side_effect=retention.RetentionError("cleanup failure")):
            with self.assertRaisesRegex(retention.RetentionError, "cleanup failure"):
                retention.cleanup(self.run_dir, self._evidence(), node="n76", dry_run=False)
        self.assertTrue(list((self.run_dir / "rollouts").glob(".retention-quarantine-*")))

    def test_apply_reports_restart_responsibility(self):
        result = retention.cleanup(self.run_dir, self._evidence(), node="n76", dry_run=False)
        self.assertTrue(result["restart_required"])
        self.assertIn("operator must restart", result["restart_responsibility"])

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

        def race(directory_fd, opened_files, expected, lease):
            directory = Path(os.readlink(f"/proc/self/fd/{directory_fd}"))
            (directory / "train_rollouts.bin").write_bytes(b"late-race")
            return original_remove(directory_fd, opened_files, expected, lease)

        with mock.patch.object(retention._WriteLease, "acquire"), mock.patch.object(retention, "_remove_tree_at", side_effect=race):
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

        with mock.patch.object(retention._WriteLease, "acquire"), mock.patch.object(retention, "_validate_held_contents", side_effect=race):
            with self.assertRaisesRegex(retention.RetentionError, "contents changed|identity changed"):
                retention.cleanup(self.run_dir, self._evidence(), node="n76", dry_run=False)
        self.assertTrue(list((self.run_dir / "rollouts").glob(".retention-quarantine-*")))

    def test_apply_race_after_per_entry_stat_before_unlink_fails(self):
        original_stat = retention.os.stat
        child = {}
        calls = {"batch": 0}

        def race(path, *args, **kwargs):
            result = original_stat(path, *args, **kwargs)
            if kwargs.get("dir_fd") is not None and path == "train_rollouts.bin":
                calls["batch"] += 1
                if calls["batch"] == 2:
                    directory = Path(os.readlink(f"/proc/self/fd/{kwargs['dir_fd']}"))
                    script = (
                        "from pathlib import Path; "
                        f"p=Path({str(directory / 'train_rollouts.bin')!r}); "
                        "stream=p.open('r+b'); stream.write(b'after-stat'); stream.flush()"
                    )
                    child["process"] = subprocess.Popen([sys.executable, "-c", script])
                    time.sleep(0.2)
            return result

        try:
            with mock.patch.object(retention.os, "stat", side_effect=race):
                with self.assertRaisesRegex(retention.RetentionError, "write lease"):
                    retention.cleanup(self.run_dir, self._evidence(), node="n76", dry_run=False)
        finally:
            process = child.get("process")
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=3)
        self.assertTrue(list((self.run_dir / "rollouts").glob(".retention-quarantine-*")))

    def test_competing_subprocess_writer_breaks_lease_and_preserves_quarantine(self):
        original_validate = retention._validate_held_contents
        child = {}

        def race(opened, expected):
            result = original_validate(opened, expected)
            directory = Path(os.readlink(f"/proc/self/fd/{opened[0]}"))
            script = (
                "from pathlib import Path; "
                f"p=Path({str(directory / 'train_rollouts.bin')!r}); "
                "stream=p.open('r+b'); stream.write(b'writer'); stream.flush()"
            )
            child["process"] = subprocess.Popen([sys.executable, "-c", script])
            time.sleep(0.2)
            return result

        with mock.patch.object(retention, "_validate_held_contents", side_effect=race):
            with self.assertRaisesRegex(retention.RetentionError, "write lease"):
                retention.cleanup(self.run_dir, self._evidence(), node="n76", dry_run=False)
        process = child["process"]
        try:
            process.terminate()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3)
        self.assertTrue(list((self.run_dir / "rollouts").glob(".retention-quarantine-*")))

    def test_unsupported_write_lease_refuses_without_deleting(self):
        with mock.patch.object(
            retention._WriteLease,
            "acquire",
            side_effect=retention.RetentionError("Linux write leases are unsupported"),
        ):
            with self.assertRaisesRegex(retention.RetentionError, "unsupported"):
                retention.cleanup(self.run_dir, self._evidence(), node="n76", dry_run=False)
        self.assertFalse((self.run_dir / "rollouts" / "step_0").exists())
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
