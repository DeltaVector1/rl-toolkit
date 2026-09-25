import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import types
import tomllib
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import training_ops.tests

spec = importlib.util.spec_from_file_location("resume_state", Path(__file__).parents[1] / "runtime/china/resume_state.py")
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


class ResumeStateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.active = Path(self.temporary.name) / "alpha/run_alpha"
        self.target = self.active / "broadcasts/step_3"
        self.target.mkdir(parents=True)
        self.weight = self.target / "adapter_model.safetensors"
        self.weight.write_bytes(b"adapter-three")
        (self.target / "adapter_config.json").write_bytes(b"{}")
        (self.target / "STABLE").touch()
        self.sha = hashlib.sha256(self.weight.read_bytes()).hexdigest()
        self.manifest = {"target_step": 3, "target_sha256": self.sha, "target_size": self.weight.stat().st_size, "config_sha256": hashlib.sha256(b"{}").hexdigest(), "config_size": 2}
        (self.target / "adapter_manifest.json").write_text(json.dumps(self.manifest))

    def check(self, role="origin", step=3, digest=""):
        return resume.check_state(str(self.active), step, role, str(self.weight), digest)

    def completed(self):
        root = self.active / "rollouts/step_3"
        trace = root / "train/effective/traces.jsonl"
        trace.parent.mkdir(parents=True)
        trace.write_text('{"reward": 1}\n')
        batch = root / "train_rollouts.bin"
        batch.write_bytes(b"batch-three")
        sha = hashlib.sha256(batch.read_bytes()).hexdigest()
        (root / "train_rollouts.bin.delivered").write_text(sha + "\n")
        return root, sha

    def partial(self, lock=True):
        trace = self.active / "rollouts/step_4/train/all/traces.jsonl"
        trace.parent.mkdir(parents=True)
        trace.write_text("partial trace")
        if lock:
            (self.active / ".batch-sync.lock").touch()
        return trace

    def test_latest_policy_requires_complete_matching_delivered_batch(self):
        root, sha = self.completed()
        self.assertEqual(resume.latest_policy(str(self.active), 1), {"step": 3, "batch_sha256": sha})
        self.assertEqual(resume.completed_batch(str(self.active), 3, sha)["records"], 1)
        for name in ("train/effective/traces.jsonl", "train_rollouts.bin", "train_rollouts.bin.delivered"):
            path = root / name
            saved = path.read_bytes()
            path.unlink()
            with self.subTest(name=name), self.assertRaises((OSError, ValueError)):
                resume.completed_batch(str(self.active), 3, sha)
            path.write_bytes(saved)
        with self.assertRaises(ValueError):
            resume.completed_batch(str(self.active), 3, "f" * 64)

    def test_latest_policy_keeps_initial_resume_without_batch(self):
        self.assertEqual(resume.latest_policy(str(self.active), 3), {"step": 3})
        with self.assertRaises(ValueError):
            resume.latest_policy(str(self.active), 4)
        with self.assertRaises(OSError):
            resume.latest_policy(str(self.active), 1)

    def test_latest_policy_allows_empty_stable_set_only_for_cold_launch(self):
        (self.target / "STABLE").unlink()
        self.assertEqual(resume.latest_policy(str(self.active), 0), {"step": None})
        with self.assertRaisesRegex(ValueError, "stable policy"):
            resume.latest_policy(str(self.active), 1)

    def test_latest_policy_rejects_malformed_policy_entry_without_stable(self):
        malformed = self.active / "broadcasts/step_0"
        malformed.unlink(missing_ok=True)
        malformed.write_text("not a directory")
        with self.assertRaisesRegex(ValueError, "not a directory"):
            resume.latest_policy(str(self.active), 0)

    def test_partial_quarantine_preserves_trace_without_reuse(self):
        trace = self.partial()
        original = Path.iterdir
        def entries(path):
            return iter([]) if path == Path("/proc") else original(path)
        with mock.patch.object(Path, "iterdir", entries):
            self.assertEqual(resume.partial_rollout(str(self.active), 3), {"partial": True})
            moved = resume.partial_rollout(str(self.active), 3, quarantine=True)
        self.assertFalse(trace.exists())
        self.assertEqual((Path(moved["quarantined"]) / "train/all/traces.jsonl").read_text(), "partial trace")
        self.assertEqual(resume.partial_rollout(str(self.active), 3), {"partial": False})

    def test_partial_creates_and_preserves_missing_publisher_lock(self):
        trace = self.partial(lock=False)
        original = Path.iterdir
        def entries(path):
            return iter([]) if path == Path("/proc") else original(path)
        with mock.patch.object(Path, "iterdir", entries):
            result = resume.partial_rollout(str(self.active), 3, quarantine=True)
        lock = self.active / ".batch-sync.lock"
        self.assertEqual(result["partial"], True)
        self.assertFalse(trace.exists())
        self.assertTrue(lock.is_file())
        self.assertFalse(lock.is_symlink())
        directory = self.active.stat()
        self.assertEqual((lock.stat().st_uid, lock.stat().st_gid), (directory.st_uid, directory.st_gid))
        self.assertEqual(stat.S_IMODE(lock.stat().st_mode), 0o600)
        with lock.open("wb"):
            pass
        with lock.open() as stream:
            resume.fcntl.flock(stream, resume.fcntl.LOCK_EX | resume.fcntl.LOCK_NB)

    def test_partial_rejects_existing_lock_owned_by_another_uid(self):
        self.partial()
        actual_fstat = resume.os.fstat
        calls = 0
        directory = self.active.stat()
        other_uid = 0 if directory.st_uid else 1
        def fstat(fd):
            nonlocal calls
            calls += 1
            value = actual_fstat(fd)
            if calls == 1:
                return types.SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_uid=directory.st_uid, st_gid=directory.st_gid)
            if calls == 2:
                return types.SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_uid=other_uid, st_gid=directory.st_gid)
            return value
        with mock.patch.object(resume.os, "fstat", side_effect=fstat):
            with self.assertRaisesRegex(ValueError, "existing publisher lock ownership"):
                resume.partial_rollout(str(self.active), 3)

    def test_partial_rejects_fifo_lock_without_blocking(self):
        self.partial(lock=False)
        os.mkfifo(self.active / ".batch-sync.lock")
        with self.assertRaisesRegex(ValueError, "publisher lock is not regular"):
            resume.partial_rollout(str(self.active), 3)

    def test_partial_rejects_missing_lock_symlink(self):
        self.partial(lock=False)
        lock = self.active / ".batch-sync.lock"
        outside = Path(self.temporary.name) / "outside-lock"
        outside.touch()
        lock.symlink_to(outside)
        with self.assertRaises(OSError):
            resume.partial_rollout(str(self.active), 3)
        self.assertTrue(lock.is_symlink())

    def test_partial_rejects_lock_creation_race(self):
        self.partial(lock=False)
        lock = self.active / ".batch-sync.lock"
        original = resume.os.open
        def raced(path, flags, *args, **kwargs):
            if path == ".batch-sync.lock" and flags & resume.os.O_CREAT:
                lock.touch()
            return original(path, flags, *args, **kwargs)
        with mock.patch.object(resume.os, "open", side_effect=raced):
            with self.assertRaisesRegex(ValueError, "changed during creation"):
                resume.partial_rollout(str(self.active), 3)
        self.assertTrue(lock.is_file())

    def test_partial_rejects_active_publisher_without_existing_lock(self):
        self.partial(lock=False)
        original = Path.read_bytes
        def read(path):
            if path == Path("/proc/123456/environ"):
                return f"SOURCE_RUN_DIR={self.active}".encode()
            return original(path)
        original_iter = Path.iterdir
        def entries(path):
            return iter([Path("/proc/123456")]) if path == Path("/proc") else original_iter(path)
        with mock.patch.object(Path, "read_bytes", read), mock.patch.object(Path, "iterdir", entries):
            with self.assertRaisesRegex(ValueError, "publisher is still present"):
                resume.partial_rollout(str(self.active), 3)
        self.assertTrue((self.active / ".batch-sync.lock").is_file())

    def test_partial_refuses_complete_unknown_and_symlink_entries(self):
        self.partial()
        target = self.active / "rollouts/step_4"
        for name in ("train_rollouts.bin", "train_rollouts.bin.delivered", "train/effective", "unknown"):
            path = target / name
            path.touch()
            with self.subTest(name=name), self.assertRaises(ValueError):
                resume.partial_rollout(str(self.active), 3, quarantine=True)
            path.unlink()
        trace = target / "train/all/traces.jsonl"
        trace.unlink()
        trace.symlink_to(self.weight)
        with self.assertRaises(ValueError):
            resume.partial_rollout(str(self.active), 3, quarantine=True)
        self.assertTrue(target.exists())

    def test_quarantine_refuses_child_symlink_swap_at_rename(self):
        trace = self.partial()
        original_iter = Path.iterdir
        def entries(path):
            return iter([]) if path == Path("/proc") else original_iter(path)
        original_rename = resume.move_no_replace
        swapped = False
        def rename(*args, **kwargs):
            nonlocal swapped
            if not swapped:
                swapped = True
                trace.unlink()
                trace.symlink_to(self.weight)
            return original_rename(*args, **kwargs)
        with mock.patch.object(Path, "iterdir", entries), mock.patch.object(resume, "move_no_replace", side_effect=rename):
            with self.assertRaises((OSError, ValueError)):
                resume.partial_rollout(str(self.active), 3, quarantine=True)
        self.assertTrue(trace.is_symlink())
        self.assertEqual(self.weight.read_bytes(), b"adapter-three")

    def test_quarantine_reports_each_changed_identity_field_and_restores(self):
        trace = self.partial()
        target = self.active / "rollouts/step_4"
        original_tree = resume.partial_tree
        fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
        for field in fields:
            with self.subTest(field=field):
                observed = {p.relative_to(target).as_posix(): p.lstat() for p in (target, *target.rglob("*"))}
                before = getattr(observed["train/all/traces.jsonl"], field)
                def changed_tree(descriptor, prefix="."):
                    current = original_tree(descriptor, prefix)
                    if prefix != ".":
                        return current
                    value = current["train/all/traces.jsonl"]
                    attributes = {key: getattr(value, key) for key in fields}
                    attributes[field] = before + 1
                    current["train/all/traces.jsonl"] = types.SimpleNamespace(**attributes)
                    return current
                with mock.patch.object(resume, "partial_tree", side_effect=changed_tree):
                    with self.assertRaisesRegex(ValueError, "validation failed; restored") as caught:
                        resume.quarantine_partial(target.parent, target, observed)
                details = str(caught.exception).split("partial rollout changed during quarantine: ", 1)[1]
                self.assertEqual(json.loads(details), {"train/all/traces.jsonl": {field: {"observed": before, "current": before + 1}}})
                self.assertEqual(trace.read_text(), "partial trace")

    def test_quarantine_ignores_moved_directory_size_change(self):
        trace = self.partial()
        target = self.active / "rollouts/step_4"
        original_tree = resume.partial_tree

        def changed_tree(descriptor, prefix="."):
            current = original_tree(descriptor, prefix)
            if prefix == ".":
                value = current["."]
                attributes = {key: getattr(value, key) for key in ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")}
                attributes["st_size"] += 1
                current["."] = types.SimpleNamespace(**attributes)
            return current

        with mock.patch.object(resume, "partial_tree", side_effect=changed_tree):
            moved = resume.quarantine_partial(target.parent, target, {
                p.relative_to(target).as_posix(): p.lstat() for p in (target, *target.rglob("*"))
            })
        self.assertFalse(target.exists())
        self.assertEqual((Path(moved) / "train/all/traces.jsonl").read_text(), "partial trace")

    def test_quarantine_rejects_nested_directory_size_changes_and_restores(self):
        self.partial()
        target = self.active / "rollouts/step_4"
        original_tree = resume.partial_tree
        observed = {p.relative_to(target).as_posix(): p.lstat() for p in (target, *target.rglob("*"))}

        for name in ("train", "train/all"):
            with self.subTest(name=name):
                def changed_tree(descriptor, prefix="."):
                    current = original_tree(descriptor, prefix)
                    if prefix == name:
                        value = current[name]
                        attributes = {key: getattr(value, key) for key in ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")}
                        attributes["st_size"] += 1
                        current[name] = types.SimpleNamespace(**attributes)
                    return current

                with mock.patch.object(resume, "partial_tree", side_effect=changed_tree):
                    with self.assertRaisesRegex(ValueError, "validation failed; restored") as caught:
                        resume.quarantine_partial(target.parent, target, observed)
                details = str(caught.exception).split("partial rollout changed during quarantine: ", 1)[1]
                self.assertEqual(json.loads(details), {name: {"st_size": {"observed": observed[name].st_size, "current": observed[name].st_size + 1}}})
                self.assertTrue(target.is_dir())
                self.assertEqual((target / "train/all/traces.jsonl").read_text(), "partial trace")

    def test_quarantine_reports_entry_changes_and_restores(self):
        self.partial()
        target = self.active / "rollouts/step_4"
        observed = {p.relative_to(target).as_posix(): p.lstat() for p in (target, *target.rglob("*"))}
        current = dict(observed)
        current["added"] = current.pop("train/all/traces.jsonl")
        with mock.patch.object(resume, "partial_tree", return_value=current):
            with self.assertRaisesRegex(ValueError, "validation failed; restored") as caught:
                resume.quarantine_partial(target.parent, target, observed)
        details = str(caught.exception).split("partial rollout changed during quarantine: ", 1)[1]
        self.assertEqual(json.loads(details), {"added": {"entry": "added"}, "train/all/traces.jsonl": {"entry": "missing"}})
        self.assertTrue(target.is_dir())

    def test_quarantine_reports_real_trace_write_at_rename_and_restores(self):
        trace = self.partial()
        target = self.active / "rollouts/step_4"
        observed = {p.relative_to(target).as_posix(): p.lstat() for p in (target, *target.rglob("*"))}
        original_rename = resume.move_no_replace
        calls = 0
        def rename(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                trace.write_text("changed partial trace")
            return original_rename(*args, **kwargs)
        with mock.patch.object(resume, "move_no_replace", side_effect=rename):
            with self.assertRaisesRegex(ValueError, "validation failed; restored") as caught:
                resume.quarantine_partial(target.parent, target, observed)
        details = str(caught.exception).split("partial rollout changed during quarantine: ", 1)[1]
        self.assertEqual(json.loads(details)["train/all/traces.jsonl"]["st_size"], {"observed": 13, "current": 21})
        self.assertEqual(trace.stat().st_ino, observed["train/all/traces.jsonl"].st_ino)
        self.assertEqual(trace.read_text(), "changed partial trace")

    def test_quarantine_never_overwrites_destination_created_at_rename(self):
        self.partial()
        original_iter = Path.iterdir
        def entries(path):
            return iter([]) if path == Path("/proc") else original_iter(path)
        original_rename = resume.move_no_replace
        created = []
        def rename(source, destination, **kwargs):
            resume.os.mkdir(destination, dir_fd=kwargs["dst_dir_fd"])
            created.append(resume.os.stat(destination, dir_fd=kwargs["dst_dir_fd"]))
            try:
                return original_rename(source, destination, **kwargs)
            finally:
                self.assertEqual(resume.os.stat(destination, dir_fd=kwargs["dst_dir_fd"]).st_ino, created[0].st_ino)
        with mock.patch.object(Path, "iterdir", entries), mock.patch.object(resume, "move_no_replace", side_effect=rename):
            with self.assertRaises(FileExistsError):
                resume.partial_rollout(str(self.active), 3, quarantine=True)
        self.assertTrue((self.active / "rollouts/step_4/train/all/traces.jsonl").is_file())

    def test_quarantine_private_directory_collision_preserves_both_paths(self):
        trace = self.partial()
        destination = self.active / "rollouts/quarantined-step_4-collision"
        destination.mkdir()
        inode = destination.stat().st_ino
        original_iter = Path.iterdir
        def entries(path):
            return iter([]) if path == Path("/proc") else original_iter(path)
        with mock.patch.object(Path, "iterdir", entries), mock.patch.object(resume.secrets, "token_hex", return_value="collision"):
            with self.assertRaises(FileExistsError):
                resume.partial_rollout(str(self.active), 3, quarantine=True)
        self.assertEqual(destination.stat().st_ino, inode)
        self.assertEqual(list(destination.iterdir()), [])
        self.assertEqual(trace.read_text(), "partial trace")

    def test_quarantine_rollback_never_replaces_a_new_active_step(self):
        trace = self.partial()
        target = self.active / "rollouts/step_4"
        original_iter = Path.iterdir
        def entries(path):
            return iter([]) if path == Path("/proc") else original_iter(path)
        original_rename = resume.move_no_replace
        calls = 0
        replacement_inode = None
        def rename(*args, **kwargs):
            nonlocal calls, replacement_inode
            calls += 1
            if calls == 1:
                trace.unlink()
                trace.symlink_to(self.weight)
                result = original_rename(*args, **kwargs)
                target.mkdir()
                replacement_inode = target.stat().st_ino
                return result
            return original_rename(*args, **kwargs)
        with mock.patch.object(Path, "iterdir", entries), mock.patch.object(resume, "move_no_replace", side_effect=rename):
            with self.assertRaisesRegex(ValueError, "rollback refused; preserved"):
                resume.partial_rollout(str(self.active), 3, quarantine=True)
        self.assertEqual(target.stat().st_ino, replacement_inode)
        self.assertEqual(list(target.iterdir()), [])
        preserved = list((self.active / "rollouts").glob("quarantined-step_4-*/step_4/train/all/traces.jsonl"))
        self.assertEqual(len(preserved), 1)
        self.assertTrue(preserved[0].is_symlink())

    def test_quarantine_refuses_rollback_for_substituted_step_inode(self):
        self.partial()
        target = self.active / "rollouts/step_4"
        original_iter = Path.iterdir
        def entries(path):
            return iter([]) if path == Path("/proc") else original_iter(path)
        original_rename = resume.move_no_replace
        def rename(*args, **kwargs):
            target.rename(target.with_name("saved-original"))
            target.mkdir()
            return original_rename(*args, **kwargs)
        with mock.patch.object(Path, "iterdir", entries), mock.patch.object(resume, "move_no_replace", side_effect=rename) as move:
            with self.assertRaisesRegex(ValueError, "identity changed; preserved"):
                resume.partial_rollout(str(self.active), 3, quarantine=True)
        self.assertEqual(move.call_count, 1)
        self.assertFalse(target.exists())
        self.assertEqual((target.with_name("saved-original") / "train/all/traces.jsonl").read_text(), "partial trace")

    def test_partial_refuses_active_publisher_lock(self):
        self.partial()
        with (self.active / ".batch-sync.lock").open() as stream:
            resume.fcntl.flock(stream, resume.fcntl.LOCK_EX | resume.fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                resume.partial_rollout(str(self.active), 3, quarantine=True)
        self.assertTrue((self.active / "rollouts/step_4").exists())

    def test_partial_refuses_matching_publisher_process(self):
        self.partial()
        original = Path.read_bytes
        def read(path):
            if path == Path("/proc/123456/cmdline"):
                return b"bash\0sync_rollout_batches.sh\0"
            if path == Path("/proc/123456/environ"):
                return f"SOURCE_RUN_DIR={self.active}".encode()
            return original(path)
        original_iter = Path.iterdir
        def entries(path):
            return iter([Path("/proc/123456")]) if path == Path("/proc") else original_iter(path)
        with mock.patch.object(Path, "read_bytes", read), mock.patch.object(Path, "iterdir", entries):
            with self.assertRaisesRegex(ValueError, "publisher is still present"):
                resume.partial_rollout(str(self.active), 3, quarantine=True)

    def test_exact_stable_identity_and_trainer_copy(self):
        self.assertEqual(self.check(), {"policy_step": 3, "next_batch": 4, "sha256": self.sha, "size": self.weight.stat().st_size, "config_sha256": self.manifest["config_sha256"], "config_size": 2})
        self.assertEqual(self.check("trainer", digest=self.sha)["next_batch"], 4)

    def test_unstable_policy_three_cannot_resume_from_stable_two(self):
        (self.target / "STABLE").unlink()
        previous = self.active / "broadcasts/step_2"
        previous.mkdir()
        (previous / "STABLE").touch()
        for role in ("origin", "orchestrator"):
            with self.subTest(role=role), self.assertRaises(OSError):
                self.check(role)

    def test_wrong_manifest_step_digest_size_and_config_fail(self):
        for field, value in (("target_step", 2), ("target_sha256", "b" * 64), ("target_size", 1), ("config_sha256", "b" * 64)):
            with self.subTest(field=field):
                (self.target / "adapter_manifest.json").write_text(json.dumps(dict(self.manifest, **{field: value})))
                with self.assertRaises(ValueError):
                    self.check()

    def test_trainer_copy_digest_must_match(self):
        with self.assertRaisesRegex(ValueError, "differs"):
            self.check("trainer", digest="b" * 64)

    def test_cached_future_binary_blocks_all_active_locations(self):
        for root in (self.active / "rollouts", self.active.parent / "rollouts", self.active / ".incoming/rollouts", self.active.parent / ".incoming/rollouts"):
            for name in ("train_rollouts.bin", "rank_0.bin"):
                with self.subTest(root=root, name=name):
                    cached = root / "step_4" / name
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    cached.write_bytes(b"old")
                    with self.assertRaisesRegex(ValueError, "cached rollout"):
                        self.check()
                    cached.unlink()

    def test_later_trace_evidence_does_not_select_progress(self):
        trace = self.active / "rollouts/step_5/train/effective/traces.jsonl"
        trace.parent.mkdir(parents=True)
        trace.write_text("old evidence")
        self.assertEqual(self.check("orchestrator")["next_batch"], 4)
        self.assertEqual(trace.read_text(), "old evidence")

    def test_saved_optimizer_state_is_not_an_adapter_resume(self):
        checkpoint = self.active.parent / "checkpoints/step_3"
        checkpoint.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "optimizer checkpoint"):
            self.check()

    def test_later_stable_policy_cannot_silently_advance_restart(self):
        stable = self.active / "broadcasts/step_4/STABLE"
        stable.parent.mkdir()
        stable.touch()
        with self.assertRaisesRegex(ValueError, "later stable policy"):
            self.check()

    def test_symlinked_policy_directory_is_rejected(self):
        self.target.rename(self.target.with_name("saved-three"))
        self.target.symlink_to(self.target.with_name("saved-three"), target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.check()

    def test_explicit_config_overrides_discovery_without_other_changes(self):
        for suffix in ("", "\n[log]\nlevel='info'\n"):
            original = "output_dir='/runs/a'\n[ckpt]\nresume_step=-1\nskip_progress=false\ninterval=5\n" + suffix
            configured = tomllib.loads(resume.explicit_config(original, 3))
            self.assertEqual(configured["ckpt"], {"resume_step": 3, "skip_progress": True, "interval": 5})
            self.assertEqual(configured["output_dir"], "/runs/a")

    def test_explicit_config_adds_missing_checkpoint_section(self):
        configured = tomllib.loads(resume.explicit_config("max_steps=50\n", 3))
        self.assertEqual(configured, {"max_steps": 50, "ckpt": {"resume_step": 3, "skip_progress": True}})

    def test_runtime_config_command_preserves_source(self):
        config = self.active / "orchestrator.toml"
        original = "max_steps=50\n[ckpt]\nresume_step=-1\nskip_progress=true\n"
        config.write_text(original)
        result = subprocess.run([sys.executable, spec.origin, "config", str(config), "3"], capture_output=True, text=True, check=True)
        self.assertEqual(tomllib.loads(result.stdout)["ckpt"]["resume_step"], 3)
        self.assertEqual(config.read_text(), original)

    def test_runtime_state_command_returns_exact_progress(self):
        result = subprocess.run([sys.executable, spec.origin, str(self.active), "3", "origin", str(self.weight), ""], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), self.check())

    def test_control_symlink_is_rejected_before_writing(self):
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        (self.active / "control").symlink_to(outside, target_is_directory=True)
        config = self.active / "config.toml"
        config.write_text("max_steps=50\n")
        with self.assertRaises(OSError):
            resume.write_config(str(config), str(self.active / "control"), 3)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.check()
        self.assertEqual(list(outside.iterdir()), [])

    def test_ancestor_symlink_is_rejected_before_writing(self):
        alias = Path(self.temporary.name) / "alias"
        alias.symlink_to(Path(self.temporary.name), target_is_directory=True)
        config = self.active / "config.toml"
        config.write_text("max_steps=50\n")
        with self.assertRaises(OSError):
            resume.write_config(str(config), str(alias / "alpha/run_alpha/control"), 3)
        self.assertFalse((self.active / "control").exists())

    def test_safe_runtime_control_is_regular_readable_and_exact(self):
        config = self.active / "config.toml"
        config.write_text("max_steps=50\n")
        generated = Path(resume.write_config(str(config), str(self.active / "control"), 3))
        self.assertEqual(generated.stat().st_mode & 0o777, 0o644)
        self.assertEqual(tomllib.loads(generated.read_text())["ckpt"], {"resume_step": 3, "skip_progress": True})


if __name__ == "__main__":
    unittest.main()
