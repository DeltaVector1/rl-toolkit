import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import tomllib
import unittest
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
