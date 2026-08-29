import hashlib
import json
import os
import stat
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/h200"))
import flashmla_overlay_preflight as overlay  # noqa: E402


ROOT = Path(__file__).parents[1]


class FlashMLAOverlayTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.deployment = self.root / "deployment"
        self.overlay = self.deployment / "cache/vllm/flashmla512"
        self.deployment.mkdir(parents=True)
        self.overlay.mkdir(parents=True)
        for relative in overlay.PATCHES + (overlay.TEST_FILE,):
            path = self.deployment / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative)
        for relative in overlay.FILES.values():
            path = self.overlay / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("pass\n")
            path.chmod(0o644)
        launcher = self.deployment / "scripts/h200/start_inference.sh"
        launcher.parent.mkdir(parents=True, exist_ok=True)
        launcher.write_text(
            '"$deployment/source/src/prime_rl/parity_router_trace.py":'
            "/app/src/prime_rl/parity_router_trace.py:ro\n"
            + "\n".join(
                f'--mount "type=bind,source=$overlay_root/{relative},'
                f'target={overlay.RUNTIME_MOUNTS[name]},readonly"'
                for name, relative in overlay.FILES.items()
            )
            + "\n"
        )
        for relative in overlay.SOURCE_FILES.values():
            path = self.deployment / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("def h200_trace_request():\n    pass\n")
        manifest = {
            "preflight_passed": True,
            "image_id": overlay.EXPECTED_IMAGE_ID,
            "files": {
                name: hashlib.sha256((self.overlay / relative).read_bytes()).hexdigest()
                for name, relative in overlay.FILES.items()
            },
            "runtime_mounts": overlay.RUNTIME_MOUNTS,
            "patches": {
                relative: hashlib.sha256((self.deployment / relative).read_bytes()).hexdigest()
                for relative in overlay.PATCHES
            },
            "test_sha256": hashlib.sha256(
                (self.deployment / overlay.TEST_FILE).read_bytes()
            ).hexdigest(),
            "source_files": {
                name: hashlib.sha256(
                    (self.deployment / relative).read_bytes()
                ).hexdigest()
                for name, relative in overlay.SOURCE_FILES.items()
            },
        }
        (self.overlay / "manifest.json").write_text(json.dumps(manifest))

    def test_manifest_accepts_two_regular_files(self):
        with mock.patch.object(overlay, "image_id", return_value=overlay.EXPECTED_IMAGE_ID):
            result = overlay.verify_manifest(self.deployment, self.overlay, "base")
        self.assertTrue(result["preflight_passed"])

    def test_missing_patch_refuses(self):
        (self.deployment / overlay.PATCHES[0]).unlink()
        with mock.patch.object(overlay, "image_id", return_value=overlay.EXPECTED_IMAGE_ID):
            with self.assertRaises(overlay.OverlayError):
                overlay.verify_manifest(self.deployment, self.overlay, "base")

    def test_missing_preflight_refuses(self):
        manifest = json.loads((self.overlay / "manifest.json").read_text())
        manifest["preflight_passed"] = False
        (self.overlay / "manifest.json").write_text(json.dumps(manifest))
        with mock.patch.object(overlay, "image_id", return_value=overlay.EXPECTED_IMAGE_ID):
            with self.assertRaises(overlay.OverlayError):
                overlay.verify_manifest(self.deployment, self.overlay, "base")

    def test_hash_mismatch_refuses(self):
        (self.overlay / overlay.FILES["envs"]).write_text("changed")
        with mock.patch.object(overlay, "image_id", return_value=overlay.EXPECTED_IMAGE_ID):
            with self.assertRaises(overlay.OverlayError):
                overlay.verify_manifest(self.deployment, self.overlay, "base")

    def test_launcher_has_exact_read_only_mounts(self):
        launcher = (ROOT / "scripts/h200/start_inference.sh").read_text()
        for name, target in overlay.RUNTIME_MOUNTS.items():
            self.assertIn(
                f"source=$overlay_root/{overlay.FILES[name]},target={target},readonly",
                launcher,
            )

    def test_manifest_requires_the_complete_runtime_mount_set(self):
        manifest = json.loads((self.overlay / "manifest.json").read_text())
        manifest.pop("runtime_mounts")
        (self.overlay / "manifest.json").write_text(json.dumps(manifest))
        with mock.patch.object(overlay, "image_id", return_value=overlay.EXPECTED_IMAGE_ID):
            with self.assertRaises(overlay.OverlayError):
                overlay.verify_manifest(self.deployment, self.overlay, "base")

    def test_restrictive_umask_makes_staging_traversable_without_write_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary) / "staging"
            old_umask = os.umask(0o077)
            try:
                staging.mkdir()
                source = staging / "vllm/model_executor/layers/example.py"
                source.parent.mkdir(parents=True)
                source.write_text("pass\n")
            finally:
                os.umask(old_umask)

            overlay._make_read_only_validation_tree(staging)

            for path in (staging, source.parent, source):
                mode = stat.S_IMODE(path.stat().st_mode)
                expected = 0o755 if path.is_dir() else 0o644
                self.assertEqual(mode, expected)
            self.assertTrue(staging.stat().st_mode & stat.S_IXOTH)
            self.assertFalse(source.stat().st_mode & stat.S_IWOTH)

    def _prepare_run(self, command, *, cwd=None, env=None):
        if command[0] != "docker":
            return
        mount = next(
            (argument for argument in command if argument.endswith(":/overlay")),
            None,
        )
        if mount is None:
            return
        staging = Path(mount[:-len(":/overlay")])
        for relative in overlay.FILES.values():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("pass\n")

    def test_every_patch_target_is_mounted_and_compiles(self):
        patch_targets = {
            line.removeprefix("+++ b/")
            for relative in overlay.PATCHES
            for line in (ROOT / relative).read_text().splitlines()
            if line.startswith("+++ b/")
        }
        self.assertEqual(len(overlay.FILES), 12)
        self.assertEqual(patch_targets, set(overlay.FILES.values()))
        self.assertEqual(set(overlay.FILES), set(overlay.RUNTIME_MOUNTS))
        for relative in overlay.FILES.values():
            path = self.overlay / relative
            compile(path.read_text(), str(path), "exec")

    def test_prepare_refreshes_old_manifest_with_all_patches_and_mounts(self):
        manifest = json.loads((self.overlay / "manifest.json").read_text())
        manifest["runtime_mounts"] = None
        (self.overlay / "manifest.json").write_text(json.dumps(manifest))
        with (
            mock.patch.object(overlay, "image_id", return_value=overlay.EXPECTED_IMAGE_ID),
            mock.patch.object(overlay, "_run", side_effect=self._prepare_run),
        ):
            refreshed = overlay.prepare(self.deployment, self.overlay, "base")

        self.assertEqual(refreshed["runtime_mounts"], overlay.RUNTIME_MOUNTS)
        self.assertEqual(set(refreshed["patches"]), set(overlay.PATCHES))
        self.assertEqual(json.loads((self.overlay / "manifest.json").read_text()), refreshed)
        self.assertEqual(list(self.deployment.glob(f".{self.overlay.name}.*")), [])

    def test_prepare_replaces_overlay_as_one_directory_transaction(self):
        replacements = []
        real_replace = overlay.os.replace

        def record_replace(source, destination):
            replacements.append((Path(source), Path(destination)))
            return real_replace(source, destination)

        with (
            mock.patch.object(overlay, "image_id", return_value=overlay.EXPECTED_IMAGE_ID),
            mock.patch.object(overlay, "_run", side_effect=self._prepare_run),
            mock.patch.object(overlay.os, "replace", side_effect=record_replace),
        ):
            overlay.prepare(self.deployment, self.overlay, "base")

        self.assertTrue(
            any(
                destination == self.overlay and source.parent == self.overlay.parent
                for source, destination in replacements
            )
        )

    def test_prepare_failure_keeps_old_overlay_and_cleans_staging(self):
        marker = self.overlay / "old-marker"
        marker.write_text("keep")

        def fail_run(command, *, cwd=None, env=None):
            if command[0] == "docker":
                raise overlay.OverlayError("pinned image failed")

        with (
            mock.patch.object(overlay, "image_id", return_value=overlay.EXPECTED_IMAGE_ID),
            mock.patch.object(overlay, "_run", side_effect=fail_run),
        ):
            with self.assertRaises(overlay.OverlayError):
                overlay.prepare(self.deployment, self.overlay, "base")

        self.assertEqual(marker.read_text(), "keep")
        self.assertEqual(list(self.deployment.glob(f".{self.overlay.name}.*")), [])


if __name__ == "__main__":
    unittest.main()
