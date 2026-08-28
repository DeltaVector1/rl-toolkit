import hashlib
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/h200"))
import flashmla_overlay_preflight as overlay  # noqa: E402


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
            path.write_text(relative)
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
        launcher = (Path(__file__).parents[1] / "scripts/h200/start_inference.sh").read_text()
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


if __name__ == "__main__":
    unittest.main()
