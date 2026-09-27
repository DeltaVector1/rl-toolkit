import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import training_ops
from training_ops.runtime import adapter_cache_cleanup


SOURCE = "http://127.0.0.1:18100/trinity-decensor-experimental-fixed/broadcasts/step_0"
KEY = "056ce0a30be48095dbd88b8c0e741146d5be162be82252eb2040ee1ec0cd9074"
LORA = "trinity-decensor-experimental-fixed-r32-node-local"


def receipt(source: str, name: str, instance: str = "a" * 32) -> dict:
    return {
        "source_url": source,
        "target_sha256": "1" * 64,
        "target_size": 1,
        "config_sha256": "2" * 64,
        "config_size": 1,
        "lora_name": name,
        "server_instance": instance,
    }


def write_cache(root: Path, source: str = SOURCE, key: str = KEY) -> Path:
    target = root / key
    target.mkdir()
    config = b"{}\n"
    model = b"model"
    files = {
        "adapter_config.json": {"size": len(config), "sha256": hashlib.sha256(config).hexdigest()},
        "adapter_model.safetensors": {"size": len(model), "sha256": hashlib.sha256(model).hexdigest()},
    }
    (target / "READY").write_text("ready\n", encoding="ascii")
    (target / "manifest.json").write_text(
        json.dumps({"version": 1, "source_url": source, "files": files}),
        encoding="utf-8",
    )
    (target / "adapter_config.json").write_bytes(config)
    (target / "adapter_model.safetensors").write_bytes(model)
    (target / "adapter_model.safetensors.sha256").write_text(
        files["adapter_model.safetensors"]["sha256"] + "\n", encoding="ascii"
    )
    return target


class ExactAdapterCacheCleanupTest(unittest.TestCase):
    def test_production_plan_pins_exact_url_and_key(self):
        plan = training_ops._exact_adapter_cache_cleanup_plan(
            config_dir="configs/trinity-decensor-experimental-fixed"
        )
        self.assertEqual(plan["run"], "trinity-decensor-experimental-fixed")
        self.assertEqual(plan["source_url"], SOURCE)
        self.assertEqual(plan["cache_key"], KEY)
        self.assertEqual(plan["cache_path"], f"/data/glm52-rl/lora-cache/{KEY}")
        self.assertEqual(plan["lora_name"], LORA)

    def test_exact_url_and_name_plan_is_supported(self):
        plan = training_ops._exact_adapter_cache_cleanup_plan(
            source_url=SOURCE,
            lora_name=LORA,
        )
        self.assertEqual(plan["cache_key"], KEY)

    def test_source_url_and_name_are_required_and_exact(self):
        with self.assertRaisesRegex(training_ops.Refused, "exact source URL"):
            training_ops._exact_adapter_cache_cleanup_plan(source_url=SOURCE)
        with self.assertRaisesRegex(training_ops.Refused, "exact loopback"):
            training_ops._exact_adapter_cache_cleanup_plan(
                source_url="https://example.invalid/run", lora_name=LORA
            )
        with self.assertRaisesRegex(training_ops.Refused, "exact step_0"):
            training_ops._exact_adapter_cache_cleanup_plan(
                source_url=SOURCE.replace("step_0", "step_1"), lora_name=LORA
            )

    def test_status_success_and_all_receipt_boundaries_fail_closed(self):
        empty = {"status": "idle", "loaded_receipts": {}}
        adapter_cache_cleanup.validate_status(
            empty,
            source_url=SOURCE,
            lora_name=LORA,
            cache_path=f"/cache/{KEY}",
            cache_root="/cache",
        )
        cases = [
            ({"status": "running", "loaded_receipts": {}}, "status is unknown"),
            ({"status": "staging", "loaded_receipts": {}}, "stage task is active"),
            ({"status": "idle"}, "loaded receipts are unknown"),
            (
                {"status": "idle", "loaded_receipts": {LORA: receipt(SOURCE, LORA)}},
                "exact adapter receipt is still loaded",
            ),
            (
                {"status": "idle", "loaded_receipts": {"other": receipt(SOURCE, "other")}},
                "exact adapter receipt is still loaded",
            ),
            (
                {"status": "idle", "loaded_receipts": {"other": receipt(SOURCE, "other", "bad")}},
                "server identity is invalid",
            ),
            (
                {"status": "idle", "loaded_receipts": {"other": {"source_url": SOURCE}}},
                "receipt contents are unknown",
            ),
            (
                {
                    "status": "ready",
                    "source_url": "http://127.0.0.1:18100/other/broadcasts/step_0",
                    "path": f"/cache/{KEY}",
                    "loaded_receipts": {},
                },
                "stage path and source URL differ",
            ),
            (
                {
                    "status": "idle",
                    "loaded_receipts": {
                        "one": receipt("http://127.0.0.1:18100/one", "one", "a" * 32),
                        "two": receipt("http://127.0.0.1:18100/two", "two", "b" * 32),
                    },
                },
                "server identity is inconsistent",
            ),
        ]
        for status, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                adapter_cache_cleanup.validate_status(
                    status,
                    source_url=SOURCE,
                    lora_name=LORA,
                    cache_path=f"/cache/{KEY}",
                    cache_root="/cache",
                )

    def test_running_and_waiting_metrics_fail_closed(self):
        valid = "vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
        adapter_cache_cleanup.validate_metrics(valid)
        for metric in (
            "vllm:num_requests_running 1\nvllm:num_requests_waiting 0\n",
            "vllm:num_requests_running 0\nvllm:num_requests_waiting 2\n",
            "vllm:num_requests_running 0\n",
        ):
            with self.assertRaises(ValueError):
                adapter_cache_cleanup.validate_metrics(metric)

    def test_directory_success_preserves_other_entries_and_rejects_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = write_cache(root)
            other = root / ("b" * 64)
            other.mkdir()
            identity = adapter_cache_cleanup.validate_directory(
                target, source_url=SOURCE, expected_key=KEY
            )
            self.assertEqual(identity, (target.stat().st_dev, target.stat().st_ino))
            self.assertTrue(other.exists())

            cases = []
            (target / "unknown").write_text("x")
            cases.append("unknown contents")
            (target / "unknown").unlink()
            (target / "manifest.json").write_text(
                json.dumps({"version": 1, "source_url": "http://wrong", "files": {}})
            )
            cases.append("manifest source")
            (target / "manifest.json").unlink()
            (target / "manifest.json").symlink_to(root / "missing")
            cases.append("manifest")
            for message in cases:
                with self.subTest(message=message), self.assertRaises(ValueError):
                    adapter_cache_cleanup.validate_directory(
                        target, source_url=SOURCE, expected_key=KEY
                    )

    def test_directory_refuses_root_target_and_file_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = write_cache(root)
            moved = root / ("c" * 64)
            target.rename(moved)
            with self.assertRaisesRegex(ValueError, "path or key mismatch"):
                adapter_cache_cleanup.validate_directory(
                    moved, source_url=SOURCE, expected_key=KEY
                )
            moved.rename(target)
            outside = root / "outside"
            outside.write_text("outside")
            (target / "adapter_config.json").unlink()
            (target / "adapter_config.json").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                adapter_cache_cleanup.validate_directory(
                    target, source_url=SOURCE, expected_key=KEY
                )
            root2 = root / "root-link"
            root2.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "real directory"):
                adapter_cache_cleanup.validate_directory(
                    root2 / KEY, source_url=SOURCE, expected_key=KEY
                )

    def test_directory_refuses_manifest_and_key_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = write_cache(root)
            manifest = json.loads((target / "manifest.json").read_text())
            manifest["source_url"] = "http://127.0.0.1:18100/other/broadcasts/step_0"
            (target / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "manifest source"):
                adapter_cache_cleanup.validate_directory(
                    target, source_url=SOURCE, expected_key=KEY
                )
            manifest["source_url"] = SOURCE
            (target / "manifest.json").write_text(json.dumps(manifest))
            (target / "adapter_model.safetensors.sha256").write_text("f" * 64)
            with self.assertRaisesRegex(ValueError, "sidecar"):
                adapter_cache_cleanup.validate_directory(
                    target, source_url=SOURCE, expected_key=KEY
                )

    def test_remote_script_is_exact_and_lock_aware(self):
        plan = training_ops._exact_adapter_cache_cleanup_plan(
            config_dir="configs/trinity-decensor-experimental-fixed"
        )
        script = training_ops._build_adapter_cache_cleanup_script(
            {**plan, "ports": [8000]}
        )
        self.assertIn("/staging_status", script)
        self.assertIn("num_requests_(running|waiting)", script)
        self.assertIn("LOCK_EX | fcntl.LOCK_NB", script)
        self.assertIn("shutil.rmtree(target)", script)
        self.assertIn(KEY, script)
        self.assertNotIn("cache_root/*", script)
        self.assertNotIn("rm -rf $cache_root", script)
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / f".{KEY}.readers.lock"
            lock_path.touch()
            import fcntl

            with lock_path.open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                probe = lock_path.open("a+")
                try:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                finally:
                    probe.close()

    def test_api_removes_one_entry_per_configured_host_from_remote_evidence(self):
        plan = training_ops._exact_adapter_cache_cleanup_plan(
            config_dir="configs/trinity-decensor-experimental-fixed"
        )
        output = json.dumps(
            {
                "status": "removed",
                "cache_key": KEY,
                "cache_path": plan["cache_path"],
                "source_url": SOURCE,
            }
        )
        calls = []

        def fake_ssh(host, script, timeout=0):
            calls.append((host, script, timeout))
            return 0, output + "\n"

        with mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh):
            result = training_ops.remove_adapter_cache(
                config_dir="configs/trinity-decensor-experimental-fixed"
            )
        self.assertEqual(result["status"], "removed")
        self.assertEqual([item[0] for item in calls], ["h200b"])
        self.assertEqual(calls[0][2], 180)


if __name__ == "__main__":
    unittest.main()
