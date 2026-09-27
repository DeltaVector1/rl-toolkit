import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import training_ops
from training_ops.runtime import adapter_cache_cleanup


CONFIG_BASE = "http://127.0.0.1:18100/trinity-decensor-experimental-fixed"
SOURCE = f"{CONFIG_BASE}/run_trinity-decensor-experimental-fixed/broadcasts/step_0"
LEGACY_SOURCE = f"{CONFIG_BASE}/broadcasts/step_0"
KEY = "f72c3222b63fe4faf9e482d6b36fb6887521aac5a76f96655e2b94a18a630f8a"
LEGACY_KEY = "056ce0a30be48095dbd88b8c0e741146d5be162be82252eb2040ee1ec0cd9074"
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


def write_cache(
    root: Path,
    source: str = SOURCE,
    key: str = KEY,
    *,
    transport_manifest: bool = False,
) -> Path:
    target = root / key
    target.mkdir()
    config = b"{}\n"
    model = bytes(index % 256 for index in range(986)) if transport_manifest else b"model"
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
    if transport_manifest:
        chunks = []
        for index, value in enumerate(model):
            chunk_hash = hashlib.sha256(bytes([value])).hexdigest()
            chunks.append(
                {
                    "codec": None,
                    "data_offset": None,
                    "file": f"adapter_model.safetensors.transport-full-zstd-{index:06d}",
                    "offset": index,
                    "raw_sha256": chunk_hash,
                    "raw_size": 1,
                    "wire_sha256": chunk_hash,
                    "wire_size": 1 if index < 985 else 6382288128 - 985,
                }
            )
        transport = {
            "base2_config_sha256": None,
            "base2_config_size": None,
            "base2_sha256": None,
            "base2_size": None,
            "base2_url": None,
            "base_config_sha256": None,
            "base_config_size": None,
            "base_sha256": None,
            "base_size": None,
            "base_url": None,
            "chunks": chunks,
            "config_sha256": files["adapter_config.json"]["sha256"],
            "config_size": files["adapter_config.json"]["size"],
            "data_offset": None,
            "mode": "full",
            "resume_identity_version": 1,
            "source_url": source,
            "target_sha256": files["adapter_model.safetensors"]["sha256"],
            "target_size": files["adapter_model.safetensors"]["size"],
            "transport": "chunked-full-zstd",
            "version": 1,
        }
        transport_path = target / "transport-manifest.json"
        transport_path.write_text(json.dumps(transport), encoding="utf-8")
        transport_path.chmod(0o600)
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

    def test_explicit_legacy_exact_url_remains_raw(self):
        plan = training_ops._exact_adapter_cache_cleanup_plan(
            source_url=LEGACY_SOURCE,
            lora_name=LORA,
        )
        self.assertEqual(plan["source_url"], LEGACY_SOURCE)
        self.assertEqual(plan["cache_key"], LEGACY_KEY)

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

    def test_receipt_urls_use_exact_source_canonicalization(self):
        base = CONFIG_BASE
        with self.assertRaisesRegex(ValueError, "exact adapter receipt is still loaded"):
            adapter_cache_cleanup.validate_status(
                {"status": "idle", "loaded_receipts": {"other": receipt(base, "other")}},
                source_url=SOURCE,
                lora_name=LORA,
                cache_path=f"/cache/{KEY}",
                cache_root="/cache",
            )
        malformed = (
            f"{base}/",
            f"{CONFIG_BASE}/run_trinity-decensor-experimental-fixed/",
            f"{SOURCE}/",
            f"{SOURCE}?query=1",
            f"{SOURCE}#fragment",
            SOURCE.replace("http://127.0.0.1", "http://user@127.0.0.1"),
            SOURCE.replace("127.0.0.1", "192.0.2.1"),
            SOURCE.replace("step_0", "step_1"),
        )
        for value in malformed:
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "loaded receipt source is invalid"
            ):
                adapter_cache_cleanup.validate_status(
                    {"status": "idle", "loaded_receipts": {"other": receipt(value, "other")}},
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
            other = root / LEGACY_KEY
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
            (target / "transport-manifest.json").write_text("{}")
            cases.append("transport manifest schema")
            (target / "transport-manifest.json").unlink()
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

    def test_measured_transport_manifest_is_strict_and_chunks_may_be_absent(self):
        def invalid(mutator):
            with tempfile.TemporaryDirectory() as directory:
                target = write_cache(Path(directory), transport_manifest=True)
                path = target / "transport-manifest.json"
                value = json.loads(path.read_text())
                mutator(value, path)
                if path.exists():
                    path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    adapter_cache_cleanup.validate_directory(
                        target, source_url=SOURCE, expected_key=KEY
                    )

        with tempfile.TemporaryDirectory() as directory:
            target = write_cache(Path(directory), transport_manifest=True)
            self.assertFalse((target / "transport-chunks").exists())
            identity = adapter_cache_cleanup.validate_directory(
                target, source_url=SOURCE, expected_key=KEY
            )
            self.assertEqual(identity, (target.stat().st_dev, target.stat().st_ino))

        invalid(lambda value, _path: value.update({"extra": None}))
        invalid(lambda value, _path: value.pop("transport"))
        invalid(lambda value, _path: value.update({"source_url": LEGACY_SOURCE}))
        invalid(lambda value, _path: value.update({"target_sha256": "f" * 64}))
        invalid(lambda value, _path: value.update({"base_url": SOURCE}))
        invalid(lambda value, _path: value["chunks"][0].update({"extra": None}))
        invalid(lambda value, _path: value["chunks"][1].update({"offset": 0}))
        invalid(lambda value, _path: value["chunks"][0].update({"file": "../escape"}))
        invalid(lambda value, _path: value["chunks"][0].update({"raw_sha256": "F" * 64}))
        invalid(lambda value, _path: value.update({"target_size": 985}))

        with tempfile.TemporaryDirectory() as directory:
            target = write_cache(Path(directory), transport_manifest=True)
            (target / "transport-manifest.json").chmod(0o644)
            with self.assertRaisesRegex(ValueError, "permissions"):
                adapter_cache_cleanup.validate_directory(
                    target, source_url=SOURCE, expected_key=KEY
                )

    def test_measured_transport_wire_total_is_6382288128(self):
        with tempfile.TemporaryDirectory() as directory:
            target = write_cache(Path(directory), transport_manifest=True)
            identity = adapter_cache_cleanup.validate_directory(
                target, source_url=SOURCE, expected_key=KEY
            )
            self.assertEqual(identity, (target.stat().st_dev, target.stat().st_ino))

    def test_fresh_preflight_uses_nested_production_cache_key(self):
        selected = {
            "trainer": "configs/trinity-decensor-experimental-fixed/trainer.toml",
            "orchestrator": "configs/trinity-decensor-experimental-fixed/orchestrator.toml",
            "inference": "configs/trinity-decensor-experimental-fixed/inference.toml",
        }
        with mock.patch.object(
            training_ops,
            "_ssh",
            return_value=(0, "FRESH_START_PRECHECK_OK"),
        ) as ssh, mock.patch.object(
            training_ops,
            "_fan",
            side_effect=lambda hosts, function: {
                host: function(host) for host in hosts
            },
        ):
            training_ops._fresh_start_preflight(
                "trinity-decensor-experimental-fixed",
                selected,
                ["n82"],
                "/data/glm52-rl/runs/trinity-decensor-experimental-fixed/run_trinity-decensor-experimental-fixed",
            )
        scripts = {call.args[0]: call.args[1] for call in ssh.call_args_list}
        self.assertIn(KEY, scripts["h200b"])
        self.assertNotIn(LEGACY_KEY, scripts["h200b"])

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

    def test_directory_refuses_file_size_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = write_cache(root)
            (target / "adapter_model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "file size differs"):
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
        self.assertIn("check_identity()\n    shutil.rmtree(target)", script)
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

    def test_remote_receipt_urls_are_canonicalized_and_fail_closed(self):
        plan = training_ops._exact_adapter_cache_cleanup_plan(
            config_dir="configs/trinity-decensor-experimental-fixed"
        )
        base = SOURCE.removesuffix("/broadcasts/step_0")
        cases = (
            (CONFIG_BASE, "exact adapter receipt is still loaded"),
            (LEGACY_SOURCE, "exact adapter receipt is still loaded"),
            (
                f"{CONFIG_BASE}/run_trinity-decensor-experimental-fixed",
                "exact adapter receipt is still loaded",
            ),
            (f"{base}/", "loaded receipt source is invalid"),
            (f"{SOURCE}/", "loaded receipt source is invalid"),
            (f"{SOURCE}?query=1", "loaded receipt source is invalid"),
            (f"{SOURCE}#fragment", "loaded receipt source is invalid"),
            (
                SOURCE.replace("http://127.0.0.1", "http://user@127.0.0.1"),
                "loaded receipt source is invalid",
            ),
            (SOURCE.replace("127.0.0.1", "192.0.2.1"), "loaded receipt source is invalid"),
            (SOURCE.replace("step_0", "step_1"), "loaded receipt source is invalid"),
        )
        for receipt_source, message in cases:
            with self.subTest(receipt_source=receipt_source):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    cache_root = root / "cache"
                    cache_root.mkdir()
                    target = write_cache(cache_root)
                    fake_bin = root / "bin"
                    fake_bin.mkdir()
                    (fake_bin / "sudo").write_text(
                        '#!/bin/sh\nexec "$@"\n', encoding="utf-8"
                    )
                    (fake_bin / "ss").write_text(
                        "#!/bin/sh\nprintf 'LISTEN 0 0 127.0.0.1:8000 0.0.0.0:* users:((\\\"server\\\",pid=123,fd=1))\\n'\n",
                        encoding="utf-8",
                    )
                    (fake_bin / "docker").write_text(
                        "#!/bin/sh\n"
                        "printf '/glm52-decensor-h200|aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa|2026-09-27T00:00:00Z|true|123\\n'\n",
                        encoding="utf-8",
                    )
                    for command in fake_bin.iterdir():
                        command.chmod(0o755)
                    status_body = json.dumps(
                        {
                            "status": "idle",
                            "loaded_receipts": {"other": receipt(receipt_source, "other")},
                        }
                    ).encode()
                    (root / "sitecustomize.py").write_text(
                        "import urllib.request\n"
                        "class Response:\n"
                        "    status = 200\n"
                        "    def __init__(self, body): self.body = body\n"
                        "    def __enter__(self): return self\n"
                        "    def __exit__(self, *_args): return False\n"
                        "    def read(self): return self.body\n"
                        "def fake_urlopen(request, timeout=30):\n"
                        "    url = request.full_url if hasattr(request, 'full_url') else str(request)\n"
                        "    if url.endswith('/health'): return Response(b'ok')\n"
                        "    if url.endswith('/metrics'): return Response(b'vllm:num_requests_running 0\\nvllm:num_requests_waiting 0\\n')\n"
                        "    if url.endswith('/staging_status'): return Response("
                        + repr(status_body)
                        + ")\n"
                        "    raise RuntimeError(url)\n"
                        "urllib.request.urlopen = fake_urlopen\n",
                        encoding="utf-8",
                    )
                    script = training_ops._build_adapter_cache_cleanup_script(
                        {
                            **plan,
                            "cache_root": str(cache_root),
                            "cache_path": str(target),
                            "ports": [8000],
                        }
                    )
                    environment = os.environ.copy()
                    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
                    environment["PYTHONPATH"] = str(root)
                    result = subprocess.run(
                        ["bash", "-c", script],
                        cwd=root,
                        env=environment,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertTrue(target.exists())

    def test_remote_identity_change_inside_lock_preserves_target(self):
        plan = training_ops._exact_adapter_cache_cleanup_plan(
            config_dir="configs/trinity-decensor-experimental-fixed"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = root / "cache"
            cache_root.mkdir()
            target = write_cache(cache_root, transport_manifest=True)
            other = cache_root / ("b" * 64)
            other.mkdir()
            fake_bin = root / "bin"
            fake_bin.mkdir()
            (fake_bin / "sudo").write_text(
                '#!/bin/sh\nexec "$@"\n', encoding="utf-8"
            )
            (fake_bin / "ss").write_text(
                "#!/bin/sh\nprintf 'LISTEN 0 0 127.0.0.1:8000 0.0.0.0:* users:((\\\"server\\\",pid=123,fd=1))\\n'\n",
                encoding="utf-8",
            )
            (fake_bin / "docker").write_text(
                "#!/bin/sh\n"
                "state=${DOCKER_STATE:?}\n"
                "count=$(cat \"$state\" 2>/dev/null || printf '0')\n"
                "count=$((count + 1))\n"
                "printf '%s' \"$count\" > \"$state\"\n"
                "if [ \"$count\" -eq 1 ]; then\n"
                "  id=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
                "else\n"
                "  id=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
                "fi\n"
                "printf '/glm52-decensor-h200|%s|2026-09-27T00:00:00Z|true|123\\n' \"$id\"\n",
                encoding="utf-8",
            )
            for command in fake_bin.iterdir():
                command.chmod(0o755)
            (root / "sitecustomize.py").write_text(
                "import json\n"
                "import urllib.request\n\n"
                "class Response:\n"
                "    status = 200\n"
                "    def __init__(self, body):\n"
                "        self.body = body\n"
                "    def __enter__(self):\n"
                "        return self\n"
                "    def __exit__(self, *_args):\n"
                "        return False\n"
                "    def read(self):\n"
                "        return self.body\n\n"
                "def fake_urlopen(request, timeout=30):\n"
                "    url = request.full_url if hasattr(request, 'full_url') else str(request)\n"
                "    if url.endswith('/health'):\n"
                "        return Response(b'ok')\n"
                "    if url.endswith('/metrics'):\n"
                "        return Response(b'vllm:num_requests_running 0\\nvllm:num_requests_waiting 0\\n')\n"
                "    if url.endswith('/staging_status'):\n"
                "        return Response(json.dumps({'status': 'idle', 'loaded_receipts': {}}).encode())\n"
                "    raise RuntimeError(url)\n\n"
                "urllib.request.urlopen = fake_urlopen\n",
                encoding="utf-8",
            )
            script = training_ops._build_adapter_cache_cleanup_script(
                {
                    **plan,
                    "cache_root": str(cache_root),
                    "cache_path": str(target),
                    "ports": [8000],
                }
            )
            docker_state = root / "docker-count"
            docker_state.write_text("0", encoding="ascii")
            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
            environment["PYTHONPATH"] = str(root)
            environment["DOCKER_STATE"] = str(docker_state)
            result = subprocess.run(
                ["bash", "-c", script],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("server identity changed", result.stderr)
            self.assertTrue(target.exists())
            self.assertTrue(other.exists())
            self.assertEqual(docker_state.read_text(encoding="ascii"), "2")

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
