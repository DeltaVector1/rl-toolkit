import hashlib
import http.client
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/prime_rl/inference/relay.py"
SPEC = importlib.util.spec_from_file_location("adapter_relay", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
relay = importlib.util.module_from_spec(SPEC)
sys.modules["adapter_relay"] = relay
SPEC.loader.exec_module(relay)


class RelayTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_url = "http://127.0.0.1:18100/glm-1/run_glm-1/broadcasts/step_1"
        self.cache = self.root / hashlib.sha256(self.source_url.encode()).hexdigest()
        (self.cache / "transport-chunks").mkdir(parents=True)
        config = b'{"r":32}\n'
        model = b"model-bytes"
        chunk = b"compressed-delta"
        (self.cache / "adapter_manifest.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "kind": "lora-adapter",
                    "mode": "delta",
                    "transport": "chunked-shuffled-xor-zstd",
                    "target_step": 1,
                    "base_step": 0,
                    "files": [],
                }
            ),
            encoding="utf-8",
        )
        (self.cache / "adapter_config.json").write_bytes(config)
        (self.cache / "adapter_model.safetensors").write_bytes(model)
        (self.cache / "transport-chunks" / "chunk-0").write_bytes(chunk)
        (self.cache.parent / f".{self.cache.name}.readers.lock").touch()
        files = {
            "adapter_manifest.json": {
                "path": "adapter_manifest.json",
                "size": (self.cache / "adapter_manifest.json").stat().st_size,
            },
            "adapter_config.json": {
                "path": "adapter_config.json",
                "size": len(config),
                "sha256": hashlib.sha256(config).hexdigest(),
            },
            "adapter_model.safetensors": {
                "path": "adapter_model.safetensors",
                "size": len(model),
                "sha256": hashlib.sha256(model).hexdigest(),
            },
            "chunk-0": {
                "path": "transport-chunks/chunk-0",
                "size": len(chunk),
                "sha256": hashlib.sha256(chunk).hexdigest(),
            },
        }
        manifest_path = self.cache / relay.relay_manifest_name
        manifest = {
            "version": 1,
            "source_url": self.source_url,
            "transport": relay.relay_transport,
            "files": files,
        }
        manifest["files"]["adapter_manifest.json"]["sha256"] = hashlib.sha256(
            (self.cache / "adapter_manifest.json").read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        (self.cache / relay.ready_marker_name).write_text("ready\n", encoding="utf-8")
        config = relay.RelayConfig(
            self.root,
            ["http://127.0.0.1:18100/glm-1"],
            18100,
            18195,
        )
        self.server = relay.RelayServer(("127.0.0.1", 0), relay.RelayHandler, config)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.temporary.cleanup()

    def request(self, path, method="GET", headers=None):
        request = Request(
            f"http://127.0.0.1:{self.server.server_address[1]}{path}",
            method=method,
            headers={"Host": "127.0.0.1:18100", **(headers or {})},
        )
        return urlopen(request)

    def test_serves_ready_files_and_ranges(self):
        with self.request("/glm-1/run_glm-1/broadcasts/step_1/adapter_config.json") as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), b'{"r":32}\n')
        with self.request(
            "/glm-1/run_glm-1/broadcasts/step_1/chunk-0",
            headers={"Range": "bytes=2-8"},
        ) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.headers["Content-Range"], "bytes 2-8/16")
            self.assertEqual(response.read(), b"mpresse")
        with self.request(
            "/glm-1/run_glm-1/broadcasts/step_1/adapter_config.json", method="HEAD"
        ) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers["Content-Length"], "9")
            self.assertEqual(response.read(), b"")
        with self.assertRaises(HTTPError) as error:
            self.request(
                "/glm-1/run_glm-1/broadcasts/step_1/adapter_config.json",
                headers={"Range": "bytes=bad-value"},
            )
        self.assertEqual(error.exception.code, 416)

    def test_does_not_serve_unknown_or_unready_data(self):
        with self.assertRaises(HTTPError) as error:
            self.request("/glm-1/run_glm-1/broadcasts/step_1/not-published")
        self.assertEqual(error.exception.code, 404)
        (self.cache / relay.ready_marker_name).unlink()
        with self.assertRaises(HTTPError) as error:
            self.request("/glm-1/run_glm-1/broadcasts/step_1/adapter_config.json")
        self.assertEqual(error.exception.code, 404)

    def test_accepts_full_bootstrap_manifest(self):
        (self.cache / "transport-chunks" / "chunk-0").unlink()
        manifest = json.loads((self.cache / relay.relay_manifest_name).read_text(encoding="utf-8"))
        manifest["transport"] = relay.full_transport
        manifest["files"].pop("chunk-0")
        (self.cache / relay.relay_manifest_name).write_text(json.dumps(manifest), encoding="utf-8")
        selected = relay._load_file(self.cache, self.source_url, "adapter_model.safetensors")
        self.assertEqual(selected.path.read_bytes(), b"model-bytes")

    def test_predictive_relay_publishes_only_verified_in_progress_chunks(self):
        source_url = "http://127.0.0.1:18100/glm-1/run_glm-1/broadcasts/step_2"
        cache = self.root / hashlib.sha256(source_url.encode()).hexdigest()
        (cache / "transport-chunks").mkdir(parents=True)
        (cache.parent / f".{cache.name}.readers.lock").touch()
        config = b'{"r":32}\n'
        model = b"model-bytes"
        chunk = b"compressed-delta"
        model_sha = hashlib.sha256(model).hexdigest()
        config_sha = hashlib.sha256(config).hexdigest()
        chunk_sha = hashlib.sha256(chunk).hexdigest()
        source = cache / "adapter_manifest.json"
        source_manifest = {
            "version": 2,
            "kind": "lora-adapter",
            "mode": "delta",
            "transport": relay.predictive_transport,
            "target_step": 2,
            "base_step": 1,
            "base2_step": 0,
            "target_url": source_url,
            "base_url": source_url.rsplit("/", 1)[0] + "/step_1",
            "base2_url": source_url.rsplit("/", 1)[0] + "/step_0",
            "target_size": len(model),
            "base_size": len(model),
            "base2_size": len(model),
            "target_sha256": model_sha,
            "base_sha256": model_sha,
            "base2_sha256": model_sha,
            "config_size": len(config),
            "config_sha256": config_sha,
            "base_config_sha256": config_sha,
            "base2_config_sha256": config_sha,
            "files": [
                {"name": "adapter_config.json", "size": len(config), "sha256": config_sha},
                {"name": "adapter_model.safetensors", "size": len(model), "sha256": model_sha},
            ],
        }
        source.write_text(json.dumps(source_manifest), encoding="utf-8")
        manifest = {
            "version": 1,
            "source_url": source_url,
            "transport": relay.predictive_transport,
            "files": {},
        }
        manifest["files"]["adapter_manifest.json"] = {
            "path": "adapter_manifest.json",
            "size": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
        manifest["files"]["adapter_config.json"] = {
            "path": "adapter_config.json",
            "size": len(config),
            "sha256": config_sha,
        }
        manifest["files"]["adapter_model.safetensors"] = {
            "path": "adapter_model.safetensors",
            "size": len(model),
            "sha256": model_sha,
        }
        manifest["files"]["chunk-0"] = {
            "path": "transport-chunks/chunk-0",
            "size": len(chunk),
            "sha256": chunk_sha,
        }
        (cache / relay.relay_manifest_name).write_text(json.dumps(manifest), encoding="utf-8")
        (cache / "adapter_config.json").write_bytes(config)
        partial = cache.with_name(f"{cache.name}.partial")
        cache.rename(partial)
        (partial / "RELAYING").write_text("relaying\n", encoding="utf-8")
        with self.request("/glm-1/run_glm-1/broadcasts/step_2/adapter_config.json") as response:
            self.assertEqual(response.read(), config)
        with self.assertRaises(HTTPError) as error:
            self.request(
                "/glm-1/run_glm-1/broadcasts/step_2/adapter_model.safetensors"
            )
        self.assertEqual(error.exception.code, 404)
        with self.assertRaises(HTTPError) as error:
            self.request("/glm-1/run_glm-1/broadcasts/step_2/chunk-0")
        self.assertEqual(error.exception.code, 503)
        (partial / "transport-chunks/chunk-0").write_bytes(chunk)
        wire_proof = {
            "type": "wire-ready",
            "file": "chunk-0",
            "wire_size": len(chunk),
            "wire_sha256": chunk_sha,
        }
        (partial / "transport-chunks/chunk-0.wire-ready").write_text(
            json.dumps(wire_proof), encoding="utf-8"
        )
        with self.request("/glm-1/run_glm-1/broadcasts/step_2/chunk-0") as response:
            self.assertEqual(response.read(), chunk)
        (partial / "transport-chunks/chunk-0.wire-ready").write_text(
            json.dumps({**wire_proof, "wire_sha256": "0" * 64}), encoding="utf-8"
        )
        with self.assertRaises(HTTPError) as error:
            self.request("/glm-1/run_glm-1/broadcasts/step_2/chunk-0")
        self.assertEqual(error.exception.code, 503)
        proof = {
            "file": "chunk-0",
            "wire_size": len(chunk),
            "wire_sha256": chunk_sha,
            "output_sha256": "0" * 64,
        }
        (partial / "transport-chunks/chunk-0.done").write_text(
            json.dumps(proof), encoding="utf-8"
        )
        with self.request("/glm-1/run_glm-1/broadcasts/step_2/chunk-0") as response:
            self.assertEqual(response.read(), chunk)

    def test_reloads_index_when_manifest_is_replaced(self):
        selected = relay._load_file(
            self.cache, self.source_url, "adapter_config.json"
        )
        self.assertEqual(selected.path.read_bytes(), b'{"r":32}\n')
        manifest = json.loads(
            (self.cache / relay.relay_manifest_name).read_text(encoding="utf-8")
        )
        manifest["transport"] = relay.full_transport
        manifest["files"].pop("chunk-0")
        (self.cache / relay.relay_manifest_name).write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        selected = relay._load_file(
            self.cache, self.source_url, "adapter_model.safetensors"
        )
        self.assertEqual(selected.path.read_bytes(), b"model-bytes")

    def test_reloads_index_when_partial_becomes_ready(self):
        partial = self.cache.with_name(f"{self.cache.name}.partial")
        self.cache.rename(partial)
        (partial / relay.ready_marker_name).unlink()
        (partial / relay.relaying_marker_name).write_text("relaying\n", encoding="utf-8")
        selected = relay._load_file(
            self.cache, self.source_url, "adapter_config.json"
        )
        self.assertEqual(selected.path.read_bytes(), b'{"r":32}\n')
        (partial / relay.relaying_marker_name).unlink()
        (partial / relay.ready_marker_name).write_text("ready\n", encoding="utf-8")
        partial.rename(self.cache)
        selected = relay._load_file(
            self.cache, self.source_url, "adapter_model.safetensors"
        )
        self.assertEqual(selected.path.read_bytes(), b"model-bytes")


if __name__ == "__main__":
    unittest.main()
