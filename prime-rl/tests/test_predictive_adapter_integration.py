import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src/prime_rl/inference/relay.py"
SPEC = importlib.util.spec_from_file_location("predictive_relay", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
relay = importlib.util.module_from_spec(SPEC)
sys.modules["predictive_relay"] = relay
SPEC.loader.exec_module(relay)


class PredictiveAdapterIntegrationTest(unittest.TestCase):
    def test_reader_lock_never_creates_a_missing_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache"
            cache.mkdir()
            with self.assertRaises(relay.RelayRequestError):
                with relay.reader_lock(cache):
                    pass
            self.assertFalse((cache.parent / f".{cache.name}.readers.lock").exists())

    def test_reader_lock_uses_existing_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache"
            cache.mkdir()
            lock = cache.parent / f".{cache.name}.readers.lock"
            lock.write_bytes(b"")
            with relay.reader_lock(cache):
                self.assertTrue(lock.is_file())

    def test_predictive_manifest_requires_adjacent_second_base(self):
        config = b"{}\n"
        model = b"target"
        manifest = {
            "version": 2,
            "kind": "lora-adapter",
            "mode": "delta",
            "transport": relay.predictive_transport,
            "target_step": 2,
            "base_step": 1,
            "base2_step": 0,
            "target_size": len(model),
            "target_sha256": hashlib.sha256(model).hexdigest(),
            "base_size": len(model),
            "base_sha256": "a" * 64,
            "base2_size": len(model),
            "base2_sha256": "b" * 64,
            "config_size": len(config),
            "config_sha256": hashlib.sha256(config).hexdigest(),
            "base_config_size": len(config),
            "base_config_sha256": "c" * 64,
            "base2_config_size": len(config),
            "base2_config_sha256": "d" * 64,
            "predictor": "bf16-float32-second-order-rne",
            "data_offset": 0,
            "target_url": "http://127.0.0.1:18100/run/broadcasts/step_2",
            "base_url": "http://127.0.0.1:18100/run/broadcasts/step_1",
            "base2_url": "http://127.0.0.1:18100/run/broadcasts/step_0",
            "files": [
                {
                    "name": "adapter_config.json",
                    "size": len(config),
                    "sha256": hashlib.sha256(config).hexdigest(),
                },
                {
                    "name": "adapter_model.safetensors",
                    "size": len(model),
                    "sha256": hashlib.sha256(model).hexdigest(),
                },
            ],
        }
        relay._validate_predictive_source_manifest(
            manifest, "http://127.0.0.1:18100/run/broadcasts/step_2"
        )
        manifest["base2_step"] = 1
        with self.assertRaises(relay.RelayRequestError):
            relay._validate_predictive_source_manifest(
                manifest, "http://127.0.0.1:18100/run/broadcasts/step_2"
            )


if __name__ == "__main__":
    unittest.main()
