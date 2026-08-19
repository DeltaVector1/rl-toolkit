import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "fp8_parity", ROOT / "scripts" / "fp8_parity.py"
)
fp8_parity = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = fp8_parity
SPEC.loader.exec_module(fp8_parity)


def _write_safetensors(path, tensors):
    offset = 0
    header = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        payload.extend(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class FP8ParityHarnessTest(unittest.TestCase):
    def test_fixture_is_reproducible_and_checksum_protected(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            fp8_parity.make_fixture(first, length=17, vocab_size=2048, seed=9)
            fp8_parity.make_fixture(second, length=17, vocab_size=2048, seed=9)
            self.assertEqual(
                np.load(first / "input_ids.npy").tolist(),
                np.load(second / "input_ids.npy").tolist(),
            )
            self.assertEqual(
                fp8_parity.fixture_digest(first),
                fp8_parity.fixture_digest(second),
            )
            data = bytearray((first / "input_ids.npy").read_bytes())
            data[-1] ^= 1
            (first / "input_ids.npy").write_bytes(data)
            with self.assertRaisesRegex(ValueError, "checksum"):
                fp8_parity.fixture_digest(first)

    def test_trace_comparison_handles_numeric_and_index_tensors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = dict(fixture_sha256="fixture", model_revision="revision")
            with fp8_parity.TraceWriter(root / "ada", side="ada", **common) as writer:
                writer.record(
                    "layer.02.output",
                    np.array([[1.0, 2.0], [3.0, 4.0]], np.float32),
                    rows=[1],
                )
                writer.record("layer.02.router.ids", [[1, 7]], semantic="indices")
            with fp8_parity.TraceWriter(root / "h200", side="h200", **common) as writer:
                writer.record(
                    "layer.02.output",
                    np.array([[99.0, 99.0], [3.0, 5.0]], np.float32),
                    rows=[1],
                )
                writer.record("layer.02.router.ids", [[1, 8]], semantic="indices")

            results = {
                result.name: result
                for result in fp8_parity.compare_traces(
                    root / "ada", root / "h200"
                )
            }
            self.assertAlmostEqual(
                results["layer.02.router.ids"].exact_fraction, 0.5
            )
            self.assertTrue(results["layer.02.output"].finite)
            self.assertGreater(results["layer.02.output"].relative_rmse, 0)
            # The unsampled first row cannot affect the comparison.
            self.assertEqual(results["layer.02.output"].shape, [1, 2])

    def test_trace_rejects_incompatible_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with fp8_parity.TraceWriter(
                root / "ada",
                side="ada",
                fixture_sha256="a",
                model_revision="revision",
            ):
                pass
            with fp8_parity.TraceWriter(
                root / "h200",
                side="h200",
                fixture_sha256="b",
                model_revision="revision",
            ):
                pass
            with self.assertRaisesRegex(ValueError, "fixture_sha256"):
                fp8_parity.compare_traces(root / "ada", root / "h200")

    def test_checkpoint_audit_accepts_partial_final_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            weight_name = "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
            scale_name = (
                "model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv"
            )
            # The payload is small; the header shape is what the zero-allocation
            # audit checks.  Hashing uses the declared byte offsets.
            _write_safetensors(
                model / "model.safetensors",
                {
                    weight_name: (
                        "F8_E4M3",
                        [576, 6144],
                        bytes(576 * 6144),
                    ),
                    scale_name: ("F32", [5, 48], bytes(5 * 48 * 4)),
                },
            )
            (model / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            weight_name: "model.safetensors",
                            scale_name: "model.safetensors",
                        }
                    }
                )
            )
            report = fp8_parity.audit_checkpoint(model)
            self.assertEqual(report["failures"], [])
            self.assertEqual(report["fp8_weight_count"], 1)
            self.assertFalse(report["weights"][0]["native_128x128"])


if __name__ == "__main__":
    unittest.main()
