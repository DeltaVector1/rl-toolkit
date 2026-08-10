import importlib.util
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPOSITORY_ROOT / "scripts" / "ops" / "test_fp8_reentrant_32k.py"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("_fp8_reentrant_32k_gate", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate_module()


class FP8Reentrant32KAccountingTest(unittest.TestCase):
    def test_default_is_32k_cp8_and_elides_owned_snapshots(self):
        shape = gate.ProbeShape()
        estimate = gate.estimate_working_set_bytes(shape, fsdp_world_size=2)

        self.assertEqual(shape.seq_len, 32768)
        self.assertEqual(shape.cp, 8)
        self.assertEqual(shape.local_tokens, 4096)
        self.assertEqual(estimate["owned_backward_snapshots"], 0)
        self.assertEqual(
            estimate["estimated_total"],
            sum(value for key, value in estimate.items() if key != "estimated_total"),
        )

    def test_estimate_grows_with_context_and_shrinks_with_more_cp(self):
        base = gate.estimate_working_set_bytes(gate.ProbeShape(), 2)
        longer = gate.estimate_working_set_bytes(
            gate.ProbeShape(seq_len=65536), 2
        )
        more_cp = gate.estimate_working_set_bytes(
            gate.ProbeShape(cp=16), 2
        )

        self.assertGreater(longer["estimated_total"], base["estimated_total"])
        self.assertLess(more_cp["estimated_total"], base["estimated_total"])

    def test_gate_rejects_reduced_context_and_misaligned_fp8_shapes(self):
        with self.assertRaisesRegex(ValueError, "below 32768"):
            gate.ProbeShape(seq_len=16384).validate()
        with self.assertRaisesRegex(ValueError, "FP8 block size"):
            gate.ProbeShape(q_rank=1500).validate()


if __name__ == "__main__":
    unittest.main()
