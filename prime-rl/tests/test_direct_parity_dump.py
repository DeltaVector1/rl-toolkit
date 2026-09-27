import ast
import json
from pathlib import Path
import tempfile
import unittest


TRAIN_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/prime_rl/trainer/rl/train.py"
)


def _load_direct_parity_helpers():
    tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"))
    wanted = {
        "_capture_direct_parity_record",
        "_write_direct_parity_artifact",
    }
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {"Path": Path, "json": json}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(TRAIN_PATH), "exec"), namespace)
    return namespace


class DirectParityDumpTest(unittest.TestCase):
    def test_five_records_are_indexed_and_written_in_order(self):
        namespace = _load_direct_parity_helpers()
        records = {}
        capture = namespace["_capture_direct_parity_record"]
        write = namespace["_write_direct_parity_artifact"]

        for index in (3, 1, 4, 0, 2):
            capture(
                records,
                {"input_ids": [index], "trainer_logprobs": [float(index)]},
                microbatch_index=index,
                microbatch_count=5,
            )

        with tempfile.TemporaryDirectory() as directory:
            path = write(Path(directory), records, microbatch_count=5)
            self.assertEqual(path, Path(directory) / "direct-parity.json")
            artifact = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(artifact["schema_version"], 2)
        self.assertEqual(artifact["microbatch_count"], 5)
        self.assertEqual(artifact["microbatch_order"], [0, 1, 2, 3, 4])
        self.assertEqual(
            [record["microbatch_index"] for record in artifact["microbatches"]],
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            [record["input_ids"] for record in artifact["microbatches"]],
            [[0], [1], [2], [3], [4]],
        )
        self.assertTrue(
            all(record["microbatch_count"] == 5 for record in artifact["microbatches"])
        )

    def test_one_microbatch_keeps_the_legacy_object(self):
        namespace = _load_direct_parity_helpers()
        records = {}
        payload = {
            "input_ids": [1, 2],
            "position_ids": [0, 1],
            "sequence_lengths": [2],
            "loss_mask": [True, False],
            "trainer_logprobs": [-1.0, -2.0],
            "reference_logprobs": [-1.1, -2.1],
            "delta": [0.1, 0.1],
        }
        namespace["_capture_direct_parity_record"](
            records, payload, microbatch_index=0, microbatch_count=1
        )

        with tempfile.TemporaryDirectory() as directory:
            path = namespace["_write_direct_parity_artifact"](
                Path(directory), records, microbatch_count=1
            )
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)

    def test_duplicate_missing_and_out_of_range_records_fail_closed(self):
        namespace = _load_direct_parity_helpers()
        capture = namespace["_capture_direct_parity_record"]
        records = {}
        capture(records, {"input_ids": [0]}, microbatch_index=0, microbatch_count=2)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            capture(records, {"input_ids": [0]}, microbatch_index=0, microbatch_count=2)
        with self.assertRaisesRegex(RuntimeError, "outside"):
            capture(records, {"input_ids": [2]}, microbatch_index=2, microbatch_count=2)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "missing"):
                namespace["_write_direct_parity_artifact"](
                    Path(directory), records, microbatch_count=2
                )

    def test_existing_artifact_is_not_overwritten(self):
        namespace = _load_direct_parity_helpers()
        records = {}
        namespace["_capture_direct_parity_record"](
            records, {"input_ids": [0]}, microbatch_index=0, microbatch_count=1
        )
        with tempfile.TemporaryDirectory() as directory:
            path = namespace["_write_direct_parity_artifact"](
                Path(directory), records, microbatch_count=1
            )
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                namespace["_write_direct_parity_artifact"](
                    Path(directory), records, microbatch_count=1
                )
            self.assertEqual(json.loads(path.read_text()), {"input_ids": [0]})

    def test_production_dump_stays_disabled_without_parity_guard(self):
        source = TRAIN_PATH.read_text(encoding="utf-8")
        start = source.index("production_dump =")
        expression = source[start : source.index("if (", start)]
        self.assertIn("config.parity_guard is not None", expression)


if __name__ == "__main__":
    unittest.main()
