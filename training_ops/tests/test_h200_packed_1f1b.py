import json
import struct
import tempfile
import tomllib
import unittest
from pathlib import Path

from training_ops.runtime.h200 import compare_trinity_packed_1f1b as comparison
from training_ops.tests.test_h200_synthetic_parity import response_for


class Packed1f1bTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            artifact = comparison.synthetic.build_artifact(
                endpoint="http://h200/inference/v1/generate",
                output_dir=Path(directory),
                post_json=lambda endpoint, request, timeout: response_for(request["token_ids"]),
            )
        cls.reference = artifact["packed_reference"]
        cls.record = {
            field: cls.reference[field]
            for field in ("input_ids", "position_ids", "sequence_lengths", "loss_mask")
        }
        logs = [struct.unpack("f", struct.pack("f", x))[0] for x in cls.reference["inference_logprobs"]]
        cls.record.update(reference_logprobs=logs, trainer_logprobs=logs)

    def artifact(self):
        return {
            "schema_version": 2, "microbatch_count": 8, "microbatch_order": list(range(8)),
            "microbatches": [dict(self.record, microbatch_index=i, microbatch_count=8) for i in range(8)],
        }

    def test_compares_every_copy_and_sequence(self):
        report = comparison.compare(self.artifact(), self.reference, self.record)
        self.assertEqual(report["total_positions"], 262144)
        self.assertEqual(report["total_retained"], 262104)
        self.assertEqual(len(report["microbatches"]), 8)
        for record in report["microbatches"]:
            self.assertEqual(record["trainer_vs_h200"]["count"], 32763)
            self.assertEqual(record["trainer_vs_gpipe"]["max_abs"], 0)
            self.assertEqual(len(record["sequences"]), 5)
        self.assertIsNone(report["qualification_pass"])

    def test_wrapper_fails_closed(self):
        for field, value in (
            ("schema_version", 1), ("microbatch_count", 5),
            ("microbatch_count", True), ("microbatch_order", list(reversed(range(8)))),
            ("microbatches", self.artifact()["microbatches"][:5]),
        ):
            with self.subTest(field=field, value=str(value)[:30]):
                artifact = self.artifact()
                artifact[field] = value
                with self.assertRaises(comparison.synthetic.SyntheticParityError):
                    comparison.compare(artifact, self.reference)

    def test_each_record_requires_identity_and_logprobs(self):
        for index in range(8):
            for field in ("microbatch_index", "microbatch_count", "trainer_logprobs", "reference_logprobs"):
                artifact = self.artifact()
                del artifact["microbatches"][index][field]
                with self.subTest(index=index, field=field):
                    with self.assertRaises(comparison.synthetic.SyntheticParityError):
                        comparison.compare(artifact, self.reference)

    def test_bindings_and_nonfinite_values_fail_closed(self):
        for field, value in (
            ("input_ids", 999999), ("position_ids", 100), ("sequence_lengths", 100),
            ("loss_mask", True), ("reference_logprobs", 7.0),
            ("trainer_logprobs", float("nan")), ("trainer_logprobs", float("inf")),
        ):
            artifact = self.artifact()
            artifact["microbatches"][7][field] = list(self.record[field])
            artifact["microbatches"][7][field][0] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(comparison.synthetic.SyntheticParityError):
                    comparison.compare(artifact, self.reference)

    def test_last_copy_error_is_measured(self):
        artifact = self.artifact()
        artifact["microbatches"][7]["trainer_logprobs"] = list(self.record["trainer_logprobs"])
        artifact["microbatches"][7]["trainer_logprobs"][1] += 1
        report = comparison.compare(artifact, self.reference)
        self.assertAlmostEqual(report["microbatches"][7]["trainer_vs_h200"]["max_abs"], 1)
        self.assertLess(report["microbatches"][0]["trainer_vs_h200"]["max_abs"], 0.001)

    def test_gpipe_requires_same_packed_reference(self):
        bad = dict(self.record)
        del bad["trainer_logprobs"]
        with self.assertRaises(comparison.synthetic.SyntheticParityError):
            comparison.compare(self.artifact(), self.reference, bad)
        with self.assertRaises(comparison.synthetic.SyntheticParityError):
            comparison.compare(self.artifact(), self.reference, self.artifact())

    def test_cli_binds_reference_file_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in (("reference", self.reference), ("trainer", self.artifact())):
                (root / name).write_text(json.dumps(value))
            args = ["--reference", str(root / "reference"), "--trainer", str(root / "trainer"),
                    "--output", str(root / "report")]
            self.assertEqual(comparison.main(args), 0)
            report = json.loads((root / "report").read_text())
            self.assertEqual(report["reference_file"]["size"], (root / "reference").stat().st_size)
            with self.assertRaises(SystemExit):
                comparison.main(args)

    def test_config_changes_only_schedule_batch_and_observation(self):
        root = Path(__file__).resolve().parents[2]
        old = tomllib.loads((root / "configs/trinity-h200-synthetic-packed-parity/trainer.toml").read_text())
        new = tomllib.loads((root / "configs/trinity-h200-synthetic-packed-1f1b-parity/trainer.toml").read_text())
        self.assertNotIn("name", new)
        self.assertEqual(Path(new["output_dir"]).name, "packed-1f1b-run-v2")
        self.assertEqual(new.pop("file_monitor"), {"filename": "trainer-metrics.jsonl"})
        self.assertNotEqual(new["output_dir"], old["output_dir"])
        self.assertEqual(new["data"]["fake"]["batch_size"], 8)
        self.assertEqual(new["model"]["pp_schedule"], "1f1b")
        new["output_dir"] = old["output_dir"]
        new["data"]["fake"]["batch_size"] = 1
        new["model"]["pp_schedule"] = "gpipe"
        self.assertEqual(new, old)

    def test_config_validates_with_trainer_schema_when_available(self):
        try:
            from prime_rl.configs.trainer import TrainerConfig
        except ModuleNotFoundError as error:
            if error.name not in {"prime_rl", "prime_rl.configs", "prime_rl.configs.trainer"}:
                raise
            self.skipTest("TrainerConfig schema is not installed in this local environment")
        from pydantic import ValidationError

        root = Path(__file__).resolve().parents[2]
        config = tomllib.loads(
            (root / "configs/trinity-h200-synthetic-packed-1f1b-parity/trainer.toml").read_text()
        )
        parsed = TrainerConfig.model_validate(config)
        self.assertEqual(parsed.output_dir.name, "packed-1f1b-run-v2")
        with self.assertRaises(ValidationError) as error:
            TrainerConfig.model_validate(dict(config, name="packed-1f1b-run"))
        self.assertTrue(any(
            item["loc"] == ("name",) and item["type"] == "extra_forbidden"
            for item in error.exception.errors()
        ))


if __name__ == "__main__":
    unittest.main()
