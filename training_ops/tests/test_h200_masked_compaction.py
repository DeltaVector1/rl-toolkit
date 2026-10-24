import hashlib
import json
import struct
import tempfile
import tomllib
import unittest
from copy import deepcopy
from pathlib import Path

from training_ops.runtime.h200 import trinity_masked_compaction as tool
from training_ops.tests import test_h200_packed_1f1b as packed_tests


class MaskedCompactionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        packed_tests.Packed1f1bTest.setUpClass()
        cls.directory = tempfile.TemporaryDirectory()
        cls.source = Path(cls.directory.name) / "source.json"
        cls.source.write_text(json.dumps(packed_tests.Packed1f1bTest.reference))
        cls.size = cls.source.stat().st_size
        cls.sha = hashlib.sha256(cls.source.read_bytes()).hexdigest()
        cls.reference = tool.derive(cls.source, cls.sha, cls.size)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def artifact(self, dense=False):
        ref = packed_tests.Packed1f1bTest.reference if dense else self.reference
        record = {
            k: ref[k]
            for k in ("input_ids", "position_ids", "sequence_lengths", "loss_mask")
        }
        logs = [
            struct.unpack("f", struct.pack("f", x))[0]
            for x in ref["inference_logprobs"]
        ]
        record.update(reference_logprobs=logs, trainer_logprobs=logs)
        return {
            "schema_version": 2,
            "microbatch_count": 8,
            "microbatch_order": list(range(8)),
            "microbatches": [
                dict(record, microbatch_index=i, microbatch_count=8) for i in range(8)
            ],
        }

    def test_exact_counts_and_all_fields_preserved(self):
        self.assertEqual(sum(self.reference["loss_mask"]), 29420)
        shifted = self.reference["loss_mask"][1:] + [False]
        self.assertEqual(
            [sum(shifted[i : i + 4096]) for i in range(0, 32768, 4096)],
            [3686, 3534, 3430, 4095, 3329, 3583, 3739, 4024],
        )
        for key, value in packed_tests.Packed1f1bTest.reference.items():
            if key != "loss_mask":
                self.assertEqual(self.reference[key], value)
        report = tool.compare(
            self.artifact(), self.reference, self.source, self.artifact(dense=True)
        )
        self.assertEqual(report["total_retained"], 235360)
        self.assertIsNone(report["qualification_pass"])
        for record in report["microbatches"]:
            self.assertEqual(record["trainer_vs_dense"]["max_abs"], 0)
            self.assertEqual(
                [s["trainer_vs_h200"]["count"] for s in record["sequences"]],
                [3686, 5069, 5990, 6912, 7763],
            )

    def test_source_hash_size_and_source_validation(self):
        for sha, size in (("0" * 64, self.size), (self.sha, self.size + 1)):
            with self.assertRaises(ValueError):
                tool.derive(self.source, sha, size)
        for field, value in (
            ("input_ids", []),
            ("routed_experts", {}),
            ("source_kind", "cached"),
            ("inference_logprobs", []),
            ("loss_mask", []),
        ):
            source = dict(packed_tests.Packed1f1bTest.reference, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                tool.validate_source(source)

    def test_reference_rejects_count_prefix_tail_tokens_routes_scores_and_metadata(
        self,
    ):
        for field, index, value in (
            ("loss_mask", 410, False),
            ("loss_mask", 0, True),
            ("loss_mask", 32767, True),
            ("input_ids", 0, 17),
            ("inference_logprobs", 20, 4.0),
        ):
            ref = deepcopy(self.reference)
            ref[field][index] = value
            with self.subTest(field=field, index=index), self.assertRaises(ValueError):
                tool.compare(self.artifact(), ref, self.source)
        for field, value in (("routed_experts", {}), ("mask_derivation", {})):
            with self.subTest(field=field), self.assertRaises(ValueError):
                tool.compare(
                    self.artifact(), dict(self.reference, **{field: value}), self.source
                )
        ref = deepcopy(self.reference)
        ref["loss_mask"][409] = True
        ref["loss_mask"][410] = False
        with self.assertRaises(ValueError):
            tool.compare(self.artifact(), ref, self.source)

    def test_record_binding_order_missing_and_nonfinite(self):
        for field in (
            "schema_version",
            "microbatch_count",
            "microbatch_order",
            "microbatches",
        ):
            artifact = self.artifact()
            del artifact[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                tool.compare(artifact, self.reference, self.source)
        artifact = self.artifact()
        artifact["microbatch_order"].reverse()
        with self.assertRaises(ValueError):
            tool.compare(artifact, self.reference, self.source)
        for field in (
            "input_ids",
            "position_ids",
            "sequence_lengths",
            "loss_mask",
            "reference_logprobs",
            "trainer_logprobs",
            "microbatch_index",
            "microbatch_count",
        ):
            artifact = self.artifact()
            del artifact["microbatches"][7][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                tool.compare(artifact, self.reference, self.source)
        for field, value in (
            ("input_ids", 17),
            ("reference_logprobs", 3.0),
            ("trainer_logprobs", float("nan")),
            ("trainer_logprobs", float("inf")),
            ("trainer_logprobs", 1000.0),
        ):
            artifact = deepcopy(self.artifact())
            artifact["microbatches"][7][field][410] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                tool.compare(artifact, self.reference, self.source)

    def test_dense_mask_and_scores_must_match_original(self):
        with self.assertRaises(ValueError):
            tool.compare(self.artifact(), self.reference, self.source, self.artifact())
        artifact = deepcopy(self.artifact(dense=True))
        artifact["microbatches"][7]["reference_logprobs"][410] += 1
        with self.assertRaises(ValueError):
            tool.compare(self.artifact(), self.reference, self.source, artifact)

    def test_cli_deterministic_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = [Path(directory) / str(i) for i in range(2)]
            for output in outputs:
                args = [
                    "derive",
                    "--source",
                    str(self.source),
                    "--source-sha256",
                    self.sha,
                    "--source-size",
                    str(self.size),
                    "--output",
                    str(output),
                ]
                self.assertEqual(tool.main(args), 0)
                with self.assertRaises(SystemExit):
                    tool.main(args)
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
            trainer = Path(directory) / "trainer.json"
            trainer.write_text(json.dumps(self.artifact()))
            args = [
                "compare",
                "--source",
                str(self.source),
                "--reference",
                str(outputs[0]),
                "--trainer",
                str(trainer),
                "--output",
                str(Path(directory) / "report.json"),
            ]
            self.assertEqual(tool.main(args), 0)
            with self.assertRaises(SystemExit):
                tool.main(args)

    def test_config_exact_baseline_and_actual_trainer_schema(self):
        from prime_rl.configs.trainer import TrainerConfig

        root = Path(__file__).resolve().parents[2]

        def config(name):
            return tomllib.loads((root / "configs" / name / "trainer.toml").read_text())

        baseline = config("trinity-h200-synthetic-packed-1f1b-zero-lora-parity")
        new = config("trinity-h200-synthetic-packed-1f1b-masked-parity")
        parsed = TrainerConfig.model_validate(new)
        self.assertEqual(parsed.output_dir.name, "packed-1f1b-masked-run")
        self.assertIsNone(parsed.ckpt)
        self.assertIsNone(parsed.model.lora.initial_adapter_path)
        self.assertFalse(parsed.data.fake.export_final_adapter)
        new["output_dir"] = baseline["output_dir"]
        new["data"]["fake"]["reference_path"] = baseline["data"]["fake"][
            "reference_path"
        ]
        self.assertEqual(new, baseline)


if __name__ == "__main__":
    unittest.main()
