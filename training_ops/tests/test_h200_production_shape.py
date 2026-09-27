import ast
import base64
import binascii
import hashlib
import json
import math
import struct
import tempfile
import tomllib
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from training_ops.runtime.h200 import trinity_production_shape as tool
from training_ops.tests.test_h200_synthetic_parity import response_for

root = Path(__file__).resolve().parents[2]


def source_objects(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


class ProductionShapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.requests = []

        def post(endpoint, request, timeout):
            cls.requests.append(request)
            return response_for(request["token_ids"])

        cls.reference = tool.build(
            endpoint="http://localhost/inference/v1/generate", post_json=post
        )

    def artifact(self):
        records = []
        for index, ref in enumerate(self.reference["microbatches"]):
            expected = tool.padded(ref)
            logs = [
                struct.unpack("f", struct.pack("f", x))[0]
                for x in expected["inference_logprobs"]
            ]
            records.append(
                dict(
                    expected,
                    reference_logprobs=logs,
                    trainer_logprobs=logs,
                    microbatch_index=index,
                    microbatch_count=8,
                )
            )
        return dict(
            schema_version=2,
            microbatch_count=8,
            microbatch_order=list(range(8)),
            microbatches=records,
        )

    def test_layout_requests_and_comparison(self):
        self.assertEqual(
            [len(r["token_ids"]) for r in self.requests],
            [8187, 7006, 6543, 6436, 4530, 32768],
        )
        self.assertEqual(len({r["cache_salt"] for r in self.requests}), 6)
        self.assertTrue(
            all(r["model"] == tool.synthetic.DEFAULT_BASE_MODEL for r in self.requests)
        )
        self.assertTrue(all("lora" not in r for r in self.requests))
        report = tool.compare(self.artifact(), self.reference)
        self.assertEqual(report["production_selected"], 228879)
        self.assertEqual(report["control_selected"], 32767)
        self.assertIsNone(report["qualification_pass"])

    def test_rejects_mutations(self):
        for field, index, value in [
            ("input_ids", 32767, 0),
            ("position_ids", 32767, 4595),
            ("loss_mask", 32767, True),
            ("reference_logprobs", 32767, -1),
            ("trainer_logprobs", 50, float("nan")),
        ]:
            artifact = self.artifact()
            artifact["microbatches"][0][field][index] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                tool.compare(artifact, self.reference)
        bad = deepcopy(self.reference)
        bad["microbatches"][7]["input_ids"][20] += 1
        with self.assertRaises(ValueError):
            tool.compare(self.artifact(), bad)
        artifact = self.artifact()
        artifact["microbatch_order"][-1] = 6
        with self.assertRaises(ValueError):
            tool.compare(artifact, self.reference)

    def test_loader_then_actual_trainer_padding(self):
        import torch

        namespace = dict(
            torch=torch,
            json=json,
            math=math,
            base64=base64,
            binascii=binascii,
            hashlib=hashlib,
            Tensor=torch.Tensor,
            TensorMicroBatch=dict,
            FakeDataLoaderConfig=object,
            get_world=lambda: None,
            get_multi_run_manager=lambda: SimpleNamespace(max_runs=1),
            _REFERENCE_ROUTE_FIELDS=set(
                self.reference["microbatches"][0]["routed_experts"]
            ),
            _MAX_REFERENCE_ROUTE_BYTES=512 * 1024 * 1024,
        )
        source_objects(
            root / "prime-rl/src/prime_rl/trainer/rl/data.py",
            {"FakeDataLoader", "_decode_reference_routed_experts"},
            namespace,
        )
        source_objects(
            root / "prime-rl/src/prime_rl/trainer/rl/train.py",
            {"pad_micro_batch_to_length"},
            namespace,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text(json.dumps(self.reference))
            config = SimpleNamespace(
                batch_size=8,
                generate_samples=False,
                ce_only=False,
                reference_path=path,
                export_final_adapter=False,
                allow_variable_length=True,
            )
            loader = namespace["FakeDataLoader"](config, 32768, 1, 0)
            batches = loader.get_batch()
            for index, batch in enumerate(batches):
                before = batch["routed_experts"].clone()
                namespace["pad_micro_batch_to_length"](batch, 32768)
                expected = tool.padded(self.reference["microbatches"][index])
                for field in (
                    "input_ids",
                    "position_ids",
                    "loss_mask",
                    "inference_logprobs",
                ):
                    self.assertEqual(
                        batch[field].tolist()[0],
                        torch.tensor(
                            expected[field], dtype=batch[field].dtype
                        ).tolist(),
                    )
                self.assertEqual(
                    batch["sequence_lengths"], expected["sequence_lengths"]
                )
                self.assertTrue(
                    torch.equal(batch["routed_experts"][:, : before.shape[1]], before)
                )
                if index < 7:
                    self.assertEqual(
                        torch.count_nonzero(batch["routed_experts"][:, -66:]).item(), 0
                    )
                # MoE replay detects padding when every top-k ID is equal.
                routes = batch["routed_experts"][0, :, 6, :]
                route_padding = routes[:, 1:].eq(routes[:, :1]).all(dim=1)
                self.assertEqual(route_padding.sum().item(), 66 if index < 7 else 0)
            config.allow_variable_length = False
            with self.assertRaises(ValueError):
                namespace["FakeDataLoader"](config, 32768, 1, 0)
            config.allow_variable_length = True
            for replacement in (
                dict(self.reference, schema_version=1),
                dict(self.reference, schema_version=2.0),
                dict(self.reference, microbatches=self.reference["microbatches"][:1]),
                dict(
                    self.reference, microbatches=[self.reference["microbatches"][0]] * 8
                ),
            ):
                path.write_text(json.dumps(replacement))
                with self.assertRaises(ValueError):
                    namespace["FakeDataLoader"](config, 32768, 1, 0)
            for field, value in (
                ("input_ids", []),
                ("input_ids", [1] * 32769),
                ("sequence_lengths", [True, 32701]),
                ("loss_mask", [1] * 32702),
                ("inference_logprobs", [float("nan")] * 32702),
            ):
                malformed = deepcopy(self.reference)
                malformed["microbatches"][0][field] = value
                path.write_text(json.dumps(malformed))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    namespace["FakeDataLoader"](config, 32768, 1, 0)
            # Schema-v1 exact-length input retains its old behavior.
            config.allow_variable_length = False
            path.write_text(
                json.dumps(dict(self.reference["microbatches"][7], schema_version=1))
            )
            self.assertEqual(
                len(namespace["FakeDataLoader"](config, 32768, 1, 0).get_batch()), 8
            )

    def test_config_schema(self):
        from prime_rl.configs.trainer import FakeDataLoaderConfig, TrainerConfig

        path = (
            root / "configs/trinity-h200-synthetic-production-shape-parity/trainer.toml"
        )
        parsed = TrainerConfig.model_validate(tomllib.loads(path.read_text()))
        self.assertTrue(parsed.data.fake.allow_variable_length)
        self.assertEqual(parsed.data.fake.batch_size, 8)
        self.assertEqual(parsed.output_dir.name, "production-shape-run")
        self.assertIsNone(parsed.ckpt)
        self.assertIsNone(parsed.model.lora.initial_adapter_path)
        self.assertFalse(parsed.data.fake.export_final_adapter)
        self.assertEqual((parsed.model.cp, parsed.model.pp, parsed.model.ep), (8, 8, 8))
        self.assertEqual(parsed.model.seq_len % parsed.model.cp, 0)
        self.assertEqual(parsed.model.pp_schedule, "1f1b")
        self.assertEqual(parsed.optim.lr, 0)
        for kwargs in (
            {"allow_variable_length": True},
            {"allow_variable_length": "true", "reference_path": "x"},
            {
                "allow_variable_length": True,
                "reference_path": "x",
                "generate_samples": True,
            },
            {"allow_variable_length": True, "reference_path": "x", "ce_only": True},
        ):
            with self.assertRaises(ValueError):
                FakeDataLoaderConfig.model_validate(kwargs)
        bad = tomllib.loads(path.read_text())
        bad["model"]["pp"] = 1
        with self.assertRaises(ValueError):
            TrainerConfig.model_validate(bad)

    def test_no_clobber_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reference.json"
            output.write_text("preserve")
            with patch.object(
                tool.synthetic, "_post_json", side_effect=AssertionError("network")
            ):
                with self.assertRaises(SystemExit):
                    tool.main(
                        [
                            "build",
                            "--endpoint",
                            "http://localhost/inference/v1/generate",
                            "--output",
                            str(output),
                        ]
                    )
            self.assertEqual(output.read_text(), "preserve")

    def test_bad_response_routes_and_scores(self):
        for response in ({}, {"prompt_logprobs": [], "choices": [{}]}):
            with self.assertRaises(ValueError):
                tool.build(
                    endpoint="http://localhost/inference/v1/generate",
                    post_json=lambda *args: response,
                )


if __name__ == "__main__":
    unittest.main()
