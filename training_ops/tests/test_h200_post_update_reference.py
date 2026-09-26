import base64
import ast
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import training_ops
from training_ops.runtime.h200.post_update_qualification import (
    PostUpdateQualificationError,
    build_post_update_reference,
    build_post_update_references,
    validate_post_update_reference,
    validate_post_update_references,
)


ROOT = Path(__file__).resolve().parents[2]
SEQ_LEN = 8192
PROMPT_TOKENS = 8160
COMPLETION_TOKENS = 32
ROUTE_SHAPE = [8191, 60, 4]


def _route_response():
    raw = bytearray(ROUTE_SHAPE[0] * ROUTE_SHAPE[1] * ROUTE_SHAPE[2])
    row_width = ROUTE_SHAPE[1] * ROUTE_SHAPE[2]
    for row in range(ROUTE_SHAPE[0]):
        for layer in range(6, ROUTE_SHAPE[1]):
            offset = row * row_width + layer * ROUTE_SHAPE[2]
            raw[offset : offset + 4] = bytes((layer, layer + 1, layer + 2, layer + 3))
    return {
        "data": base64.b64encode(raw).decode(),
        "dtype": "uint8",
        "shape": ROUTE_SHAPE,
        "start": 0,
    }


class H200PostUpdateReferenceTest(unittest.TestCase):
    def receipt(self):
        return {
            "target_sha256": "a" * 64,
            "target_size": 100,
            "config_sha256": "b" * 64,
            "config_size": 200,
            "source_url": "http://h200b:18100/post-update/step_0",
            "lora_name": "policy-0-slot0",
            "server_instance": "h200b-instance-1",
        }

    def input_record(self):
        return {
            "source_kind": "operator_non_rollout",
            "input_ids": list(range(PROMPT_TOKENS)),
            "sequence_lengths": [PROMPT_TOKENS],
        }

    def response(self):
        generated = list(range(10000, 10032))
        return {
            "choices": [{
                "token_ids": generated,
                "finish_reason": "length",
                "logprobs": {
                    "content": [
                        {"token": f"token_id:{token}", "logprob": -0.25}
                        for token in generated
                    ]
                },
                "routed_experts": _route_response(),
            }]
        }

    def build(self, output: Path):
        receipt = self.receipt()
        response = self.response()
        return build_post_update_reference(
            endpoint="http://h200b:18100/inference/v1/generate",
            input_record=self.input_record(),
            output_path=output,
            base_model="/data/trinity-thinking-fp8",
            policy_model="policy-0-slot0",
            expected_policy_identity=receipt,
            policy_receipt=receipt,
            cache_salt="fresh-post-update-decode",
            readiness_endpoint="http://h200b:18100/inference/v1/staging-status",
            post_json=lambda *_: response,
            get_json=lambda *_: receipt,
        )

    def test_builder_emits_one_decode_reference_with_aligned_advantages(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.build(Path(directory) / "reference.json")
        self.assertEqual(result["request_mode"], "serialized_generate_completion_logprobs")
        self.assertEqual(len(result["input_ids"]), SEQ_LEN)
        self.assertEqual(result["sequence_lengths"], [SEQ_LEN])
        self.assertEqual(result["loss_mask"], [False] * PROMPT_TOKENS + [True] * COMPLETION_TOKENS)
        self.assertEqual(result["advantages"], [0.0] * PROMPT_TOKENS + [1.0] * COMPLETION_TOKENS)
        self.assertEqual(result["qualification_binding"]["route_shape"], ROUTE_SHAPE)
        self.assertNotIn("prefill_rescore_comparison", result)
        self.assertEqual(result["qualification_binding"]["advantage_value"], 1.0)

    def test_validator_binds_config_receipt_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.build(Path(directory) / "reference.json")
        self.assertEqual(
            validate_post_update_reference(
                result,
                expected_seq_len=SEQ_LEN,
                expected_policy_identity=self.receipt(),
            ),
            result,
        )
        for field, value in (
            ("advantages", [0.0] * SEQ_LEN),
            ("request_sha256", "0" * 64),
            ("response_sha256", "0" * 64),
        ):
            changed = copy.deepcopy(result)
            if field in {"request_sha256", "response_sha256"}:
                changed["qualification_binding"][field] = value
            else:
                changed[field] = value
            with self.subTest(field=field), self.assertRaises(PostUpdateQualificationError):
                validate_post_update_reference(
                    changed,
                    expected_seq_len=SEQ_LEN,
                    expected_policy_identity=self.receipt(),
                )
        for label, mutate in (
            ("under_length", lambda item: item["input_ids"].pop()),
            ("over_length", lambda item: item["input_ids"].append(1)),
            ("mask", lambda item: item["loss_mask"].__setitem__(0, True)),
            ("rescore", lambda item: item.__setitem__("request_mode", "serialized_generate_decode256_prefill_rescore")),
            ("stale_receipt", lambda item: item["policy_identity_after"].__setitem__("server_instance", "old")),
            ("response", lambda item: item["responses"]["decode"]["choices"][0].__setitem__("finish_reason", "stop")),
        ):
            changed = copy.deepcopy(result)
            mutate(changed)
            with self.subTest(label=label), self.assertRaises(PostUpdateQualificationError):
                validate_post_update_reference(
                    changed,
                    expected_seq_len=SEQ_LEN,
                    expected_policy_identity=self.receipt(),
                )

    def test_eight_microbatches_are_distinct_actual_decode_references(self):
        records = []
        responses = []
        for index in range(8):
            record = self.input_record()
            record["input_ids"] = [token + index * 10000 for token in record["input_ids"]]
            records.append(record)
            response = self.response()
            response["choices"][0]["token_ids"] = [10000 + index * 100 + token for token in range(32)]
            response["choices"][0]["logprobs"]["content"] = [
                {"token": f"token_id:{token}", "logprob": -0.25}
                for token in response["choices"][0]["token_ids"]
            ]
            responses.append(response)
        with tempfile.TemporaryDirectory() as directory:
            result = build_post_update_references(
                endpoint="http://h200b:18100/inference/v1/generate",
                input_records=records,
                output_path=Path(directory) / "reference.json",
                base_model="/data/trinity-thinking-fp8",
                policy_model="policy-0-slot0",
                expected_policy_identity=self.receipt(),
                policy_receipt=self.receipt(),
                cache_salt="fresh-post-update-decode",
                readiness_endpoint="http://h200b:18100/inference/v1/staging-status",
                post_json=lambda *_: responses.pop(0),
                get_json=lambda *_: self.receipt(),
            )
        self.assertEqual(len(result["microbatches"]), 8)
        self.assertEqual(
            len({item["qualification_binding"]["input_sha256"] for item in result["microbatches"]}),
            8,
        )
        self.assertEqual(
            len({item["qualification_binding"]["response_sha256"] for item in result["microbatches"]}),
            8,
        )
        validate_post_update_references(
            result,
            expected_seq_len=SEQ_LEN,
            expected_policy_identity=self.receipt(),
        )

    def test_fake_loader_source_consumes_reference_advantages(self):
        source = (ROOT / "prime-rl/src/prime_rl/trainer/rl/data.py").read_text()
        tree = ast.parse(source)
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_get_reference_micro_batch"
        )
        function.decorator_list = []
        function.returns = None
        for argument in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs):
            argument.annotation = None

        class FakeTensor:
            def __init__(self, values):
                self.values = list(values)

            @property
            def shape(self):
                return (len(self.values),)

            def unsqueeze(self, _dimension):
                return FakeTensor([self.values])

            def __setitem__(self, index, value):
                self.values[index] = value

        class FakeTorch:
            long = "long"
            float = "float"
            bool = "bool"
            int32 = "int32"

            @staticmethod
            def tensor(values, dtype=None):
                return FakeTensor(values)

            @staticmethod
            def arange(length):
                return FakeTensor(range(length))

            @staticmethod
            def cat(values):
                flattened = []
                for value in values:
                    flattened.extend(value.values)
                return FakeTensor(flattened)

            @staticmethod
            def zeros(length, dtype=None):
                return FakeTensor([0] * length)

            @staticmethod
            def ones_like(value):
                return FakeTensor([1] * len(value.values))

        namespace = {"torch": FakeTorch()}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "data.py", "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            reference = self.build(Path(directory) / "reference.json")
        reference["_routed_experts_tensor"] = None
        loader = SimpleNamespace(
            multi_run_manager=SimpleNamespace(max_runs=1),
        )
        batch = namespace["_get_reference_micro_batch"](
            loader,
            reference,
        )
        self.assertEqual(
            batch["advantages"].values[0],
            [0.0] * PROMPT_TOKENS + [1.0] * COMPLETION_TOKENS,
        )

    def test_qualify_validates_post_update_reference_before_ssh(self):
        events = []
        with mock.patch.object(
            training_ops,
            "_validate_post_update_qualification_reference",
            side_effect=lambda config, path: events.append("reference"),
        ), mock.patch.object(
            training_ops,
            "_trinity_trainer_loss_preflight",
            side_effect=lambda nodes: events.append("image"),
        ), mock.patch.object(training_ops, "_ssh", return_value=(1, "unexpected ssh")):
            with self.assertRaises(training_ops.Refused):
                training_ops.qualify(
                    "post-update-reference-test",
                    nodes=["n82"],
                    config_dir="configs/trinity-h200-post-update-qualification",
                )
        self.assertEqual(events, ["reference", "image"])


if __name__ == "__main__":
    unittest.main()
