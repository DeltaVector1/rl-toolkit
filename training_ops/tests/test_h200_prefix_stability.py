import base64
import copy
import importlib.util
import json
import math
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "h200_prefix_stability",
    ROOT / "training_ops/runtime/h200/prefix_stability.py",
)
assert SPEC and SPEC.loader
prefix_stability = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prefix_stability)

BUILD_SPEC = importlib.util.spec_from_file_location(
    "trinity_parity_reference",
    ROOT / "training_ops/runtime/h200/build_trinity_parity_reference.py",
)
assert BUILD_SPEC and BUILD_SPEC.loader
reference_builder = importlib.util.module_from_spec(BUILD_SPEC)
BUILD_SPEC.loader.exec_module(reference_builder)


def route_for(token_count: int) -> dict:
    raw = bytearray()
    for index in range(token_count):
        raw.extend((0,) * 12)
        raw.extend((1 + index % 2, 2 + index % 2, 2 + index % 2, 3 + index % 2))
    return {
        "data": base64.b64encode(raw).decode(),
        "dtype": "uint8",
        "shape": [token_count, 8, 2],
        "start": 0,
    }


class H200PrefixStabilityTest(unittest.TestCase):
    def receipt(self):
        return {
            "target_sha256": "a" * 64,
            "target_size": 123,
            "config_sha256": "b" * 64,
            "config_size": 456,
            "source_url": "http://engine:18100/policy-0",
            "lora_name": "policy-0-slot0",
            "server_instance": "h200-instance-1",
        }

    def readiness(self, receipt=None):
        return {"loaded_receipt": receipt or self.receipt()}

    def build_reference(self, directory: Path) -> Path:
        prompt_ids = list(range(256))
        completion_ids = list(range(900, 1156))
        full_ids = prompt_ids + completion_ids
        decode = {
            "choices": [{
                "token_ids": completion_ids,
                "logprobs": {
                    "content": [
                        {"token": f"token_id:{token}", "logprob": -0.1 - index / 1000}
                        for index, token in enumerate(completion_ids)
                    ]
                },
                "routed_experts": route_for(511),
                "finish_reason": "length",
            }]
        }
        prefill = {
            "prompt_logprobs": [
                None,
                *[
                    {str(token): {"logprob": -0.2 - index / 1000}}
                    for index, token in enumerate(full_ids[1:], 1)
                ],
            ],
            "choices": [{"routed_experts": route_for(512)}],
        }
        calls = []

        def post(_endpoint, _body, _timeout):
            calls.append(1)
            return decode if len(calls) == 1 else prefill

        record = {
            "source_kind": "operator_non_rollout",
            "input_ids": prompt_ids,
            "sequence_lengths": [256],
            "expected_route_shape": [512, 8, 2],
        }
        output = directory / "reference.json"
        reference_builder.build_reference(
            endpoint="http://engine:8000/inference/v1/generate",
            input_record=record,
            output_path=output,
            base_model="base",
            policy="policy-0",
            policy_model="policy-0-slot0",
            expected_policy_identity=self.receipt(),
            cache_salt="original-decode",
            rescore_cache_salt="original-rescore",
            prefill_rescore=True,
            readiness_endpoint="http://engine:8000/inference/v1/staging-status",
            minimum_tokens=1,
            post_json=post,
            get_json=lambda *_: self.readiness(),
        )
        return output

    def diagnostic_responses(self, input_ids):
        responses = []
        for prefix_length in prefix_stability.PREFIX_LENGTHS:
            if prefix_length == 256:
                responses.append({
                    "choices": [{
                        "token_ids": [900],
                        "logprobs": {
                            "content": [{"token": "token_id:900", "logprob": -0.1}]
                        },
                        "routed_experts": route_for(256),
                    }]
                })
                continue
            responses.append({
                "prompt_logprobs": [
                    None,
                    *[
                        {str(token): {"logprob": -0.2 - index / 1000}}
                        for index, token in enumerate(input_ids[1:prefix_length], 1)
                    ],
                ],
                "choices": [{"routed_experts": route_for(prefix_length)}],
            })
        return responses

    def execute(
        self,
        mutate_response=None,
        readiness=None,
        after=None,
        endpoint="http://engine:8000/inference/v1/generate",
        readiness_endpoint="http://engine:8000/inference/v1/staging-status",
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = self.build_reference(root)
            reference = json.loads(reference_path.read_text())
            responses = self.diagnostic_responses(reference["input_ids"])
            calls = []
            post_responses = [copy.deepcopy(response) for response in responses]
            if mutate_response:
                mutate_response(post_responses)

            def post(_endpoint, body, _timeout):
                calls.append(body)
                return post_responses[len(calls) - 1]

            readiness_values = readiness or [self.readiness()] * 6
            states = iter(readiness_values)
            output = root / "diagnostic.json"
            result = prefix_stability.run_diagnostic(
                reference_path=reference_path,
                endpoint=endpoint,
                readiness_endpoint=readiness_endpoint,
                expected_policy_identity=self.receipt(),
                policy_model="policy-0-slot0",
                fresh_salt_prefix="prefix-test",
                output_path=output,
                post_json=post,
                get_json=lambda *_: next(states),
            )
            if after is not None:
                after(result, output)
            return result, calls, output

    def test_exact_requests_and_suffix_prefix_alignment(self):
        result, calls, output = self.execute()
        self.assertEqual([len(call["token_ids"]) for call in calls], [256, 257, 288, 384, 512])
        self.assertEqual([call["cache_salt"] for call in calls], [
            "prefix-test-256", "prefix-test-257", "prefix-test-288",
            "prefix-test-384", "prefix-test-512",
        ])
        self.assertEqual(calls[0]["sampling_params"], {
            "max_tokens": 1, "temperature": 0.0, "top_p": 1.0,
            "logprobs": 1, "min_tokens": 1,
        })
        self.assertEqual(calls[1]["sampling_params"]["prompt_logprobs"], 1)
        self.assertNotIn("inference_logprobs", result["original_evidence"])
        self.assertEqual(result["requests"][0]["generated_token_id"], 900)
        self.assertIsNotNone(output)

        self.assertEqual(
            [(entry["stage"], entry["request_index"]) for entry in result["receipt_sequence"]],
            [
                ("before", 0),
                ("between", 0),
                ("between", 1),
                ("between", 2),
                ("between", 3),
                ("after", 4),
            ],
        )
        self.assertEqual(
            [entry["actual_token_position"] for entry in result["requests"]],
            [256, 256, 287, 383, 511],
        )

    def test_receipt_change_is_rejected(self):
        changed = self.receipt()
        changed["server_instance"] = "h200-instance-2"
        with self.assertRaisesRegex(prefix_stability.PrefixStabilityError, "receipt"):
            self.execute(readiness=[self.readiness()] * 2 + [self.readiness(changed)])

    def test_response_tokens_routes_and_logprobs_are_rejected_when_tampered(self):
        def tamper(responses):
            responses[0]["choices"][0]["token_ids"] = [901]

        with self.assertRaisesRegex(prefix_stability.PrefixStabilityError, "first token"):
            self.execute(mutate_response=tamper)

        def tamper_route(responses):
            responses[-1]["choices"][0]["routed_experts"]["data"] = "AAAA"

        with self.assertRaises(prefix_stability.PrefixStabilityError):
            self.execute(mutate_response=tamper_route)

        def tamper_score(responses):
            responses[1]["prompt_logprobs"][1] = {"999": {"logprob": -0.2}}

        with self.assertRaisesRegex(prefix_stability.PrefixStabilityError, "token key"):
            self.execute(mutate_response=tamper_score)

    def test_hash_tamper_and_no_clobber(self):
        original_bytes = []

        def check_no_clobber(result, output):
            original_bytes.append(output.read_bytes())
            with self.assertRaisesRegex(prefix_stability.PrefixStabilityError, "clobber"):
                prefix_stability.publish_diagnostic(result, output)
            self.assertEqual(output.read_bytes(), original_bytes[0])

        result, _, output = self.execute(after=check_no_clobber)
        changed = copy.deepcopy(result)
        changed["requests"][0]["response"]["choices"][0]["finish_reason"] = "stop"
        with self.assertRaisesRegex(prefix_stability.PrefixStabilityError, "response hash"):
            prefix_stability.validate_diagnostic(changed)
        self.assertTrue(original_bytes)

    def test_default_cli_requires_all_inputs(self):
        with self.assertRaises(SystemExit) as raised:
            prefix_stability.main([])
        self.assertEqual(raised.exception.code, 2)

    def test_endpoints_must_bind_to_same_origin_and_distinct_paths(self):
        for endpoint, readiness_endpoint in (
            (
                "https://other-engine:8000/inference/v1/generate",
                "http://engine:8000/inference/v1/staging-status",
            ),
            (
                "http://engine:8000/inference/v1/generate",
                "http://engine:8000/inference/v1/generate",
            ),
        ):
            with self.subTest(endpoint=endpoint, readiness_endpoint=readiness_endpoint):
                with self.assertRaisesRegex(prefix_stability.PrefixStabilityError, "endpoint"):
                    self.execute(endpoint=endpoint, readiness_endpoint=readiness_endpoint)

    def test_validate_diagnostic_recomputes_collections_and_derived_evidence(self):
        result, _, _ = self.execute()
        missing = copy.deepcopy(result)
        del missing["comparisons"]["common_prefix_routes"]
        with self.assertRaises(prefix_stability.PrefixStabilityError):
            prefix_stability.validate_diagnostic(missing)

        changed_comparison = copy.deepcopy(result)
        changed_comparison["comparisons"]["common_prefix_routes"][0]["different_row_count"] += 1
        with self.assertRaises(prefix_stability.PrefixStabilityError):
            prefix_stability.validate_diagnostic(changed_comparison)

        changed_delta = copy.deepcopy(result)
        changed_delta["original_evidence"]["first_decode_score_delta"] += 1.0
        with self.assertRaises(prefix_stability.PrefixStabilityError):
            prefix_stability.validate_diagnostic(changed_delta)

        nonfinite = copy.deepcopy(result)
        nonfinite["original_evidence"]["first_decoded_logprob"] = math.nan
        with self.assertRaises(prefix_stability.PrefixStabilityError):
            prefix_stability.validate_diagnostic(nonfinite)

    def test_original_reference_receipt_evidence_is_present(self):
        result, _, _ = self.execute()
        self.assertIn("reference_policy_receipt", result["original_evidence"])
        self.assertIn("reference_policy_receipt_sha256", result["original_evidence"])

    def test_metadata_receipts_and_positions_are_bound(self):
        result, _, _ = self.execute()
        changed_stage = copy.deepcopy(result)
        changed_stage["receipt_sequence"][1]["stage"] = "after"
        with self.subTest(case="stage"):
            with self.assertRaises(prefix_stability.PrefixStabilityError):
                prefix_stability.validate_diagnostic(changed_stage)

        changed_index = copy.deepcopy(result)
        changed_index["receipt_sequence"][2]["request_index"] = 4
        with self.subTest(case="request index"):
            with self.assertRaises(prefix_stability.PrefixStabilityError):
                prefix_stability.validate_diagnostic(changed_index)

        changed_position = copy.deepcopy(result)
        changed_position["requests"][0]["actual_token_position"] = 255
        with self.subTest(case="actual token position"):
            with self.assertRaises(prefix_stability.PrefixStabilityError):
                prefix_stability.validate_diagnostic(changed_position)

        changed_identity = copy.deepcopy(result)
        changed_identity["expected_policy_identity"]["server_instance"] = "joint-substitution"
        for entry in changed_identity["receipt_sequence"]:
            entry["receipt"]["server_instance"] = "joint-substitution"
        with self.subTest(case="joint receipt substitution"):
            with self.assertRaises(prefix_stability.PrefixStabilityError):
                prefix_stability.validate_diagnostic(changed_identity)

    def test_route_comparison_reports_sparse_layer_sets_and_order(self):
        def route(rows):
            raw = bytearray()
            for layer_values in rows:
                raw.extend((0,) * 12)
                for values in layer_values:
                    raw.extend(values)
            encoded = base64.b64encode(raw).decode()
            return prefix_stability._normalise_route(
                {"data": encoded, "dtype": "uint8", "shape": [len(rows), 8, 2], "start": 0},
                len(rows),
            )

        left = route([((1, 2), (3, 4))])
        swapped_layers = route([((3, 4), (1, 2))])
        swapped_order = route([((2, 1), (3, 4))])
        layer_difference = prefix_stability._route_comparison(
            left, swapped_layers, "left", "swapped_layers"
        )
        self.assertEqual(layer_difference["different_set_layer_count"], 2)
        self.assertEqual(layer_difference["order_only_layer_count"], 0)
        self.assertEqual(layer_difference["different_row_count"], 1)
        order_difference = prefix_stability._route_comparison(
            left, swapped_order, "left", "swapped_order"
        )
        self.assertEqual(order_difference["different_set_layer_count"], 0)
        self.assertEqual(order_difference["order_only_layer_count"], 1)
        self.assertEqual(order_difference["different_row_count"], 1)

    def test_stale_same_pid_temp_is_not_removed(self):
        stale_bytes = b"stale temp from another invocation\n"

        def check_stale_temp(result, output):
            output.unlink()
            stale = output.with_name(
                f"{output.name}.partial.prefix-stability.{os.getpid()}"
            )
            stale.write_bytes(stale_bytes)
            with self.assertRaisesRegex(prefix_stability.PrefixStabilityError, "temporary"):
                prefix_stability.publish_diagnostic(result, output)
            self.assertFalse(output.exists())
            self.assertEqual(stale.read_bytes(), stale_bytes)
            stale.unlink()

        self.execute(after=check_stale_temp)


if __name__ == "__main__":
    unittest.main()
