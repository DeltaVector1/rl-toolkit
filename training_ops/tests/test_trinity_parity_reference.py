import base64
import hashlib
import importlib.util
import json
import tempfile
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "trinity_parity_reference",
    ROOT / "training_ops/runtime/h200/build_trinity_parity_reference.py",
)
assert SPEC and SPEC.loader
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


def route_for(token_count: int) -> dict:
    raw = bytearray()
    for index in range(token_count):
        raw.extend((0,) * 12)
        raw.extend((1 + index % 2, 2 + index % 2, 2 + index % 2, 3 + index % 2))
    encoded = base64.b64encode(raw).decode()
    # This is the serialized endpoint schema. The helper adds the trainer
    # reference envelope after validating it.
    return {
        "data": encoded,
        "dtype": "uint8",
        "shape": [token_count, 8, 2],
        "start": 0,
    }


class TrinityParityReferenceTest(unittest.TestCase):
    def record(self, token_count=4):
        return {
            "source_kind": "operator_non_rollout",
            "input_ids": list(range(100, 100 + token_count)),
            "sequence_lengths": [token_count],
            "expected_route_shape": [token_count, 8, 2],
        }

    def response(self, token_count=4):
        return {
            "prompt_logprobs": [
                None,
                *[
                    {str(token): {"logprob": -float(index)}}
                    for index, token in enumerate(range(101, 100 + token_count), 1)
                ],
            ],
            "choices": [{"routed_experts": route_for(token_count)}],
        }

    def test_builds_reference_and_selects_policy_model(self):
        calls = []

        def post(url, body, timeout):
            calls.append((url, body, timeout))
            return self.response()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reference.json"
            result = reference.build_reference(
                endpoint="http://engine:8000/inference/v1/generate",
                input_record=self.record(),
                output_path=output,
                base_model="/data/trinity-thinking-fp8",
                policy="policy-0",
                policy_model="policy-0-slot0",
                minimum_tokens=1,
                post_json=post,
            )
            self.assertEqual(calls[0][0], "http://engine:8000/inference/v1/generate")
            self.assertEqual(calls[0][1]["model"], "policy-0-slot0")
            self.assertEqual(calls[0][1]["token_ids"], self.record()["input_ids"])
            self.assertEqual(
                calls[0][1]["sampling_params"],
                {
                    "max_tokens": 1,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "prompt_logprobs": 1,
                },
            )
            self.assertEqual(result["inference_logprobs"], [0.0, -1.0, -2.0, -3.0])
            self.assertEqual(
                set(result["routed_experts"]),
                {"schema_version", "encoding", "dtype", "shape", "dense_layers", "experts", "sha256", "data"},
            )
            self.assertEqual(result["routed_experts"]["dense_layers"], 6)
            self.assertEqual(result["routed_experts"]["experts"], 256)
            self.assertEqual(
                result["routed_experts"]["sha256"],
                hashlib.sha256(base64.b64decode(route_for(4)["data"])).hexdigest(),
            )
            self.assertEqual(json.loads(output.read_text()), result)

    def test_fresh_input_mode_writes_exact_diverse_non_rollout_sequence(self):
        def encode(text):
            # The production path uses AutoTokenizer. This fixture keeps the
            # test local while preserving deterministic, diverse IDs.
            return [index % 257 for index, _ in enumerate(text.split())]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "input.json"
            result = reference.write_input_record(
                output,
                "/data/trinity-thinking-fp8",
                token_count=4097,
                encode=encode,
            )
            self.assertEqual(len(result["input_ids"]), 4097)
            self.assertEqual(result["sequence_lengths"], [4097])
            self.assertEqual(result["source_kind"], "deterministic_local_tokenizer_seed")
            self.assertGreaterEqual(len(set(result["input_ids"])), 64)
            self.assertEqual(json.loads(output.read_text()), result)

    def test_short_or_mismatched_reference_is_refused(self):
        with self.assertRaisesRegex(reference.ReferenceError, "at least 4097"):
            reference.build_reference(
                endpoint="http://engine:8000/inference/v1/generate",
                input_record=self.record(),
                output_path=Path("unused.json"),
                base_model="base",
                policy="base",
                post_json=lambda *_: self.response(),
            )

        def wrong_logprobs(*_):
            response = self.response()
            response["prompt_logprobs"] = response["prompt_logprobs"][:-1]
            return response

        with self.assertRaisesRegex(reference.ReferenceError, "logprob count"):
            reference.build_reference(
                endpoint="http://engine:8000/inference/v1/generate",
                input_record=self.record(),
                output_path=Path("unused.json"),
                base_model="base",
                policy="base",
                minimum_tokens=1,
                post_json=wrong_logprobs,
            )

    def test_route_checksum_and_expected_shape_are_fail_closed(self):
        def bad_checksum(*_):
            response = self.response()
            response["choices"][0]["routed_experts"]["data"] = "!" * 88
            return response

        with self.assertRaisesRegex(reference.ReferenceError, "base64"):
            reference.build_reference(
                endpoint="http://engine:8000/inference/v1/generate",
                input_record=self.record(),
                output_path=Path("unused.json"),
                base_model="base",
                policy="base",
                minimum_tokens=1,
                post_json=bad_checksum,
            )

        record = self.record()
        record["expected_route_sha256"] = "f" * 64
        with self.assertRaisesRegex(reference.ReferenceError, "digest"):
            reference.build_reference(
                endpoint="http://engine:8000/inference/v1/generate",
                input_record=record,
                output_path=Path("unused.json"),
                base_model="base",
                policy="base",
                minimum_tokens=1,
                post_json=lambda *_: self.response(),
            )

        with self.assertRaisesRegex(reference.ReferenceError, "shape"):
            reference.build_reference(
                endpoint="http://engine:8000/inference/v1/generate",
                input_record=self.record(),
                output_path=Path("unused.json"),
                base_model="base",
                policy="base",
                minimum_tokens=1,
                expected_route_shape=[4, 60, 4],
                post_json=lambda *_: self.response(),
            )

    def test_prompt_logprob_token_key_must_match_input_token(self):
        def wrong_token_key(*_):
            response = self.response()
            response["prompt_logprobs"][1] = {"999": {"logprob": -1.0}}
            return response

        with self.assertRaisesRegex(reference.ReferenceError, "no unique value"):
            reference.build_reference(
                endpoint="http://engine:8000/inference/v1/generate",
                input_record=self.record(),
                output_path=Path("unused.json"),
                base_model="base",
                policy="base",
                minimum_tokens=1,
                post_json=wrong_token_key,
            )

    def test_policy_zero_requires_explicit_loaded_model_name(self):
        with self.assertRaisesRegex(reference.ReferenceError, "selected policy model"):
            reference.build_reference(
                endpoint="http://engine:8000/inference/v1/generate",
                input_record=self.record(),
                output_path=Path("unused.json"),
                base_model="base",
                policy="policy-0",
                minimum_tokens=1,
                post_json=lambda *_: self.response(),
            )

    def test_rollout_source_is_rejected(self):
        record = self.record()
        record["source_kind"] = "rollout_batch"
        with self.assertRaisesRegex(reference.ReferenceError, "non-rollout"):
            reference.build_reference(
                endpoint="http://engine:8000/inference/v1/generate",
                input_record=record,
                output_path=Path("unused.json"),
                base_model="base",
                policy="base",
                minimum_tokens=1,
                post_json=lambda *_: self.response(),
            )

    def test_long_context_profile_keeps_native_fp8_and_strict_guard(self):
        with (ROOT / "configs/trinity-h200-parity-production/trainer.toml").open("rb") as stream:
            config = tomllib.load(stream)
        self.assertEqual(config["model"]["seq_len"], 8192)
        self.assertEqual(config["model"]["name"], "/data/trinity-thinking-fp8")
        self.assertEqual(config["model"]["quantization"]["type"], "fp8")
        self.assertTrue(config["enable_router_replay"])
        self.assertEqual(config["parity_guard"]["max_unmasked_mismatch_kl"], 0.1)
        self.assertEqual(config["parity_guard"]["max_masked_mismatch_kl"], 0.5)
        self.assertEqual(config["parity_guard"]["max_masked_fraction"], 0.06)


if __name__ == "__main__":
    unittest.main()
