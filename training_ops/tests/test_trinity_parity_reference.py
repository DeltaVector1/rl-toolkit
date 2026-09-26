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
    def policy_receipt(self):
        return {
            "target_sha256": "a" * 64,
            "target_size": 123,
            "config_sha256": "b" * 64,
            "config_size": 456,
            "source_url": "http://engine:18100/policy-0",
            "lora_name": "policy-0-slot0",
            "server_instance": "h200b-instance-1",
        }

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

    def policy_response(self, prompt_count=4, completion_count=32):
        generated_ids = list(range(900, 900 + completion_count))
        return {
            "prompt_logprobs": self.response(prompt_count)["prompt_logprobs"],
            "choices": [{
                "token_ids": generated_ids,
                "logprobs": {
                    "content": [
                        {"token": f"token_id:{token}", "logprob": -0.01 * index}
                        for index, token in enumerate(generated_ids)
                    ]
                },
                "finish_reason": "length",
                "routed_experts": route_for(prompt_count + completion_count - 1),
            }],
        }

    def readiness(self, receipt=None):
        return {"loaded_receipt": receipt or self.policy_receipt()}

    def test_builds_reference_and_selects_policy_model(self):
        calls = []

        def post(url, body, timeout):
            calls.append((url, body, timeout))
            return self.policy_response()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reference.json"
            policy_record = self.record()
            policy_record.pop("expected_route_shape")
            result = reference.build_reference(
                endpoint="http://engine:8000/inference/v1/generate",
                input_record=policy_record,
                output_path=output,
                base_model="/data/trinity-thinking-fp8",
                policy="policy-0",
                policy_model="policy-0-slot0",
                expected_policy_identity=self.policy_receipt(),
                cache_salt="qualifier-policy-0-fresh-1",
                readiness_endpoint="http://engine:8000/inference/v1/staging-status",
                trainer_seq_len=36,
                minimum_tokens=1,
                post_json=post,
                get_json=lambda *_: self.readiness(),
            )
            self.assertEqual(calls[0][0], "http://engine:8000/inference/v1/generate")
            self.assertEqual(calls[0][1]["model"], "policy-0-slot0")
            self.assertEqual(calls[0][1]["token_ids"], self.record()["input_ids"])
            self.assertEqual(calls[0][1]["cache_salt"], "qualifier-policy-0-fresh-1")
            self.assertNotIn("cache_salt", calls[0][1]["sampling_params"])
            self.assertEqual(
                calls[0][1]["sampling_params"],
                {
                    "max_tokens": 32,
                    "min_tokens": 32,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "prompt_logprobs": 1,
                    "logprobs": 1,
                },
            )
            self.assertEqual(len(result["input_ids"]), 36)
            self.assertEqual(result["inference_logprobs"][:4], [0.0, -1.0, -2.0, -3.0])
            self.assertEqual(result["loss_mask"], [False] * 4 + [True] * 32)
            self.assertEqual(result["sequence_lengths"], [36])
            self.assertEqual(
                set(result["routed_experts"]),
                {"schema_version", "encoding", "dtype", "shape", "dense_layers", "experts", "sha256", "data"},
            )
            self.assertEqual(result["routed_experts"]["dense_layers"], 6)
            self.assertEqual(result["routed_experts"]["experts"], 256)
            self.assertEqual(
                result["routed_experts"]["sha256"],
                hashlib.sha256(base64.b64decode(route_for(35)["data"])).hexdigest(),
            )
            self.assertEqual(result["policy_identity"], self.policy_receipt())
            self.assertEqual(result["cache_salt"], "qualifier-policy-0-fresh-1")
            self.assertEqual(result["request_mode"], "serialized_generate_completion_prompt_logprobs")
            self.assertEqual(result["generated_token_count"], 32)
            self.assertEqual(result["policy_identity_before"], self.policy_receipt())
            self.assertEqual(result["policy_identity_after"], self.policy_receipt())
            self.assertEqual(result["sampling_params"]["prompt_logprobs"], 1)
            self.assertEqual(json.loads(output.read_text()), result)

    def test_policy_zero_requires_matching_readiness_receipt(self):
        with self.assertRaisesRegex(reference.ReferenceError, "readiness receipt"):
            reference.build_reference(
                endpoint="http://engine:8000/inference/v1/generate",
                input_record=self.record(),
                output_path=Path("unused.json"),
                base_model="base",
                policy="policy-0",
                policy_model="policy-0-slot0",
                cache_salt="fresh",
                minimum_tokens=1,
                post_json=lambda *_: self.policy_response(),
            )

        with self.assertRaisesRegex(reference.ReferenceError, "fresh cache_salt"):
            reference.build_reference(
                endpoint="http://engine:8000/inference/v1/generate",
                input_record=self.record(),
                output_path=Path("unused.json"),
                base_model="base",
                policy="policy-0",
                policy_model="policy-0-slot0",
                policy_receipt=self.policy_receipt(),
                expected_policy_identity=self.policy_receipt(),
                readiness_endpoint="http://engine:8000/inference/v1/staging-status",
                minimum_tokens=1,
                post_json=lambda *_: self.policy_response(),
                get_json=lambda *_: self.readiness(),
            )

        receipt = self.policy_receipt()
        receipt["target_sha256"] = "c" * 64
        with self.assertRaisesRegex(reference.ReferenceError, "differs in target_sha256"):
            reference.build_reference(
                endpoint="http://engine:8000/inference/v1/generate",
                input_record=self.record(),
                output_path=Path("unused.json"),
                base_model="base",
                policy="policy-0",
                policy_model="policy-0-slot0",
                policy_receipt=receipt,
                expected_policy_identity=self.policy_receipt(),
                cache_salt="fresh",
                readiness_endpoint="http://engine:8000/inference/v1/staging-status",
                trainer_seq_len=36,
                minimum_tokens=1,
                post_json=lambda *_: self.policy_response(),
                get_json=lambda *_: self.readiness(receipt),
            )

    def test_policy0_profile_matches_trainer_seq_len_and_fake_loader_contract(self):
        with (ROOT / "configs/trinity-h200-parity-production/trainer.toml").open("rb") as stream:
            trainer_seq_len = tomllib.load(stream)["model"]["seq_len"]
        self.assertEqual(trainer_seq_len, 8192)
        record = self.record(token_count=8160)
        record.pop("expected_route_shape")
        with tempfile.TemporaryDirectory() as directory:
            result = reference.build_reference(
                endpoint="http://engine:8000/inference/v1/generate",
                input_record=record,
                output_path=Path(directory) / "reference.json",
                base_model="base",
                policy="policy-0",
                policy_model="policy-0-slot0",
                expected_policy_identity=self.policy_receipt(),
                cache_salt="fresh-profile",
                readiness_endpoint="http://engine:8000/inference/v1/staging-status",
                post_json=lambda *_: self.policy_response(prompt_count=8160),
                trainer_seq_len=trainer_seq_len,
                get_json=lambda *_: self.readiness(),
            )
        self.assertEqual(len(result["input_ids"]), trainer_seq_len)
        self.assertEqual(result["sequence_lengths"], [trainer_seq_len])
        self.assertEqual(sum(result["loss_mask"]), 32)
        self.assertEqual(result["routed_experts"]["shape"][0], trainer_seq_len - 1)
        self.assertEqual(len(result["inference_logprobs"]), trainer_seq_len)
        self.assertTrue(
            set(("input_ids", "inference_logprobs", "loss_mask", "sequence_lengths", "routed_experts"))
            <= result.keys()
        )
        data_source = (ROOT / "prime-rl/src/prime_rl/trainer/rl/data.py").read_text()
        self.assertIn("shape[0] in {token_count - 1, token_count}", data_source)
        self.assertIn("routed.new_zeros((1, routed.shape[1], routed.shape[2]))", data_source)
        self.assertIn("torch.count_nonzero(sparse[-1]).item()", data_source)
        self.assertNotIn("~sparse_nonzero", data_source)

    def test_route_normalizer_allows_expert_zero_when_topk_is_one(self):
        raw = bytes((0,) * (8 * 1 * 4))
        route = {
            "data": base64.b64encode(raw).decode(),
            "dtype": "uint8",
            "shape": [4, 8, 1],
            "start": 0,
        }
        normalised = reference._normalise_route(route, 4)
        self.assertEqual(normalised["shape"], [4, 8, 1])

    def test_policy_zero_requires_active_receipt_to_stay_stable(self):
        changed = self.policy_receipt()
        changed["server_instance"] = "h200b-instance-2"
        states = iter((self.readiness(), self.readiness(changed)))
        policy_record = self.record()
        policy_record.pop("expected_route_shape")
        with self.assertRaisesRegex(reference.ReferenceError, "differs in server_instance"):
            reference.build_reference(
                endpoint="http://engine:8000/inference/v1/generate",
                input_record=policy_record,
                output_path=Path("unused.json"),
                base_model="base",
                policy="policy-0",
                policy_model="policy-0-slot0",
                policy_receipt=self.policy_receipt(),
                expected_policy_identity=self.policy_receipt(),
                cache_salt="fresh",
                readiness_endpoint="http://engine:8000/inference/v1/staging-status",
                trainer_seq_len=36,
                minimum_tokens=1,
                post_json=lambda *_: self.policy_response(),
                get_json=lambda *_: next(states),
            )

    def test_policy_zero_rejects_readiness_endpoint_for_another_engine(self):
        with self.assertRaisesRegex(reference.ReferenceError, "active generate endpoint"):
            reference.build_reference(
                endpoint="http://engine:8000/inference/v1/generate",
                input_record=self.record(),
                output_path=Path("unused.json"),
                base_model="base",
                policy="policy-0",
                policy_model="policy-0-slot0",
                policy_receipt=self.policy_receipt(),
                expected_policy_identity=self.policy_receipt(),
                cache_salt="fresh",
                readiness_endpoint="http://other-engine:8000/inference/v1/staging-status",
                minimum_tokens=1,
                post_json=lambda *_: self.policy_response(),
                get_json=lambda *_: self.readiness(),
            )

    def test_policy_zero_rejects_active_source_url_change(self):
        changed = self.policy_receipt()
        changed["source_url"] = "http://other-engine:18100/policy-0"
        states = iter((self.readiness(), self.readiness(changed)))
        policy_record = self.record()
        policy_record.pop("expected_route_shape")
        with self.assertRaisesRegex(reference.ReferenceError, "differs in source_url"):
            reference.build_reference(
                endpoint="http://engine:8000/inference/v1/generate",
                input_record=policy_record,
                output_path=Path("unused.json"),
                base_model="base",
                policy="policy-0",
                policy_model="policy-0-slot0",
                policy_receipt=self.policy_receipt(),
                expected_policy_identity=self.policy_receipt(),
                cache_salt="fresh",
                readiness_endpoint="http://engine:8000/inference/v1/staging-status",
                trainer_seq_len=36,
                minimum_tokens=1,
                post_json=lambda *_: self.policy_response(),
                get_json=lambda *_: next(states),
            )

    def test_completion_token_key_and_count_are_fail_closed(self):
        def wrong_key(*_):
            response = self.policy_response()
            response["choices"][0]["logprobs"]["content"][0]["token"] = "token_id:999"
            return response

        with self.assertRaisesRegex(reference.ReferenceError, "wrong token key"):
            reference.build_reference(
                endpoint="http://engine:8000/inference/v1/generate",
                input_record=self.record(),
                output_path=Path("unused.json"),
                base_model="base",
                policy="policy-0",
                policy_model="policy-0-slot0",
                policy_receipt=self.policy_receipt(),
                expected_policy_identity=self.policy_receipt(),
                cache_salt="fresh",
                readiness_endpoint="http://engine:8000/inference/v1/staging-status",
                trainer_seq_len=36,
                minimum_tokens=1,
                post_json=wrong_key,
                get_json=lambda *_: self.readiness(),
            )

    def test_base_policy_keeps_no_adapter_receipt_or_cache_salt(self):
        calls = []

        def post(url, body, timeout):
            calls.append(body)
            return self.response()

        with tempfile.TemporaryDirectory() as directory:
            result = reference.build_reference(
                endpoint="http://engine:8000/inference/v1/generate",
                input_record=self.record(),
                output_path=Path(directory) / "reference.json",
                base_model="base-model",
                policy="base",
                minimum_tokens=1,
                post_json=post,
            )
        self.assertEqual(calls[0]["model"], "base-model")
        self.assertNotIn("cache_salt", calls[0])
        self.assertIsNone(result["policy_identity"])
        self.assertIsNone(result["cache_salt"])
        self.assertEqual(result["request_mode"], "serialized_generate_prompt_logprobs")
        self.assertEqual(result["input_ids"], self.record()["input_ids"])
        self.assertNotIn("loss_mask", result)

    def test_fresh_input_mode_writes_exact_diverse_non_rollout_sequence(self):
        def encode(text):
            # The production path uses AutoTokenizer. This fixture keeps the
            # test local while preserving deterministic, diverse IDs.
            return [index for index, _ in enumerate(text.split())]

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

    def test_seed_generation_is_deterministic_and_exactly_sized(self):
        def encode(text):
            return [index for index, _ in enumerate(text.split())]

        with tempfile.TemporaryDirectory() as directory:
            first = reference.write_input_record(
                Path(directory) / "first.json",
                "/data/trinity-thinking-fp8",
                token_count=4097,
                encode=encode,
            )
            second = reference.write_input_record(
                Path(directory) / "second.json",
                "/data/trinity-thinking-fp8",
                token_count=4097,
                encode=encode,
            )
        self.assertEqual(first["input_ids"], second["input_ids"])
        self.assertEqual(first["sequence_lengths"], [4097])
        self.assertEqual(len(first["input_ids"]), 4097)

    def test_pathological_repeated_seed_is_refused(self):
        def repeated_encode(text):
            return [100 + index % 64 for index, _ in enumerate(text.split())]

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(reference.ReferenceError, "repeated 16-token"):
                reference.write_input_record(
                    Path(directory) / "repeated.json",
                    "/data/trinity-thinking-fp8",
                    token_count=4097,
                    encode=repeated_encode,
                )

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

    def test_prompt_logprobs_requires_full_and_well_formed_values(self):
        def missing_logprobs(*_):
            response = self.response()
            response.pop("prompt_logprobs")
            return response

        with self.assertRaisesRegex(reference.ReferenceError, "logprob count"):
            reference.build_reference(
                endpoint="http://engine:8000/inference/v1/generate",
                input_record=self.record(),
                output_path=Path("unused.json"),
                base_model="base",
                policy="base",
                minimum_tokens=1,
                post_json=missing_logprobs,
            )

        def malformed_logprobs(*_):
            response = self.response()
            response["prompt_logprobs"][2] = {"102": {}}
            return response

        with self.assertRaisesRegex(reference.ReferenceError, "not numeric"):
            reference.build_reference(
                endpoint="http://engine:8000/inference/v1/generate",
                input_record=self.record(),
                output_path=Path("unused.json"),
                base_model="base",
                policy="base",
                minimum_tokens=1,
                post_json=malformed_logprobs,
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
        self.assertEqual(
            config["model"]["lora"]["initial_adapter_path"],
            "/data/glm52-rl/qualifications/trinity-h200-parity-production/policy-0/adapter_model.safetensors",
        )
        self.assertEqual(config["model"]["quantization"]["type"], "fp8")
        self.assertTrue(config["enable_router_replay"])
        self.assertEqual(config["parity_guard"]["max_unmasked_mismatch_kl"], 0.1)
        self.assertEqual(config["parity_guard"]["max_masked_mismatch_kl"], 0.5)
        self.assertEqual(config["parity_guard"]["max_masked_fraction"], 0.06)


if __name__ == "__main__":
    unittest.main()
