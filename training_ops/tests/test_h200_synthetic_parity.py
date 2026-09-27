import base64
import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import tomllib


ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = ROOT / "training_ops/runtime/h200/build_trinity_synthetic_parity.py"
SPEC = importlib.util.spec_from_file_location("synthetic_parity", BUILDER_PATH)
assert SPEC and SPEC.loader
synthetic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(synthetic)


def route_for(token_count: int, *, layers: int = 8, top_k: int = 2) -> dict:
    raw = bytearray()
    for index in range(token_count):
        for layer in range(layers):
            raw.extend(
                ((layer + index) % 256, (layer + index + 1) % 256)[:top_k]
            )
    return {
        "data": base64.b64encode(raw).decode(),
        "dtype": "uint8",
        "shape": [token_count, layers, top_k],
        "start": 0,
    }


def response_for(tokens: list[int]) -> dict:
    return {
        "choices": [
            {
                "prompt_logprobs": [
                    None,
                    *[
                        {str(token): {"logprob": -0.01 * index}}
                        for index, token in enumerate(tokens[1:], 1)
                    ],
                ],
                "routed_experts": route_for(len(tokens)),
            }
        ]
    }


class SyntheticParityTest(unittest.TestCase):
    def test_sequences_have_required_lengths_sum_and_diversity(self):
        sequences = synthetic.make_synthetic_sequences()
        self.assertEqual(
            [len(item["input_ids"]) for item in sequences],
            [4096, 5632, 6656, 7680, 8704],
        )
        self.assertEqual(sum(len(item["input_ids"]) for item in sequences), 32768)
        self.assertGreater(len(set(sequences[0]["input_ids"])), 64)
        self.assertGreater(
            len(set(tuple(sequences[0]["input_ids"][i : i + 16]) for i in range(4081))),
            4000,
        )

    def test_sequences_are_deterministic_and_seeded(self):
        first = synthetic.make_synthetic_sequences()
        second = synthetic.make_synthetic_sequences()
        changed = synthetic.make_synthetic_sequences(seed="other-seed")
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_source_is_synthetic_and_does_not_accept_rollout_records(self):
        self.assertEqual(synthetic.SOURCE_KIND, "synthetic")
        with self.assertRaises(synthetic.SyntheticParityError):
            synthetic.validate_source_kind({"source_kind": "rollout_batch"})

    def test_request_descriptions_are_base_model_and_exact_sampling(self):
        sequences = synthetic.make_synthetic_sequences()
        requests = synthetic.make_request_descriptions(sequences, "base-model")
        self.assertEqual(len(requests), 5)
        for request, sequence in zip(requests, sequences):
            self.assertEqual(request["model"], "base-model")
            self.assertEqual(request["token_ids"], sequence["input_ids"])
            self.assertEqual(
                request["sampling_params"],
                {
                    "max_tokens": 1,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "prompt_logprobs": 1,
                },
            )
            self.assertNotIn("lora_name", request)

    def test_build_packed_and_individual_artifacts_with_hash_bindings(self):
        calls = []
        sequences = synthetic.make_synthetic_sequences()

        def post(_endpoint, request, _timeout):
            calls.append(request)
            return response_for(request["token_ids"])

        with tempfile.TemporaryDirectory() as directory:
            artifact = synthetic.build_artifact(
                endpoint="http://h200/inference/v1/generate",
                output_path=Path(directory) / "synthetic.json",
                base_model="base-model",
                post_json=post,
                sequences=sequences,
            )
            written = json.loads((Path(directory) / "synthetic.json").read_text())
        self.assertEqual(len(calls), 5)
        self.assertEqual(artifact, written)
        self.assertEqual(artifact["source_kind"], "synthetic")
        self.assertEqual(artifact["sequence_lengths"], [4096, 5632, 6656, 7680, 8704])
        self.assertEqual(artifact["total_tokens"], 32768)
        self.assertEqual(artifact["topology"], {"cp": 8, "pp": 8, "ep": 8})
        manifest = artifact["manifest"]
        self.assertEqual(manifest["request_sha256"], [synthetic.json_sha256(item) for item in calls])
        self.assertEqual(len(manifest["response_sha256"]), 5)
        self.assertEqual(manifest["route_shapes"], [[len(item["input_ids"]), 8, 2] for item in sequences])
        packed = artifact["packed"]
        self.assertEqual(packed["input_ids"], [token for item in sequences for token in item["input_ids"]])
        self.assertEqual(packed["sequence_lengths"], artifact["sequence_lengths"])
        self.assertEqual(
            packed["routed_experts"]["shape"][0],
            len(packed["input_ids"]),
        )
        self.assertEqual(len(artifact["individual"]["microbatches"]), 5)
        for sequence, microbatch in zip(sequences, artifact["individual"]["microbatches"]):
            self.assertEqual(len(microbatch["input_ids"]), 8704)
            self.assertEqual(
                microbatch["loss_mask"],
                [True] * len(sequence["input_ids"])
                + [False] * (8704 - len(sequence["input_ids"])),
            )
            self.assertEqual(microbatch["route_mask"], microbatch["loss_mask"])
            self.assertEqual(microbatch["routed_experts"]["shape"][0], 8704)
        synthetic.validate_artifact(artifact)

    def test_prompt_logprob_alignment_and_route_tamper_rejection(self):
        sequences = synthetic.make_synthetic_sequences()

        def bad_prompt(_endpoint, request, _timeout):
            response = response_for(request["token_ids"])
            response["choices"][0]["prompt_logprobs"][2] = {"999": {"logprob": -1}}
            return response

        with self.assertRaisesRegex(synthetic.SyntheticParityError, "prompt logprob"):
            synthetic.build_artifact(
                endpoint="http://h200/inference/v1/generate",
                output_path=Path("unused.json"),
                base_model="base-model",
                post_json=bad_prompt,
                sequences=sequences,
            )

        def no_route(_endpoint, request, _timeout):
            response = response_for(request["token_ids"])
            del response["choices"][0]["routed_experts"]
            return response

        with self.assertRaisesRegex(synthetic.SyntheticParityError, "routed experts"):
            synthetic.build_artifact(
                endpoint="http://h200/inference/v1/generate",
                output_path=Path("unused.json"),
                base_model="base-model",
                post_json=no_route,
                sequences=sequences,
            )

        def post(_endpoint, request, _timeout):
            return response_for(request["token_ids"])

        artifact = synthetic.build_artifact(
            endpoint="http://h200/inference/v1/generate",
            output_path=Path("unused.json"),
            base_model="base-model",
            post_json=post,
            sequences=sequences,
        )
        changed = copy.deepcopy(artifact)
        changed["packed"]["routed_experts"]["data"] = base64.b64encode(b"bad").decode()
        with self.assertRaisesRegex(synthetic.SyntheticParityError, "packed route"):
            synthetic.validate_artifact(changed)

    def test_configs_are_native_fp8_no_lora_and_keep_gates(self):
        config_path = ROOT / "configs/trinity-h200-synthetic-packed-parity/trainer.toml"
        with config_path.open("rb") as stream:
            config = tomllib.load(stream)
        self.assertEqual(config["max_steps"], 1)
        self.assertTrue(config["enable_router_replay"])
        self.assertEqual(config["model"]["name"], "/data/trinity-thinking-fp8")
        self.assertNotIn("lora", config["model"])
        self.assertEqual((config["model"]["cp"], config["model"]["pp"], config["model"]["ep"]), (8, 8, 8))
        self.assertEqual(config["model"]["pp_transfer_sparse_indices"], False)
        self.assertEqual(config["model"]["quantization"]["type"], "fp8")
        self.assertEqual(config["optim"]["lr"], 0.0)
        self.assertEqual(config["parity_guard"], {
            "max_unmasked_mismatch_kl": 0.1,
            "max_masked_mismatch_kl": 0.5,
            "max_masked_fraction": 0.06,
        })


if __name__ == "__main__":
    unittest.main()
