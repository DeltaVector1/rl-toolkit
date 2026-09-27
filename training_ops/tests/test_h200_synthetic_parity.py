import base64
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import tomllib

import training_ops


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
            if layer < 6:
                raw.extend((0,) * top_k)
            else:
                raw.extend((layer + 1, layer + 2)[:top_k])
    return {
        "data": base64.b64encode(raw).decode(),
        "dtype": "uint8",
        "shape": [token_count, layers, top_k],
        "start": 0,
    }


def response_for(tokens: list[int]) -> dict:
    return {
        "prompt_logprobs": [
            None,
            *[
                {str(token): {"logprob": -0.01 * index}}
                for index, token in enumerate(tokens[1:], 1)
            ],
        ],
        "choices": [
            {
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
                output_dir=Path(directory),
                base_model="base-model",
                post_json=post,
                sequences=sequences,
            )
            written = {
                "manifest": json.loads((Path(directory) / "manifest.json").read_text()),
                "packed_reference": json.loads((Path(directory) / "packed-reference.json").read_text()),
                "individual_reference": json.loads((Path(directory) / "individual-reference.json").read_text()),
            }
            self.assertTrue((Path(directory) / "manifest.json").is_file())
            self.assertTrue((Path(directory) / "packed-reference.json").is_file())
            self.assertTrue((Path(directory) / "individual-reference.json").is_file())
        self.assertEqual(len(calls), 5)
        self.assertEqual(artifact["manifest"], written["manifest"])
        self.assertEqual(artifact["packed_reference"], written["packed_reference"])
        self.assertEqual(artifact["individual_reference"], written["individual_reference"])
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
        self.assertEqual(artifact["manifest"]["route_construction"], "byte_concat(individual_routes)")
        self.assertEqual(artifact["manifest"]["diagnostic_name"], synthetic.DIAGNOSTIC_NAME)
        self.assertEqual(artifact["manifest"]["token_count"], 32768)
        self.assertEqual(artifact["manifest"]["pp_transfer_sparse_indices"], False)
        route_fields = {"schema_version", "encoding", "dtype", "shape", "dense_layers", "experts", "sha256", "data"}
        packed_reference = written["packed_reference"]
        self.assertEqual(sum(packed_reference["sequence_lengths"]), len(packed_reference["input_ids"]))
        self.assertEqual(len(packed_reference["inference_logprobs"]), len(packed_reference["input_ids"]))
        self.assertEqual(set(packed_reference["routed_experts"]), route_fields)
        individual_reference = written["individual_reference"]
        self.assertEqual(individual_reference["schema_version"], 2)
        self.assertEqual(len(individual_reference["microbatches"]), 5)
        for sequence, microbatch in zip(sequences, artifact["individual"]["microbatches"]):
            self.assertEqual(len(microbatch["input_ids"]), 8704)
            self.assertEqual(
                microbatch["loss_mask"],
                [False] + [True] * (len(sequence["input_ids"]) - 1)
                + [False] * (8704 - len(sequence["input_ids"])),
            )
            self.assertEqual(microbatch["route_mask"], microbatch["loss_mask"])
            self.assertEqual(microbatch["routed_experts"]["shape"][0], 8704)
            self.assertEqual(sum(microbatch["sequence_lengths"]), len(microbatch["input_ids"]))
            self.assertEqual(set(microbatch["routed_experts"]), route_fields)
        synthetic.validate_artifact(artifact)


    def test_prompt_logprob_alignment_and_route_tamper_rejection(self):
        sequences = synthetic.make_synthetic_sequences()

        def bad_prompt(_endpoint, request, _timeout):
            response = response_for(request["token_ids"])
            response["prompt_logprobs"][2] = {"999": {"logprob": -1}}
            return response

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(synthetic.SyntheticParityError, "prompt logprob"):
                synthetic.build_artifact(
                    endpoint="http://h200/inference/v1/generate",
                    output_dir=Path(directory),
                    base_model="base-model",
                    post_json=bad_prompt,
                    sequences=sequences,
                )

        def no_route(_endpoint, request, _timeout):
            response = response_for(request["token_ids"])
            del response["choices"][0]["routed_experts"]
            return response

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(synthetic.SyntheticParityError, "routed experts"):
                synthetic.build_artifact(
                    endpoint="http://h200/inference/v1/generate",
                    output_dir=Path(directory),
                    base_model="base-model",
                    post_json=no_route,
                    sequences=sequences,
                )

        def post(_endpoint, request, _timeout):
            return response_for(request["token_ids"])

        with tempfile.TemporaryDirectory() as directory:
            artifact = synthetic.build_artifact(
                endpoint="http://h200/inference/v1/generate",
                output_dir=Path(directory),
                base_model="base-model",
                post_json=post,
                sequences=sequences,
            )
            changed = copy.deepcopy(artifact)
            changed["packed"]["routed_experts"]["data"] = base64.b64encode(b"bad").decode()
            with self.assertRaisesRegex(synthetic.SyntheticParityError, "packed route"):
                synthetic.validate_artifact(changed)
            with self.assertRaisesRegex(synthetic.SyntheticParityError, "already exists"):
                synthetic.build_artifact(
                    endpoint="http://h200/inference/v1/generate",
                    output_dir=Path(directory),
                    base_model="base-model",
                    post_json=post,
                    sequences=sequences,
                )

    def test_top_level_prompt_logprobs_are_required(self):
        def choice_only(_endpoint, request, _timeout):
            response = response_for(request["token_ids"])
            response["choices"][0]["prompt_logprobs"] = response.pop("prompt_logprobs")
            return response

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(synthetic.SyntheticParityError, "top-level"):
                synthetic.build_artifact(
                    endpoint="http://h200/inference/v1/generate",
                    output_dir=Path(directory),
                    base_model="base-model",
                    post_json=choice_only,
                )

    def test_comparison_binds_positions_and_detects_tamper(self):
        def post(_endpoint, request, _timeout):
            return response_for(request["token_ids"])

        with tempfile.TemporaryDirectory() as directory:
            artifact = synthetic.build_artifact(
                endpoint="http://h200/inference/v1/generate",
                output_dir=Path(directory),
                base_model="base-model",
                post_json=post,
            )
            packed = copy.deepcopy(artifact["packed_reference"])
            individual = copy.deepcopy(artifact["individual_reference"])
            packed_reference = packed.pop("inference_logprobs")
            packed["reference_logprobs"] = packed_reference
            packed["trainer_logprobs"] = [
                value + (0.0002 if flag else 5.0)
                for value, flag in zip(packed_reference, packed["loss_mask"])
            ]
            for index, microbatch in enumerate(individual["microbatches"]):
                microbatch["microbatch_index"] = index
                microbatch["microbatch_count"] = 5
                reference_values = microbatch.pop("inference_logprobs")
                microbatch["reference_logprobs"] = reference_values
                microbatch["trainer_logprobs"] = [
                    value + (0.0002 if flag else 5.0)
                    for value, flag in zip(reference_values, microbatch["loss_mask"])
                ]
            report = synthetic.compare_packed_individual(packed, individual)
            self.assertEqual(report["token_count"], 32768)
            self.assertEqual(report["totals"]["packed_vs_individual"]["count"], 32763)
            self.assertEqual(report["totals"]["packed_vs_reference"]["count"], 32763)
            self.assertEqual(report["totals"]["packed_vs_reference"]["mismatch_count"], 32763)
            self.assertGreater(report["totals"]["packed_vs_reference"]["mean_kl"], 0.0)
            individual["microbatches"][0]["input_ids"][1] += 1
            with self.assertRaisesRegex(synthetic.SyntheticParityError, "input IDs"):
                synthetic.compare_packed_individual(packed, individual)

    def test_endpoint_requires_exact_serialized_generate_path(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(synthetic.SyntheticParityError, "inference/v1/generate"):
                synthetic.build_artifact(
                    endpoint="http://h200/v1/generate",
                    output_dir=Path(directory),
                )

    def test_configs_are_native_fp8_no_lora_and_keep_gates(self):
        config_specs = (
            (
                ROOT / "configs/trinity-h200-synthetic-packed-parity",
                32768,
                1,
                "packed-reference.json",
                "packed-run",
            ),
            (
                ROOT / "configs/trinity-h200-synthetic-individual-parity",
                8704,
                5,
                "individual-reference.json",
                "individual-run",
            ),
        )
        for config_root, expected_seq_len, expected_batch, reference_name, output_name in (
            config_specs
        ):
            config_path = config_root / "trainer.toml"
            self.assertTrue(config_path.is_file())
            with config_path.open("rb") as stream:
                config = tomllib.load(stream)
            self.assertEqual(config["max_steps"], 1)
            self.assertTrue(config["enable_router_replay"])
            self.assertEqual(config["data"]["fake"]["batch_size"], expected_batch)
            self.assertTrue(config["data"]["fake"]["reference_path"].endswith(reference_name))
            self.assertTrue(config["output_dir"].endswith(output_name))
            self.assertEqual(config["model"]["seq_len"], expected_seq_len)
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
        selected_packed = training_ops._config_dir(
            "configs/trinity-h200-synthetic-packed-parity",
            require_inference=True,
        )
        selected_individual = training_ops._config_dir(
            "configs/trinity-h200-synthetic-individual-parity"
        )
        self.assertEqual(
            selected_packed["trainer"],
            "configs/trinity-h200-synthetic-packed-parity/trainer.toml",
        )
        self.assertEqual(
            selected_individual["trainer"],
            "configs/trinity-h200-synthetic-individual-parity/trainer.toml",
        )
        with (ROOT / "configs/trinity-h200-synthetic-packed-parity/inference.toml").open("rb") as stream:
            inference = tomllib.load(stream)
        self.assertFalse(inference["enable_lora"])
        self.assertEqual(inference["model"]["max_model_len"], 32768)


if __name__ == "__main__":
    unittest.main()
