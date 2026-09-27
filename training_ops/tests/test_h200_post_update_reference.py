import base64
import ast
import copy
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import training_ops
import training_ops.runtime.h200.post_update_qualification as post_update
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
            "request_id": "server-response-default",
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

    def test_validator_recomputes_stored_request_and_response_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.build(Path(directory) / "reference.json")
        for field in ("request_sha256", "response_sha256"):
            changed = copy.deepcopy(result)
            changed[field]["decode"] = "0" * 64
            with self.subTest(field=field), self.assertRaises(PostUpdateQualificationError):
                validate_post_update_reference(
                    changed,
                    expected_seq_len=SEQ_LEN,
                    expected_policy_identity=self.receipt(),
                )

    def test_builder_adds_distinct_request_ids_and_binds_response_fields(self):
        records = []
        responses = []
        for index in range(8):
            record = self.input_record()
            record["input_ids"] = [token + index * 10000 for token in record["input_ids"]]
            records.append(record)
            response = self.response()
            response["request_id"] = f"server-response-{index}"
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
        ids = [item["request_bodies"]["decode"].get("request_id") for item in result["microbatches"]]
        self.assertEqual(len(ids), 8)
        self.assertEqual(len(set(ids)), 8)
        self.assertTrue(all(isinstance(value, str) and value for value in ids))
        server_ids = [item["responses"]["decode"].get("request_id") for item in result["microbatches"]]
        self.assertEqual(len(set(server_ids)), 8)
        for item in result["microbatches"]:
            self.assertEqual(
                item["qualification_binding"]["request_sha256"],
                item["request_sha256"]["decode"],
            )
            self.assertEqual(
                item["qualification_binding"]["client_request_id"],
                item["request_bodies"]["decode"]["request_id"],
            )
            self.assertEqual(
                item["qualification_binding"]["server_response_id"],
                item["responses"]["decode"]["request_id"],
            )

        duplicate_server = copy.deepcopy(result)
        duplicate_server["microbatches"][1]["responses"]["decode"]["request_id"] = server_ids[0]
        duplicate_server["microbatches"][1]["server_response_id"] = server_ids[0]
        duplicate_server["microbatches"][1]["qualification_binding"] = post_update._post_update_binding(
            duplicate_server["microbatches"][1]
        )
        duplicate_server["microbatches"][1]["response_sha256"]["decode"] = duplicate_server["microbatches"][1]["qualification_binding"]["response_sha256"]
        with self.assertRaises(PostUpdateQualificationError):
            validate_post_update_references(
                duplicate_server,
                expected_seq_len=SEQ_LEN,
                expected_policy_identity=self.receipt(),
            )


    def test_server_response_id_may_differ_from_client_request_id(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.build(Path(directory) / "reference.json")
        client_id = result["qualification_binding"]["client_request_id"]
        server_id = result["qualification_binding"]["server_response_id"]
        self.assertNotEqual(client_id, server_id)
        self.assertEqual(server_id, "server-response-default")

    def test_failed_post_publishes_raw_id_diagnostic_without_clobbering(self):
        output = None
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reference.json"
            with self.assertRaises(PostUpdateQualificationError):
                build_post_update_reference(
                    endpoint="http://h200b:18100/inference/v1/generate",
                    input_record=self.input_record(),
                    output_path=output,
                    base_model="/data/trinity-thinking-fp8",
                    policy_model="policy-0-slot0",
                    expected_policy_identity=self.receipt(),
                    policy_receipt=self.receipt(),
                    cache_salt="fresh-post-update-decode",
                    readiness_endpoint="http://h200b:18100/inference/v1/staging-status",
                    post_json=lambda *_: {"unexpected": True},
                    get_json=lambda *_: self.receipt(),
                )
            diagnostic = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(diagnostic["kind"], "post_update_post_failure")
            self.assertNotIn("request_id", diagnostic["response"])
            original = output.read_bytes()
            with self.assertRaises(PostUpdateQualificationError):
                build_post_update_reference(
                    endpoint="http://h200b:18100/inference/v1/generate",
                    input_record=self.input_record(),
                    output_path=output,
                    base_model="/data/trinity-thinking-fp8",
                    policy_model="policy-0-slot0",
                    expected_policy_identity=self.receipt(),
                    policy_receipt=self.receipt(),
                    cache_salt="fresh-post-update-decode",
                    readiness_endpoint="http://h200b:18100/inference/v1/staging-status",
                    post_json=lambda *_: {"request_id": "other-server-id"},
                    get_json=lambda *_: self.receipt(),
                )
            self.assertEqual(output.read_bytes(), original)

    def test_eight_reference_failure_keeps_diagnostic_at_final_path(self):
        records = []
        for index in range(8):
            record = self.input_record()
            record["input_ids"] = [token + index * 10000 for token in record["input_ids"]]
            records.append(record)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reference.json"
            with self.assertRaises(PostUpdateQualificationError):
                build_post_update_references(
                    endpoint="http://h200b:18100/inference/v1/generate",
                    input_records=records,
                    output_path=output,
                    base_model="/data/trinity-thinking-fp8",
                    policy_model="policy-0-slot0",
                    expected_policy_identity=self.receipt(),
                    policy_receipt=self.receipt(),
                    cache_salt="fresh-post-update-decode",
                    readiness_endpoint="http://h200b:18100/inference/v1/staging-status",
                    post_json=lambda *_: {"choices": []},
                    get_json=lambda *_: self.receipt(),
                )
            diagnostic = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(diagnostic["kind"], "post_update_post_failure")
            self.assertEqual(diagnostic["response"], {"choices": []})

    def test_validator_rejects_reused_salts_and_unbound_response_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.build(Path(directory) / "reference.json")
        changed = copy.deepcopy(result)
        changed["cache_salt"] = "same-salt"
        changed["request_bodies"]["decode"]["cache_salt"] = "same-salt"
        changed["request_bodies"]["decode"].pop("request_id")
        changed["request_bodies"]["decode"]["request_id"] = post_update._json_sha256(
            changed["request_bodies"]["decode"]
        )
        changed["qualification_binding"] = post_update._post_update_binding(changed)
        with self.assertRaises(PostUpdateQualificationError):
            validate_post_update_reference(
                changed,
                expected_seq_len=SEQ_LEN,
                expected_policy_identity=self.receipt(),
            )
        changed = copy.deepcopy(result)
        changed["request_bodies"]["decode"]["sampling_params"]["temperature"] = 0.2
        changed["request_bodies"]["decode"].pop("request_id")
        changed["request_bodies"]["decode"]["request_id"] = post_update._json_sha256(
            changed["request_bodies"]["decode"]
        )
        changed["qualification_binding"] = post_update._post_update_binding(changed)
        with self.assertRaises(PostUpdateQualificationError):
            validate_post_update_reference(
                changed,
                expected_seq_len=SEQ_LEN,
                expected_policy_identity=self.receipt(),
            )
        changed = copy.deepcopy(result)
        changed["responses"]["decode"]["choices"][0]["routed_experts"]["data"] = base64.b64encode(b"bad").decode()
        changed["qualification_binding"] = post_update._post_update_binding(changed)
        with self.assertRaises(PostUpdateQualificationError):
            validate_post_update_reference(
                changed,
                expected_seq_len=SEQ_LEN,
                expected_policy_identity=self.receipt(),
            )
        changed = copy.deepcopy(result)
        changed["generated_token_ids"][0] += 1
        changed["qualification_binding"] = post_update._post_update_binding(changed)
        with self.assertRaises(PostUpdateQualificationError):
            validate_post_update_reference(
                changed,
                expected_seq_len=SEQ_LEN,
                expected_policy_identity=self.receipt(),
            )
        changed = copy.deepcopy(result)
        changed["inference_logprobs"][-1] = -0.5
        changed["qualification_binding"] = post_update._post_update_binding(changed)
        with self.assertRaises(PostUpdateQualificationError):
            validate_post_update_reference(
                changed,
                expected_seq_len=SEQ_LEN,
                expected_policy_identity=self.receipt(),
            )

    def test_builder_rejects_duplicate_inputs_before_requests(self):
        calls = []
        records = [self.input_record() for _ in range(8)]
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(PostUpdateQualificationError):
            build_post_update_references(
                endpoint="http://h200b:18100/inference/v1/generate",
                input_records=records,
                output_path=Path(directory) / "reference.json",
                base_model="/data/trinity-thinking-fp8",
                policy_model="policy-0-slot0",
                expected_policy_identity=self.receipt(),
                policy_receipt=self.receipt(),
                cache_salt="fresh-post-update-decode",
                readiness_endpoint="http://h200b:18100/inference/v1/staging-status",
                post_json=lambda *_: calls.append(True) or self.response(),
                get_json=lambda *_: self.receipt(),
            )
        self.assertEqual(calls, [])

    def test_final_reference_publication_is_no_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reference.json"
            self.build(output)
            original = output.read_bytes()
            with self.assertRaises(PostUpdateQualificationError):
                self.build(output)
            self.assertEqual(output.read_bytes(), original)

    def test_qualify_requires_caller_supplied_reference_identity(self):
        config = {
            "data": {"fake": {"batch_size": 8}},
            "model": {"seq_len": 8192},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text(json.dumps({"schema_version": 1, "microbatches": []}))
            with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(training_ops.Refused):
                training_ops._validate_post_update_qualification_reference(config, path)

    def test_remote_reference_validation_does_not_require_controller_path_and_cleans_temp(self):
        payload = b"remote-reference"
        digest = hashlib.sha256(payload).hexdigest()
        expected = {
            "final_reference_sha256": digest,
            "final_reference_size": len(payload),
            "server_instance": self.receipt()["server_instance"],
            "adapter_sha256": self.receipt()["target_sha256"],
            "adapter_size": self.receipt()["target_size"],
            "config_sha256": self.receipt()["config_sha256"],
            "config_size": self.receipt()["config_size"],
        }
        identity_output = f"QUAL_FILE reference.json {digest} {len(payload)}\n"
        copied = []

        def fake_scp(command, **kwargs):
            target = Path(command[-1])
            target.write_bytes(payload)
            copied.append(target)
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch.object(training_ops, "_ssh", return_value=(0, identity_output)), mock.patch.object(
            training_ops, "_run_process", side_effect=fake_scp
        ), mock.patch.object(
            training_ops,
            "_policy0_post_update_reference_copy",
            return_value={"sha256": digest, "size": len(payload)},
        ) as strict:
            result = training_ops._validate_post_update_remote_reference(
                {"data": {"fake": {"batch_size": 8}}, "model": {"seq_len": 8192}},
                "/data/qualifications/post-update/reference.json",
                expected,
            )
        self.assertEqual(result, {"sha256": digest, "size": len(payload)})
        strict.assert_called_once()
        self.assertEqual(len(copied), 1)
        self.assertFalse(copied[0].exists())

    def test_remote_reference_rejects_identity_mismatch_or_source_change_before_strict_validation(self):
        payload = b"remote-reference"
        digest = hashlib.sha256(payload).hexdigest()
        expected = {
            "final_reference_sha256": digest,
            "final_reference_size": len(payload),
            "server_instance": self.receipt()["server_instance"],
            "adapter_sha256": self.receipt()["target_sha256"],
            "adapter_size": self.receipt()["target_size"],
            "config_sha256": self.receipt()["config_sha256"],
            "config_size": self.receipt()["config_size"],
        }
        wrong = "QUAL_FILE reference.json " + "f" * 64 + " 99\n"
        with mock.patch.object(training_ops, "_ssh", return_value=(0, wrong)), mock.patch.object(
            training_ops, "_run_process"
        ) as scp, mock.patch.object(training_ops, "_policy0_post_update_reference_copy") as strict:
            with self.assertRaisesRegex(training_ops.Refused, "identity"):
                training_ops._validate_post_update_remote_reference(
                    {}, "/data/qualifications/post-update/reference.json", expected
                )
        scp.assert_not_called()
        strict.assert_not_called()

        changed = b"changed-reference"
        changed_digest = hashlib.sha256(changed).hexdigest()
        outputs = iter([
            f"QUAL_FILE reference.json {digest} {len(payload)}\n",
            f"QUAL_FILE reference.json {changed_digest} {len(changed)}\n",
        ])

        def fake_scp_change(command, **kwargs):
            Path(command[-1]).write_bytes(payload)
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch.object(training_ops, "_ssh", side_effect=lambda *args, **kwargs: (0, next(outputs))), mock.patch.object(
            training_ops, "_run_process", side_effect=fake_scp_change
        ), mock.patch.object(training_ops, "_policy0_post_update_reference_copy") as strict:
            with self.assertRaisesRegex(training_ops.Refused, "changed"):
                training_ops._validate_post_update_remote_reference(
                    {}, "/data/qualifications/post-update/reference.json", expected
                )
        strict.assert_not_called()

    def test_remote_identity_script_rejects_missing_and_symlink_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing-reference.json"
            result = subprocess.run(
                ["bash", "-c", training_ops._policy0_reference_source_identity_script(str(missing))],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            target = root / "target-reference.json"
            target.write_bytes(b"reference")
            alias = root / "reference.json"
            alias.symlink_to(target)
            result = subprocess.run(
                ["bash", "-c", training_ops._policy0_reference_source_identity_script(str(alias))],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)

    def test_remote_fetch_and_strict_failure_cleanup_temporary_copy(self):
        payload = b"remote-reference"
        digest = hashlib.sha256(payload).hexdigest()
        expected = {
            "final_reference_sha256": digest,
            "final_reference_size": len(payload),
            "server_instance": self.receipt()["server_instance"],
            "adapter_sha256": self.receipt()["target_sha256"],
            "adapter_size": self.receipt()["target_size"],
            "config_sha256": self.receipt()["config_sha256"],
            "config_size": self.receipt()["config_size"],
        }
        identity_output = f"QUAL_FILE reference.json {digest} {len(payload)}\n"
        for fetch_result, strict_error in ((1, None), (0, "strict mutation")):
            paths = []

            def fake_fetch(command, **kwargs):
                target = Path(command[-1])
                target.write_bytes(payload)
                paths.append(target)
                return mock.Mock(returncode=fetch_result, stdout="", stderr="fetch failed")

            strict_side_effect = (
                mock.Mock(side_effect=training_ops.Refused(strict_error))
                if strict_error
                else mock.Mock(return_value={"sha256": digest, "size": len(payload)})
            )
            with self.subTest(fetch_result=fetch_result, strict_error=strict_error), mock.patch.object(
                training_ops, "_ssh", return_value=(0, identity_output)
            ), mock.patch.object(training_ops, "_run_process", side_effect=fake_fetch), mock.patch.object(
                training_ops, "_policy0_post_update_reference_copy", strict_side_effect
            ):
                if fetch_result:
                    message = "fetch failed"
                else:
                    message = "strict mutation"
                with self.assertRaisesRegex(training_ops.Refused, message):
                    training_ops._validate_post_update_remote_reference(
                        {}, "/data/qualifications/post-update/reference.json", expected
                    )
            self.assertEqual(len(paths), 1)
            self.assertFalse(paths[0].exists())

    def test_qualify_stops_before_loss_or_launch_when_remote_reference_fails(self):
        events = []
        with mock.patch.object(
            training_ops,
            "_validate_post_update_remote_reference",
            side_effect=lambda *args: events.append("reference") or (_ for _ in ()).throw(training_ops.Refused("remote reference")),
        ), mock.patch.object(training_ops, "_trinity_trainer_loss_preflight", side_effect=lambda *args: events.append("loss")), mock.patch.object(
            training_ops, "_deploy_dependencies", side_effect=lambda *args: events.append("deploy")
        ), mock.patch.object(training_ops, "_ssh") as ssh:
            with self.assertRaisesRegex(training_ops.Refused, "remote reference"):
                training_ops.qualify(
                    "remote-reference-failure",
                    nodes=["n82"],
                    config_dir="configs/trinity-h200-post-update-qualification",
                )
        self.assertEqual(events, ["reference"])
        ssh.assert_not_called()

    def test_export_mode_rejects_singleton_reference_cycling(self):
        source = (ROOT / "prime-rl/src/prime_rl/trainer/rl/data.py").read_text()
        self.assertIn(
            "config.export_final_adapter and len(self.references) != self.num_micro_batches",
            source,
        )

    def test_builder_cli_requires_exactly_eight_records(self):
        arguments = [
            "--endpoint", "http://h200b:18100/inference/v1/generate",
            "--output", "/tmp/post-update-reference.json",
            "--base-model", "/data/trinity-thinking-fp8",
            "--policy-model", "policy-0-slot0",
            "--policy", "policy-0",
            "--expected-policy-identity", "/tmp/expected.json",
            "--policy-receipt", "/tmp/receipt.json",
            "--cache-salt", "fresh-post-update-decode",
            "--readiness-endpoint", "http://h200b:18100/inference/v1/staging-status",
        ]
        for index in range(7):
            arguments.extend(["--input-record", f"/tmp/input-{index}.json"])
        with self.assertRaises(SystemExit):
            post_update.main(arguments)

    def test_builder_cli_forwards_policy_and_rejects_ambiguous_policy(self):
        arguments = [
            "--endpoint", "http://h200b:18100/inference/v1/generate",
            "--output", "/tmp/post-update-reference.json",
            "--base-model", "/data/trinity-thinking-fp8",
            "--policy-model", "trinity-h200-post-update-qualification-slot1",
            "--policy", "policy-1",
            "--expected-policy-identity", "/tmp/expected.json",
            "--policy-receipt", "/tmp/receipt.json",
            "--cache-salt", "fresh-post-update-decode",
            "--readiness-endpoint", "http://h200b:18100/inference/v1/staging-status",
        ]
        for index in range(8):
            arguments.extend(["--input-record", f"/tmp/input-{index}.json"])

        with mock.patch.object(
            post_update, "build_post_update_references", return_value={}
        ) as build:
            post_update.main(arguments)
        self.assertEqual(build.call_args.kwargs["policy"], "policy-1")

        for policy_args in ([], ["--policy", "policy-2"]):
            invalid = [item for item in arguments if item not in {"--policy", "policy-1"}]
            invalid.extend(policy_args)
            with mock.patch.object(
                post_update, "build_post_update_references", return_value={}
            ) as build:
                with self.assertRaises(SystemExit):
                    post_update.main(invalid)
                build.assert_not_called()

    def test_eight_microbatches_are_distinct_actual_decode_references(self):
        records = []
        responses = []
        for index in range(8):
            record = self.input_record()
            record["input_ids"] = [token + index * 10000 for token in record["input_ids"]]
            records.append(record)
            response = self.response()
            response["request_id"] = f"server-response-{index}"
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
            "_validate_post_update_remote_reference",
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
