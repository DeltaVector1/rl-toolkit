import base64
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "training_ops/runtime/h200/rescore_fixed_sequences.py"
SPEC = importlib.util.spec_from_file_location("fixed_sequence_rescore", SCRIPT)
assert SPEC and SPEC.loader
fixed = importlib.util.module_from_spec(SPEC)


def _load_module():
    SPEC.loader.exec_module(fixed)
    return fixed


fixed = _load_module()


class FixedSequenceRescoreTest(unittest.TestCase):
    def setUp(self):
        self.model = "policy-0-slot0"
        self.receipt = {
            "target_sha256": "a" * 64,
            "target_size": 123,
            "config_sha256": "b" * 64,
            "config_size": 456,
            "source_url": "http://h200b:18100/qualified/policy0",
            "lora_name": self.model,
            "server_instance": "h200b-instance-1",
        }
        self.lengths = [3, 2, 4, 3, 5]
        self.ids = list(range(100, 117))
        self.positions = []
        for length in self.lengths:
            self.positions.extend(range(length))
        self.mask = [False, True, True, False, True, False, True, True, True, False,
                     True, True, False, True, True, True, False]
        self.reference = [0.0] + [-0.1 * index for index in range(1, len(self.ids))]
        self.trainer = [0.0] + [-0.2 * index for index in range(1, len(self.ids))]
        self.record = {
            "input_ids": self.ids,
            "position_ids": self.positions,
            "sequence_lengths": self.lengths,
            "loss_mask": self.mask,
            "trainer_logprobs": self.trainer,
            "reference_logprobs": self.reference,
            "delta": [a - b for a, b in zip(self.trainer, self.reference)],
        }

    def _route(self, token_count):
        raw = bytearray()
        for index in range(token_count):
            raw.extend((0,) * 12)
            raw.extend((1 + index % 2, 2 + index % 2, 3 + index % 2, 4 + index % 2))
        return {
            "data": base64.b64encode(raw).decode(),
            "dtype": "uint8",
            "shape": [token_count, 8, 2],
            "start": 0,
        }

    def _response(self, values=None, ids=None):
        ids = self.ids if ids is None else ids
        values = self.reference if values is None else values
        return {
            "choices": [{"routed_experts": self._route(len(ids))}],
            "prompt_logprobs": [
                None if index == 0 else {str(token): {"logprob": value}}
                for index, (token, value) in enumerate(zip(ids, values))
            ],
        }

    def _run(self, response_values=None, get_json=None, **kwargs):
        calls = []
        post_response = kwargs.pop("post_response", None)
        policy_model = kwargs.pop("policy_model", self.model)

        def post(_endpoint, body, _timeout):
            calls.append(copy.deepcopy(body))
            if post_response is not None:
                return post_response
            start = sum(self.lengths[: len(calls) - 1])
            length = self.lengths[len(calls) - 1]
            values = response_values
            if values is not None:
                values = values[start:start + length]
            else:
                values = self.reference[start:start + length]
            return self._response(values, body["token_ids"])

        if get_json is None:
            def get_json(*_):
                return {"loaded_receipts": {self.model: self.receipt}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = fixed.run_rescore(
                reference_path=self._write_reference(directory),
                endpoint="http://h200b:18100/inference/v1/generate",
                readiness_endpoint="http://h200b:18100/inference/v1/staging-status",
                expected_policy_identity=self.receipt,
                policy_model=policy_model,
                fresh_salt_prefix="test-rescore",
                output_path=output,
                post_json=post,
                get_json=get_json,
                **kwargs,
            )
            saved = json.loads(output.read_text())
        return result, saved, calls

    def _write_reference(self, directory):
        path = Path(directory) / "parity-step1.json"
        path.write_text(json.dumps(self.record))
        return path

    def test_requests_are_exact_five_sequences_with_fresh_salts_and_no_tokenizer(self):
        with mock.patch.dict("sys.modules", {"transformers": None}):
            result, saved, calls = self._run()
        self.assertEqual(len(calls), 5)
        offset = 0
        for index, (call, length) in enumerate(zip(calls, self.lengths)):
            self.assertEqual(call["model"], self.model)
            self.assertEqual(call["token_ids"], self.ids[offset:offset + length])
            self.assertEqual(call["sampling_params"], {
                "max_tokens": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "prompt_logprobs": 1,
            })
            self.assertEqual(call["cache_salt"], f"test-rescore-{index}")
            offset += length
        self.assertEqual(len({call["cache_salt"] for call in calls}), 5)
        self.assertEqual(result["sequence_count"], 5)
        self.assertEqual(saved["sequence_count"], 5)

    def test_receipt_is_checked_before_and_after_each_request_and_all(self):
        statuses = []

        def get_json(*_):
            statuses.append(1)
            return {"loaded_receipts": {self.model: self.receipt}}

        self._run(get_json=get_json)
        self.assertEqual(len(statuses), 12)

    def test_receipt_drift_fails_closed(self):
        changed = dict(self.receipt, server_instance="h200b-instance-2")
        values = iter([self.receipt, changed])

        def get_json(*_):
            return {"loaded_receipts": {self.model: next(values)}}

        with self.assertRaisesRegex(fixed.FixedSequenceRescoreError, "receipt"):
            self._run(get_json=get_json)

    def test_missing_and_misaligned_prompt_logprobs_fail_closed(self):
        malformed = self._response()
        malformed["prompt_logprobs"] = malformed["prompt_logprobs"][:-1]
        with self.assertRaisesRegex(fixed.FixedSequenceRescoreError, "logprob"):
            self._run(response_values=None, post_response=malformed)

    def test_length_and_first_none_alignment_fail_closed(self):
        malformed = self._response()
        malformed["prompt_logprobs"][0] = {str(self.ids[0]): {"logprob": 0.0}}
        malformed["prompt_logprobs"] = malformed["prompt_logprobs"][: self.lengths[0]]
        with self.assertRaisesRegex(fixed.FixedSequenceRescoreError, "first"):
            self._run(post_response=malformed)

    def test_masked_padding_is_excluded_from_decision_but_reported(self):
        fresh = list(self.reference)
        fresh[-1] = 999.0
        result, _, _ = self._run(response_values=fresh)
        self.assertEqual(result["selected_tokens"], sum(self.mask))
        self.assertEqual(result["masked_tokens"], len(self.mask) - sum(self.mask))
        self.assertEqual(result["comparisons"]["fresh_vs_reference"]["selected"]["max_abs"], 0.0)
        self.assertEqual(result["decision"], "fresh_matches_stored_rollout_but_trainer_differs")

    def test_masked_padded_fifth_sentinels_are_allowed_and_excluded(self):
        self.assertEqual(self.lengths[-1], 5)
        self.assertFalse(self.mask[-1])
        self.record["reference_logprobs"][-1] = -9999.0
        self.record["trainer_logprobs"][-1] = -9999.0
        self.record["delta"][-1] = -9999.0
        result, _, _ = self._run()
        self.assertEqual(result["masked_tokens"], len(self.mask) - sum(self.mask))
        self.assertEqual(
            result["comparisons"]["fresh_vs_reference"]["all"]["count"],
            len(self.ids) - 1,
        )
        self.assertEqual(
            result["comparisons"]["fresh_vs_trainer"]["all"]["count"],
            len(self.ids) - 1,
        )

    def test_masked_sentinel_delta_validation_is_skipped_only_for_masked_position(self):
        self.record["reference_logprobs"][-1] = -9999.0
        self.record["trainer_logprobs"][-1] = -9999.0
        self.record["delta"][-1] = -9999.0
        loaded = fixed.load_parity_record(self.record)
        self.assertEqual(loaded["reference_logprobs"][-1], -9999.0)
        self.assertEqual(loaded["trainer_logprobs"][-1], -9999.0)
        self.assertEqual(loaded["delta"][-1], -9999.0)

    def test_selected_stored_sentinel_fails_closed_for_each_field(self):
        for field in ("reference_logprobs", "trainer_logprobs", "delta"):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.record)
                changed[field][1] = -9999.0
                with self.assertRaisesRegex(fixed.FixedSequenceRescoreError, "selected|loss_mask"):
                    fixed.load_parity_record(changed)

    def test_masked_fresh_sentinel_is_excluded_and_selected_fresh_sentinel_fails(self):
        fresh = list(self.reference)
        fresh[-1] = -9999.0
        result, _, _ = self._run(response_values=fresh)
        self.assertEqual(
            result["comparisons"]["fresh_vs_reference"]["all"]["count"],
            len(self.ids) - 1,
        )
        fresh[1] = -9999.0
        with self.assertRaisesRegex(fixed.FixedSequenceRescoreError, "selected|loss_mask"):
            self._run(response_values=fresh)

    def test_decision_classes_cover_all_three_diagnoses(self):
        fresh_reference = list(self.reference)
        fresh_trainer = list(self.trainer)
        result, _, _ = self._run(response_values=fresh_reference)
        self.assertEqual(result["decision"], "fresh_matches_stored_rollout_but_trainer_differs")
        result, _, _ = self._run(response_values=[value + 1.0 for value in self.reference])
        self.assertEqual(result["decision"], "fresh_differs_from_stored_rollout")
        result, _, _ = self._run(response_values=fresh_trainer)
        self.assertEqual(result["decision"], "fresh_matches_trainer_but_stored_rollout_differs")

    def test_output_contains_only_hashes_counts_numeric_stats_and_identity(self):
        _, saved, _ = self._run()
        encoded = json.dumps(saved)
        for forbidden in ("input_ids", "token_ids", "prompt_logprobs", '"response":', '"data":'):
            self.assertNotIn(forbidden, saved)
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(len(saved["sequences"]), 5)
        for sequence in saved["sequences"]:
            self.assertEqual(len(sequence["sequence_sha256"]), 64)
            self.assertEqual(len(sequence["route_sha256"]), 64)
            self.assertIn("comparisons", sequence)
        self.assertEqual(saved["policy_identity"], self.receipt)

    def test_policy_model_mismatch_is_rejected(self):
        with self.assertRaisesRegex(fixed.FixedSequenceRescoreError, "LoRA"):
            self._run(policy_model="policy-0-slot1")

    def test_delta_and_input_arrays_are_validated(self):
        changed = copy.deepcopy(self.record)
        changed["delta"][1] += 1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parity-step1.json"
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(fixed.FixedSequenceRescoreError, "delta"):
                fixed.load_parity_record(path)


if __name__ == "__main__":
    unittest.main()
