import unittest
from unittest.mock import patch

from training_ops import prepare_policy0_qualification
from training_ops.runtime.h200.prepare_trinity_policy0_qualification import (
    QualificationError,
    validate_lora_config,
    validate_reference,
)


EXPECTED_LORA = {
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
        "down_proj", "shared_expert", "experts",
    ],
}


def reference_payload():
    return {
        "policy": "policy-0",
        "request_mode": "serialized_generate_completion_prompt_logprobs",
        "source_kind": "operator_non_rollout",
        "input_ids": [10, 11, 12, 13],
        "inference_logprobs": [0.0, -1.0, -2.0, -3.0],
        "loss_mask": [False, False, True, True],
        "generated_token_ids": [12, 13],
        "generated_token_count": 2,
        "sequence_lengths": [4],
        "routed_experts": {"shape": [4, 60, 4]},
        "policy_identity_before": {"server_instance": "h200"},
        "policy_identity_after": {"server_instance": "h200"},
        "cache_salt": "fresh-policy-0",
    }


class TrinityPolicy0StagingTest(unittest.TestCase):
    def test_validates_exact_adapter_settings(self):
        self.assertEqual(validate_lora_config(EXPECTED_LORA, EXPECTED_LORA)["r"], 32)
        changed = dict(EXPECTED_LORA)
        changed["target_modules"] = list(EXPECTED_LORA["target_modules"][:-1])
        with self.assertRaisesRegex(QualificationError, "target_modules"):
            validate_lora_config(changed, EXPECTED_LORA)

    def test_rejects_invalid_reference_alignment_or_cached_rollout(self):
        self.assertEqual(validate_reference(reference_payload())["policy"], "policy-0")
        changed = reference_payload()
        changed["source_kind"] = "rollout_batch"
        with self.assertRaisesRegex(QualificationError, "cached rollout"):
            validate_reference(changed)
        changed = reference_payload()
        changed["routed_experts"] = {"shape": [3, 60, 4]}
        with self.assertRaisesRegex(QualificationError, "routes"):
            validate_reference(changed)

    def test_plan_uses_all_configured_trainers_without_live_action(self):
        plan = prepare_policy0_qualification(
            "configs/trinity-h200-parity-production", apply=False
        )
        self.assertEqual(
            plan["nodes"], ["n76", "n77", "n78", "n80", "n82", "n83", "n84", "n85"]
        )
        self.assertNotIn("n79", plan["nodes"])
        self.assertTrue(plan["adapter_path"].endswith("/policy-0/adapter_model.safetensors"))
        self.assertTrue(plan["reference_path"].endswith("/trinity-h200-parity-production/reference.json"))
        self.assertIn("serialized_generate_completion_prompt_logprobs", plan["verify_script"])
        self.assertIn("sha256sum", plan["verify_script"])

    def test_stage_accepts_all_verified_hosts_and_refuses_one_bad_hash(self):
        manifest = "\n".join([
            "QUAL_FILE adapter_model.safetensors " + "a" * 64 + " 10",
            "QUAL_FILE adapter_config.json " + "b" * 64 + " 20",
            "QUAL_FILE reference.json " + "c" * 64 + " 30",
        ])
        with patch("training_ops._ssh", side_effect=[(0, "verified"), (0, manifest)]) as ssh:
            with patch("training_ops._copy_resume_adapter") as copy:
                with patch(
                    "training_ops._fan",
                    return_value={host: (0, "verified") for host in [
                        "n76", "n77", "n78", "n80", "n82", "n83", "n84", "n85"
                    ]},
                ) as fan:
                    result = prepare_policy0_qualification(
                        "configs/trinity-h200-parity-production", apply=True
                    )
        self.assertTrue(result["staged"])
        self.assertEqual(copy.call_count, 3)
        self.assertEqual(fan.call_count, 1)
        self.assertGreaterEqual(ssh.call_count, 1)

        bad_results = {
            host: (0 if host != "n83" else 1, "bad" if host == "n83" else "verified")
            for host in ["n76", "n77", "n78", "n80", "n82", "n83", "n84", "n85"]
        }
        with patch("training_ops._ssh", side_effect=[(0, "verified"), (0, manifest)]):
            with patch("training_ops._copy_resume_adapter"):
                with patch("training_ops._fan", return_value=bad_results):
                    with self.assertRaisesRegex(RuntimeError, "verification failed"):
                        prepare_policy0_qualification(
                            "configs/trinity-h200-parity-production", apply=True
                        )


if __name__ == "__main__":
    unittest.main()
