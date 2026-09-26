import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

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
            "configs/trinity-h200-parity-production",
            apply=False,
            source_adapter_dir="/data/glm52-rl/preserved-step0",
            source_reference_path="/data/glm52-rl/fresh-h200/reference.json",
        )
        self.assertEqual(
            plan["nodes"], ["n76", "n77", "n78", "n80", "n82", "n83", "n84", "n85"]
        )
        self.assertNotIn("n79", plan["nodes"])
        self.assertTrue(plan["adapter_path"].endswith("/policy-0/adapter_model.safetensors"))
        self.assertTrue(plan["reference_path"].endswith("/trinity-h200-parity-production/reference.json"))
        self.assertEqual(plan["source_adapter_host"], "n82")
        self.assertEqual(plan["source_reference_host"], "h200b")
        self.assertIn("preserved-step0/adapter_model.safetensors", plan["source_adapter_script"])
        self.assertIn("fresh-h200/reference.json", plan["transfer_reference_script"])
        self.assertIn("serialized_generate_completion_prompt_logprobs", plan["verify_script"])
        self.assertIn("sha256sum", plan["verify_script"])
        self.assertIn("read(1024 * 1024)", plan["verify_script"])
        self.assertNotIn('open(path, "rb").read()', plan["verify_script"])
        self.assertIn("destination conflicts", plan["stage_adapter_script"])
        self.assertIn("trap 'rm -f -- \"$tmp\"' EXIT", plan["stage_adapter_script"])
        self.assertIn("tmp.policy0", plan["transfer_reference_script"])
        self.assertIn("china-trainer", plan["transfer_reference_script"])
        self.assertIn("scp -q", plan["transfer_reference_script"])
        self.assertNotIn("ssh -n", plan["transfer_reference_script"])

    def test_stage_accepts_all_verified_hosts_and_refuses_one_bad_hash(self):
        source_manifest = "\n".join([
            "QUAL_FILE adapter_model.safetensors " + "a" * 64 + " 10",
            "QUAL_FILE adapter_config.json " + "b" * 64 + " 20",
        ])
        reference_manifest = "QUAL_FILE reference.json " + "c" * 64 + " 30"
        with patch("training_ops._ssh", side_effect=[
            (0, source_manifest), (0, reference_manifest), (0, "transferred"),
            (0, "staged"), (0, "staged"),
        ]) as ssh:
            with patch("training_ops._copy_resume_adapter") as copy:
                with patch(
                    "training_ops._fan",
                    return_value={host: (0, "verified") for host in [
                        "n76", "n77", "n78", "n80", "n82", "n83", "n84", "n85"
                    ]},
                ) as fan:
                    result = prepare_policy0_qualification(
                        "configs/trinity-h200-parity-production",
                        apply=True,
                        source_adapter_dir="/data/glm52-rl/preserved-step0",
                        source_reference_path="/data/glm52-rl/fresh-h200/reference.json",
                    )
        self.assertTrue(result["staged"])
        self.assertEqual(copy.call_count, 3)
        self.assertEqual(fan.call_count, 1)
        self.assertGreaterEqual(ssh.call_count, 1)

        bad_results = {
            host: (0 if host != "n83" else 1, "bad" if host == "n83" else "verified")
            for host in ["n76", "n77", "n78", "n80", "n82", "n83", "n84", "n85"]
        }
        with patch("training_ops._ssh", side_effect=[
            (0, source_manifest), (0, reference_manifest), (0, "transferred"),
            (0, "staged"), (0, "staged"),
        ]):
            with patch("training_ops._copy_resume_adapter"):
                with patch("training_ops._fan", return_value=bad_results):
                    with self.assertRaisesRegex(RuntimeError, "verification failed"):
                        prepare_policy0_qualification(
                            "configs/trinity-h200-parity-production",
                            apply=True,
                            source_adapter_dir="/data/glm52-rl/preserved-step0",
                            source_reference_path="/data/glm52-rl/fresh-h200/reference.json",
                        )

    def test_stage_refuses_wrong_source_identity(self):
        with self.assertRaisesRegex(RuntimeError, "source adapter host"):
            prepare_policy0_qualification(
                "configs/trinity-h200-parity-production",
                source_adapter_dir="/data/glm52-rl/preserved-step0",
                source_reference_path="/data/glm52-rl/fresh-h200/reference.json",
                source_adapter_host="n80",
            )

    def test_stage_refuses_missing_source_identity_before_transfer(self):
        incomplete = "QUAL_FILE adapter_model.safetensors " + "a" * 64 + " 10"
        with patch("training_ops._ssh", return_value=(0, incomplete)) as ssh:
            with self.assertRaisesRegex(RuntimeError, "source step-0 adapter identity"):
                prepare_policy0_qualification(
                    "configs/trinity-h200-parity-production",
                    apply=True,
                    source_adapter_dir="/data/glm52-rl/preserved-step0",
                    source_reference_path="/data/glm52-rl/fresh-h200/reference.json",
                )
        self.assertEqual(ssh.call_count, 1)

    def test_reference_transfer_uses_stdin_capable_remote_bash_stub(self):
        from training_ops import _policy0_transfer_reference_script

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "reference.json"
            target = root / "destination" / "reference.json"
            source.write_text("fresh-reference")
            stub_dir = root / "bin"
            stub_dir.mkdir()
            ssh = stub_dir / "ssh"
            ssh.write_text(
                "#!/bin/sh\n"
                "last=\"\"\n"
                "while [ $# -gt 0 ]; do\n"
                "  if [ \"$1\" = bash ]; then shift; exec bash \"$@\"; fi\n"
                "  last=\"$1\"; shift\n"
                "done\n"
                "case \"$last\" in mkdir*|rm*) sh -c \"$last\";; esac\n"
                "exit 0\n"
            )
            scp = stub_dir / "scp"
            scp.write_text(
                "#!/bin/sh\n"
                "previous=\"\"; last=\"\"\n"
                "for arg in \"$@\"; do previous=\"$last\"; last=\"$arg\"; done\n"
                "source=\"$previous\"\n"
                "destination=\"$last\"\n"
                "destination=${destination#*:}\n"
                "cp -- \"$source\" \"$destination\"\n"
            )
            for stub in (ssh, scp):
                stub.chmod(stub.stat().st_mode | 0o111)
            script = _policy0_transfer_reference_script(
                str(source), str(target), "stub"
            )
            result = subprocess.run(
                ["bash", "-c", script],
                env={"PATH": str(stub_dir) + os.pathsep + os.environ["PATH"]},
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(target.read_text(), "fresh-reference")


if __name__ == "__main__":
    unittest.main()
