import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import training_ops
import training_ops.tests  # Install the local-only test guard before child tests.


class InferenceProbeTest(unittest.TestCase):
    def run_probe(self, *, policy_lora_name="", policy_identity=None, receipt=None):
        settings = {
            "gpu_count": 1,
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "ports": [18000],
        }
        with mock.patch.object(training_ops, "_inference_config", return_value=settings):
            script = training_ops._inference_script(
                "h200b",
                "/models/glm52",
                "glm-engine",
                policy_lora_name=policy_lora_name,
                run="test-run",
                policy_identity=policy_identity,
            )
        stage = {"status": "idle"}
        if receipt is not None:
            stage["loaded_receipts"] = {policy_identity["slot"]: receipt}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sudo").write_text(
                '#!/bin/sh\nexec "$@"\n', encoding="utf-8"
            )
            (root / "docker").write_text(
                '#!/bin/sh\n'
                'case "$*" in\n'
                '  inspect*) printf "/glm-engine true running 123\\n" ;;\n'
                '  *logs*) : ;;\n'
                'esac\n',
                encoding="utf-8",
            )
            (root / "curl").write_text(
                '#!/bin/sh\n'
                'case "$*" in\n'
                '  *staging_status*) printf \'%s\' "$STAGE" ;;\n'
                '  *v1/completions*) printf 000 ;;\n'
                'esac\n',
                encoding="utf-8",
            )
            (root / "ss").write_text("#!/bin/sh\n", encoding="utf-8")
            (root / "nvidia-smi").write_text("#!/bin/sh\nprintf '0\\n'\n", encoding="utf-8")
            for command in ("sudo", "docker", "curl", "ss", "nvidia-smi"):
                (root / command).chmod(0o755)
            result = subprocess.run(
                ["bash", "-c", script],
                env={
                    "PATH": f"{root}:/usr/bin:/bin",
                    "STAGE": json.dumps(stage),
                },
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        return training_ops._kv(result.stdout)

    def test_generation_timeout_does_not_change_exact_policy_to_mismatch(self):
        identity = {
            "step": 1,
            "slot": "adapter-slot1",
            "target_sha256": "a" * 64,
            "target_size": 1,
            "config_sha256": "b" * 64,
            "config_size": 1,
            "source_url": "source",
        }
        report = self.run_probe(
            policy_lora_name="adapter",
            policy_identity=identity,
            receipt={k: identity[k] for k in identity if k != "step" and k != "slot"},
        )
        self.assertEqual(report["GEN"].split()[0], "000")
        self.assertEqual(report["PROBES"], "0 1")
        self.assertEqual(report["POLICY"], "ready 1")

    def test_missing_exact_policy_receipt_is_still_a_mismatch(self):
        identity = {
            "step": 1,
            "slot": "adapter-slot1",
            "target_sha256": "a" * 64,
            "target_size": 1,
            "config_sha256": "b" * 64,
            "config_size": 1,
            "source_url": "source",
        }
        report = self.run_probe(policy_lora_name="adapter", policy_identity=identity)
        self.assertEqual(report["GEN"].split()[0], "POLICY_MISMATCH")
        self.assertEqual(report["PROBES"], "0 1")
        self.assertEqual(report["POLICY"], "mismatch 1")


if __name__ == "__main__":
    unittest.main()
