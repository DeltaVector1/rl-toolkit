import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training_ops import (
    Refused,
    _policy0_controller_transfer_reference,
    _policy0_qualification_verify_script,
    _policy0_transfer_reference_script,
    prepare_policy0_qualification,
)
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
OLD_SHA256 = "ad0dc097157b516e1a5ddc19124dbf10241c0245ce7a9a712c2e0ea78dbb42c4"
OLD_SIZE = 2_838_023


def reference_payload():
    return {
        "policy": "policy-0",
        "request_mode": "serialized_generate_completion_logprobs",
        "source_kind": "operator_non_rollout",
        "input_ids": [10, 11, 12, 13],
        "inference_logprobs": [0.0, 0.0, -2.0, -3.0],
        "loss_mask": [False, False, True, True],
        "generated_token_ids": [12, 13],
        "generated_token_count": 2,
        "sequence_lengths": [4],
        "routed_experts": {"shape": [3, 60, 4]},
        "policy_identity_before": {"server_instance": "h200"},
        "policy_identity_after": {"server_instance": "h200"},
        "cache_salt": "fresh-policy-0",
    }


def _identity_bytes(value: bytes) -> tuple[str, int]:
    return hashlib.sha256(value).hexdigest(), len(value)


class TrinityPolicy0StagingTest(unittest.TestCase):
    def run_install(
        self,
        *,
        final=None,
        partial=b"new",
        archive=None,
        old_sha=OLD_SHA256,
        old_size=OLD_SIZE,
        expected_new_sha=None,
        expected_new_size=None,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "qualification" / "reference.json"
            destination.parent.mkdir()
            partial_path = destination.with_name("reference.partial.policy0.unique")
            partial_path.write_bytes(partial)
            if final is not None:
                destination.write_bytes(final)
            if archive is not None:
                destination.with_name(destination.name + ".preserved." + old_sha).write_bytes(archive)
            new_sha, new_size = _identity_bytes(partial)
            expected_new_sha = expected_new_sha or new_sha
            expected_new_size = new_size if expected_new_size is None else expected_new_size
            script = _policy0_transfer_reference_script(
                str(partial_path), str(destination), expected_new_sha, expected_new_size,
                old_sha, old_size
            )
            result = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
            archive_path = destination.with_name(destination.name + ".preserved." + old_sha)
            return result, {
                "final": destination.read_bytes() if destination.exists() else None,
                "partial_exists": partial_path.exists(),
                "archive": archive_path.read_bytes() if archive_path.exists() else None,
            }

    def test_validates_exact_adapter_settings_and_reference(self):
        self.assertEqual(validate_lora_config(EXPECTED_LORA, EXPECTED_LORA)["r"], 32)
        changed = dict(EXPECTED_LORA)
        changed["target_modules"] = list(EXPECTED_LORA["target_modules"][:-1])
        with self.assertRaisesRegex(QualificationError, "target_modules"):
            validate_lora_config(changed, EXPECTED_LORA)
        self.assertEqual(validate_reference(reference_payload())["policy"], "policy-0")
        changed = reference_payload()
        changed["source_kind"] = "rollout_batch"
        with self.assertRaisesRegex(QualificationError, "cached rollout"):
            validate_reference(changed)
        changed = reference_payload()
        changed["routed_experts"] = {"shape": [4, 60, 4]}
        with self.assertRaisesRegex(QualificationError, "routes"):
            validate_reference(changed)
        changed = reference_payload()
        changed["inference_logprobs"][1] = -0.1
        with self.assertRaisesRegex(QualificationError, "prompt inference logprobs"):
            validate_reference(changed)

    def test_plan_uses_controller_route_and_expected_old_options(self):
        plan = prepare_policy0_qualification(
            "configs/trinity-h200-parity-production",
            source_adapter_dir="/data/glm52-rl/preserved-step0",
            source_reference_path="/data/glm52-rl/fresh-h200/reference.json",
            expected_old_reference_sha256=OLD_SHA256,
            expected_old_reference_size=OLD_SIZE,
        )
        self.assertEqual(
            plan["nodes"], ["n76", "n77", "n78", "n80", "n82", "n83", "n84", "n85"]
        )
        self.assertEqual(plan["source_reference_host"], "h200b")
        self.assertEqual(plan["expected_old_reference_sha256"], OLD_SHA256)
        self.assertEqual(plan["expected_old_reference_size"], OLD_SIZE)
        self.assertNotIn("china-trainer", plan["transfer_reference_script"])
        self.assertNotIn("adapter_model.safetensors", plan["transfer_reference_script"])
        self.assertIn("flock -n", plan["transfer_reference_script"])
        self.assertIn("fsync", plan["transfer_reference_script"])
        self.assertIn("preserved.", plan["transfer_reference_script"])
        self.assertIn("sha256sum", plan["verify_script"])

    def test_api_requires_expected_old_pair(self):
        kwargs = {
            "source_adapter_dir": "/data/glm52-rl/preserved-step0",
            "source_reference_path": "/data/glm52-rl/fresh-h200/reference.json",
        }
        with self.assertRaisesRegex(Refused, "and size are both required"):
            prepare_policy0_qualification(
                "configs/trinity-h200-parity-production",
                expected_old_reference_sha256=OLD_SHA256,
                **kwargs,
            )
        with self.assertRaisesRegex(Refused, "and size are both required"):
            prepare_policy0_qualification(
                "configs/trinity-h200-parity-production",
                expected_old_reference_size=2_838_023,
                **kwargs,
            )

    def test_rejects_wrong_source_host_and_incomplete_source_identity(self):
        kwargs = {
            "source_adapter_dir": "/data/glm52-rl/preserved-step0",
            "source_reference_path": "/data/glm52-rl/fresh-h200/reference.json",
            "expected_old_reference_sha256": OLD_SHA256,
            "expected_old_reference_size": OLD_SIZE,
        }
        with self.assertRaisesRegex(Refused, "source adapter host"):
            prepare_policy0_qualification(
                "configs/trinity-h200-parity-production",
                source_adapter_host="n80",
                **kwargs,
            )
        incomplete = "QUAL_FILE adapter_model.safetensors " + "a" * 64 + " 10"
        with patch("training_ops._ssh", return_value=(0, incomplete)) as ssh:
            with self.assertRaisesRegex(RuntimeError, "source step-0 adapter identity"):
                prepare_policy0_qualification(
                    "configs/trinity-h200-parity-production", apply=True, **kwargs
                )
        self.assertEqual(ssh.call_count, 1)
    def test_stage_keeps_adapter_and_config_staging_separate(self):
        source_manifest = "\n".join([
            "QUAL_FILE adapter_model.safetensors " + "a" * 64 + " 10",
            "QUAL_FILE adapter_config.json " + "b" * 64 + " 20",
        ])
        reference_manifest = "QUAL_FILE reference.json " + "c" * 64 + " 30"
        nodes = ["n76", "n77", "n78", "n80", "n82", "n83", "n84", "n85"]
        with patch("training_ops._ssh", side_effect=[
            (0, source_manifest), (0, reference_manifest), (0, "staged"), (0, "staged")
        ]), patch(
            "training_ops._policy0_controller_transfer_reference",
            return_value={host: {"ok": True, "message": "installed"} for host in nodes},
        ) as transfer, patch("training_ops._copy_resume_adapter") as copy, patch(
            "training_ops._fan",
            return_value={host: (0, "verified") for host in nodes},
        ):
            result = prepare_policy0_qualification(
                "configs/trinity-h200-parity-production",
                apply=True,
                source_adapter_dir="/data/glm52-rl/preserved-step0",
                source_reference_path="/data/glm52-rl/fresh-h200/reference.json",
                expected_old_reference_sha256=OLD_SHA256,
                expected_old_reference_size=OLD_SIZE,
            )
        self.assertTrue(result["staged"])
        transfer.assert_called_once()
        self.assertEqual(copy.call_count, 2)
        self.assertTrue(all("reference.json" not in call.args[0] for call in copy.call_args_list))
        self.assertEqual(copy.call_args_list[0].kwargs["target_path"].split("/")[-1], "adapter_model.safetensors")
        self.assertEqual(copy.call_args_list[1].kwargs["target_path"].split("/")[-1], "adapter_config.json")
    def test_reference_script_preserves_exact_old_and_installs_new(self):
        old = b"old-reference"
        old_sha, old_size = _identity_bytes(old)
        result, state = self.run_install(
            final=old, partial=b"new-reference", old_sha=old_sha, old_size=old_size
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(state["final"], b"new-reference")
        self.assertEqual(state["archive"], old)
        self.assertFalse(state["partial_exists"])
        self.assertIn("status=installed", result.stdout)
        self.assertIn(f"old_sha={old_sha}", result.stdout)
        self.assertIn("preserved=", result.stdout)
        self.assertIn(".preserved.", result.stdout)
        self.assertIn(f"new_sha={hashlib.sha256(b'new-reference').hexdigest()}", result.stdout)

    def test_reference_script_already_new_reports_valid_preserved_archive(self):
        old = b"old-reference"
        old_sha, old_size = _identity_bytes(old)
        result, state = self.run_install(
            final=b"new-reference", partial=b"new-reference", archive=old,
            old_sha=old_sha, old_size=old_size,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(state["final"], b"new-reference")
        self.assertEqual(state["archive"], old)
        self.assertFalse(state["partial_exists"])
        self.assertIn("status=idempotent", result.stdout)
        self.assertIn(f"preserved=", result.stdout)
        self.assertIn(f".preserved.{old_sha}", result.stdout)

    def test_reference_script_already_new_without_archive_is_not_recoverable(self):
        result, state = self.run_install(final=b"new-reference", partial=b"new-reference")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not recoverable", result.stderr)
        self.assertEqual(state["final"], b"new-reference")
        self.assertFalse(state["partial_exists"])

    def test_reference_script_already_new_rejects_corrupt_archive(self):
        old = b"old-reference"
        old_sha, old_size = _identity_bytes(old)
        result, state = self.run_install(
            final=b"new-reference", partial=b"new-reference", archive=b"corrupt",
            old_sha=old_sha, old_size=old_size,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("archive conflicts", result.stderr)
        self.assertEqual(state["final"], b"new-reference")
        self.assertFalse(state["partial_exists"])

    def test_reference_script_missing_installs_without_archive(self):
        result, state = self.run_install(final=None, partial=b"new-reference")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(state["final"], b"new-reference")
        self.assertFalse(state["partial_exists"])
        self.assertIsNone(state["archive"])

    def test_reference_script_rejects_wrong_old_and_cleans_partial(self):
        result, state = self.run_install(final=b"wrong-old", partial=b"new-reference")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(state["final"], b"wrong-old")
        self.assertFalse(state["partial_exists"])

    def test_reference_script_rejects_symlink_and_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "reference.json"
            partial = root / "reference.partial"
            partial.write_bytes(b"new")
            target.symlink_to(partial)
            sha, size = _identity_bytes(b"new")
            result = subprocess.run(
                ["bash", "-c", _policy0_transfer_reference_script(str(partial), str(target), sha, size)],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "reference.json"
            target.mkdir()
            partial = root / "reference.partial"
            partial.write_bytes(b"new")
            sha, size = _identity_bytes(b"new")
            result = subprocess.run(
                ["bash", "-c", _policy0_transfer_reference_script(str(partial), str(target), sha, size)],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)

    def test_reference_script_rejects_archive_collision_and_corrupt_upload(self):
        old = b"old-reference"
        old_sha, old_size = _identity_bytes(old)
        result, state = self.run_install(
            final=old, partial=b"new-reference", archive=b"different-archive", old_sha=old_sha, old_size=old_size
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(state["final"], old)
        self.assertFalse(state["partial_exists"])
        expected_sha, expected_size = _identity_bytes(b"new-reference")
        result, state = self.run_install(
            final=old,
            partial=b"corrupt",
            expected_new_sha=expected_sha,
            expected_new_size=expected_size,
            old_sha=old_sha,
            old_size=old_size,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(state["final"], old)
        self.assertFalse(state["partial_exists"])

    def test_reference_script_rejects_nonregular_archive(self):
        old = b"old-reference"
        old_sha, old_size = _identity_bytes(old)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "reference.json"
            target.write_bytes(b"new-reference")
            partial = root / "reference.partial"
            partial.write_bytes(b"new-reference")
            archive = root / (target.name + ".preserved." + old_sha)
            archive.mkdir()
            new_sha, new_size = _identity_bytes(b"new-reference")
            result = subprocess.run(
                ["bash", "-c", _policy0_transfer_reference_script(
                    str(partial), str(target), new_sha, new_size, old_sha, old_size
                )],
                text=True,
                capture_output=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not a regular file", result.stderr)

    def test_reference_script_interrupted_upload_can_retry_cleanly(self):
        old = b"old-reference"
        new = b"new-reference"
        old_sha, old_size = _identity_bytes(old)
        new_sha, new_size = _identity_bytes(new)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "reference.json"
            target.write_bytes(old)
            partial = root / "reference.partial"
            partial.write_bytes(b"corrupt-upload")
            script = _policy0_transfer_reference_script(
                str(partial), str(target), new_sha, new_size, old_sha, old_size
            )
            interrupted = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
            self.assertNotEqual(interrupted.returncode, 0)
            self.assertFalse(partial.exists())
            partial.write_bytes(new)
            retried = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
            self.assertEqual(retried.returncode, 0, retried.stderr)
            archive = target.with_name(target.name + ".preserved." + old_sha)
            self.assertEqual(target.read_bytes(), new)
            self.assertEqual(archive.read_bytes(), old)
    def test_reference_script_race_lock_refuses_second_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "reference.json"
            lock = Path(str(target) + ".policy0.lock")
            partial = root / "reference.partial"
            partial.write_bytes(b"new")
            sha, size = _identity_bytes(b"new")
            holder = subprocess.Popen(["flock", "-n", str(lock), "sleep", "1"])
            try:
                result = subprocess.run(
                    ["bash", "-c", _policy0_transfer_reference_script(str(partial), str(target), sha, size)],
                    text=True,
                    capture_output=True,
                )
            finally:
                holder.wait()
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(target.exists())
            self.assertFalse(partial.exists())

    def test_reference_script_rejects_equal_size_relative_symlink_race(self):
        old = b"old-reference"
        old_sha, old_size = _identity_bytes(old)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "reference.json"
            relative_target = root / "same-size-target"
            relative_target.write_bytes(old)
            target.write_bytes(old)
            partial = root / "reference.partial"
            partial.write_bytes(b"new-reference")
            wrapper_dir = root / "bin"
            wrapper_dir.mkdir()
            wrapper = wrapper_dir / "python3"
            wrapper.write_text(
                "#!/bin/sh\n"
                f"if [ \"$2\" = {str(target)!r} ]; then\n"
                f"  rm -f -- {str(target)!r}\n"
                f"  ln -s -- same-size-target {str(target)!r}\n"
                "fi\n"
                "exec /usr/bin/python3 \"$@\"\n",
                encoding="utf-8",
            )
            wrapper.chmod(0o755)
            new_sha, new_size = _identity_bytes(b"new-reference")
            environment = dict(os.environ)
            environment["PATH"] = str(wrapper_dir) + os.pathsep + environment["PATH"]
            result = subprocess.run(
                ["bash", "-c", _policy0_transfer_reference_script(
                    str(partial), str(target), new_sha, new_size, old_sha, old_size
                )],
                text=True,
                capture_output=True,
                env=environment,
            )
            self.assertTrue(target.is_symlink(), result.stderr)
        self.assertNotEqual(result.returncode, 0)

    def test_controller_downloads_once_and_uploads_all_trainers(self):
        source_bytes = b"verified-reference"
        source_sha, source_size = _identity_bytes(source_bytes)
        nodes = ["n76", "n77", "n78", "n80", "n82", "n83", "n84", "n85"]
        process_commands = []

        def run_process(command, *, stdin, timeout):
            process_commands.append(command)
            if ":" in command[-1]:
                source = Path(command[-2])
                destination = Path(command[-1].split(":", 1)[1])
            elif len(process_commands) == 1:
                source = None
                destination = Path(command[-1])
            else:
                source = Path(command[-2])
                destination = Path(command[-1])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source is None:
                destination.write_bytes(source_bytes)
            else:
                shutil.copyfile(source, destination)
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_bytes(source_bytes)
            with patch("training_ops._run_process", side_effect=run_process), patch(
                "training_ops._ssh", return_value=(0, "REFERENCE_INSTALL status=installed")
            ) as ssh:
                result = _policy0_controller_transfer_reference(
                    "h200b", str(source), str(Path(directory) / "qualification" / "reference.json"),
                    nodes, {"sha256": source_sha, "size": source_size}, OLD_SHA256, 2_838_023
                )
        self.assertEqual(set(result), set(nodes))
        self.assertTrue(all(record["ok"] for record in result.values()))
        self.assertEqual(len(process_commands), 1 + len(nodes))
        self.assertEqual(ssh.call_count, 2 * len(nodes))
        self.assertTrue(process_commands[0][-2].startswith("root@38.255.28.19:"))
        target_commands = process_commands[1:]
        self.assertEqual(
            {command[-1].split(":", 1)[0] for command in target_commands},
            set(nodes),
        )
        self.assertTrue(all("adapter_model" not in " ".join(command) for command in process_commands))
        self.assertTrue(all("china-trainer" not in " ".join(command) for command in process_commands))

    def test_controller_rejects_corrupt_download(self):
        def corrupt_download(command, *, stdin, timeout):
            Path(command[-1]).write_bytes(b"corrupt")
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_bytes(b"verified")
            with patch("training_ops._run_process", side_effect=corrupt_download), patch(
                "training_ops._ssh"
            ) as ssh:
                with self.assertRaisesRegex(Refused, "identity differs"):
                    _policy0_controller_transfer_reference(
                        "h200b", str(source), str(Path(directory) / "reference.json"), ["n76"],
                        {"sha256": hashlib.sha256(b"verified").hexdigest(), "size": 8},
                        OLD_SHA256, 2_838_023,
                    )
            self.assertEqual(ssh.call_count, 0)

    def test_controller_transfer_failure_has_structured_phase_and_cleanup(self):
        calls = []

        def process(command, *, stdin, timeout):
            calls.append(command)
            if len(calls) == 1:
                Path(command[-1]).write_bytes(b"verified")
                return subprocess.CompletedProcess(command, 0, "", "")
            return subprocess.CompletedProcess(command, 9, "", "upload failed")

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_bytes(b"verified")
            with patch("training_ops._run_process", side_effect=process), patch(
                "training_ops._ssh", side_effect=[(0, "parent"), (0, "cleanup")]
            ) as ssh:
                with self.assertRaisesRegex(RuntimeError, "phase.*upload"):
                    _policy0_controller_transfer_reference(
                        "h200b", str(source), str(Path(directory) / "reference.json"), ["n76"],
                        {"sha256": hashlib.sha256(b"verified").hexdigest(), "size": 8},
                        OLD_SHA256, 2_838_023,
                    )
            self.assertEqual(ssh.call_count, 2)
            self.assertIn("partial", ssh.call_args_list[-1].args[1])

    def test_controller_failure_retains_complete_multi_host_results(self):
        source_bytes = b"verified"
        source_sha, source_size = _identity_bytes(source_bytes)
        all_results = {
            "n76": {
                "ok": True, "phase": "install", "preserved_path": "/ref.preserved.old",
            },
            "n77": {
                "ok": False, "phase": "upload", "message": "exit 9",
            },
        }
        def fetch(command, *, stdin, timeout):
            Path(command[-1]).write_bytes(source_bytes)
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_bytes(source_bytes)
            with patch("training_ops._run_process", side_effect=fetch), patch(
                "training_ops._fan", return_value=all_results
            ):
                with self.assertRaises(RuntimeError) as raised:
                    _policy0_controller_transfer_reference(
                        "h200b", str(source), str(Path(directory) / "reference.json"),
                        list(all_results), {"sha256": source_sha, "size": source_size},
                        OLD_SHA256, OLD_SIZE,
                    )
        self.assertEqual(raised.exception.results, all_results)
        self.assertEqual(raised.exception.failures, {"n77": all_results["n77"]})

    def test_final_manifest_verification_checks_exact_hash_and_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter_model.safetensors"
            config = root / "adapter_config.json"
            reference = root / "reference.json"
            adapter.write_bytes(b"adapter")
            config.write_text(json.dumps({
                "r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.0,
                "target_modules": EXPECTED_LORA["target_modules"],
            }), encoding="utf-8")
            reference_data = reference_payload()
            reference_data["policy_identity_before"] = {
                "server_instance": "h200",
                "target_sha256": hashlib.sha256(b"adapter").hexdigest(),
                "target_size": 7,
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "config_size": config.stat().st_size,
            }
            reference_data["policy_identity_after"] = dict(reference_data["policy_identity_before"])
            reference.write_text(json.dumps(reference_data), encoding="utf-8")
            script = _policy0_qualification_verify_script(
                str(adapter), str(config), str(reference), EXPECTED_LORA,
                {
                    "adapter_model.safetensors": {"sha256": hashlib.sha256(b"adapter").hexdigest(), "size": 7},
                    "adapter_config.json": {"sha256": hashlib.sha256(config.read_bytes()).hexdigest(), "size": config.stat().st_size},
                    "reference.json": {"sha256": hashlib.sha256(reference.read_bytes()).hexdigest(), "size": reference.stat().st_size},
                },
            )
            result = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            adapter.write_bytes(b"changed")
            failed = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
            self.assertNotEqual(failed.returncode, 0)

    def test_cli_options_parse_size_and_help(self):
        from training_ops import _prepare_policy0_action, usage

        with self.assertRaisesRegex(Refused, "and size are both required"):
            _prepare_policy0_action([
                "--source-adapter-dir", "/data/glm52-rl/preserved-step0",
                "--source-reference-path", "/data/glm52-rl/fresh-h200/reference.json",
                "--expected-old-reference-sha256", OLD_SHA256,
            ])
        with self.assertRaisesRegex(Refused, "must be an integer"):
            _prepare_policy0_action(["--expected-old-reference-size", "not-an-int"])
        self.assertIn("--expected-old-reference-sha256 SHA256", usage)
        self.assertIn("--expected-old-reference-size BYTES", usage)


if __name__ == "__main__":
    unittest.main()
