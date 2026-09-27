import gzip
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
    _policy0_reference_lan_fanout_script,
    _policy0_reference_seed_prepare_script,
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
POST_UPDATE_IDENTITY = {
    "final_reference_sha256": "c" * 64,
    "final_reference_size": 123456,
    "server_instance": "h200b-instance-1",
    "adapter_sha256": "a" * 64,
    "adapter_size": 100,
    "config_sha256": "b" * 64,
    "config_size": 200,
}


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
        "source_input_sha256": "43a2d66e5f850f64677fc2fae986fcdbdf190e6b6f026167f70b21d8860826b3",
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
        use_expected_old=True,
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
            script_args = (old_sha, old_size) if use_expected_old else (None, None)
            script = _policy0_transfer_reference_script(
                str(partial_path), str(destination), expected_new_sha, expected_new_size,
                *script_args
            )
            result = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
            archive_path = destination.with_name(destination.name + ".preserved." + old_sha)
            return result, {
                "final": destination.read_bytes() if destination.exists() else None,
                "partial_exists": partial_path.exists(),
                "archive": archive_path.read_bytes() if archive_path.exists() else None,
                "archive_path": str(archive_path),
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

    def test_post_update_plan_requires_pinned_identity_and_uses_collection_validator(self):
        kwargs = {
            "source_adapter_dir": "/data/glm52-rl/preserved-step0",
            "source_reference_path": "/data/glm52-rl/fresh-h200/reference.json",
            "expected_old_reference_sha256": OLD_SHA256,
            "expected_old_reference_size": OLD_SIZE,
        }
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
            Refused, "post-update reference identity is not configured"
        ):
            prepare_policy0_qualification(
                "configs/trinity-h200-post-update-qualification", **kwargs
            )
        with patch.dict(
            os.environ,
            {"PRIME_RL_POST_UPDATE_REFERENCE_IDENTITY": json.dumps(POST_UPDATE_IDENTITY)},
            clear=True,
        ):
            plan = prepare_policy0_qualification(
                "configs/trinity-h200-post-update-qualification", **kwargs
            )
        self.assertEqual(plan["post_update_reference_identity"], POST_UPDATE_IDENTITY)
        self.assertIn("exactly eight microbatches", plan["source_reference_script"])
        self.assertIn("request_without_id", plan["source_reference_script"])
        self.assertNotIn("reference is not a policy-0 reference", plan["source_reference_script"])

    def test_old_parity_plan_does_not_require_post_update_identity(self):
        kwargs = {
            "source_adapter_dir": "/data/glm52-rl/preserved-step0",
            "source_reference_path": "/data/glm52-rl/fresh-h200/reference.json",
            "expected_old_reference_sha256": OLD_SHA256,
            "expected_old_reference_size": OLD_SIZE,
        }
        with patch.dict(os.environ, {}, clear=True):
            plan = prepare_policy0_qualification(
                "configs/trinity-h200-parity-production", **kwargs
            )
        self.assertNotIn("post_update_reference_identity", plan)
        self.assertIn("reference is not policy-0", plan["source_reference_script"])

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
        self.assertIn(f"preserved={state['archive_path']}", result.stdout)
        self.assertIn(f"new_sha={hashlib.sha256(b'new-reference').hexdigest()}", result.stdout)

    def test_reference_script_reuses_existing_archive_and_reports_path(self):
        old = b"old-reference"
        old_sha, old_size = _identity_bytes(old)
        result, state = self.run_install(
            final=old, partial=b"new-reference", archive=old,
            old_sha=old_sha, old_size=old_size,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(state["final"], b"new-reference")
        self.assertEqual(state["archive"], old)
        self.assertFalse(state["partial_exists"])
        self.assertIn(f"preserved={state['archive_path']}", result.stdout)

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
        self.assertIn(f"preserved={state['archive_path']}", result.stdout)

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
        result, state = self.run_install(
            final=None, partial=b"new-reference", use_expected_old=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(state["final"], b"new-reference")
        self.assertFalse(state["partial_exists"])
        self.assertIsNone(state["archive"])

    def test_reference_script_explicit_old_missing_refuses_and_no_old_retry_installs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "reference.json"
            partial = root / "reference.partial"
            partial.write_bytes(b"new-reference")
            new_sha, new_size = _identity_bytes(b"new-reference")
            explicit = _policy0_transfer_reference_script(
                str(partial), str(target), new_sha, new_size, OLD_SHA256, OLD_SIZE
            )
            refused = subprocess.run(["bash", "-c", explicit], text=True, capture_output=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("explicit expected-old", refused.stderr)
            self.assertFalse(target.exists())
            self.assertFalse(partial.exists())

            partial.write_bytes(b"new-reference")
            retry = _policy0_transfer_reference_script(
                str(partial), str(target), new_sha, new_size
            )
            installed = subprocess.run(["bash", "-c", retry], text=True, capture_output=True)
            self.assertEqual(installed.returncode, 0, installed.stderr)
            self.assertEqual(target.read_bytes(), b"new-reference")
            self.assertFalse(partial.exists())

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

    def test_reference_script_rolls_back_late_symlink_at_exchange_boundary(self):
        self._assert_exchange_boundary_injection("symlink")

    def test_reference_script_rolls_back_late_directory_at_exchange_boundary(self):
        self._assert_exchange_boundary_injection("directory")

    def test_reference_script_fsyncs_before_reporting_verified_rollback(self):
        result, state, events = self._run_transient_exchange_mismatch(fsync_failure=None)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rollback verified", result.stderr)
        self.assertEqual(events[-2:], ["fsync-file", "fsync-directory"])
        self.assertEqual(state["final"], b"old-reference")
        self.assertIsNone(state["partial"])

    def test_reference_script_file_fsync_failure_retains_rollback_evidence(self):
        result, state, events = self._run_transient_exchange_mismatch(fsync_failure="file")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rollback could not be verified", result.stderr)
        self.assertEqual(events[-2:], ["target-read", "fsync-file-failed"])
        self.assertNotIn("fsync-directory", events)
        self.assertEqual(state["final"], b"old-reference")
        self.assertEqual(state["partial"], b"new-reference")

    def test_reference_script_directory_fsync_failure_retains_rollback_evidence(self):
        result, state, events = self._run_transient_exchange_mismatch(fsync_failure="directory")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rollback could not be verified", result.stderr)
        self.assertEqual(events[-2:], ["fsync-file", "fsync-directory-failed"])
        self.assertEqual(state["final"], b"old-reference")
        self.assertEqual(state["partial"], b"new-reference")

    def _run_transient_exchange_mismatch(self, *, fsync_failure):
        old = b"old-reference"
        new = b"new-reference"
        old_sha, old_size = _identity_bytes(old)
        new_sha, new_size = _identity_bytes(new)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "reference.json"
            target.write_bytes(old)
            partial = root / "reference.partial"
            partial.write_bytes(new)
            exchange_marker = root / "exchange.marker"
            mismatch_marker = root / "mismatch.marker"
            target_calls = root / "target.calls"
            events_path = root / "events"
            wrapper_dir = root / "bin"
            wrapper_dir.mkdir()
            wrapper = wrapper_dir / "python3"
            failure_mode = str(fsync_failure or "none")
            wrapper.write_text(
                "#!/bin/sh\n"
                f"if [ \"$2\" = {str(partial)!r} ] && [ \"$3\" = {str(target)!r} ] && [ ! -e {str(exchange_marker)!r} ]; then\n"
                f"  touch -- {str(exchange_marker)!r}\n"
                "  exec /usr/bin/python3 \"$@\"\n"
                "fi\n"
                f"if [ \"$2\" = {str(target)!r} ] && [ -e {str(exchange_marker)!r} ] && [ ! -e {str(mismatch_marker)!r} ]; then\n"
                f"  touch -- {str(mismatch_marker)!r}\n"
                "  /usr/bin/python3 \"$@\" >/dev/null\n"
                "  printf '%s %s\\n' " + "0" * 64 + " 0\n"
                "  exit 0\n"
                "fi\n"
                f"if [ \"$2\" = {str(target)!r} ] && [ -e {str(mismatch_marker)!r} ]; then\n"
                f"  target_call=$(cat -- {str(target_calls)!r} 2>/dev/null || printf '0')\n"
                "  target_call=$((target_call + 1))\n"
                f"  printf '%s\\n' \"$target_call\" > {str(target_calls)!r}\n"
                "  if [ \"$target_call\" -eq 1 ]; then\n"
                f"    printf '%s\\n' target-read >> {str(events_path)!r}\n"
                f"  elif [ {failure_mode!r} = file ]; then\n"
                f"    printf '%s\\n' fsync-file-failed >> {str(events_path)!r}\n"
                "    exit 96\n"
                "  else\n"
                f"    printf '%s\\n' fsync-file >> {str(events_path)!r}\n"
                "  fi\n"
                "fi\n"
                f"if [ \"$2\" = {str(target.parent)!r} ] && [ -e {str(mismatch_marker)!r} ] && [ {failure_mode!r} = directory ]; then\n"
                f"  printf '%s\\n' fsync-directory-failed >> {str(events_path)!r}\n"
                "  exit 97\n"
                "fi\n"
                f"if [ \"$2\" = {str(target.parent)!r} ] && [ -e {str(mismatch_marker)!r} ]; then\n"
                f"  printf '%s\\n' fsync-directory >> {str(events_path)!r}\n"
                "fi\n"
                "exec /usr/bin/python3 \"$@\"\n",
                encoding="utf-8",
            )
            wrapper.chmod(0o755)
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
            state = {
                "final": target.read_bytes() if target.exists() else None,
                "partial": partial.read_bytes() if partial.exists() else None,
            }
            events = events_path.read_text(encoding="utf-8").splitlines()
            return result, state, events

    def _assert_exchange_boundary_injection(self, kind):
        old = b"old-reference"
        new = b"new-reference"
        old_sha, old_size = _identity_bytes(old)
        new_sha, new_size = _identity_bytes(new)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "reference.json"
            target.write_bytes(old)
            partial = root / "reference.partial"
            partial.write_bytes(new)
            marker = root / "exchanged.marker"
            wrapper_dir = root / "bin"
            wrapper_dir.mkdir()
            replacement = (
                f"ln -s -- {str(partial)!r} {str(target)!r}"
                if kind == "symlink"
                else f"mkdir -- {str(target)!r}"
            )
            wrapper = wrapper_dir / "python3"
            wrapper.write_text(
                "#!/bin/sh\n"
                f"if [ \"$2\" = {str(partial)!r} ] && [ \"$3\" = {str(target)!r} ] && [ ! -e {str(marker)!r} ]; then\n"
                f"  touch -- {str(marker)!r}\n"
                "  /usr/bin/python3 \"$@\"\n"
                "  status=$?\n"
                f"  rm -rf -- {str(target)!r}\n"
                f"  {replacement}\n"
                "  exit $status\n"
                "fi\n"
                "exec /usr/bin/python3 \"$@\"\n",
                encoding="utf-8",
            )
            wrapper.chmod(0o755)
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
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertEqual(target.read_bytes(), old)
            self.assertTrue(partial.is_symlink() if kind == "symlink" else partial.is_dir())
            self.assertTrue(marker.exists())
            self.assertEqual(
                (root / (target.name + ".preserved." + old_sha)).read_bytes(), old
            )

    def test_controller_downloads_once_and_uploads_all_trainers(self):
        source_bytes = b"verified-reference"
        source_sha, source_size = _identity_bytes(source_bytes)
        nodes = ["n76", "n77", "n78", "n80", "n82", "n83", "n84", "n85"]
        process_commands = []
        compressed_uploads = []

        def run_process(command, *, stdin, timeout):
            process_commands.append(command)
            if ":" in command[-1]:
                source = Path(command[-2])
                destination = Path(command[-1].split(":", 1)[1])
                if len(process_commands) == 2:
                    compressed_uploads.append(source.read_bytes())
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
        self.assertTrue(all(record["seed_cleanup"] == "cleaned" for record in result.values()))
        self.assertEqual(len(process_commands), 2)
        self.assertEqual(gzip.decompress(compressed_uploads[0]), source_bytes)
        self.assertGreaterEqual(ssh.call_count, 1 + 2 * len(nodes))
        self.assertTrue(process_commands[0][-2].startswith("root@38.255.28.19:"))
        self.assertTrue(process_commands[1][-1].split(":", 1)[0].endswith("n82"))
        self.assertTrue(all("adapter_model" not in " ".join(command) for command in process_commands))
        self.assertTrue(all("china-trainer" not in " ".join(command) for command in process_commands))
        fanout_scripts = [call.args[1] for call in ssh.call_args_list if "scp -q" in call.args[1]]
        self.assertEqual(len(fanout_scripts), len(nodes) - 1)
        for host in nodes:
            if host != "n82":
                self.assertTrue(any(
                    f"10.92.20.{host.removeprefix('n')}" in script
                    for script in fanout_scripts
                ))
        self.assertTrue(all("reference.json.partial.policy0." in script for script in fanout_scripts))
        self.assertTrue(all(call.args[0] != "h200b" for call in ssh.call_args_list))

    def _write_compressed_seed(self, path, payload):
        with path.open("wb") as stream:
            with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as compressed:
                compressed.write(payload)

    def test_seed_prepare_preserves_foreign_file_on_o_excl_race(self):
        payload = b"verified-reference"
        raw_sha, raw_size = _identity_bytes(payload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compressed = root / "seed.json.gz"
            self._write_compressed_seed(compressed, payload)
            compressed_sha, compressed_size = _identity_bytes(compressed.read_bytes())
            raw = root / "reference.partial"
            wrapper_dir = root / "bin"
            wrapper_dir.mkdir()
            wrapper = wrapper_dir / "python3"
            wrapper.write_text(
                "#!/bin/sh\n"
                f"if [ \"$#\" -ge 3 ] && [ \"$3\" = {str(raw)!r} ]; then\n"
                f"  printf foreign > {str(raw)!r}\n"
                "fi\n"
                "exec /usr/bin/python3 \"$@\"\n",
                encoding="utf-8",
            )
            wrapper.chmod(0o755)
            environment = dict(os.environ)
            environment["PATH"] = str(wrapper_dir) + os.pathsep + environment["PATH"]
            result = subprocess.run(
                ["bash", "-c", _policy0_reference_seed_prepare_script(
                    str(compressed), str(raw), compressed_sha, compressed_size, raw_sha, raw_size
                )],
                text=True,
                capture_output=True,
                env=environment,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(raw.read_bytes(), b"foreign")
            self.assertFalse(compressed.exists())

    def test_seed_prepare_cleans_own_incomplete_decompression(self):
        payload = b"verified-reference"
        raw_sha, raw_size = _identity_bytes(payload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compressed = root / "seed.json.gz"
            self._write_compressed_seed(compressed, payload)
            damaged = bytearray(compressed.read_bytes())
            damaged[-1] ^= 1
            compressed.write_bytes(damaged)
            compressed_sha, compressed_size = _identity_bytes(bytes(damaged))
            raw = root / "reference.partial"
            result = subprocess.run(
                ["bash", "-c", _policy0_reference_seed_prepare_script(
                    str(compressed), str(raw), compressed_sha, compressed_size, raw_sha, raw_size
                )],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(raw.exists())
            self.assertFalse(compressed.exists())

    def test_seed_prepare_rejects_corrupt_uploaded_bytes(self):
        payload = b"verified-reference"
        raw_sha, raw_size = _identity_bytes(payload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compressed = root / "seed.json.gz"
            self._write_compressed_seed(compressed, payload)
            expected_sha, expected_size = _identity_bytes(compressed.read_bytes())
            damaged = bytearray(compressed.read_bytes())
            damaged[-1] ^= 1
            compressed.write_bytes(damaged)
            raw = root / "reference.partial"
            result = subprocess.run(
                ["bash", "-c", _policy0_reference_seed_prepare_script(
                    str(compressed), str(raw), expected_sha, expected_size, raw_sha, raw_size
                )],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(raw.exists())
            self.assertFalse(compressed.exists())

    def test_seed_prepare_rejects_raw_manifest_mismatch_and_cleans(self):
        payload = b"verified-reference"
        raw_sha, raw_size = _identity_bytes(payload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compressed = root / "seed.json.gz"
            self._write_compressed_seed(compressed, payload)
            compressed_sha, compressed_size = _identity_bytes(compressed.read_bytes())
            raw = root / "reference.partial"
            result = subprocess.run(
                ["bash", "-c", _policy0_reference_seed_prepare_script(
                    str(compressed), str(raw), compressed_sha, compressed_size,
                    "0" * 64, raw_size + 1,
                )],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(raw.exists())
            self.assertFalse(compressed.exists())

    def test_seed_prepare_preserves_conflicting_raw_partial(self):
        payload = b"verified-reference"
        raw_sha, raw_size = _identity_bytes(payload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compressed = root / "seed.json.gz"
            self._write_compressed_seed(compressed, payload)
            compressed_sha, compressed_size = _identity_bytes(compressed.read_bytes())
            raw = root / "reference.partial"
            raw.write_bytes(b"foreign")
            result = subprocess.run(
                ["bash", "-c", _policy0_reference_seed_prepare_script(
                    str(compressed), str(raw), compressed_sha, compressed_size, raw_sha, raw_size
                )],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(raw.read_bytes(), b"foreign")
            self.assertFalse(compressed.exists())

    def test_seed_prepare_retry_after_interrupted_upload(self):
        payload = b"verified-reference"
        raw_sha, raw_size = _identity_bytes(payload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compressed = root / "seed.json.gz"
            compressed.write_bytes(b"interrupted")
            raw = root / "reference.partial"
            expected_sha, expected_size = _identity_bytes(gzip.compress(payload, mtime=0))
            failed = subprocess.run(
                ["bash", "-c", _policy0_reference_seed_prepare_script(
                    str(compressed), str(raw), expected_sha, expected_size, raw_sha, raw_size
                )],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse(compressed.exists())
            self._write_compressed_seed(compressed, payload)
            expected_sha, expected_size = _identity_bytes(compressed.read_bytes())
            succeeded = subprocess.run(
                ["bash", "-c", _policy0_reference_seed_prepare_script(
                    str(compressed), str(raw), expected_sha, expected_size, raw_sha, raw_size
                )],
                text=True,
                capture_output=True,
            )
            self.assertEqual(succeeded.returncode, 0, succeeded.stderr)
            self.assertEqual(raw.read_bytes(), payload)

    def test_lan_fanout_refuses_peer_conflict_without_overwrite(self):
        payload = b"verified-reference"
        expected_sha, expected_size = _identity_bytes(payload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "seed.partial"
            target = root / "peer.partial"
            source.write_bytes(payload)
            target.write_bytes(b"foreign")
            wrapper_dir = root / "bin"
            wrapper_dir.mkdir()
            (wrapper_dir / "ssh").write_text(
                "#!/bin/sh\n"
                "last=\"\"\n"
                "for value in \"$@\"; do last=\"$value\"; done\n"
                "exec bash -c \"$last\"\n",
                encoding="utf-8",
            )
            marker = root / "scp-called"
            (wrapper_dir / "scp").write_text(
                "#!/bin/sh\n"
                f"touch -- {str(marker)!r}\n"
                "exit 0\n",
                encoding="utf-8",
            )
            for wrapper in (wrapper_dir / "ssh", wrapper_dir / "scp"):
                wrapper.chmod(0o755)
            environment = dict(os.environ)
            environment["PATH"] = str(wrapper_dir) + os.pathsep + environment["PATH"]
            result = subprocess.run(
                ["bash", "-c", _policy0_reference_lan_fanout_script(
                    str(source), str(target), "10.92.20.77", expected_sha, expected_size
                )],
                text=True,
                capture_output=True,
                env=environment,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(target.read_bytes(), b"foreign")
            self.assertFalse(marker.exists())

    def test_lan_fanout_cleans_owned_partial_after_upload_failure(self):
        payload = b"verified-reference"
        expected_sha, expected_size = _identity_bytes(payload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "seed.partial"
            target = root / "peer.partial"
            source.write_bytes(payload)
            wrapper_dir = root / "bin"
            wrapper_dir.mkdir()
            (wrapper_dir / "ssh").write_text(
                "#!/bin/sh\n"
                "last=\"\"\n"
                "for value in \"$@\"; do last=\"$value\"; done\n"
                "exec bash -c \"$last\"\n",
                encoding="utf-8",
            )
            (wrapper_dir / "scp").write_text(
                "#!/bin/sh\n"
                "previous=\"\"\n"
                "source=\"\"\n"
                "destination=\"\"\n"
                "for value in \"$@\"; do source=\"$previous\"; previous=\"$value\"; done\n"
                "destination=\"${previous#*:}\"\n"
                "cp -- \"$source\" \"$destination\"\n"
                "exit 9\n",
                encoding="utf-8",
            )
            for wrapper in (wrapper_dir / "ssh", wrapper_dir / "scp"):
                wrapper.chmod(0o755)
            environment = dict(os.environ)
            environment["PATH"] = str(wrapper_dir) + os.pathsep + environment["PATH"]
            result = subprocess.run(
                ["bash", "-c", _policy0_reference_lan_fanout_script(
                    str(source), str(target), "10.92.20.77", expected_sha, expected_size
                )],
                text=True,
                capture_output=True,
                env=environment,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(target.exists())

    def test_lan_fanout_success_keeps_verified_partial_for_install(self):
        payload = b"verified-reference"
        expected_sha, expected_size = _identity_bytes(payload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "seed.partial"
            target = root / "peer.partial"
            source.write_bytes(payload)
            wrapper_dir = root / "bin"
            wrapper_dir.mkdir()
            (wrapper_dir / "ssh").write_text(
                "#!/bin/sh\n"
                "last=\"\"\n"
                "for value in \"$@\"; do last=\"$value\"; done\n"
                "exec bash -c \"$last\"\n",
                encoding="utf-8",
            )
            (wrapper_dir / "scp").write_text(
                "#!/bin/sh\n"
                "previous=\"\"\n"
                "source=\"\"\n"
                "destination=\"\"\n"
                "for value in \"$@\"; do source=\"$previous\"; previous=\"$value\"; done\n"
                "destination=\"${previous#*:}\"\n"
                "cp -- \"$source\" \"$destination\"\n",
                encoding="utf-8",
            )
            for wrapper in (wrapper_dir / "ssh", wrapper_dir / "scp"):
                wrapper.chmod(0o755)
            environment = dict(os.environ)
            environment["PATH"] = str(wrapper_dir) + os.pathsep + environment["PATH"]
            result = subprocess.run(
                ["bash", "-c", _policy0_reference_lan_fanout_script(
                    str(source), str(target), "10.92.20.77", expected_sha, expected_size
                )],
                text=True,
                capture_output=True,
                env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(target.read_bytes(), payload)

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
                "training_ops._ssh", side_effect=[(0, "cleanup")]
            ) as ssh:
                with self.assertRaisesRegex(RuntimeError, "phase.*upload"):
                    _policy0_controller_transfer_reference(
                        "h200b", str(source), str(Path(directory) / "reference.json"), ["n76"],
                        {"sha256": hashlib.sha256(b"verified").hexdigest(), "size": 8},
                        OLD_SHA256, 2_838_023,
                    )
            self.assertEqual(ssh.call_count, 2)
            self.assertIn("partial.policy0", ssh.call_args_list[-1].args[1])

    def test_controller_seed_cleanup_failure_preserves_install_evidence(self):
        source_bytes = b"verified-reference"
        source_sha, source_size = _identity_bytes(source_bytes)
        install_report = (
            f"REFERENCE_INSTALL status=installed old_sha={OLD_SHA256} "
            f"old_size={OLD_SIZE} preserved=/tmp/reference.preserved.{OLD_SHA256} "
            f"new_sha={source_sha} new_size={source_size}"
        )

        for cleanup_result in ((9, "cleanup denied"), OSError("cleanup socket closed")):
            with self.subTest(cleanup_result=cleanup_result):
                process_calls = []

                def process(command, *, stdin, timeout):
                    process_calls.append(command)
                    if len(process_calls) == 1:
                        Path(command[-1]).write_bytes(source_bytes)
                    else:
                        remote_path = Path(command[-1].split(":", 1)[1])
                        remote_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(command[-2], remote_path)
                    return subprocess.CompletedProcess(command, 0, "", "")

                with tempfile.TemporaryDirectory() as directory:
                    source = Path(directory) / "source.json"
                    source.write_bytes(source_bytes)
                    ssh_results = [
                        (0, "guard"),
                        (0, "seed"),
                        (0, "parent"),
                        (0, install_report),
                        cleanup_result,
                    ]
                    with patch("training_ops._run_process", side_effect=process), patch(
                        "training_ops._ssh", side_effect=ssh_results
                    ):
                        with self.assertRaises(RuntimeError) as raised:
                            _policy0_controller_transfer_reference(
                                "h200b", str(source), str(Path(directory) / "reference.json"),
                                ["n82"], {"sha256": source_sha, "size": source_size},
                                OLD_SHA256, OLD_SIZE,
                            )
                record = raised.exception.results["n82"]
                self.assertFalse(record["ok"])
                self.assertEqual(record["phase"], "cleanup")
                self.assertEqual(record["status"], "installed")
                self.assertEqual(record["new_sha"], source_sha)
                self.assertIn("seed cleanup", record["message"])
                self.assertIn(record["seed_compressed_path"], record["message"])
                self.assertIn(record["seed_partial_path"], record["message"])
                self.assertEqual(record["cleanup_paths"], [
                    record["seed_compressed_path"], record["seed_partial_path"]
                ])

    def test_controller_install_failure_keeps_rollback_foreign_path_and_full_results(self):
        old = b"old-reference"
        new = b"new-reference"
        old_sha, old_size = _identity_bytes(old)
        source_sha, source_size = _identity_bytes(new)
        nodes = ["n76"]
        process_calls = []

        def transfer(command, *, stdin, timeout):
            process_calls.append(command)
            if len(process_calls) == 1:
                Path(command[-1]).write_bytes(new)
            else:
                remote_path = Path(command[-1].split(":", 1)[1])
                remote_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(command[-2], remote_path)
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "qualification" / "reference.json"
            target.parent.mkdir(parents=True)
            target.write_bytes(old)
            foreign = root / "foreign-reference"
            foreign.write_bytes(b"foreign-displaced")
            marker = root / "exchange.marker"
            wrapper_dir = root / "bin"
            wrapper_dir.mkdir()
            wrapper = wrapper_dir / "python3"
            wrapper.write_text(
                "#!/bin/sh\n"
                f"if [ \"$3\" = {str(target)!r} ] && [ ! -e {str(marker)!r} ]; then\n"
                f"  touch -- {str(marker)!r}\n"
                "  /usr/bin/python3 \"$@\"\n"
                "  status=$?\n"
                f"  rm -rf -- {str(target)!r}\n"
                f"  ln -s -- {str(foreign)!r} {str(target)!r}\n"
                "  exit $status\n"
                "fi\n"
                "exec /usr/bin/python3 \"$@\"\n",
                encoding="utf-8",
            )
            wrapper.chmod(0o755)
            environment = dict(os.environ)
            environment["PATH"] = str(wrapper_dir) + os.pathsep + environment["PATH"]

            def remote(host, script, timeout):
                if "scp -q" in script and "partial=" in script:
                    values = {}
                    for line in script.splitlines():
                        if line.startswith(("source=", "partial=")):
                            values[line.split("=", 1)[0]] = line.split("=", 1)[1].strip("'")
                    destination = Path(values["partial"])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(values["source"], destination)
                    return 0, "verified"
                result = subprocess.run(
                    ["bash", "-c", script],
                    text=True,
                    capture_output=True,
                    env=environment,
                )
                return result.returncode, result.stdout + result.stderr

            with patch("training_ops._run_process", side_effect=transfer), patch(
                "training_ops._ssh", side_effect=remote
            ):
                with self.assertRaises(RuntimeError) as raised:
                    _policy0_controller_transfer_reference(
                        "h200b", "/source/reference.json", str(target), nodes,
                        {"sha256": source_sha, "size": source_size},
                        old_sha, old_size,
                    )

            results = raised.exception.results
            self.assertEqual(set(results), set(nodes))
            record = results["n76"]
            self.assertFalse(record["ok"])
            self.assertEqual(record["phase"], "install")
            self.assertIn("rollback could not be verified", record["message"])
            self.assertEqual(record["message"], record["remote_diagnostic"])
            self.assertTrue(Path(record["partial_path"]).is_symlink())
            self.assertEqual(Path(record["partial_path"]).readlink(), foreign)
            self.assertEqual(target.read_bytes(), old)
            self.assertEqual(
                target.with_name(target.name + ".preserved." + old_sha).read_bytes(), old
            )

    def test_controller_failure_retains_complete_multi_host_results(self):
        source_bytes = b"verified"
        source_sha, source_size = _identity_bytes(source_bytes)
        all_results = {
            "n76": {
                "ok": True, "phase": "fanout", "message": "verified",
            },
            "n77": {
                "ok": False, "phase": "fanout", "message": "exit 9",
            },
        }
        calls = []
        def fetch(command, *, stdin, timeout):
            calls.append(command)
            if len(calls) == 1:
                Path(command[-1]).write_bytes(source_bytes)
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_bytes(source_bytes)
            with patch("training_ops._run_process", side_effect=fetch), patch(
                "training_ops._ssh", side_effect=[(0, "seed"), (0, "cleanup")]
            ) as ssh, patch("training_ops._fan", return_value=all_results):
                with self.assertRaises(RuntimeError) as raised:
                    _policy0_controller_transfer_reference(
                        "h200b", str(source), str(Path(directory) / "reference.json"),
                        list(all_results), {"sha256": source_sha, "size": source_size},
                        OLD_SHA256, OLD_SIZE,
                    )
        results = raised.exception.results
        self.assertEqual(set(results), set(all_results))
        self.assertTrue(all(result["phase"] == "fanout" for result in results.values()))
        self.assertTrue(results["n76"]["ok"])
        self.assertFalse(results["n77"]["ok"])
        self.assertIn("installer not started", results["n76"]["message"])
        self.assertIn("partial.policy0", results["n76"]["partial_path"])
        self.assertIn("seed_compressed_sha256", results["n76"])
        self.assertEqual(ssh.call_count, 3)

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
