import copy
import json
import os
import re
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import training_ops


class H200ParityTest(unittest.TestCase):
    def _operation_ssh(
        self,
        commands,
        *,
        fail_start=False,
        fail_work_prep=False,
        fail_refresh=False,
        fail_capture=False,
        fail_cleanup=False,
        fail_fixture_backup=False,
        fail_capture_backup=False,
        fail_rollback=False,
        lose_fixture_ack=False,
        invalid_fixture_ack=False,
        lose_capture_ack=False,
        invalid_capture_ack=False,
        interrupt_fifo=False,
        main_status="absent",
    ):
        secret = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_"
        secret_inputs = []
        transaction = {"state": None}

        def call(host, command, timeout=120):
            del host, timeout
            commands.append(command)
            if "sha256sum --" in command:
                return 0, "config-sha\n"
            if "mkfifo -m 0600" in command:
                return 0, "fifo-ready\n"
            if "work-ready" in command:
                if fail_work_prep:
                    return 1, "permission denied"
                return 0, "work-ready\n"
            if "outputs-ready" in command:
                return 0, "outputs-ready\n"
            if "refresh-fixture" in command:
                if fail_refresh:
                    return 1, "refresh failed"
                return 0, json.dumps(
                    {
                        "fixture_sha256": "1" * 64,
                        "source_fixture_sha256": "2" * 64,
                        "token_ids_sha256": "3" * 64,
                        "positions_sha256": "4" * 64,
                        "advantages_sha256": "5" * 64,
                        "token_export_sha256": "6" * 64,
                        "rank": 56,
                    }
                ) + "\n"
            if "state=created" in command and ".fixture-publish-failed-" in command:
                transaction["state"]["phase"] = "fixture_published"
                transaction["state"]["fixture"]["action"] = "replaced"
                if lose_fixture_ack:
                    return 124, "fixture acknowledgement lost"
                if invalid_fixture_ack:
                    return 0, "invalid\n"
                return 0, "replaced\n"
            if "state=created" in command and ".h200-capture-previous-" in command:
                transaction["state"]["phase"] = "capture_published"
                transaction["state"]["capture"]["action"] = "replaced"
                if lose_capture_ack:
                    return 124, "capture acknowledgement lost"
                if invalid_capture_ack:
                    return 0, "invalid\n"
                return 0, "replaced\n"
            if " committed none committed" in command:
                transaction["state"]["phase"] = "committed"
                return 0, ""
            if " rolled_back none rolled_back" in command:
                if fail_rollback:
                    return 73, "rollback stopped"
                transaction["state"]["phase"] = "rolled_back"
                return 0, ""
            if command.startswith("test -f ") and command.endswith("transaction.json"):
                return (0, "") if transaction["state"] is not None else (1, "")
            if "docker ps -a" in command and "glm52-optims-h200-" in command:
                return 0, "absent\n"
            if "docker ps -a" in command and "glm52-optims-h200" in command:
                return 0, f"{main_status}\n"
            if "start_inference.sh" in command and fail_start:
                return 1, "start rejected"
            if "capture-h200" in command and fail_capture:
                return 1, f"capture rejected {secret}"
            if (
                fail_cleanup
                and "bootstrap.fifo" in command
                and ".diagnostic-source-" in command
                and "sudo rm -f --" in command
            ):
                return 1, "cleanup failed"
            if (
                fail_fixture_backup
                and ".fixture-previous-" in command
                and "rm -rf --" in command
                and ".fixture-failed-" not in command
            ):
                return 1, "fixture backup retained"
            if (
                fail_capture_backup
                and ".h200-capture-previous-" in command
                and "rm -rf --" in command
                and ".h200-capture-failed-" not in command
            ):
                return 1, "capture backup retained"
            return 0, ""

        def secret_call(host, command, provided, label, timeout):
            secret_inputs.append(provided)
            code, output = call(host, command, timeout)
            clean = training_ops._redact_sensitive(output, (provided,))
            if code:
                raise RuntimeError(f"{label} failed on {host}: {clean}")
            return clean

        def receive(host, command, timeout):
            commands.append(command)
            if interrupt_fifo:
                raise KeyboardInterrupt
            self.assertEqual(host, "h200a")
            self.assertEqual(timeout, 7230)
            return secret

        call.transaction = transaction
        call.receive = receive
        return secret, call, secret_call, secret_inputs

    def _run_operation(self, ssh_call, secret_call, secret, *, keep_engine=False):
        metadata = json.dumps({"model_revision": "a" * 40, "adapter_sha256": "b" * 64})
        source_root = f"{training_ops.parity_trace_root}/qualification-1"

        def write_transaction(_host, _path, state, _sensitive):
            ssh_call.transaction["state"] = copy.deepcopy(state)

        def transaction_snapshot(_host, _path, _sensitive):
            state = copy.deepcopy(ssh_call.transaction["state"])
            if state is None:
                raise RuntimeError("remote transaction state is missing")
            observed = {}
            for kind in ("fixture", "capture"):
                artifact = state[kind]
                action = artifact["action"]
                for key in ("live", "previous", "candidate", "failed"):
                    exists = (key == "live" and action in {"created", "replaced"}) or (
                        key == "previous" and action == "replaced"
                    )
                    digest = None
                    if key == "live" and exists:
                        digest = artifact["new_manifest_sha256"]
                    elif key == "previous" and exists:
                        digest = artifact["old_manifest_sha256"]
                    observed[f"{kind}.{key}"] = {
                        "exists": exists,
                        "symlink": False,
                        "manifest_sha256": digest,
                    }
            return {"state": state, "observed": observed}

        def manifest_sha(_host, path, _label, _sensitive):
            if ".diagnostic-work-1111111111111111/capture" in path:
                return "7" * 64
            if path == training_ops.glm_parity_paths["h200_capture"]:
                return "8" * 64
            raise AssertionError(f"unexpected manifest path: {path}")

        with patch.object(
            training_ops.secrets, "token_hex", return_value="1" * 16
        ), patch.object(
            training_ops,
            "_diagnostic_capture_preflight",
            return_value={"files": 20, "estimated_bytes": 1},
        ), patch.object(
            training_ops,
            "_resolve_glm52_parity_diagnostic_source",
            return_value=(
                training_ops.trainer_head,
                "n85",
                56,
                source_root,
                json.loads(metadata),
            ),
        ), patch.object(
            training_ops,
            "_resolve_glm52_parity_token_export",
            return_value=f"{training_ops.parity_token_root}/step_0/rank_56.jsonl",
        ), patch.object(training_ops, "_remote_ok", return_value="source-ready"), patch.object(
            training_ops, "_sha256_file", return_value="config-sha"
        ), patch.object(training_ops, "_copy_parity_diagnostic_input") as copy_input, patch.object(
            training_ops, "_copy_parity_refresh_file"
        ), patch.object(
            training_ops,
            "_write_parity_transaction_state",
            side_effect=write_transaction,
        ), patch.object(
            training_ops,
            "_parity_transaction_snapshot",
            side_effect=transaction_snapshot,
        ), patch.object(
            training_ops,
            "_remote_manifest_sha256",
            side_effect=manifest_sha,
        ), patch.object(training_ops, "_h200_parity_container_preflight"), patch.object(
            training_ops, "_assert_h200_parity_port_free"
        ), patch.object(
            training_ops,
            "_h200_parity_listener_identity",
            return_value={
                "listener_pid": 1234,
                "container_init_pid": 1234,
                "listener_relation": "init",
                "container_id": "d" * 64,
                "host_boot_id": "11111111-2222-3333-4444-555555555555",
            },
        ), patch.object(
            training_ops,
            "_h200_capture_bootstrap_identity",
            return_value={
                "status": "ready",
                "process_id": 1,
                "boot_nonce": "e" * 32,
                "token_sha256": training_ops.hashlib.sha256(
                    secret.encode("ascii")
                ).hexdigest(),
            },
        ), patch.object(
            training_ops, "_configured_h200_engine_startup_grace", return_value=7200
        ), patch.object(training_ops, "_ssh", side_effect=ssh_call), patch.object(
            training_ops, "_sensitive_ssh_input", side_effect=secret_call
        ), patch.object(
            training_ops, "_receive_parity_bootstrap", side_effect=ssh_call.receive
        ):
            result = training_ops.diagnose_glm52_policy_parity("qualification-1", keep_engine=keep_engine)
        self.last_diagnostic_copy = copy_input.call_args
        return result

    @staticmethod
    def _source_evidence(pp_by_host, session):
        session.setdefault("dense_layers", 3)
        session.setdefault("hidden_layers", 78)
        modules = {
            "prime_rl.models.glm": {
                "location": "mounted",
                "sha256": "c" * 64,
            }
        }
        records = [
                {
                    "schema_version": 1,
                    "kind": "glm52_policy_parity_trainer_runtime",
                    "host": host,
                    "pp_rank": pp_rank,
                    "cp_rank": 0,
                    "global_rank": pp_rank * 8,
                    "pp_size": 8,
                    "cp_size": 8,
                    "batch_size": 8,
                    "image_digest": "sha256:" + "d" * 64,
                    "owned_layers": list(range(3 + pp_rank, 78, 8)),
                    "modules": modules,
                }
            for host, pp_rank in pp_by_host.items()
        ]
        return json.dumps(
            {
                "union": {
                    "kind": "glm52_policy_parity_trainer_provenance_union",
                    "session": session,
                    "records": records,
                    "binding_sha256": "e" * 64,
                    "fixture_source_sha256": "f" * 64,
                },
                "manifest": {
                    "kind": "glm52_policy_parity_trainer_fixture_source",
                    "rank": 56,
                    "file": "trainer-fixture-source.npz",
                    "file_sha256": "f" * 64,
                    "binding_sha256": "e" * 64,
                },
            }
        )

    def test_diagnostic_source_uses_provenance_not_trainer_list_order(self):
        hosts = ["n76", "n77", "n78", "n80", "n82", "n83", "n84", "n85"]
        pp_by_host = {"n82": 0}
        pp_by_host.update(
            {host: rank for rank, host in enumerate((h for h in hosts if h != "n82"), 1)}
        )
        session = {
            "side": "trainer",
            "model_revision": "a" * 40,
            "adapter_sha256": "b" * 64,
        }
        queried = []

        def remote(host, command, label, timeout=120, connect_timeout=None):
            del command, label
            self.assertEqual(timeout, 60)
            self.assertEqual(connect_timeout, 45)
            queried.append(host)
            return self._source_evidence(pp_by_host, session)

        with patch.object(training_ops, "trainer_hosts", hosts), patch.object(
            training_ops, "trainer_head", "n82"
        ), patch.object(training_ops, "_remote_ok", side_effect=remote):
            source_host, final_host, final_rank, source_root, source_session = (
                training_ops._resolve_glm52_parity_diagnostic_source("qualification-1")
            )

        self.assertEqual(source_host, "n82")
        self.assertEqual(final_host, "n85")
        self.assertEqual(final_rank, 56)
        self.assertEqual(
            source_root,
            f"{training_ops.parity_trace_root}/qualification-1",
        )
        self.assertEqual(source_session, session)
        self.assertEqual(queried, ["n82"])

    def test_provenance_timeout_stops_serial_read_before_h200(self):
        hosts = ["n76", "n77", "n78", "n80", "n82", "n83", "n84", "n85"]
        queried = []

        def remote(host, command, label, timeout=120, connect_timeout=None):
            del command, label
            self.assertEqual(timeout, 60)
            self.assertEqual(connect_timeout, 45)
            queried.append(host)
            raise RuntimeError("banner exchange timed out")

        with patch.object(training_ops, "trainer_hosts", hosts), patch.object(
            training_ops, "trainer_head", "n82"
        ), patch.object(
            training_ops, "_remote_ok", side_effect=remote
        ), patch.object(
            training_ops,
            "_diagnostic_capture_preflight",
            return_value={"files": 20, "estimated_bytes": 1},
        ), patch.object(training_ops, "_ssh") as h200, self.assertRaisesRegex(
            RuntimeError, "banner exchange timed out"
        ):
            training_ops.diagnose_glm52_policy_parity("qualification-1")

        self.assertEqual(queried, ["n82"])
        h200.assert_not_called()

    def test_diagnostic_source_refuses_missing_or_duplicate_pp0(self):
        hosts = ["n76", "n77", "n78", "n80", "n82", "n83", "n84", "n85"]
        session = {
            "side": "trainer",
            "model_revision": "a" * 40,
            "adapter_sha256": "b" * 64,
        }
        for pp_by_host in (
            {host: rank + 1 for rank, host in enumerate(hosts)},
            {host: (0 if rank < 2 else rank - 1) for rank, host in enumerate(hosts)},
        ):
            with self.subTest(pp_by_host=pp_by_host):
                def remote(
                    host, command, label, timeout=120, connect_timeout=None
                ):
                    del command, label
                    self.assertEqual(timeout, 60)
                    self.assertEqual(connect_timeout, 45)
                    self.assertEqual(host, "n82")
                    return self._source_evidence(pp_by_host, session)

                with patch.object(training_ops, "trainer_hosts", hosts), patch.object(
                    training_ops, "trainer_head", "n82"
                ), patch.object(training_ops, "_remote_ok", side_effect=remote):
                    with self.assertRaisesRegex(training_ops.Refused, "one PP0/CP0"):
                        training_ops._resolve_glm52_parity_diagnostic_source(
                            "qualification-1"
                        )

    def test_diagnostic_source_refuses_pp0_that_is_not_trainer_head(self):
        hosts = ["n76", "n77", "n78", "n80", "n82", "n83", "n84", "n85"]
        session = {
            "side": "trainer",
            "model_revision": "a" * 40,
            "adapter_sha256": "b" * 64,
        }
        pp_by_host = {host: rank for rank, host in enumerate(hosts)}

        def remote(host, command, label, timeout=120, connect_timeout=None):
            del command, label
            self.assertEqual(timeout, 60)
            self.assertEqual(connect_timeout, 45)
            self.assertEqual(host, "n82")
            return self._source_evidence(pp_by_host, session)

        with patch.object(training_ops, "trainer_hosts", hosts), patch.object(
            training_ops, "trainer_head", "n82"
        ), patch.object(training_ops, "_remote_ok", side_effect=remote):
            with self.assertRaisesRegex(training_ops.Refused, "trainer head"):
                training_ops._resolve_glm52_parity_diagnostic_source("qualification-1")

    def test_provenance_ssh_uses_one_bounded_connect_override(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ready\n", stderr=""
        )
        with patch.object(
            training_ops.subprocess, "run", return_value=completed
        ) as run:
            result = training_ops._ssh(
                "n76", "true", timeout=60, connect_timeout=45
            )

        expected = list(training_ops.ssh_command)
        option = expected.index("ConnectTimeout=25")
        expected[option] = "ConnectTimeout=45"
        self.assertEqual(result, (0, "ready\n"))
        self.assertEqual(run.call_args.args[0], expected + ["n76", "true"])
        self.assertEqual(run.call_args.kwargs["timeout"], 60)
        self.assertIs(run.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(
            [value for value in run.call_args.args[0] if value.startswith("ConnectTimeout=")],
            ["ConnectTimeout=45"],
        )

    def test_provenance_ssh_adds_no_multiplexing_cleanup_state(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        with patch.object(
            training_ops.subprocess, "run", return_value=completed
        ) as run:
            training_ops._ssh("n76", "true", connect_timeout=45)

        arguments = run.call_args.args[0]
        self.assertNotIn("-M", arguments)
        self.assertNotIn("-O", arguments)
        self.assertFalse(
            any(
                value.startswith(("ControlMaster=", "ControlPath=", "ControlPersist="))
                for value in arguments
            )
        )
        run.assert_called_once()

    def test_provenance_ssh_timeout_is_bounded_and_not_retried(self):
        with patch.object(
            training_ops.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=60),
        ) as run:
            result = training_ops._ssh(
                "n76", "true", timeout=60, connect_timeout=45
            )

        self.assertEqual(result, (124, "TIMEOUT"))
        self.assertEqual(run.call_args.kwargs["timeout"], 60)
        self.assertIn("ConnectTimeout=45", run.call_args.args[0])
        run.assert_called_once()

    def test_provenance_ssh_refuses_invalid_connect_timeout(self):
        for value in (True, 0, 121, "60"):
            with self.subTest(value=value), patch.object(
                training_ops.subprocess, "run"
            ) as run, self.assertRaisesRegex(
                training_ops.Refused, "integer from 1 to 120"
            ):
                training_ops._ssh("n76", "true", connect_timeout=value)
            run.assert_not_called()

    def test_final_pp_token_export_requires_one_stable_exact_source(self):
        expected = f"{training_ops.parity_token_root}/step_0/rank_56.jsonl"
        with patch.object(training_ops, "_remote_ok", return_value=expected) as remote:
            actual = training_ops._resolve_glm52_parity_token_export("n85", 56)
        self.assertEqual(actual, expected)
        command = remote.call_args.args[1]
        self.assertIn("-maxdepth 2", command)
        self.assertIn("rank_56.jsonl", command)
        self.assertIn("/STABLE", command)
        with patch.object(
            training_ops,
            "_remote_ok",
            return_value=f"{expected}\n{training_ops.parity_token_root}/step_1/rank_56.jsonl",
        ):
            with self.assertRaisesRegex(training_ops.Refused, "unique and exact"):
                training_ops._resolve_glm52_parity_token_export("n85", 56)

    def test_diagnostic_capture_preflight_fits_existing_h200_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "h200.toml"
            config.write_text(
                "[env_vars]\n"
                'PRIME_PARITY_DIAGNOSTICS = "1"\n'
                'PRIME_PARITY_DIAGNOSTIC_TOKENS = "256"\n'
                'PRIME_PARITY_DIAGNOSTIC_MAX_FILES = "64"\n'
                'PRIME_PARITY_DIAGNOSTIC_MAX_BYTES = "100000000"\n'
            )
            result = training_ops._diagnostic_capture_preflight(config, "h200")
        self.assertEqual(result, {"files": 20, "estimated_bytes": 96_485_376})

    def test_sensitive_ssh_input_keeps_token_out_of_command_and_redacts_error(self):
        secret = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_"
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr=f"remote repeated {secret}"
        )
        with patch.object(training_ops.subprocess, "run", return_value=completed) as run:
            with self.assertRaises(RuntimeError) as caught:
                training_ops._sensitive_ssh_input(
                    "h200a", "run isolated command", secret, "diagnostic", 30
                )
        self.assertNotIn(secret, " ".join(run.call_args.args[0]))
        self.assertEqual(run.call_args.kwargs["input"], secret + "\n")
        self.assertNotIn(secret, str(caught.exception))
        self.assertIn("[redacted]", str(caught.exception))

    def test_bootstrap_receive_uses_stdout_only_and_never_exposes_token(self):
        secret = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_"
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=secret + "\n", stderr=""
        )
        with patch.object(training_ops.subprocess, "run", return_value=completed) as run:
            token = training_ops._receive_parity_bootstrap("h200a", "read fifo", 30)
        self.assertEqual(token, secret)
        self.assertNotIn(secret, " ".join(run.call_args.args[0]))
        self.assertIs(run.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(run.call_args.kwargs["timeout"], 30)

        failed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout=secret + "\n", stderr=secret
        )
        with patch.object(
            training_ops.subprocess, "run", return_value=failed
        ), self.assertRaises(RuntimeError) as caught:
            training_ops._receive_parity_bootstrap("h200a", "read fifo", 30)
        self.assertNotIn(secret, str(caught.exception))

    def test_bootstrap_receive_command_propagates_read_and_cleanup_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing.fifo"
            read_command = training_ops._parity_bootstrap_receive_command(
                str(missing), 1
            ).replace("sudo ", "")
            read = subprocess.run(
                ["bash", "-c", read_command],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(read.returncode, 0)
            self.assertFalse(missing.exists())

            retained = root / "bootstrap.fifo"
            retained.write_text("x" * 64 + "\n")
            cleanup_command = training_ops._parity_bootstrap_receive_command(
                str(retained), 1
            )
            cleanup = subprocess.run(
                [
                    "bash",
                    "-c",
                    "sudo() { if test \"$1\" = rm; then return 93; fi; \"$@\"; }; "
                    + cleanup_command,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(cleanup.returncode, 0)
            self.assertTrue(retained.exists())

            retained_regular = root / "retained-regular.fifo"
            retained_regular.write_text("x" * 64 + "\n")
            regular_command = training_ops._parity_bootstrap_receive_command(
                str(retained_regular), 1
            )
            regular_cleanup = subprocess.run(
                [
                    "bash",
                    "-c",
                    "sudo() { if test \"$1\" = rm; then return 0; fi; \"$@\"; }; "
                    + regular_command,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(regular_cleanup.returncode, 0)
            self.assertTrue(retained_regular.is_file())

            retained_symlink = root / "retained-symlink.fifo"
            retained_symlink.symlink_to(root / "missing-target")
            symlink_command = training_ops._parity_bootstrap_receive_command(
                str(retained_symlink), 1
            )
            symlink_cleanup = subprocess.run(
                [
                    "bash",
                    "-c",
                    "sudo() { "
                    "if test \"$1\" = timeout; then printf '%064d\\n' 0; return 0; fi; "
                    "if test \"$1\" = rm; then return 0; fi; "
                    "\"$@\"; }; "
                    + symlink_command,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(symlink_cleanup.returncode, 0)
            self.assertTrue(retained_symlink.is_symlink())

    def test_temporary_cleanup_keeps_early_failure_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            transaction = run / "transaction.json"
            transaction.write_text("state")
            diagnostic_input = root / "diagnostic-input"
            diagnostic_input.mkdir()
            command = training_ops._parity_temporary_cleanup_command(
                bootstrap_fifo=str(root / "missing-fifo"),
                transaction_path=str(transaction),
                run_directory=str(run),
                source_diagnostic_input=str(diagnostic_input),
                source_session=str(root / "missing-session"),
                source_compact=str(root / "missing-compact"),
                source_compact_manifest=str(root / "missing-manifest"),
                container_work_directory=str(root / "missing-work"),
                retain_container_work=False,
            )
            completed = subprocess.run(
                ["bash", "-c", "sudo() { \"$@\"; }; " + command],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertTrue(diagnostic_input.is_dir())

    def test_temporary_cleanup_rejects_retained_file_and_symlink(self):
        for kind in ("file", "symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                run = root / "run"
                run.mkdir()
                transaction = run / "transaction.json"
                transaction.write_text("state")
                diagnostic_input = root / "diagnostic-input"
                if kind == "file":
                    diagnostic_input.write_text("retained")
                else:
                    diagnostic_input.symlink_to(root / "missing-target")
                command = training_ops._parity_temporary_cleanup_command(
                    bootstrap_fifo=str(root / "missing-fifo"),
                    transaction_path=str(transaction),
                    run_directory=str(run),
                    source_diagnostic_input=str(diagnostic_input),
                    source_session=str(root / "missing-session"),
                    source_compact=str(root / "missing-compact"),
                    source_compact_manifest=str(root / "missing-manifest"),
                    container_work_directory=str(root / "missing-work"),
                    retain_container_work=False,
                )
                completed = subprocess.run(
                    [
                        "bash",
                        "-c",
                        "rm() { return 0; }; sudo() { \"$@\"; }; " + command,
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertTrue(diagnostic_input.exists() or diagnostic_input.is_symlink())

    def test_diagnostic_capture_preflight_refuses_existing_cap_overflow(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "h200.toml"
            config.write_text(
                "[env_vars]\n"
                'PRIME_PARITY_DIAGNOSTICS = "1"\n'
                'PRIME_PARITY_DIAGNOSTIC_TOKENS = "256"\n'
                'PRIME_PARITY_DIAGNOSTIC_MAX_FILES = "64"\n'
                'PRIME_PARITY_DIAGNOSTIC_MAX_BYTES = "96485375"\n'
            )
            with self.assertRaises(training_ops.Refused):
                training_ops._diagnostic_capture_preflight(config, "h200")

    def test_post_launch_inspection_accepts_exact_command_and_config(self):
        trace = (
            "/data/glm52-rl/qualifications/glm52-policy-parity/"
            ".diagnostic-work-1111111111111111/h200-router-trace"
        )
        command = [
            "@",
            training_ops.glm_parity_inference_config,
            "--vllm-extra",
            training_ops.glm_parity_vllm_extra,
        ]
        inspect = "\n".join(
            [
                json.dumps(True),
                json.dumps(training_ops.inference_image),
                json.dumps(command),
                json.dumps(
                    [
                        "PRIME_PARITY_BOOTSTRAP_FIFO=/run/prime-parity-bootstrap",
                        f"PRIME_PARITY_ROUTER_TRACE_DIR={trace}",
                    ]
                ),
                json.dumps(
                    [
                        {
                            "Type": "bind",
                            "Source": (
                                "/data/glm52-rl/qualifications/glm52-policy-parity/"
                                ".diagnostic-run-1111111111111111/bootstrap.fifo"
                            ),
                            "Destination": "/run/prime-parity-bootstrap",
                            "RW": True,
                        }
                    ]
                ),
                json.dumps("no"),
            ]
        )
        config = json.dumps(
            {
                "max_model_len": 32769,
                "tp": 8,
                "dp": 1,
                "api_servers": 1,
                "lora_rank": 32,
                "router_trace": True,
            }
        )
        with patch.object(training_ops, "_ssh", side_effect=[(0, inspect), (0, config)]):
            result = training_ops._h200_parity_container_preflight("h200a", trace)
        self.assertEqual(result["config"]["max_model_len"], 32769)

    def test_post_launch_inspection_refuses_wrong_config_command(self):
        trace = (
            "/data/glm52-rl/qualifications/glm52-policy-parity/"
            ".diagnostic-work-1111111111111111/h200-router-trace"
        )
        command = [
            "@",
            f"{training_ops.remote_repo}/configs/h200/inference.toml",
            "--vllm-extra",
            training_ops.glm_parity_vllm_extra,
        ]
        inspect = "\n".join(
            [
                json.dumps(True),
                json.dumps(training_ops.inference_image),
                json.dumps(command),
                json.dumps(
                    [
                        "PRIME_PARITY_BOOTSTRAP_FIFO=/run/prime-parity-bootstrap",
                        f"PRIME_PARITY_ROUTER_TRACE_DIR={trace}",
                    ]
                ),
                json.dumps(
                    [
                        {
                            "Type": "bind",
                            "Source": (
                                "/data/glm52-rl/qualifications/glm52-policy-parity/"
                                ".diagnostic-run-1111111111111111/bootstrap.fifo"
                            ),
                            "Destination": "/run/prime-parity-bootstrap",
                            "RW": True,
                        }
                    ]
                ),
                json.dumps("no"),
            ]
        )
        with patch.object(training_ops, "_ssh", return_value=(0, inspect)):
            with self.assertRaises(training_ops.Refused):
                training_ops._h200_parity_container_preflight("h200a", trace)

    def test_post_launch_inspection_refuses_token_in_container_environment(self):
        trace = (
            "/data/glm52-rl/qualifications/glm52-policy-parity/"
            ".diagnostic-work-1111111111111111/h200-router-trace"
        )
        secret = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_"
        command = [
            "@",
            training_ops.glm_parity_inference_config,
            "--vllm-extra",
            training_ops.glm_parity_vllm_extra,
        ]
        inspect = "\n".join(
            [
                json.dumps(True),
                json.dumps(training_ops.inference_image),
                json.dumps(command),
                json.dumps(
                    [
                        "PRIME_PARITY_BOOTSTRAP_FIFO=/run/prime-parity-bootstrap",
                        f"PRIME_PARITY_ROUTER_TRACE_DIR={trace}",
                        f"PRIME_PARITY_BOOTSTRAP_TOKEN={secret}",
                    ]
                ),
                json.dumps([]),
                json.dumps("no"),
            ]
        )
        with patch.object(training_ops, "_ssh", return_value=(0, inspect)):
            with self.assertRaisesRegex(training_ops.Refused, "metadata contains") as caught:
                training_ops._h200_parity_container_preflight("h200a", trace)
        self.assertNotIn(secret, str(caught.exception))

    def test_parity_port_preflight_refuses_listener_or_protected_inference(self):
        with patch.object(
            training_ops, "_ssh", return_value=(0, "port-8000-free\n")
        ) as ssh:
            training_ops._assert_h200_parity_port_free("h200a")
        command = ssh.call_args.args[1]
        self.assertIn("sport = :8000", command)
        for name in training_ops.protected_inference:
            self.assertIn(re.escape(name), command)

        for output in ("port-8000-in-use", "protected-inference-active"):
            with self.subTest(output=output), patch.object(
                training_ops, "_ssh", return_value=(74, output)
            ), self.assertRaises(training_ops.Refused):
                training_ops._assert_h200_parity_port_free("h200a")

    def test_parity_port_preflight_propagates_docker_and_ss_failures(self):
        command = training_ops._h200_parity_port_preflight_command()
        cases = {
            "docker": "docker() { return 91; }; ss() { return 0; };",
            "ss": "docker() { return 0; }; ss() { return 92; };",
        }
        for name, functions in cases.items():
            with self.subTest(name=name):
                script = f"sudo() {{ \"$@\"; }}; {functions} {command}"
                result = subprocess.run(
                    ["bash", "-c", script],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("port-8000-free", result.stdout)

    def test_listener_identity_binds_port_to_running_isolated_container(self):
        identity = {
            "listener_pid": 1235,
            "container_init_pid": 1234,
            "listener_relation": "child",
            "container_id": "a" * 64,
            "host_boot_id": "11111111-2222-3333-4444-555555555555",
        }
        with patch.object(
            training_ops, "_sensitive_ssh", return_value=json.dumps(identity)
        ) as remote:
            result = training_ops._h200_parity_listener_identity(
                "h200a", ("secret",), 7200
            )
        self.assertEqual(result, identity)
        command = remote.call_args.args[1]
        self.assertIn(".State.Running", command)
        self.assertIn("sport = :8000", command)
        self.assertIn(".State.Pid", command)
        self.assertNotIn('test "$listener_pid" = "$container_pid" ||', command)
        self.assertIn("/cgroup", command)
        self.assertIn("$container_id", command)
        self.assertIn("container-image-mismatch", command)
        self.assertIn("container-command-mismatch", command)
        self.assertIn("container-restart-policy-invalid", command)
        self.assertEqual(remote.call_args.args[4], 7230)

    def test_listener_identity_shell_accepts_child_only_inside_container_cgroup(self):
        command = training_ops._h200_parity_listener_identity_command(0)
        container_id = "a" * 64
        expected_command = json.dumps(
            [
                "@",
                training_ops.glm_parity_inference_config,
                "--vllm-extra",
                training_ops.glm_parity_vllm_extra,
            ],
            separators=(",", ":"),
        )
        common = (
            "sudo() { \"$@\"; }; "
            "docker() { printf '%s\\n' true 101 "
            f"{container_id} {shlex.quote(json.dumps(training_ops.inference_image))} "
            f"{shlex.quote(expected_command)} 0 {shlex.quote(json.dumps('no'))}; }}; "
            "ss() { printf '%s\\n' 'LISTEN 0 128 *:8000 users:((\"python\",pid=202,fd=9))'; }; "
        )
        inside = (
            common
            + "cat() { case \"$*\" in *cgroup) printf '%s\\n' '0::/docker/"
            + container_id
            + "';; *) printf '%s\\n' 11111111-2222-3333-4444-555555555555;; esac; }; "
            + command
        )
        accepted = subprocess.run(
            ["bash", "-c", inside], text=True, capture_output=True, check=False
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        identity = json.loads(accepted.stdout)
        self.assertEqual(identity["listener_pid"], 202)
        self.assertEqual(identity["container_init_pid"], 101)
        self.assertEqual(identity["listener_relation"], "child")

        outside = (
            common
            + "cat() { case \"$*\" in *cgroup) printf '%s\\n' '0::/docker/"
            + ("b" * 64)
            + "';; *) printf '%s\\n' 11111111-2222-3333-4444-555555555555;; esac; }; "
            + command
        )
        rejected = subprocess.run(
            ["bash", "-c", outside], text=True, capture_output=True, check=False
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("parity-listener-error:listener-outside-container", rejected.stderr)

    def test_listener_identity_shell_failures_are_detailed(self):
        command = training_ops._h200_parity_listener_identity_command(0)
        container_id = "a" * 64
        expected_command = json.dumps(
            [
                "@",
                training_ops.glm_parity_inference_config,
                "--vllm-extra",
                training_ops.glm_parity_vllm_extra,
            ],
            separators=(",", ":"),
        )
        inspect = (
            "docker() { printf '%s\\n' true 101 "
            f"{container_id} {shlex.quote(json.dumps(training_ops.inference_image))} "
            f"{shlex.quote(expected_command)} 0 {shlex.quote(json.dumps('no'))}; }}; "
        )
        cases = {
            "container-inspect-failed": "docker() { return 71; }; ss() { :; };",
            "container-not-running": inspect.replace("true 101", "false 101")
            + "ss() { :; };",
            "container-image-mismatch": inspect.replace(
                shlex.quote(json.dumps(training_ops.inference_image)),
                shlex.quote(json.dumps("wrong-image")),
            )
            + "ss() { :; };",
            "container-command-mismatch": inspect.replace(
                shlex.quote(expected_command), shlex.quote(json.dumps(["wrong"]))
            )
            + "ss() { :; };",
            "container-restarted": inspect.replace(" 0 ", " 1 ")
            + "ss() { :; };",
            "container-restart-policy-invalid": inspect.replace(
                shlex.quote(json.dumps("no")), shlex.quote(json.dumps("always"))
            )
            + "ss() { :; };",
            "listener-timeout": inspect + "ss() { :; }; sleep() { :; };",
            "multiple-listeners": inspect
            + "ss() { printf '%s\\n' 'LISTEN pid=202,' 'LISTEN pid=203,'; };",
            "multiple-listener-processes": inspect
            + "ss() { printf '%s\\n' 'LISTEN pid=202,pid=203,'; };",
            "listener-pid-missing": inspect
            + "ss() { printf '%s\\n' 'LISTEN 0 128 *:8000'; };",
            "listener-probe-failed": inspect + "ss() { return 72; };",
            "listener-cgroup-unreadable": inspect
            + "ss() { printf '%s\\n' 'LISTEN pid=202,'; }; cat() { return 73; };",
        }
        for detail, functions in cases.items():
            with self.subTest(detail=detail):
                script = f"sudo() {{ \"$@\"; }}; {functions} {command}"
                result = subprocess.run(
                    ["bash", "-c", script],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"parity-listener-error:{detail}", result.stderr)

    def test_capture_records_boot_nonce_and_bootstrap_hash(self):
        secret = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_"
        identity = {
            "status": "ready",
            "process_id": 1,
            "boot_nonce": "f" * 32,
            "token_sha256": training_ops.hashlib.sha256(
                secret.encode("ascii")
            ).hexdigest(),
        }
        capture = (
            "/data/glm52-rl/qualifications/glm52-policy-parity/"
            ".diagnostic-work-1111111111111111/capture"
        )
        with patch.object(
            training_ops, "_sensitive_ssh", return_value=json.dumps(identity)
        ):
            self.assertEqual(
                training_ops._h200_capture_bootstrap_identity(
                    "h200a", capture, secret
                ),
                identity,
            )
        wrong = {**identity, "token_sha256": "0" * 64}
        with patch.object(
            training_ops, "_sensitive_ssh", return_value=json.dumps(wrong)
        ), self.assertRaisesRegex(training_ops.Refused, "identity is invalid"):
            training_ops._h200_capture_bootstrap_identity("h200a", capture, secret)

    def test_diagnostic_operation_redacts_token_and_cleans_up_after_failure(self):
        commands = []
        secret, ssh_call, secret_call, _ = self._operation_ssh(
            commands, fail_capture=True
        )
        with self.assertRaises(RuntimeError) as caught:
            self._run_operation(ssh_call, secret_call, secret, keep_engine=True)
        self.assertNotIn(secret, str(caught.exception))
        self.assertIn("[redacted]", str(caught.exception))
        self.assertTrue(
            any(
                'name=glm52-optims-h200' in command
                and 'sudo docker rm -f "$name"' in command
                for command in commands
            )
        )
        rollback = next(
            command
            for command in commands
            if ".fixture-failed-1111111111111111" in command
        )
        self.assertIn(".fixture-previous-1111111111111111", rollback)
        self.assertIn(".fixture-failed-1111111111111111", rollback)
        self.assertTrue(
            any(
                'sudo rm -f -- "$state"' in command
                and 'sudo rmdir -- "$run"' in command
                for command in commands
            )
        )

    def test_diagnostic_operation_uses_memory_token_and_exact_cleanup(self):
        commands = []
        secret, ssh_call, secret_call, secret_inputs = self._operation_ssh(
            commands, main_status="exited"
        )
        result = self._run_operation(ssh_call, secret_call, secret)
        self.assertNotIn(secret, json.dumps(result, sort_keys=True))
        self.assertEqual(result["cleanup_warnings"], [])
        self.assertEqual(result["listener_identity"]["listener_pid"], 1234)
        self.assertEqual(result["bootstrap_identity"]["boot_nonce"], "e" * 32)
        self.assertEqual(secret_inputs, [secret])
        self.assertEqual(self.last_diagnostic_copy.args[0], training_ops.trainer_head)
        self.assertNotEqual(self.last_diagnostic_copy.args[0], training_ops.trainer_hosts[0])
        self.assertFalse(any(secret in command for command in commands))
        start = next(
            command
            for command in commands
            if "start_inference.sh" in command and "prime_deployment=" in command
        )
        self.assertIn("prime_require_absent=1", start)
        self.assertIn("prime_parity_bootstrap_fifo=", start)
        self.assertTrue(any("capture-h200" in command for command in commands))
        self.assertFalse(
            any("PRIME_PARITY_BOOTSTRAP_TOKEN" in command for command in commands)
        )
        capture = next(command for command in commands if "capture-h200" in command)
        self.assertIn("docker run --rm -i", capture)
        self.assertNotIn(" -e ", capture)
        fifo_setup = next(command for command in commands if "mkfifo -m 0600" in command)
        self.assertIn("chown 1000:1000", fifo_setup)
        self.assertIn("1000:1000:600", fifo_setup)
        fifo_receive = next(
            command
            for command in commands
            if "timeout --foreground" in command and "bootstrap.fifo" in command
        )
        self.assertIn("timeout --foreground 7200", fifo_receive)
        self.assertIn("cat --", fifo_receive)
        self.assertNotIn("timeout --foreground 180", fifo_receive)
        fifo_cleanup = next(
            command
            for command in commands
            if "sudo rmdir --" in command and "bootstrap.fifo" in command
        )
        self.assertIn("sudo rmdir --", fifo_cleanup)
        self.assertFalse(any("training.env" in command for command in commands))
        self.assertFalse(
            any(
                "trainer-traces" in command and "rm " in command
                for command in commands
            )
        )
        removals = [command for command in commands if "docker rm" in command]
        self.assertIn(
            "sudo docker rm glm52-optims-h200 >/dev/null",
            removals,
        )
        final_cleanup = next(command for command in removals if 'rm -f "$name"' in command)
        self.assertIn('name=glm52-optims-h200', final_cleanup)
        self.assertIn('name=^/${name}$', final_cleanup)
        self.assertFalse(any("*" in command for command in removals))
        self.assertTrue(
            any(
                ".fixture-previous-1111111111111111" in command
                and "rm -rf --" in command
                for command in commands
            )
        )
        self.assertTrue(
            any(
                ".h200-capture-previous-1111111111111111" in command
                and "rm -rf --" in command
                for command in commands
            )
        )
        self.assertTrue(
            any(
                'sudo rm -f -- "$state"' in command
                and 'sudo rmdir -- "$run"' in command
                for command in commands
            )
        )

        work_setup = next(command for command in commands if "work-ready" in command)
        refresh = next(command for command in commands if "refresh-fixture" in command)
        capture = next(command for command in commands if "capture-h200" in command)
        start = next(command for command in commands if "prime_parity_trace_dir=" in command)
        self.assertLess(commands.index(work_setup), commands.index(refresh))
        self.assertIn("install -d -o 1000 -g 1000 -m 0750", work_setup)
        self.assertIn("1000:1000:750", work_setup)
        work = (
            "/data/glm52-rl/qualifications/glm52-policy-parity/"
            ".diagnostic-work-1111111111111111"
        )
        self.assertIn(f"{work}/fixture", refresh)
        self.assertIn(f"{work}/diagnostic.npz", refresh)
        self.assertIn(f"{work}/capture", capture)
        self.assertIn(f"{work}/h200-router-trace", start)

    def test_container_write_inventory_refuses_unprepared_path(self):
        paths = training_ops._parity_container_write_paths("1" * 16)
        self.assertEqual(
            set(paths),
            {
                "work",
                "fixture",
                "diagnostic",
                "capture",
                "router_trace",
                "bootstrap_fifo",
                "lora_cache",
                "vllm_cache",
            },
        )
        paths["capture"] = "/tmp/unprepared"
        with self.assertRaisesRegex(training_ops.Refused, "inventory"):
            training_ops._prepare_parity_container_work("h200a", paths, ())

    def test_container_work_preparation_creates_exact_owned_empty_dirs(self):
        exact_work = (
            "/data/glm52-rl/qualifications/glm52-policy-parity/"
            ".diagnostic-work-1111111111111111"
        )
        paths = training_ops._parity_container_write_paths("1" * 16)
        with tempfile.TemporaryDirectory() as temporary:
            local = Path(temporary) / ".diagnostic-work-1111111111111111"

            def run_prepare(_host, command, _label, _sensitive, _timeout):
                local_command = command.replace(exact_work, str(local)).replace(
                    "sudo install", "install"
                )
                local_command = local_command.replace("sudo -u '#1000' ", "")
                completed = subprocess.run(
                    ["bash", "-c", local_command],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if completed.returncode:
                    raise RuntimeError(completed.stderr)
                return completed.stdout

            with patch.object(training_ops, "_sensitive_ssh", side_effect=run_prepare):
                training_ops._prepare_parity_container_work("h200a", paths, ())
            status = local.stat()
            self.assertEqual(status.st_uid, 1000)
            self.assertEqual(status.st_gid, 1000)
            self.assertEqual(status.st_mode & 0o777, 0o750)
            self.assertEqual([path.name for path in local.iterdir()], ["h200-router-trace"])
            self.assertEqual(list((local / "h200-router-trace").iterdir()), [])

    def test_container_refresh_outputs_are_verified_before_publish(self):
        exact_work = (
            "/data/glm52-rl/qualifications/glm52-policy-parity/"
            ".diagnostic-work-1111111111111111"
        )
        paths = training_ops._parity_container_write_paths("1" * 16)
        with tempfile.TemporaryDirectory() as temporary:
            local = Path(temporary) / ".diagnostic-work-1111111111111111"
            local.mkdir(mode=0o750)
            os.chmod(local, 0o750)
            trace = local / "h200-router-trace"
            trace.mkdir(mode=0o750)
            os.chmod(trace, 0o750)
            fixture = local / "fixture"
            fixture.mkdir()
            os.chmod(fixture, 0o755)
            manifest = fixture / "manifest.json"
            manifest.write_text("{}")
            os.chmod(manifest, 0o644)
            diagnostic = local / "diagnostic.npz"
            diagnostic.write_bytes(b"diagnostic")
            os.chmod(diagnostic, 0o644)

            def run_verify(_host, command, _label, _sensitive, _timeout):
                local_command = command.replace(exact_work, str(local)).replace(
                    "sudo -u '#1000' ", ""
                )
                completed = subprocess.run(
                    ["bash", "-c", local_command],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if completed.returncode:
                    raise RuntimeError(completed.stderr)
                return completed.stdout

            with patch.object(training_ops, "_sensitive_ssh", side_effect=run_verify):
                training_ops._verify_parity_container_outputs(
                    "h200a", paths, "refresh", ()
                )

            os.chmod(local / "diagnostic.npz", 0o666)
            with patch.object(training_ops, "_sensitive_ssh", side_effect=run_verify):
                with self.assertRaises(RuntimeError):
                    training_ops._verify_parity_container_outputs(
                        "h200a", paths, "refresh", ()
                    )

    def test_container_permission_failure_stops_before_container_run(self):
        commands = []
        secret, ssh_call, secret_call, _ = self._operation_ssh(
            commands, fail_work_prep=True
        )
        with self.assertRaisesRegex(RuntimeError, "permission denied"):
            self._run_operation(ssh_call, secret_call, secret)
        self.assertTrue(any("work-ready" in command for command in commands))
        self.assertFalse(any("refresh-fixture" in command for command in commands))
        self.assertFalse(any("capture-h200" in command for command in commands))
        self.assertFalse(
            any(
                "start_inference.sh" in command and "prime_deployment=" in command
                for command in commands
            )
        )
        self.assertTrue(
            any(
                ".diagnostic-work-1111111111111111" in command
                and "sudo rm -rf --" in command
                for command in commands
            )
        )

    def test_container_work_cleanup_uses_sudo_after_refresh_failure(self):
        commands = []
        secret, ssh_call, secret_call, _ = self._operation_ssh(
            commands, fail_refresh=True
        )
        with self.assertRaisesRegex(RuntimeError, "refresh failed"):
            self._run_operation(ssh_call, secret_call, secret)
        cleanup = next(
            command
            for command in commands
            if "sudo rm -rf --" in command
            and ".diagnostic-work-1111111111111111" in command
            and "realpath -e" in command
        )
        self.assertIn("test ! -L", cleanup)
        self.assertIn("sudo rm -rf --", cleanup)

    def test_kept_engine_retains_its_exact_router_trace_work(self):
        commands = []
        secret, ssh_call, secret_call, _ = self._operation_ssh(commands)
        result = self._run_operation(
            ssh_call, secret_call, secret, keep_engine=True
        )
        work = (
            "/data/glm52-rl/qualifications/glm52-policy-parity/"
            ".diagnostic-work-1111111111111111"
        )
        self.assertTrue(result["engine_kept"])
        self.assertEqual(result["container_work"], work)
        cleanup = next(
            command
            for command in commands
            if "bootstrap.fifo" in command and ".diagnostic-source-" in command
        )
        self.assertIn(f'realpath -e {work}', cleanup)
        self.assertNotIn(f"sudo rm -rf -- {work}", cleanup)

    def test_diagnostic_operation_reconciles_lost_or_invalid_publish_ack(self):
        cases = (
            {"lose_fixture_ack": True},
            {"invalid_fixture_ack": True},
            {"lose_capture_ack": True},
            {"invalid_capture_ack": True},
        )
        for options in cases:
            with self.subTest(options=options):
                commands = []
                secret, ssh_call, secret_call, _ = self._operation_ssh(commands, **options)
                result = self._run_operation(ssh_call, secret_call, secret)
                self.assertEqual(result["capture"], training_ops.glm_parity_paths["h200_capture"])
                self.assertEqual(ssh_call.transaction["state"]["phase"], "committed")

    def test_pair_rollback_second_move_failure_never_leaves_mixed_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_live = root / "fixture"
            fixture_previous = root / "fixture-previous"
            fixture_failed = root / "fixture-failed"
            capture_live = root / "capture"
            capture_previous = root / "capture-previous"
            capture_failed = root / "capture-failed"
            for path, value in (
                (fixture_live, "new-fixture"),
                (fixture_previous, "old-fixture"),
                (capture_live, "new-capture"),
                (capture_previous, "old-capture"),
            ):
                path.mkdir()
                (path / "manifest.json").write_text(value)
            tools = root / "tools"
            tools.mkdir()
            move_count = root / "move-count"
            fake_move = tools / "mv"
            fake_move.write_text(
                "#!/bin/sh\n"
                "count=0\n"
                'if test -f "$MV_COUNT_FILE"; then count=$(cat "$MV_COUNT_FILE"); fi\n'
                "count=$((count+1))\n"
                'printf \'%s\\n\' "$count" > "$MV_COUNT_FILE"\n'
                'if test "$count" = 2; then exit 72; fi\n'
                'exec /bin/mv "$@"\n'
            )
            fake_move.chmod(0o755)
            command = training_ops._parity_pair_rollback_command(
                [
                    {
                        "live": str(fixture_live),
                        "previous": str(fixture_previous),
                        "failed": str(fixture_failed),
                        "replaced": True,
                    },
                    {
                        "live": str(capture_live),
                        "previous": str(capture_previous),
                        "failed": str(capture_failed),
                        "replaced": True,
                    },
                ],
                "true",
            )
            environment = os.environ.copy()
            environment["PATH"] = f"{tools}:/usr/bin:/bin"
            environment["MV_COUNT_FILE"] = str(move_count)
            completed = subprocess.run(
                ["bash", "-c", command],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 72)
            self.assertEqual((fixture_live / "manifest.json").read_text(), "new-fixture")
            self.assertEqual((capture_live / "manifest.json").read_text(), "new-capture")
            self.assertEqual((fixture_previous / "manifest.json").read_text(), "old-fixture")
            self.assertEqual((capture_previous / "manifest.json").read_text(), "old-capture")
            self.assertFalse(fixture_failed.exists())
            self.assertFalse(capture_failed.exists())

    def test_transaction_snapshot_command_reports_all_remote_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for kind in ("fixture", "capture"):
                paths[kind] = {}
                for key in ("live", "previous", "candidate", "failed"):
                    path = root / f"{kind}-{key}"
                    paths[kind][key] = str(path)
                    if key in {"live", "previous"}:
                        path.mkdir()
                        (path / "manifest.json").write_text(f"{kind}-{key}")
            state = {
                "schema_version": 1,
                "fixture": {"paths": paths["fixture"]},
                "capture": {"paths": paths["capture"]},
            }
            transaction = root / "transaction.json"
            transaction.write_text(json.dumps(state))

            def run_snapshot(_host, command, _label, _sensitive, _timeout):
                arguments = shlex.split(command)
                self.assertEqual(arguments[:2], ["sudo", "python3"])
                arguments[-1] = str(transaction)
                completed = subprocess.run(arguments[1:], text=True, capture_output=True, check=False)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                return completed.stdout

            exact_path = (
                "/data/glm52-rl/qualifications/glm52-policy-parity/" ".diagnostic-run-1111111111111111/transaction.json"
            )
            with patch.object(training_ops, "_sensitive_ssh", side_effect=run_snapshot), patch.object(
                training_ops, "_validate_parity_transaction_state"
            ):
                snapshot = training_ops._parity_transaction_snapshot("h200a", exact_path, ())
            self.assertEqual(snapshot["state"], state)
            self.assertTrue(snapshot["observed"]["fixture.live"]["exists"])
            self.assertTrue(snapshot["observed"]["capture.previous"]["exists"])
            self.assertFalse(snapshot["observed"]["fixture.failed"]["exists"])

    def test_transaction_state_rejects_paths_outside_exact_nonce(self):
        path = (
            "/data/glm52-rl/qualifications/glm52-policy-parity/"
            ".diagnostic-run-1111111111111111/transaction.json"
        )
        root = "/data/glm52-rl/qualifications/glm52-policy-parity"
        writes = training_ops._parity_container_write_paths("1" * 16)
        state = {
            "schema_version": 1,
            "nonce": "1111111111111111",
            "phase": "fixture_ready",
            "fixture": {
                "action": None,
                "new_manifest_sha256": "1" * 64,
                "old_manifest_sha256": "2" * 64,
                "paths": {
                    "live": training_ops.glm_parity_paths["fixture"],
                    "previous": f"{root}/.fixture-previous-1111111111111111",
                    "candidate": writes["fixture"],
                    "failed": f"{root}/.fixture-failed-1111111111111111",
                },
            },
            "capture": {
                "action": None,
                "new_manifest_sha256": None,
                "old_manifest_sha256": None,
                "paths": {
                    "live": training_ops.glm_parity_paths["h200_capture"],
                    "previous": f"{root}/.h200-capture-previous-1111111111111111",
                    "candidate": writes["capture"],
                    "failed": f"{root}/.h200-capture-failed-1111111111111111",
                },
            },
        }
        training_ops._validate_parity_transaction_state(path, state)
        state["capture"]["paths"]["failed"] = "/tmp/outside"
        with self.assertRaisesRegex(RuntimeError, "capture paths"):
            training_ops._validate_parity_transaction_state(path, state)

    def test_rolled_back_transaction_cleanup_removes_only_regular_state_and_empty_dir(self):
        exact_directory = (
            "/data/glm52-rl/qualifications/glm52-policy-parity/"
            ".diagnostic-run-1111111111111111"
        )
        exact_state = f"{exact_directory}/transaction.json"
        with tempfile.TemporaryDirectory() as temporary:
            local_directory = Path(temporary) / ".diagnostic-run-1111111111111111"
            local_directory.mkdir()
            local_state = local_directory / "transaction.json"
            local_state.write_text(json.dumps({"phase": "rolled_back"}))

            def run_cleanup(_host, command, _label, _sensitive, _timeout):
                local_command = command.replace(exact_state, str(local_state)).replace(
                    exact_directory, str(local_directory)
                )
                local_command = local_command.replace("sudo rm", "rm").replace(
                    "sudo rmdir", "rmdir"
                )
                completed = subprocess.run(
                    ["bash", "-c", local_command],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if completed.returncode:
                    raise RuntimeError(completed.stderr)
                return completed.stdout

            with patch.object(training_ops, "_sensitive_ssh", side_effect=run_cleanup):
                training_ops._remove_completed_parity_transaction(
                    "h200a", exact_state, exact_directory, ()
                )
            self.assertFalse(local_state.exists())
            self.assertFalse(local_directory.exists())

    def test_completed_transaction_cleanup_retains_state_when_dir_is_not_empty(self):
        exact_directory = (
            "/data/glm52-rl/qualifications/glm52-policy-parity/"
            ".diagnostic-run-1111111111111111"
        )
        exact_state = f"{exact_directory}/transaction.json"
        with tempfile.TemporaryDirectory() as temporary:
            local_directory = Path(temporary) / ".diagnostic-run-1111111111111111"
            local_directory.mkdir()
            local_state = local_directory / "transaction.json"
            local_state.write_text("state")
            (local_directory / "recovery").write_text("retain")

            def run_cleanup(_host, command, _label, _sensitive, _timeout):
                local_command = command.replace(exact_state, str(local_state)).replace(
                    exact_directory, str(local_directory)
                )
                local_command = local_command.replace("sudo rm", "rm").replace(
                    "sudo rmdir", "rmdir"
                )
                completed = subprocess.run(
                    ["bash", "-c", local_command],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if completed.returncode:
                    raise RuntimeError("not empty")
                return completed.stdout

            with patch.object(training_ops, "_sensitive_ssh", side_effect=run_cleanup):
                with self.assertRaisesRegex(RuntimeError, "not empty"):
                    training_ops._remove_completed_parity_transaction(
                        "h200a", exact_state, exact_directory, ()
                    )
            self.assertTrue(local_state.is_file())
            self.assertTrue(local_directory.is_dir())


    def test_diagnostic_operation_cleans_exact_container_after_start_failure(self):
        commands = []
        secret, ssh_call, secret_call, _ = self._operation_ssh(commands, fail_start=True)
        with self.assertRaises(RuntimeError) as caught:
            self._run_operation(ssh_call, secret_call, secret, keep_engine=True)
        self.assertNotIn(secret, str(caught.exception))
        self.assertTrue(
            any(
                'name=glm52-optims-h200' in command
                and 'sudo docker rm -f "$name"' in command
                for command in commands
            )
        )

    def test_diagnostic_operation_reports_cleanup_failure(self):
        commands = []
        secret, ssh_call, secret_call, _ = self._operation_ssh(
            commands, fail_cleanup=True
        )
        with self.assertRaisesRegex(RuntimeError, "artifact cleanup failed"):
            self._run_operation(ssh_call, secret_call, secret)
        rollback = next(
            command
            for command in commands
            if ".fixture-failed-1111111111111111" in command
            and ".h200-capture-failed-1111111111111111" in command
        )
        self.assertIn(".fixture-previous-1111111111111111", rollback)
        self.assertIn(".h200-capture-previous-1111111111111111", rollback)

    def test_committed_fixture_backup_cleanup_failure_is_only_a_warning(self):
        commands = []
        secret, ssh_call, secret_call, _ = self._operation_ssh(
            commands, fail_fixture_backup=True
        )
        result = self._run_operation(ssh_call, secret_call, secret)
        self.assertEqual(len(result["cleanup_warnings"]), 2)
        self.assertIn("fixture backup cleanup failed", result["cleanup_warnings"][0])
        self.assertIn("parity transaction state retained at", result["cleanup_warnings"][1])
        self.assertFalse(any('sudo rm -f -- "$state"' in command for command in commands))
        self.assertTrue(
            any(
                ".h200-capture-previous-1111111111111111" in command
                and "rm -rf --" in command
                for command in commands
            )
        )

    def test_committed_capture_backup_cleanup_failure_is_only_a_warning(self):
        commands = []
        secret, ssh_call, secret_call, _ = self._operation_ssh(
            commands, fail_capture_backup=True
        )
        result = self._run_operation(ssh_call, secret_call, secret)
        self.assertEqual(len(result["cleanup_warnings"]), 2)
        self.assertIn("capture backup cleanup failed", result["cleanup_warnings"][0])
        self.assertIn("parity transaction state retained at", result["cleanup_warnings"][1])
        self.assertFalse(any('sudo rm -f -- "$state"' in command for command in commands))

    def test_incomplete_rollback_retains_exact_transaction_state(self):
        commands = []
        secret, ssh_call, secret_call, _ = self._operation_ssh(
            commands, fail_capture=True, fail_rollback=True
        )
        with self.assertRaises(RuntimeError) as caught:
            self._run_operation(ssh_call, secret_call, secret)
        notes = "\n".join(getattr(caught.exception, "__notes__", ()))
        self.assertIn("rollback did not complete", notes)
        self.assertIn(
            "/data/glm52-rl/qualifications/glm52-policy-parity/"
            ".diagnostic-run-1111111111111111",
            notes,
        )
        self.assertFalse(any('sudo rm -f -- "$state"' in command for command in commands))

    def test_diagnostic_operation_cleans_fifo_and_container_on_interrupt(self):
        commands = []
        secret, ssh_call, secret_call, _ = self._operation_ssh(
            commands, interrupt_fifo=True
        )
        with self.assertRaises(KeyboardInterrupt):
            self._run_operation(ssh_call, secret_call, secret, keep_engine=True)
        self.assertTrue(
            any('sudo docker rm -f "$name"' in command for command in commands)
        )
        self.assertTrue(
            any(
                "sudo rm -f --" in command
                and "bootstrap.fifo" in command
                and "sudo rmdir --" in command
                for command in commands
            )
        )

    def test_cleanup_removes_only_the_exact_exited_parity_container(self):
        inspection = "\n".join(
            (
                json.dumps(training_ops.inference_image),
                json.dumps(
                    [
                        "@",
                        training_ops.glm_parity_inference_config,
                        "--vllm-extra",
                        training_ops.glm_parity_vllm_extra,
                    ]
                ),
                json.dumps("no"),
            )
        )
        with patch.object(
            training_ops,
            "_isolated_container_status",
            side_effect=["absent", "absent", "exited"],
        ) as status, patch.object(
            training_ops, "_sensitive_ssh", return_value=inspection
        ), patch.object(
            training_ops, "_remote_ok", return_value="removed\n"
        ) as remove:
            result = training_ops.cleanup_glm52_policy_parity_engine()
        self.assertEqual(result, "removed")
        self.assertEqual(
            [call.args[1] for call in status.call_args_list],
            [
                training_ops.glm_inference_container + "-0",
                training_ops.glm_inference_container + "-1",
                training_ops.glm_inference_container,
            ],
        )
        command = remove.call_args.args[1]
        self.assertIn("name=glm52-optims-h200", command)
        self.assertNotIn(training_ops.protected_inference[0], command)
        self.assertNotIn("rm -f", command)
        syntax = subprocess.run(
            ["bash", "-n"], input=command, text=True, capture_output=True, check=False
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_cleanup_refuses_a_running_parity_container(self):
        with patch.object(
            training_ops,
            "_isolated_container_status",
            side_effect=["absent", "absent", "running"],
        ), patch.object(training_ops, "_remote_ok") as remove, self.assertRaisesRegex(
            training_ops.Refused, "only an exited container"
        ):
            training_ops.cleanup_glm52_policy_parity_engine()
        remove.assert_not_called()


if __name__ == "__main__":
    unittest.main()
