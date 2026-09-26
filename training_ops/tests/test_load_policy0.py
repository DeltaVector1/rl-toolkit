import copy
import io
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from training_ops import (
    _load_policy0_action,
    _policy0_load_engine_state_script,
    _policy0_load_service_state_script,
    _policy0_remote_client_op,
    policy0_remote_load_timeout_seconds,
    policy0_remote_stage_timeout_seconds,
)
from training_ops.runtime.h200.load_policy0 import (
    EXPECTED_LORA,
    Policy0LoadError,
    build_plan,
    execute_policy0_load,
    validate_metric_samples,
)


IDENTITY = {
    "source_url": "http://127.0.0.1:18100/run/step_0",
    "target_sha256": "a" * 64,
    "target_size": 123,
    "config_sha256": "b" * 64,
    "config_size": 456,
}


def plan():
    return build_plan(
        config_dir="configs/example",
        source_run="run",
        source_adapter_dir="/data/glm52-rl/runs/run/run_run/broadcasts/step_0",
        source_config_path="/data/glm52-rl/runs/run/run_run/broadcasts/step_0/adapter_config.json",
        source_url=IDENTITY["source_url"],
        qualification_slot="qualification-policy-0",
        admin_urls=["http://127.0.0.1:8000/v1"],
        origin_urls=["http://127.0.0.1:8000/v1"],
        trainer_hosts=["n76", "n82"],
        orchestrator_host="h200b",
        inference_hosts=["h200b"],
        engine_ports={"h200b": [8000]},
        engine_container="glm52-decensor-h200",
    )


def state(*, active=False, known=True, running=0, waiting=0, stage_status="idle", api_pid=321):
    return {
        "services": {
            "trainers": {
                "n76": {"active": active, "known": known},
                "n82": {"active": active, "known": known},
            },
            "orchestrator": {"active": active, "known": known},
        },
        "engine": {
            "healthy": True,
            "container_id": "container-a",
            "container_started_at": "2026-09-26T00:00:00Z",
            "api_pid": api_pid,
            "api_start_time": "1001",
            "running": running,
            "waiting": waiting,
            "stage_status": stage_status,
            "active_stage": stage_status == "staging",
            "stage_conflict": stage_status == "staging",
            "endpoints": ["http://127.0.0.1:8000"],
            "loaded_receipts": {},
        },
    }


def source():
    return {"stable": True, "identity": IDENTITY.copy(), "lora": {**EXPECTED_LORA, "target_modules": list(EXPECTED_LORA["target_modules"])}}


def receipt(server="instance-a"):
    return {**IDENTITY, "lora_name": "qualification-policy-0", "server_instance": server}


def run_engine_probe(
    script,
    inspect_line,
    *,
    inspect_after="",
    listener_initial="LISTEN users:((\"python\",pid=321,fd=8))",
    listener_after="",
    docker_id="a" * 64,
    listener_exe="/usr/bin/python3.12",
    container_exe="/usr/bin/python3.12",
    listener_exe_identity="8:123",
    container_exe_identity="8:123",
    listener_title="PRIME-RL::Inference",
    listener_comm="PRIME-RL::Infer",
    cgroup_id=None,
    nspid_host=321,
    nspid_container=1,
    nspid_status="",
    start_ticks_after="",
):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        if not nspid_status:
            nspid_status = f"Name:\tfixture\nNSpid:\t{nspid_host}\t{nspid_container}\n"
        (root / "status").write_text(nspid_status)
        (root / "sudo").write_text(
            """#!/bin/sh
set -eu
case "$1" in
  docker)
    count=0
    if [ -f "$DOCKER_COUNT" ]; then IFS= read -r count <"$DOCKER_COUNT"; fi
    printf '1\\n' >"$DOCKER_COUNT"
    if [ "$count" = 0 ] || [ -z "$DOCKER_INSPECT_AFTER" ]; then printf '%s\\n' "$DOCKER_INSPECT"; else printf '%s\\n' "$DOCKER_INSPECT_AFTER"; fi
    ;;
  ss)
    count=0
    if [ -f "$SS_COUNT" ]; then IFS= read -r count <"$SS_COUNT"; fi
    printf '1\\n' >"$SS_COUNT"
    if [ "$count" = 0 ] || [ -z "$DOCKER_LISTENER_AFTER" ]; then printf '%s\\n' "$DOCKER_LISTENER_INITIAL"; else printf '%s\\n' "$DOCKER_LISTENER_AFTER"; fi
    ;;
  cat)
    case "$*" in
      */cmdline) printf '%s\\0' "$LISTENER_TITLE" ;;
      */comm) printf '%s\\n' "$LISTENER_COMM" ;;
      */cgroup) printf '0::/docker/%s\\n' "$CGROUP_ID" ;;
      *) exit 1 ;;
    esac
    ;;
  readlink)
    case "$*" in
      */root/proc/1/exe) printf '%s\\n' "$CONTAINER_EXE" ;;
      *) printf '%s\\n' "$LISTENER_EXE" ;;
    esac
    ;;
  stat)
    case "$*" in
      */root/proc/1/exe) printf '%s\\n' "$CONTAINER_EXE_IDENTITY" ;;
      *) printf '%s\\n' "$LISTENER_EXE_IDENTITY" ;;
    esac
    ;;
  awk)
    case "$*" in
      */status) /usr/bin/awk "$2" "$3" "$4" "$NSPID_STATUS_FILE" ;;
      */stat)
        count=0
        if [ -f "$STAT_COUNT" ]; then IFS= read -r count <"$STAT_COUNT"; fi
        printf '1\\n' >"$STAT_COUNT"
        if [ "$count" != 0 ] && [ -n "$START_TICKS_AFTER" ]; then printf '%s\\n' "$START_TICKS_AFTER"; else printf '12345\\n'; fi
        ;;
      *) exit 1 ;;
    esac
    ;;
  *) exit 1 ;;
esac
"""
        )
        (root / "curl").write_text(
            """#!/bin/sh
set -eu
case "$*" in
  *metrics*) printf '%s\\n' 'vllm:num_requests_running 0' 'vllm:num_requests_waiting 0' ;;
  *staging_status*) printf '%s\\n' '{"status":"idle","loaded_receipts":{}}' ;;
  *) printf '%s\\n' '{}' ;;
esac
"""
        )
        for path in (root / "sudo", root / "curl"):
            path.chmod(0o755)
        environment = os.environ.copy()
        environment["PATH"] = f"{directory}:{environment['PATH']}"
        environment["DOCKER_INSPECT"] = inspect_line
        environment["DOCKER_INSPECT_AFTER"] = inspect_after
        environment["DOCKER_LISTENER_INITIAL"] = listener_initial
        environment["DOCKER_LISTENER_AFTER"] = listener_after
        environment["DOCKER_ID"] = docker_id
        environment["CGROUP_ID"] = cgroup_id or docker_id
        environment["LISTENER_EXE"] = listener_exe
        environment["CONTAINER_EXE"] = container_exe
        environment["LISTENER_EXE_IDENTITY"] = listener_exe_identity
        environment["CONTAINER_EXE_IDENTITY"] = container_exe_identity
        environment["LISTENER_TITLE"] = listener_title
        environment["LISTENER_COMM"] = listener_comm
        environment["NSPID_HOST"] = str(nspid_host)
        environment["NSPID_CONTAINER"] = str(nspid_container)
        environment["NSPID_STATUS_FILE"] = str(root / "status")
        environment["START_TICKS_AFTER"] = start_ticks_after
        environment["DOCKER_COUNT"] = str(root / "docker-count")
        environment["SS_COUNT"] = str(root / "ss-count")
        environment["STAT_COUNT"] = str(root / "stat-count")
        return subprocess.run(
            ["bash", "-ceu", script],
            text=True,
            capture_output=True,
            env=environment,
        )


class Policy0LoadTest(unittest.TestCase):
    def test_success_uses_stage_without_delta_and_prefetched_load(self):
        before = state()
        final = state()
        other = receipt("instance-b")
        other["lora_name"] = "other-slot"
        final["engine"]["loaded_receipts"] = {
            "qualification-policy-0": receipt(),
            "other-slot": other,
        }
        calls = []

        def stage(url, **kwargs):
            calls.append(("stage", url, kwargs))
            return {"adapter_identity": IDENTITY.copy()}

        def load(slot, url, **kwargs):
            calls.append(("load", slot, url, kwargs))
            return [receipt()]

        states = iter([before, copy.deepcopy(before), final])
        result = execute_policy0_load(
            plan(),
            read_state=lambda: next(states),
            read_source=source,
            stage=stage,
            load=load,
        )
        self.assertEqual(result["status"], "loaded")
        self.assertFalse(calls[0][2]["require_delta"])
        self.assertIsNone(calls[0][2]["policy_step"])
        self.assertTrue(calls[1][3]["prefetched"])
        self.assertEqual(calls[1][3]["expected_identity"], IDENTITY)

    def test_active_or_unknown_services_refuse_without_start_calls(self):
        for changed in (
            state(active=True),
            {**state(), "services": {**state()["services"], "orchestrator": {"active": False, "known": False}}},
        ):
            with self.subTest(changed=changed):
                starts = []
                with self.assertRaises(Policy0LoadError):
                    execute_policy0_load(
                        plan(),
                        read_state=lambda changed=changed: changed,
                        read_source=source,
                        stage=lambda *a, **k: starts.append("stage"),
                        load=lambda *a, **k: starts.append("load"),
                    )
                self.assertEqual(starts, [])

    def test_waiting_running_and_stage_conflicts_refuse(self):
        for changed in (state(waiting=1), state(running=1), state(stage_status="staging")):
            with self.subTest(changed=changed):
                with self.assertRaises(Policy0LoadError):
                    execute_policy0_load(
                        plan(),
                        read_state=lambda changed=changed: changed,
                        read_source=source,
                        stage=lambda *a, **k: {"adapter_identity": IDENTITY.copy()},
                        load=lambda *a, **k: [receipt()],
                    )

    def test_source_and_stage_identity_mismatch_refuse(self):
        bad_source = source()
        bad_source["stable"] = False
        with self.assertRaisesRegex(Policy0LoadError, "STABLE"):
            execute_policy0_load(
                plan(), read_state=state, read_source=lambda: bad_source,
                stage=lambda *a, **k: {"adapter_identity": IDENTITY.copy()},
                load=lambda *a, **k: [receipt()],
            )
        changed = IDENTITY.copy()
        changed["target_sha256"] = "c" * 64
        with self.assertRaisesRegex(Policy0LoadError, "staged adapter differs"):
            execute_policy0_load(
                plan(), read_state=state, read_source=source,
                stage=lambda *a, **k: {"adapter_identity": changed},
                load=lambda *a, **k: [receipt()],
            )

    def test_engine_identity_and_final_server_receipt_mismatch_refuse(self):
        changed = state(api_pid=999)
        final = state()
        final["engine"]["loaded_receipts"] = {"qualification-policy-0": receipt("instance-b")}
        changed_states = iter([state(), changed, final])
        with self.assertRaisesRegex(Policy0LoadError, "engine identity changed"):
            execute_policy0_load(
                plan(), read_state=lambda: next(changed_states), read_source=source,
                stage=lambda *a, **k: {"adapter_identity": IDENTITY.copy()},
                load=lambda *a, **k: [receipt()],
            )
        states = iter([state(), state(), final])
        with self.assertRaisesRegex(Policy0LoadError, "final staging receipts"):
            execute_policy0_load(
                plan(), read_state=lambda: next(states), read_source=source,
                stage=lambda *a, **k: {"adapter_identity": IDENTITY.copy()},
                load=lambda *a, **k: [receipt()],
            )

    def test_empty_endpoint_plan_refuses(self):
        with self.assertRaisesRegex(Policy0LoadError, "admin endpoints"):
            build_plan(
                config_dir="configs/example", source_run="run",
                source_adapter_dir="/data/run/step_0",
                source_config_path="/data/run/step_0/adapter_config.json",
                source_url=IDENTITY["source_url"], qualification_slot="slot",
                admin_urls=[], origin_urls=["http://127.0.0.1:8000/v1"],
                trainer_hosts=["n76"], orchestrator_host="h200b",
                inference_hosts=["h200b"], engine_ports={"h200b": [8000]},
                engine_container="engine",
            )

    def test_mixed_metric_samples_refuse_even_if_last_sample_is_zero(self):
        with self.assertRaisesRegex(Policy0LoadError, "running samples"):
            validate_metric_samples({"running": [1, 0], "waiting": [0, 0]})
        validate_metric_samples({"running": [0, 0], "waiting": [0]})

    def test_occupied_target_refuses_before_stage(self):
        occupied = state()
        occupied["engine"]["loaded_receipts"] = {
            "qualification-policy-0": receipt()
        }
        calls = []
        with self.assertRaisesRegex(Policy0LoadError, "already occupied"):
            execute_policy0_load(
                plan(), read_state=lambda: occupied, read_source=source,
                stage=lambda *a, **k: calls.append("stage"),
                load=lambda *a, **k: calls.append("load"),
            )
        self.assertEqual(calls, [])

    def test_plan_action_makes_no_live_call(self):
        args = [
            "configs/trinity-decensor-experimental-fixed", "plan",
            "--source-run", "trinity-decensor-experimental-fixed",
            "--source-adapter-dir", "/data/glm52-rl/runs/trinity-decensor-experimental-fixed/run_trinity-decensor-experimental-fixed/broadcasts/step_0",
            "--qualification-slot", "trinity-decensor-experimental-fixed-r32-node-local",
        ]
        with patch("training_ops._ssh") as ssh, redirect_stdout(io.StringIO()) as output:
            result = _load_policy0_action(args)
        ssh.assert_not_called()
        self.assertEqual(
            result["source_url"],
            "http://127.0.0.1:18100/trinity-decensor-experimental-fixed/run_trinity-decensor-experimental-fixed/broadcasts/step_0",
        )
        self.assertEqual(result["source_adapter_host"], "n82")
        self.assertIn("exact H200 container ID", output.getvalue())

    def test_plan_refuses_source_outside_configured_source_dir(self):
        with self.assertRaisesRegex(RuntimeError, "outside weight_broadcast.source_dir"):
            _load_policy0_action([
                "configs/trinity-decensor-experimental-fixed", "plan",
                "--source-run", "trinity-decensor-experimental-fixed",
                "--source-adapter-dir", "/data/glm52-rl/other/trinity-decensor-experimental-fixed/broadcasts/step_0",
                "--qualification-slot", "trinity-decensor-experimental-fixed-r32-node-local",
            ])

    def test_live_scripts_are_read_only_and_pin_engine_identity(self):
        service = _policy0_load_service_state_script("glm52-decensor-trainer")
        engine = _policy0_load_engine_state_script("glm52-decensor-h200", [8000])
        self.assertIn("docker inspect", service)
        self.assertNotIn("docker start", service)
        self.assertIn(".Id", engine)
        self.assertIn(".State.StartedAt", engine)
        self.assertIn("ss -H -ltnp", engine)
        self.assertIn("/proc/$api_pid/cmdline", engine)
        self.assertIn("num_requests_(running|waiting)", engine)
        self.assertIn("staging_status", engine)
        self.assertNotIn("docker restart", engine)

    def test_engine_probe_uses_literal_pipe_delimiters_and_parses_valid_identity(self):
        engine = _policy0_load_engine_state_script("glm52-decensor-h200", [8000])
        self.assertIn("{{.Name}}|{{.Id}}|{{.State.StartedAt}}|{{.State.Running}}|{{.State.Pid}}", engine)
        self.assertIn("IFS='|' read -r name container_id container_started_at running container_pid", engine)
        self.assertIn("/proc/$api_pid/comm", engine)
        self.assertNotIn(r"{{.Name}}\t{{.Id}}", engine)
        result = run_engine_probe(
            engine,
            f"/glm52-decensor-h200|{'a' * 64}|2026-09-26T00:00:00Z|true|321",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        parsed = json.loads(result.stdout)
        self.assertEqual(parsed["container_id"], "a" * 64)
        self.assertEqual(parsed["container_started_at"], "2026-09-26T00:00:00Z")
        self.assertEqual(parsed["api_pid"], 321)

    def test_engine_probe_rejects_malformed_or_mismatched_identity(self):
        engine = _policy0_load_engine_state_script("glm52-decensor-h200", [8000])
        cases = (
            (r"/glm52-decensor-h200\t" + "a" * 64 + r"\tstarted\ttrue\t321", "identity-mismatch"),
            ("/other-engine|" + "a" * 64 + "|started|true|321", "identity-mismatch"),
            ("/glm52-decensor-h200|" + "a" * 64 + "|started|true|", "identity-mismatch"),
            ("/glm52-decensor-h200|" + "a" * 64 + "|started|true|extra", "identity-mismatch"),
            ("/glm52-decensor-h200|" + "a" * 64 + "|started|true|321\nsecond", "identity-mismatch"),
            ("/glm52-decensor-h200||started|true|321", "identity-mismatch"),
            ("/glm52-decensor-h200|" + "a" * 64 + "||true|321", "identity-mismatch"),
            ("/glm52-decensor-h200|" + "a" * 64 + "|started|true|0", "identity-mismatch"),
            ("/glm52-decensor-h200|" + "a" * 64 + "|started|false|321", "engine-stopped"),
        )
        for inspect_line, error in cases:
            with self.subTest(inspect_line=inspect_line):
                result = run_engine_probe(engine, inspect_line)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)

    def test_engine_probe_rejects_listener_identity_mismatches(self):
        engine = _policy0_load_engine_state_script("glm52-decensor-h200", [8000])
        valid = f"/glm52-decensor-h200|{'a' * 64}|2026-09-26T00:00:00Z|true|321"
        cases = (
            (dict(inspect_line=valid.replace("|321", "|322")), "listener PID differs"),
            (dict(inspect_line=valid, listener_title="PRIME-RL::Infer"), "command title"),
            (dict(inspect_line=valid, listener_comm="PRIME-RL::Inference"), "comm"),
            (dict(inspect_line=valid, listener_exe="/usr/bin/python3.11"), "executable is not"),
            (dict(inspect_line=valid, cgroup_id="b" * 64), "cgroup"),
            (dict(inspect_line=valid, nspid_host=999), "NSpid"),
            (dict(inspect_line=valid, container_exe_identity="8:999"), "executable identity"),
        )
        for kwargs, error in cases:
            with self.subTest(error=error):
                result = run_engine_probe(engine, **kwargs)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)

    def test_engine_probe_parses_exactly_one_nspid_row_with_real_awk(self):
        engine = _policy0_load_engine_state_script("glm52-decensor-h200", [8000])
        valid = f"/glm52-decensor-h200|{'a' * 64}|2026-09-26T00:00:00Z|true|321"
        invalid_statuses = (
            "Name:\tfixture\nNSpid:\t321\t1\t2\n",
            "Name:\tfixture\nNSpid:\t321\t1\nNSpid:\t321\t1\n",
            "Name:\tfixture\n",
            "Name:\tfixture\nNSpid:\t321x\t1\n",
            "Name:\tfixture\nNSpid:\t321\t01\n",
        )
        for status in invalid_statuses:
            with self.subTest(status=status):
                result = run_engine_probe(engine, valid, nspid_status=status)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("listener NSpid", result.stderr)

    def test_engine_probe_rejects_multiple_or_changed_listeners(self):
        engine = _policy0_load_engine_state_script("glm52-decensor-h200", [8000])
        valid = f"/glm52-decensor-h200|{'a' * 64}|2026-09-26T00:00:00Z|true|321"
        result = run_engine_probe(
            engine,
            valid,
            listener_initial='LISTEN users:(("python",pid=321,fd=8),("python",pid=322,fd=9))',
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not unique", result.stderr)
        result = run_engine_probe(
            engine,
            valid,
            inspect_after=f"/glm52-decensor-h200|{'b' * 64}|2026-09-26T00:00:00Z|true|321",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("engine identity changed after health checks", result.stderr)
        result = run_engine_probe(engine, valid, start_ticks_after="67890")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("listener start identity changed", result.stderr)

    def test_remote_client_call_is_h200_local_and_controller_has_no_client_import(self):
        response = '{"ok":true,"result":{"adapter_identity":{}}}'
        with patch("training_ops._ssh", return_value=(0, response)) as ssh:
            result = _policy0_remote_client_op(plan(), "stage")
        self.assertEqual(result["adapter_identity"], {})
        script = ssh.call_args.args[1]
        self.assertIn("docker exec -i", script)
        self.assertIn("/app/.venv/bin/python", script)
        self.assertIn("127.0.0.1:8000", script)
        self.assertNotIn("from prime_rl", inspect.getsource(__import__("training_ops")._policy0_load_apply))
        self.assertEqual(ssh.call_args.kwargs["timeout"], policy0_remote_stage_timeout_seconds)

    def test_remote_client_stage_timeout_covers_server_stage_budget(self):
        response = '{"ok":true,"result":{"adapter_identity":{}}}'
        with patch("training_ops._ssh", return_value=(0, response)) as ssh:
            _policy0_remote_client_op(plan(), "stage")
        self.assertGreater(policy0_remote_stage_timeout_seconds, 900)
        self.assertEqual(ssh.call_args.kwargs["timeout"], 1860)
        with patch("training_ops._ssh", return_value=(0, response)) as ssh:
            _policy0_remote_client_op(plan(), "load", IDENTITY)
        self.assertEqual(ssh.call_args.kwargs["timeout"], policy0_remote_load_timeout_seconds)

    def test_remote_client_error_category_is_preserved(self):
        with patch("training_ops._ssh", return_value=(1, "{\"ok\":false,\"category\":\"identity\",\"error\":\"receipt mismatch\"}")):
            with self.assertRaisesRegex(RuntimeError, r"\[identity\].*receipt mismatch"):
                _policy0_remote_client_op(plan(), "load", IDENTITY)


if __name__ == "__main__":
    unittest.main()
