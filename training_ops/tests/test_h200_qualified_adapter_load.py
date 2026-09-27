import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import training_ops as training_ops_module
from training_ops import Refused, _qualified_adapter_load_action, actions
from training_ops.runtime.h200.qualified_adapter_load import (
    QualifiedAdapterLoadError,
    build_publish_script,
    build_publication_cleanup_script,
    build_result_receipt_script,
    build_rollback_script,
    derive_plan,
    parse_delta_receipt,
    parse_result_receipt,
    parse_publish_result,
)
from training_ops import _qualified_receipt_from_output, _qualified_adapter_receipt_script


def receipt():
    model = {"sha256": "a" * 64, "size": 100}
    config = {"sha256": "b" * 64, "size": 200}
    manifest = {"sha256": "c" * 64, "size": 900}
    stable = {"sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "size": 0}
    artifacts = {
        "adapter_model.safetensors": model,
        "adapter_config.json": config,
        "adapter_manifest.json": manifest,
        "adapter_model.safetensors.sha256": {"sha256": "d" * 64, "size": 65},
        "STABLE": stable,
    }
    return {
        "schema_version": 1,
        "kind": "h200_step1_adapter_delta",
        "run_id": "step1-run",
        "policy0_adapter": {"sha256": "9" * 64, "size": 110},
        "policy0_config": {"sha256": "8" * 64, "size": 200},
        "step1_adapter": model,
        "step1_config": config,
        "step1_manifest_sha256": manifest["sha256"],
        "trainer_config": {"sha256": "7" * 64, "size": 321},
        "identities_before": {"policy0_adapter": {"sha256": "9" * 64, "size": 110}, "policy0_config": {"sha256": "8" * 64, "size": 200}, "step1_adapter": model, "step1_config": config, "step1_manifest": manifest, "artifacts": artifacts},
        "identities_after": {"policy0_adapter": {"sha256": "9" * 64, "size": 110}, "policy0_config": {"sha256": "8" * 64, "size": 200}, "step1_adapter": model, "step1_config": config, "step1_manifest": manifest, "artifacts": artifacts},
        "expected_identity": {
            "run_id": "step1-run", "reference_sha256": "f" * 64,
            "trainer_config_sha256": "7" * 64, "trainer_config_size": 321,
            "code_sha256": "6" * 64, "code_size": 654,
            "loss_sha256": "5" * 64, "loss_size": 16446, "loss_mode": "644",
        },
        "trainer_evidence": {
            "run_id": "step1-run", "completed_optimizer_steps": 1,
            "learning_rate": 8e-6, "loss": 1.25, "grad_norm": 0.5,
            "forward_backward_seconds": 10.0, "export_seconds": 2.5,
            "export_completed": True, "initial_adapter_sha256": "9" * 64,
            "initial_adapter_size": 110, "final_adapter_sha256": "a" * 64,
            "final_adapter_size": 100, "reference_sha256": "f" * 64,
            "trainer_config_sha256": "7" * 64, "trainer_config_size": 321,
            "code_sha256": "6" * 64, "code_size": 654,
            "loss_identity": {"sha256": "5" * 64, "size": 16446, "mode": "644"},
        },
        "changed_element_count": 1, "changed_tensor_count": 1,
        "max_abs_delta": 0.5, "l2_delta": 0.5,
    }


def result_receipt():
    source_url = "http://127.0.0.1:18100/qualified-adapters/" + "c" * 64 + "/step_1"
    source_dir = "/data/glm52-rl/runs/qualified-adapters/" + "c" * 64 + "/step_1"
    artifacts = receipt()["identities_after"]["artifacts"]
    adapter = {"source_url": source_url, "target_sha256": "a" * 64, "target_size": 100, "config_sha256": "b" * 64, "config_size": 200}
    return {
        "schema_version": 1,
        "kind": "h200_qualified_adapter_load",
        "status": "loaded",
        "qualification_slot": "trinity-h200-post-update-qualification-slot1",
        "source_url": source_url,
        "source_adapter_dir": source_dir,
        "manifest_sha256": "c" * 64,
        "receipt_sha256": "f" * 64,
        "adapter_identity": adapter,
        "published": {"status": "published", "source_url": source_url, "source_adapter_dir": source_dir, "manifest_sha256": "c" * 64, "owner_token": "8:99:123:456", "files": artifacts},
        "stage_result": {"status": "ok", "adapter_identity": adapter},
        "load_result": [{"server_instance": "instance-a"}],
        "server_instances": ["instance-a"],
        "engine_identity_before": {"container_id": "container-a", "container_started_at": "2026-09-27T00:00:00Z", "api_pid": 321, "api_start_time": "1001"},
        "engine_identity_after": {"container_id": "container-a", "container_started_at": "2026-09-27T00:00:00Z", "api_pid": 321, "api_start_time": "1001"},
        "phase_seconds": {"preflight_seconds": 1.0, "publish_seconds": 2.0, "stage_seconds": 3.0, "load_seconds": 4.0},
    }


class QualifiedAdapterLoadTest(unittest.TestCase):
    def apply_plan(self):
        plan = derive_plan(
            "configs/trinity-h200-post-update-qualification", receipt(), "f" * 64,
            qualification_root="/data/q", runs_root="/data/runs", trainer_head="n82",
            inference_host="h200b", inference_port=8000, direct_route_port=18100,
        )
        plan.update(
            trainer_hosts=["n76", "n77", "n78", "n80", "n82", "n83", "n84", "n85"],
            orchestrator_host="h200b", inference_hosts=["h200b"],
            engine_ports={"h200b": [8000]},
            source_config_path=plan["source_adapter_dir"] + "/adapter_config.json",
        )
        return plan

    def live_state(self, loaded):
        services = {"trainers": {host: {"known": True, "active": False} for host in self.apply_plan()["trainer_hosts"]}, "orchestrator": {"known": True, "active": False}}
        return {"services": services, "engine": {"healthy": True, "container_id": "container-a", "container_started_at": "2026-09-27T00:00:00Z", "api_pid": 321, "api_start_time": "1001", "running": 0, "waiting": 0, "active_stage": False, "stage_conflict": False, "stage_status": "idle", "endpoints": ["http://127.0.0.1:8000"], "loaded_receipts": loaded}}

    def test_receipt_requires_exact_hash_schema_and_identity_chain(self):
        raw = json.dumps(receipt(), sort_keys=True, separators=(",", ":")).encode() + b"\n"
        expected = hashlib.sha256(raw).hexdigest()
        parsed = parse_delta_receipt(raw, expected)
        self.assertEqual(parsed["kind"], "h200_step1_adapter_delta")
        for wrong in ("0" * 64, expected[:-1] + "0"):
            with self.subTest(wrong=wrong), self.assertRaisesRegex(QualifiedAdapterLoadError, "receipt"):
                parse_delta_receipt(raw, wrong)
        broken = receipt()
        broken["identities_after"]["step1_adapter"] = {"sha256": "9" * 64, "size": 100}
        broken_raw = json.dumps(broken, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        with self.assertRaisesRegex(QualifiedAdapterLoadError, "identity"):
            parse_delta_receipt(broken_raw, hashlib.sha256(broken_raw).hexdigest())

    def test_remote_receipt_framing_preserves_exact_terminal_newline(self):
        raw = json.dumps(receipt(), sort_keys=True, separators=(",", ":")).encode() + b"\n"
        expected = hashlib.sha256(raw).hexdigest()
        output = (
            f"QUALIFIED_RECEIPT_SHA256 sha256={expected} size={len(raw)}\n"
            "QUALIFIED_RECEIPT_BEGIN\n"
            + base64.b64encode(raw).decode()
            + "\nQUALIFIED_RECEIPT_END\n"
        )
        parsed = _qualified_receipt_from_output(output, expected)
        self.assertEqual(parsed["kind"], "h200_step1_adapter_delta")
        self.assertIn("base64 -w0", _qualified_adapter_receipt_script("/data/glm52-rl/qualifications/trinity-h200-post-update-qualification/step1-delta-receipt.json"))

    def test_receipt_rejects_zero_or_nonfinite_delta_and_identity_no_change(self):
        for field, value in (("changed_element_count", 0), ("changed_tensor_count", 0), ("max_abs_delta", 0.0), ("l2_delta", float("nan"))):
            mutated = json.loads(json.dumps(receipt()))
            mutated[field] = value
            payload = json.dumps(mutated, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            with self.subTest(field=field), self.assertRaisesRegex(QualifiedAdapterLoadError, "delta"):
                parse_delta_receipt(payload, hashlib.sha256(payload).hexdigest())
        mutated = json.loads(json.dumps(receipt()))
        mutated["trainer_evidence"]["final_adapter_sha256"] = mutated["trainer_evidence"]["initial_adapter_sha256"]
        mutated["trainer_evidence"]["final_adapter_size"] = mutated["trainer_evidence"]["initial_adapter_size"]
        payload = json.dumps(mutated, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        with self.assertRaisesRegex(QualifiedAdapterLoadError, "initial.*final"):
            parse_delta_receipt(payload, hashlib.sha256(payload).hexdigest())

    def test_plan_binds_canonical_output_step1_route_and_fixed_slot(self):
        plan = derive_plan(
            "configs/trinity-h200-post-update-qualification",
            receipt(),
            "f" * 64,
            qualification_root="/data/glm52-rl/qualifications/trinity-h200-post-update-qualification",
            runs_root="/data/glm52-rl/runs",
            trainer_head="n82",
            inference_host="h200b",
            inference_port=8000,
            direct_route_port=18100,
        )
        self.assertEqual(plan["qualification_slot"], "trinity-h200-post-update-qualification-slot1")
        self.assertTrue(plan["source_adapter_dir"].endswith("/step_1"))
        self.assertEqual(plan["source_url"], "http://127.0.0.1:18100/qualified-adapters/" + "c" * 64 + "/step_1")
        self.assertEqual(plan["source_adapter_dir"], "/data/glm52-rl/runs/qualified-adapters/" + "c" * 64 + "/step_1")
        with self.assertRaisesRegex(QualifiedAdapterLoadError, "canonical"):
            derive_plan("configs/alias", receipt(), "f" * 64, qualification_root="/data/x", runs_root="/data/y", trainer_head="n82", inference_host="h200b", inference_port=8000, direct_route_port=18100)

    def test_publish_script_is_independent_atomic_and_fail_closed(self):
        plan = derive_plan(
            "configs/trinity-h200-post-update-qualification",
            receipt(),
            "f" * 64,
            qualification_root="/data/q",
            runs_root="/data/runs",
            trainer_head="n82",
            inference_host="h200b",
            inference_port=8000,
            direct_route_port=18100,
        )
        script = build_publish_script(plan)
        for text in ("cp --reflink=auto", "renameat2", "RENAME_NOREPLACE", "source_before", "source_after", "adapter_model.safetensors", "adapter_config.json", "STABLE", "18100"):
            self.assertIn(text, script)
        self.assertNotIn("ln --", script)
        self.assertIn("test ! -L", script)
        self.assertIn("existing destination", script)
        for text in ("%y", "%z", "directory entry", "route_manifest", "route_config", "target_step", "mode"):
            self.assertIn(text, script)
        self.assertIn('test ! -L "$runs"', script)

    def test_result_receipt_is_atomic_no_clobber(self):
        plan = derive_plan(
            "configs/trinity-h200-post-update-qualification", receipt(), "f" * 64,
            qualification_root="/data/q", runs_root="/data/runs", trainer_head="n82",
            inference_host="h200b", inference_port=8000, direct_route_port=18100,
        )
        script = build_result_receipt_script(plan, "e30=")
        self.assertIn("renameat2", script)
        self.assertIn("RENAME_NOREPLACE", script)
        self.assertNotIn("mv --", script)

    def test_result_receipt_requires_exact_bytes_and_schema(self):
        result = result_receipt()
        raw = json.dumps(result, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        parsed = parse_result_receipt(raw, hashlib.sha256(raw).hexdigest())
        self.assertEqual(parsed["status"], "loaded")
        with self.assertRaisesRegex(QualifiedAdapterLoadError, "SHA256"):
            parse_result_receipt(raw, "0" * 64)
        extra = dict(result)
        extra["unexpected"] = True
        extra_raw = json.dumps(extra, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        with self.assertRaisesRegex(QualifiedAdapterLoadError, "schema"):
            parse_result_receipt(extra_raw, hashlib.sha256(extra_raw).hexdigest())

    def test_cleanup_requires_session_owner_token(self):
        plan = self.apply_plan()
        script = build_publication_cleanup_script(plan, owner_token="8:99:123:456")
        self.assertIn("owner_token", script)
        self.assertIn("%d:%i:%s:%y:%z", script)

    def test_qualified_load_rejects_failed_or_ready_engine_before_publish(self):
        plan = self.apply_plan()
        for stage_status in ("ready", "failed", "staging"):
            state = self.live_state({})
            state["engine"]["stage_status"] = stage_status
            with self.subTest(stage_status=stage_status), self.assertRaisesRegex(Refused, "idle"):
                training_ops_module._qualified_check_engine(plan, state)

    def test_slot_rollback_cannot_remove_publication_without_owner_proof(self):
        with self.assertRaisesRegex(QualifiedAdapterLoadError, "owner-token"):
            build_rollback_script(self.apply_plan(), remove_published=True)

    def test_publish_result_rejects_mismatched_source_or_slot(self):
        plan = derive_plan(
            "configs/trinity-h200-post-update-qualification", receipt(), "f" * 64,
            qualification_root="/data/q", runs_root="/data/runs", trainer_head="n82",
            inference_host="h200b", inference_port=8000, direct_route_port=18100,
        )
        result = {"status": "published", "source_url": plan["source_url"], "source_adapter_dir": plan["source_adapter_dir"], "manifest_sha256": "c" * 64, "owner_token": "8:99:123:456", "files": receipt()["identities_after"]["artifacts"]}
        parsed = parse_publish_result(json.dumps(result), plan)
        self.assertEqual(parsed["manifest_sha256"], "c" * 64)
        result["source_url"] += "/wrong"
        with self.assertRaisesRegex(QualifiedAdapterLoadError, "source"):
            parse_publish_result(json.dumps(result), plan)

    def test_command_is_separate_from_policy0_and_requires_pinned_receipt(self):
        self.assertIn("load-qualified-adapter", actions)
        with self.assertRaisesRegex(Refused, "delta-receipt-path"):
            _qualified_adapter_load_action(["configs/trinity-h200-post-update-qualification", "plan"])
        with self.assertRaisesRegex(Refused, "canonical"):
            _qualified_adapter_load_action(["configs/alias", "plan", "--delta-receipt-path", "/tmp/r", "--expected-receipt-sha256", "a" * 64])

    def test_apply_uses_stage_then_prefetched_load_and_preserves_existing_slot(self):
        plan = self.apply_plan()
        expected = {
            "source_url": plan["source_url"],
            "target_sha256": "a" * 64,
            "target_size": 100,
            "config_sha256": "b" * 64,
            "config_size": 200,
        }
        old = {"source_url": "http://old/step_0", "target_sha256": "1" * 64, "target_size": 10, "config_sha256": "2" * 64, "config_size": 20, "lora_name": "policy0", "server_instance": "instance-a"}
        new = {**expected, "lora_name": plan["qualification_slot"], "server_instance": "instance-a"}
        states = [self.live_state({"policy0": old}), self.live_state({"policy0": old}), self.live_state({"policy0": old, plan["qualification_slot"]: new})]
        publish = f"QUALIFIED_PUBLISH status=published source_url={plan['source_url']} source_adapter_dir={plan['source_adapter_dir']} manifest_sha256={plan['manifest_sha256']} owner_token=8:99:123:456\n"
        calls = []
        def ssh(host, script, timeout=120):
            calls.append((host, script))
            if "QUALIFIED_PUBLISH" in script:
                return 0, publish
            return 0, "QUALIFIED_RESULT_RECEIPT path=/data/q/qualified-adapter-load-receipt.json sha256=" + "3" * 64 + " size=10\n"
        with patch.object(training_ops_module, "_policy0_load_live_state", side_effect=states), patch.object(training_ops_module, "_policy0_remote_client_op", side_effect=[{"status": "ok", "adapter_identity": expected}, [new]]), patch.object(training_ops_module, "_ssh", side_effect=ssh), patch.object(training_ops_module, "_qualified_result_receipt_from_output", return_value={"status": "loaded"}):
            result = training_ops_module._qualified_adapter_load_apply(plan)
        self.assertEqual(result["status"], "loaded")
        self.assertEqual(result["qualification_slot"], plan["qualification_slot"])
        self.assertIn("QUALIFIED_PUBLISH", calls[0][1])
        self.assertEqual(len(calls), 2)

    def test_apply_rolls_back_session_owned_slot_when_stage_identity_differs(self):
        plan = self.apply_plan()
        states = [self.live_state({})]
        publish = f"QUALIFIED_PUBLISH status=published source_url={plan['source_url']} source_adapter_dir={plan['source_adapter_dir']} manifest_sha256={plan['manifest_sha256']} owner_token=8:99:123:456\n"
        with patch.object(training_ops_module, "_policy0_load_live_state", side_effect=states), patch.object(training_ops_module, "_policy0_remote_client_op", return_value={"status": "ok", "adapter_identity": {}}), patch.object(training_ops_module, "_ssh", return_value=(0, publish)), patch.object(training_ops_module, "_qualified_adapter_rollback", return_value="QUALIFIED_ROLLBACK slot=removed"):
            with self.assertRaisesRegex(Refused, "stage identity"):
                training_ops_module._qualified_adapter_load_apply(plan)


if __name__ == "__main__":
    unittest.main()
