import ast
import base64
import datetime
import gzip
import hashlib
import io
import json
import os
import shlex
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import patch

import tomllib

import training_ops


def _trainer_inspect_record(run_records, command):
    return json.dumps({"run_records": run_records, "command": command})


class EdgeOperationsTest(unittest.TestCase):
    def _run_tunnel_start_script(
        self, script, commands, outcomes, kill_race_pids=()
    ):
        processes = {}
        latest = {}
        killed = []
        printed = []
        next_pid = 3000
        now = 0.0

        class Process:
            def __init__(self, pid, command, outcome):
                self.pid = pid
                self.command = command
                self.outcome = outcome
                self.started_at = now
                self.killed = False

            def poll(self):
                return 1 if self.outcome == "exit" else -15 if self.killed else None

        def popen(command, **_kwargs):
            nonlocal next_pid
            next_pid += 1
            port = int(command[command.index("-L") + 1].split(":")[1])
            outcome = outcomes[port].pop(0)
            process = Process(next_pid, command, outcome)
            processes[next_pid] = process
            latest[port] = process
            return process

        def open_process(path, mode="r", *_args, **_kwargs):
            pid = int(str(path).split("/")[2])
            process = processes[pid]
            if process.killed or process.outcome == "exit":
                raise FileNotFoundError(path)
            command = ["wrong"] if process.outcome == "wrong" else process.command
            return io.BytesIO(b"\0".join(value.encode() for value in command) + b"\0")

        class Socket:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def settimeout(self, _timeout):
                pass

            def connect_ex(self, address):
                process = latest.get(address[1])
                ready = process and (
                    process.outcome == "success"
                    or (
                        isinstance(process.outcome, tuple)
                        and process.outcome[0] == "slow"
                        and now - process.started_at >= process.outcome[1]
                    )
                )
                return 0 if ready and not process.killed else 1

        def killpg(pid, action):
            if pid in kill_race_pids and action == signal.SIGTERM:
                raise ProcessLookupError(pid)
            killed.append((pid, action))
            processes[pid].killed = True

        def monotonic():
            return now

        def sleep(seconds):
            nonlocal now
            now += seconds

        class Response(io.BytesIO):
            status = 206
            headers = {
                "Content-Encoding": "gzip",
                "X-Prime-Raw-Length": "1024",
            }

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        encoded = base64.b64encode(
            json.dumps(commands, separators=(",", ":")).encode()
        ).decode()
        self._tunnel_simulation = processes, killed, printed
        with (
            patch("subprocess.Popen", side_effect=popen),
            patch("socket.socket", return_value=Socket()),
            patch("builtins.open", side_effect=open_process),
            patch("os.killpg", side_effect=killpg),
            patch("time.monotonic", side_effect=monotonic),
            patch("time.sleep", side_effect=sleep),
            patch(
                "urllib.request.urlopen",
                side_effect=lambda *_args, **_kwargs: Response(gzip.compress(b"x" * 1024)),
            ),
            patch("builtins.print", side_effect=lambda value: printed.append(value)),
            patch.object(
                sys,
                "argv",
                ["tunnels", "http://127.0.0.1:18200/file", encoded],
            ),
        ):
            exec(compile(script, "<tunnel-start>", "exec"), {})
        return processes, killed, printed

    def qualification_evidence(self, profile_name="glm52-native-b2048"):
        with (
            training_ops.repo_root
            / f"configs/qualification/{profile_name}.toml"
        ).open("rb") as stream:
            profile = tomllib.load(stream)
        now = datetime.datetime(2026, 8, 24, 12, 0, tzinfo=datetime.timezone.utc)
        hosts = [training_ops.trainer_head] + [
            host
            for host in training_ops.trainer_hosts
            if host != training_ops.trainer_head
        ]
        source_files = {
            path: hashlib.sha256(path.encode()).hexdigest()
            for path in training_ops._mounts_in(
                training_ops.repo_root / "scripts/china/cluster.sh"
            )
        }
        source_files["scripts/china/cluster.sh"] = "2" * 64
        session = {
            "schema_version": 1,
            "kind": "glm52_trainer_qualification_session",
            "session_id": "1" * 32,
            "profile": profile_name,
            "created_utc": (now - datetime.timedelta(minutes=2)).isoformat(),
            "config_sha256": hashlib.sha256(
                (
                    training_ops.repo_root
                    / f"configs/qualification/{profile_name}.toml"
                ).read_bytes()
            ).hexdigest(),
            "source_sha256": hashlib.sha256(
                (json.dumps(source_files, sort_keys=True, separators=(",", ":")) + "\n").encode()
            ).hexdigest(),
            "source_files": source_files,
            "trainer_hosts": hosts,
            "trainer_head": training_ops.trainer_head,
            "image_digests": {host: "sha256:" + "b" * 64 for host in hosts},
            "topology": {
                "world_size": 64,
                "pp": 8,
                "cp": 8,
                "ep": 8,
                "dp_replicate": 1,
                "dp_shard": 1,
            },
        }
        if profile_name == "glm52-native-b2048":
            session["deadlines"] = {
                "session_seconds": 3600,
                "startup_seconds": 900,
                "forward_backward_seconds": 900,
                "profiler_finalize_seconds": 120,
            }
        count = 2 if profile_name == "glm52-native-b2048" else 20
        runtime_ranks = [
            {
                "global_rank": rank,
                "local_rank": rank % 8,
                "host": hosts[rank // 8],
                "image_digest": session["image_digests"][hosts[rank // 8]],
                "source_sha256": session["source_sha256"],
                "launcher_sha256": source_files["scripts/china/cluster.sh"],
            }
            for rank in range(64)
        ]
        result = {
            "schema_version": 1,
            "kind": "glm52_trainer_qualification",
            "session_id": session["session_id"],
            "profile": profile_name,
            "session_sha256": training_ops._canonical_json_sha256(session),
            "config_sha256": session["config_sha256"],
            "source_sha256": session["source_sha256"],
            "trainer_head": training_ops.trainer_head,
            "trainer_head_image_digest": session["image_digests"][training_ops.trainer_head],
            "completed_utc": (now - datetime.timedelta(minutes=1)).isoformat(),
            "topology": session["topology"],
            "fake_data": {
                "batch_size": 64,
                "samples_per_micro_batch": 32,
                "group_size": 16,
                "sequence_length": 32768,
                "rollout_cache": False,
            },
            "lora": {
                "rank": 32,
                "alpha": 64,
                "targets": sorted(profile["model"]["lora"]["target_modules"]),
            },
            "max_steps": count,
            "steps": [
                {
                    "step": step,
                    "data/rollout_rows": 2048.0,
                    "loss/mean": 0.5,
                    "optim/grad_norm": 0.8,
                    "optim/lr": 8e-6,
                    "perf/peak_memory": 44.0,
                    "perf/forward_backward_peak_memory": 43.5,
                    "time/forward_backward": 300.0,
                    "input_sha256": hashlib.sha256(str(step).encode()).hexdigest(),
                }
                for step in range(1, count + 1)
            ],
            "metrics_path": profile["output_dir"] + "/trainer-metrics.jsonl",
            "metrics_sha256": "c" * 64,
            "runtime_ranks": runtime_ranks,
            "runtime_sha256": hashlib.sha256(
                (
                    json.dumps(
                        runtime_ranks, sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                ).encode()
            ).hexdigest(),
            "adapter": None,
        }
        if count == 20:
            result["adapter"] = {
                "path": profile["output_dir"] + "/run_trainer_only/broadcasts/step_20/adapter_model.safetensors",
                "sha256": "d" * 64,
                "size": 30_767_549_784,
                "step": 20,
                "config_path": profile["output_dir"] + "/run_trainer_only/broadcasts/step_20/adapter_config.json",
                "config_sha256": "e" * 64,
                "stable_path": profile["output_dir"] + "/run_trainer_only/broadcasts/step_20/STABLE",
                "stable_sha256": "f" * 64,
                "checksum_path": profile["output_dir"] + "/run_trainer_only/broadcasts/step_20/adapter_model.safetensors.sha256",
                "checksum_sha256": "0" * 64,
                "checksum_value": "d" * 64,
                "config": {
                    "r": 32,
                    "lora_alpha": 64,
                    "lora_dropout": 0.0,
                    "target_modules": sorted(training_ops.qualification_adapter_targets),
                },
                "inventory": {
                    "leaf_targets": sorted(training_ops.qualification_adapter_targets),
                    "tensor_count": 116_448,
                    "attention_tensors": 780,
                    "dense_mlp_tensors": 18,
                    "shared_expert_tensors": 450,
                    "routed_expert_tensors": 115_200,
                    "key_sha256": "1" * 64,
                },
                "layout": {
                    "header_bytes": 15_379_792,
                    "tensor_bytes": 30_752_169_984,
                    "file_bytes": 30_767_549_784,
                },
            }
        return session, result, profile, now

    def test_trainer_qualification_evidence_is_exact_and_fresh(self):
        for name in ("glm52-native-b2048", "glm52-native-stability20"):
            session, result, profile, now = self.qualification_evidence(name)
            self.assertEqual(
                training_ops._validate_trainer_qualification(
                    session, result, profile, name, now=now
                ),
                [],
            )

    def test_trainer_qualification_rejects_source_manifest_set_mismatch(self):
        session, result, profile, now = self.qualification_evidence()
        session["source_files"].pop(next(iter(session["source_files"])))
        failures = training_ops._validate_trainer_qualification(
            session, result, profile, "glm52-native-b2048", now=now
        )
        self.assertTrue(any("launcher mounts" in failure for failure in failures))

        session, result, profile, now = self.qualification_evidence()
        session["source_files"]["source/unexpected.py"] = "0" * 64
        failures = training_ops._validate_trainer_qualification(
            session, result, profile, "glm52-native-b2048", now=now
        )
        self.assertTrue(any("launcher mounts" in failure for failure in failures))

    def test_trainer_qualification_rejects_bad_batch_time_memory_and_adapter(self):
        session, result, profile, now = self.qualification_evidence(
            "glm52-native-stability20"
        )
        result["steps"][0]["data/rollout_rows"] = 2047
        result["steps"][1]["time/forward_backward"] = 901
        result["steps"][2]["perf/peak_memory"] = 46.1
        result["adapter"] = None
        failures = training_ops._validate_trainer_qualification(
            session, result, profile, "glm52-native-stability20", now=now
        )
        self.assertTrue(any("2048 samples" in failure for failure in failures))
        self.assertTrue(any("trainer time" in failure for failure in failures))
        self.assertTrue(any("memory" in failure for failure in failures))
        self.assertTrue(any("adapter" in failure for failure in failures))

    def test_trainer_qualification_rejects_reused_input_stale_session_and_rank(self):
        session, result, profile, now = self.qualification_evidence(
            "glm52-native-stability20"
        )
        result["steps"][1]["input_sha256"] = result["steps"][0][
            "input_sha256"
        ]
        session["created_utc"] = (now - datetime.timedelta(hours=25)).isoformat()
        result["session_sha256"] = training_ops._canonical_json_sha256(session)
        result["runtime_ranks"][9]["host"] = "n82"
        failures = training_ops._validate_trainer_qualification(
            session, result, profile, "glm52-native-stability20", now=now
        )
        self.assertTrue(any("reused" in failure for failure in failures))
        self.assertTrue(any("stale" in failure for failure in failures))
        self.assertTrue(any("runtime rank" in failure for failure in failures))

    def test_stability20_rejects_unstable_time_and_growing_memory(self):
        session, result, profile, now = self.qualification_evidence(
            "glm52-native-stability20"
        )
        result["steps"][-2]["time/forward_backward"] = 500
        result["steps"][-1]["time/forward_backward"] = 500
        for row in result["steps"][-5:]:
            row["perf/forward_backward_peak_memory"] = 44.1
        failures = training_ops._validate_trainer_qualification(
            session, result, profile, "glm52-native-stability20", now=now
        )
        self.assertTrue(any("trainer time is not stable" in failure for failure in failures))
        self.assertTrue(any("memory use grew" in failure for failure in failures))

    def test_stability20_export_peak_does_not_change_forward_backward_trend(self):
        session, result, profile, now = self.qualification_evidence(
            "glm52-native-stability20"
        )
        result["steps"][-1]["perf/peak_memory"] = 45.9
        self.assertEqual(
            training_ops._validate_trainer_qualification(
                session, result, profile, "glm52-native-stability20", now=now
            ),
            [],
        )

    def test_safetensors_layout_requires_exact_contiguous_tensor_bytes(self):
        header = {
            "a": {
                "dtype": "BF16",
                "shape": [2, 3],
                "data_offsets": [0, 12],
            },
            "b": {
                "dtype": "BF16",
                "shape": [4],
                "data_offsets": [12, 20],
            },
        }
        self.assertEqual(
            training_ops._safetensors_layout(header, 128),
            {"header_bytes": 128, "tensor_bytes": 20, "file_bytes": 156},
        )
        header["b"]["data_offsets"] = [13, 21]
        with self.assertRaisesRegex(training_ops.Refused, "contiguous"):
            training_ops._safetensors_layout(header, 128)

    def test_stability20_adapter_size_must_equal_header_layout(self):
        session, result, profile, now = self.qualification_evidence(
            "glm52-native-stability20"
        )
        result["adapter"]["size"] += 1
        failures = training_ops._validate_trainer_qualification(
            session, result, profile, "glm52-native-stability20", now=now
        )
        self.assertTrue(any("adapter evidence" in failure for failure in failures))

    def test_trainer_qualification_rejects_adapter_path_config_and_checksum(self):
        session, result, profile, now = self.qualification_evidence(
            "glm52-native-stability20"
        )
        result["adapter"]["path"] = profile["output_dir"] + "/adapter.safetensors"
        result["adapter"]["config"]["r"] = 16
        result["adapter"]["inventory"]["shared_expert_tensors"] = 449
        result["adapter"]["checksum_value"] = "0" * 64
        failures = training_ops._validate_trainer_qualification(
            session, result, profile, "glm52-native-stability20", now=now
        )
        self.assertTrue(any("adapter path" in failure for failure in failures))
        self.assertTrue(any("adapter configuration" in failure for failure in failures))
        self.assertTrue(any("tensor coverage" in failure for failure in failures))
        self.assertTrue(any("checksum content" in failure for failure in failures))

    def test_trainer_qualification_rejects_overlong_fresh_session(self):
        session, result, profile, now = self.qualification_evidence(
            "glm52-native-b2048"
        )
        session["created_utc"] = (now - datetime.timedelta(hours=5)).isoformat()
        result["session_sha256"] = training_ops._canonical_json_sha256(session)
        failures = training_ops._validate_trainer_qualification(
            session, result, profile, "glm52-native-b2048", now=now
        )
        self.assertTrue(any("stale" in failure for failure in failures))

    def test_stability20_recovery_keeps_original_source_and_marks_missing_inputs(self):
        session, result, profile, now = self.qualification_evidence(
            "glm52-native-stability20"
        )
        session["source_files"][training_ops.qualification_train_source] = (
            training_ops.legacy_stability20_train_sha256
        )
        session["source_sha256"] = hashlib.sha256(
            (
                json.dumps(
                    session["source_files"], sort_keys=True, separators=(",", ":")
                )
                + "\n"
            ).encode()
        ).hexdigest()
        result["source_sha256"] = session["source_sha256"]
        result["session_sha256"] = training_ops._canonical_json_sha256(session)
        for record in result["runtime_ranks"]:
            record["source_sha256"] = session["source_sha256"]
        result["runtime_sha256"] = hashlib.sha256(
            (
                json.dumps(
                    result["runtime_ranks"], sort_keys=True, separators=(",", ":")
                )
                + "\n"
            ).encode()
        ).hexdigest()
        for row in result["steps"]:
            row.pop("input_sha256")
            row.pop("perf/forward_backward_peak_memory")
        result["steps"][-1]["perf/peak_memory"] = 44.9921875
        result["recovery"] = {
            "reason": "adapter_leaf_target_metadata",
            "original_train_sha256": training_ops.legacy_stability20_train_sha256,
            "input_checks": "executed_not_persisted",
            "runtime_records": "derived_from_session",
            "finalizer_sha256": "2" * 64,
            "metrics_archive_path": (
                f"{profile['output_dir']}/{profile['file_monitor']['filename']}.gz"
            ),
            "metrics_archive_sha256": "3" * 64,
            "forward_backward_memory": (
                "steps_1_19_total_without_adapter_export"
            ),
            "step20_forward_backward_peak": "not_persisted",
            "step20_total_export_peak_gib": result["steps"][-1][
                "perf/peak_memory"
            ],
        }
        self.assertEqual(
            training_ops._validate_trainer_qualification(
                session, result, profile, "glm52-native-stability20", now=now
            ),
            [],
        )
        result["recovery"]["step20_total_export_peak_gib"] = 44.0
        failures = training_ops._validate_trainer_qualification(
            session, result, profile, "glm52-native-stability20", now=now
        )
        self.assertTrue(any("export memory differs" in failure for failure in failures))
        result["recovery"]["step20_total_export_peak_gib"] = result["steps"][-1][
            "perf/peak_memory"
        ]
        result["recovery"]["original_train_sha256"] = "0" * 64
        failures = training_ops._validate_trainer_qualification(
            session, result, profile, "glm52-native-stability20", now=now
        )
        self.assertTrue(any("recovery" in failure for failure in failures))

    def test_stability20_recovery_rejects_changed_train_source(self):
        with (
            training_ops.repo_root / training_ops.production_stability20_profile
        ).open("rb") as stream:
            profile = tomllib.load(stream)
        source_files = {
            source: hashlib.sha256(source.encode()).hexdigest()
            for source in training_ops._qualification_source_paths()
        }
        source_files[training_ops.qualification_train_source] = (
            training_ops.legacy_stability20_train_sha256
        )
        current_sources = dict(source_files)
        current_sources[training_ops.qualification_train_source] = (
            hashlib.sha256(
                (
                    training_ops.repo_root / training_ops.qualification_train_source
                ).read_bytes()
            ).hexdigest()
        )
        result = {
            "recovery": {
                "reason": "adapter_leaf_target_metadata",
                "original_train_sha256": (
                    training_ops.legacy_stability20_train_sha256
                ),
                "input_checks": "executed_not_persisted",
                "runtime_records": "derived_from_session",
                "finalizer_sha256": "1" * 64,
                "metrics_archive_path": (
                    f"{profile['output_dir']}/{profile['file_monitor']['filename']}.gz"
                ),
                "metrics_archive_sha256": "2" * 64,
                "forward_backward_memory": (
                    "steps_1_19_total_without_adapter_export"
                ),
                "step20_forward_backward_peak": "not_persisted",
                "step20_total_export_peak_gib": 44.9921875,
            }
        }
        session = {"source_files": source_files}
        self.assertNotEqual(
            hashlib.sha256(
                (
                    training_ops.repo_root / training_ops.qualification_train_source
                ).read_bytes()
            ).hexdigest(),
            training_ops.approved_recovery_train_sha256,
        )
        self.assertFalse(
            training_ops._qualification_sources_match(
                session,
                result,
                profile,
                "glm52-native-stability20",
                current_sources,
            )
        )
        changed_other = dict(current_sources)
        changed_other["source/file-0.py"] = "3" * 64
        self.assertFalse(
            training_ops._qualification_sources_match(
                session,
                result,
                profile,
                "glm52-native-stability20",
                changed_other,
            )
        )
        changed_train = dict(current_sources)
        changed_train[training_ops.qualification_train_source] = "4" * 64
        self.assertFalse(
            training_ops._qualification_sources_match(
                session,
                result,
                profile,
                "glm52-native-stability20",
                changed_train,
            )
        )
        result["recovery"]["unexpected"] = True
        self.assertFalse(
            training_ops._qualification_sources_match(
                session,
                result,
                profile,
                "glm52-native-stability20",
                current_sources,
            )
        )

    def test_stability20_recovery_write_is_create_only_and_atomic(self):
        captured = {}

        def remote(_host, script, _label, timeout):
            captured["script"] = script
            captured["timeout"] = timeout
            return ""

        destination = (
            "/data/glm52-rl/qualifications/glm52-native-stability20/"
            "qualification-result.json"
        )
        with patch.object(training_ops, "_remote_ok", side_effect=remote):
            training_ops._write_remote_qualification_result(destination, {"ok": True})
        script = captured["script"]
        embedded = script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        compile(embedded, "<qualification-recovery-write>", "exec")
        self.assertIn("qualification result already exists", embedded)
        self.assertIn("os.O_EXCL | os.O_NOFOLLOW", embedded)
        self.assertIn("os.link(", embedded)
        self.assertEqual(captured["timeout"], 60)

    def test_stability20_stat_parser_accepts_actual_gnu_stat_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "adapter file"
            path.write_bytes(b"weights")
            output = subprocess.run(
                ["stat", "-c", "%n|%s|%Y", "--", str(path)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        size, modified = training_ops._parse_qualification_file_stats(
            output, {str(path)}
        )[str(path)]
        self.assertEqual(size, 7)
        self.assertGreater(modified, 0)

    def test_stability20_completion_allows_one_second_monitor_flush(self):
        paths = {
            "metrics": "metrics.jsonl",
            "adapter": "adapter.safetensors",
            "config": "adapter_config.json",
            "stable": "STABLE",
            "checksum": "adapter.safetensors.sha256",
        }
        stats = {
            paths["metrics"]: (100, 1787711315),
            paths["adapter"]: (200, 1787711295),
            paths["config"]: (300, 1787711295),
            paths["stable"]: (0, 1787711314),
            paths["checksum"]: (65, 1787711314),
        }
        self.assertEqual(
            training_ops._qualification_recovery_completed_epoch(
                stats, paths, 1787711200
            ),
            1787711315,
        )

    def test_stability20_completion_rejects_late_monitor_flush(self):
        paths = {
            "metrics": "metrics.jsonl",
            "adapter": "adapter.safetensors",
            "config": "adapter_config.json",
            "stable": "STABLE",
            "checksum": "adapter.safetensors.sha256",
        }
        stats = {
            paths["metrics"]: (100, 1787711317),
            paths["adapter"]: (200, 1787711295),
            paths["config"]: (300, 1787711295),
            paths["stable"]: (0, 1787711314),
            paths["checksum"]: (65, 1787711314),
        }
        with self.assertRaisesRegex(training_ops.Refused, "monitor flush"):
            training_ops._qualification_recovery_completed_epoch(
                stats, paths, 1787711200
            )

    def qualification_metric_fragments(self, timestamp=150.0):
        rows = []
        for step in (1, 2):
            step_time = timestamp + step
            rows.extend(
                (
                    {
                        "step": step,
                        "time": step_time,
                        "data/rollout_rows": 2048,
                        "perf/peak_memory": 44.0,
                    },
                    {
                        "step": step,
                        "time": step_time + 0.1,
                        "optim/lr": 8e-6,
                        "optim/grad_norm": 0.02,
                    },
                    {
                        "step": step,
                        "time": step_time + 0.2,
                        "loss/mean": 0.001,
                    },
                    {
                        "step": step,
                        "time": step_time + 0.3,
                        "time/forward_backward": 700.0,
                    },
                    {"step": step, "time": step_time + 0.4, "system/free": 1},
                )
            )
        return rows

    def test_stability20_metric_fragments_merge_per_step(self):
        metrics = "".join(
            json.dumps(row) + "\n" for row in self.qualification_metric_fragments()
        )
        steps = training_ops._qualification_recovery_steps(
            metrics, 100.0, 200, expected_steps=2
        )
        self.assertEqual([row["step"] for row in steps], [1, 2])
        self.assertEqual(steps[0]["data/rollout_rows"], 2048)
        self.assertEqual(steps[1]["time/forward_backward"], 700.0)

    def test_stability20_metric_fragments_exclude_older_runs(self):
        rows = self.qualification_metric_fragments()
        older = [
            {**row, "time": 99.0, "data/rollout_rows": 1}
            for row in rows
        ]
        metrics = "".join(json.dumps(row) + "\n" for row in older + rows)
        steps = training_ops._qualification_recovery_steps(
            metrics, 100.0, 200, expected_steps=2
        )
        self.assertEqual([row["data/rollout_rows"] for row in steps], [2048, 2048])

    def test_stability20_metric_fragments_reject_duplicate_required_value(self):
        rows = self.qualification_metric_fragments()
        rows.append(
            {"step": 1, "time": 152.0, "loss/mean": rows[2]["loss/mean"]}
        )
        metrics = "".join(json.dumps(row) + "\n" for row in rows)
        with self.assertRaisesRegex(training_ops.Refused, "repeats loss/mean"):
            training_ops._qualification_recovery_steps(
                metrics, 100.0, 200, expected_steps=2
            )

    def test_stability20_metric_gzip_restore_is_create_only(self):
        payload = b'{"step":1,"time":150}\n'
        archive_bytes = gzip.compress(payload, mtime=0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "trainer-metrics.jsonl.gz"
            archive.write_bytes(archive_bytes)
            os.utime(archive, (1787711315, 1787711315))
            script = training_ops._qualification_metrics_restore_script(
                str(root),
                "trainer-metrics.jsonl",
                archive.name,
                hashlib.sha256(archive_bytes).hexdigest(),
                len(archive_bytes),
                hashlib.sha256(payload).hexdigest(),
                len(payload),
            )
            subprocess.run(["bash", "-c", script], check=True, capture_output=True)
            restored = root / "trainer-metrics.jsonl"
            self.assertEqual(restored.read_bytes(), payload)
            self.assertEqual(archive.read_bytes(), archive_bytes)
            self.assertEqual(int(restored.stat().st_mtime), 1787711315)
            repeated = subprocess.run(
                ["bash", "-c", script], check=False, capture_output=True, text=True
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("already exists", repeated.stderr)
            self.assertEqual(restored.read_bytes(), payload)

    def test_remote_file_reader_streams_without_replacing_source(self):
        payload = b"evidence\n"
        encoded = base64.b64encode(gzip.compress(payload)).decode()
        with patch.object(training_ops, "_ssh", return_value=(0, encoded)) as remote:
            self.assertEqual(
                training_ops._remote_file_bytes("trainer", "/data/evidence", 100),
                payload,
            )
        command = remote.call_args.args[1]
        self.assertIn("gzip -1 -c --", command)

    def step20_h200_evidence(self):
        _session, stability, _trainer_profile, now = self.qualification_evidence(
            "glm52-native-stability20"
        )
        with (
            training_ops.repo_root / training_ops.production_step20_h200_profile
        ).open("rb") as stream:
            profile = tomllib.load(stream)
        capture_sha256 = "6" * 64
        adapter_sha256 = stability["adapter"]["sha256"]
        guard = {"passed": True, "checks": {"unmasked_mismatch_kl": True}}
        capture = {
            "schema_version": 1,
            "kind": "glm52_policy_parity_h200_capture",
            "created_utc": (now - datetime.timedelta(minutes=1)).isoformat(),
            "adapter_sha256": adapter_sha256,
            "adapter_loaded": True,
            "adapter_path": training_ops.glm_step20_h200_gate["adapter"],
            "graph_eager_guard": guard,
            "arrays": {
                "full_logprobs": {"sha256": "7" * 64},
                "router_ids": {"sha256": "8" * 64},
                "router_weights": {"sha256": "9" * 64},
            },
        }
        parity_report = {
            "created_utc": (now - datetime.timedelta(seconds=30)).isoformat(),
            "adapter_sha256": adapter_sha256,
            "h200_capture_manifest_sha256": capture_sha256,
            "graph_eager_guard": guard,
        }
        settings = {
            name: profile[name]
            for name in (
                "requests",
                "group_size",
                "max_inflight",
                "context_length",
                "max_output_tokens",
                "min_output_tokens",
                "temperature",
                "top_p",
                "seed",
                "request_timeout_seconds",
                "max_retries",
                "retry_count_max",
                "qualification_deadline_seconds",
                "tensor_parallel_size",
                "cuda_graph",
                "cuda_graph_capture_size",
                "prefix_cache",
                "kv_cache_dtype",
                "max_num_seqs",
                "max_num_batched_tokens",
                "batch_seconds_min",
                "batch_seconds_max",
                "peak_vram_mib_max",
                "evidence_max_age_seconds",
            )
        }
        result = {
            "schema_version": 1,
            "kind": "glm52_step20_h200_gate",
            "created_utc": now.isoformat(),
            "profile_sha256": hashlib.sha256(
                (
                    training_ops.repo_root
                    / training_ops.production_step20_h200_profile
                ).read_bytes()
            ).hexdigest(),
            "settings": settings,
            "adapter": {
                "name": profile["model"],
                "h200_path": training_ops.glm_step20_h200_gate["adapter"],
                "qualification_result_sha256": "a" * 64,
                "qualification_session_sha256": stability["session_sha256"],
                "trainer_path": stability["adapter"]["path"],
                **{
                    name: stability["adapter"][name]
                    for name in (
                        "sha256",
                        "size",
                        "config_sha256",
                        "stable_sha256",
                        "checksum_sha256",
                    )
                },
            },
            "engine": {
                "config": profile["engine_config"],
                "config_sha256": hashlib.sha256(
                    (training_ops.repo_root / profile["engine_config"]).read_bytes()
                ).hexdigest(),
                "image_digest": "sha256:" + "b" * 64,
                **{
                    name: profile[name]
                    for name in (
                        "tensor_parallel_size",
                        "cuda_graph",
                        "cuda_graph_capture_size",
                        "prefix_cache",
                        "kv_cache_dtype",
                        "max_num_seqs",
                        "max_num_batched_tokens",
                    )
                },
            },
            "parity": {
                "adapter_sha256": adapter_sha256,
                "capture_manifest_sha256": capture_sha256,
                "graph_eager_guard": guard,
                "router_runtime_marker_sha256": "0" * 64,
                "array_sha256": {
                    name: value["sha256"] for name, value in capture["arrays"].items()
                },
            },
            "measurements": {
                "batch_seconds": 600.0,
                "completed_requests": 1024,
                "failed_requests": 0,
                "retries": 2,
                "retry_errors": ["H200RequestError: reset"] * 2,
                "output_tokens": 1024 * 16384,
                "output_tokens_per_second": (1024 * 16384) / 600,
                "max_prompt_tokens": 1024,
                "peak_vram_mib": [120_000] * 8,
                "latency_seconds": {
                    "mean": 100.0,
                    "p50": 90.0,
                    "p95": 200.0,
                    "p99": 300.0,
                    "max": 400.0,
                },
                "first_errors": [],
            },
        }
        return result, profile, stability, parity_report, capture, now

    def test_step20_h200_gate_binds_fresh_adapter_parity_and_engine(self):
        result, profile, stability, report, capture, now = self.step20_h200_evidence()
        self.assertEqual(
            training_ops._validate_step20_h200_gate(
                result,
                profile,
                stability,
                report,
                capture,
                qualification_result_sha256="a" * 64,
                capture_manifest_sha256="6" * 64,
                expected_engine_digest="sha256:" + "b" * 64,
                now=now,
            ),
            [],
        )

    def test_step20_h200_gate_rejects_mixed_or_stale_evidence(self):
        result, profile, stability, report, capture, now = self.step20_h200_evidence()
        result["adapter"]["sha256"] = "f" * 64
        result["measurements"]["batch_seconds"] = 721.0
        capture["created_utc"] = (now - datetime.timedelta(hours=25)).isoformat()
        profile["max_inflight"] = 127
        failures = training_ops._validate_step20_h200_gate(
            result,
            profile,
            stability,
            report,
            capture,
            qualification_result_sha256="a" * 64,
            capture_manifest_sha256="6" * 64,
            expected_engine_digest="sha256:" + "b" * 64,
            now=now,
        )
        self.assertTrue(any("adapter identity" in failure for failure in failures))
        self.assertTrue(any("production shape" in failure for failure in failures))
        self.assertTrue(any("480-720" in failure for failure in failures))
        self.assertTrue(any("stale" in failure for failure in failures))

    def test_step20_h200_adapter_stage_uses_exact_candidate_and_hashes(self):
        payloads = {
            "adapter_model.safetensors": b"weights",
            "adapter_config.json": b"{}\n",
            "STABLE": b"stable\n",
        }
        payloads["adapter_model.safetensors.sha256"] = (
            hashlib.sha256(payloads["adapter_model.safetensors"]).hexdigest()
            + "\n"
        ).encode()
        adapter = {
            "path": "/data/glm52-rl/qualifications/test/adapter_model.safetensors",
            "sha256": hashlib.sha256(payloads["adapter_model.safetensors"]).hexdigest(),
            "size": len(payloads["adapter_model.safetensors"]),
            "config_path": "/source/adapter_config.json",
            "config_sha256": hashlib.sha256(payloads["adapter_config.json"]).hexdigest(),
            "stable_path": "/source/STABLE",
            "stable_sha256": hashlib.sha256(payloads["STABLE"]).hexdigest(),
            "checksum_path": "/source/adapter_model.safetensors.sha256",
            "checksum_sha256": hashlib.sha256(
                payloads["adapter_model.safetensors.sha256"]
            ).hexdigest(),
        }
        result_sha256 = hashlib.sha256(b"qualification\n").hexdigest()
        payloads["qualification-result.json"] = b"qualification\n"
        uploads = []
        remote = []
        direct = []

        def download(source, destination):
            destination.write_bytes(payloads[Path(source).name])

        def command(arguments, _label, timeout):
            uploads.append((arguments, timeout))

        def direct_download(*arguments):
            direct.append(arguments)
            return {
                "bytes": adapter["size"],
                "seconds": 1.0,
                "sha256": adapter["sha256"],
            }

        def remote_ok(_host, command_text, _label, timeout=60):
            remote.append((command_text, timeout))
            return "installed\n" if "sudo python3" in command_text else ""

        tunnel_records = [
            {
                "pid": 1000 + port,
                "port": port,
                "cmdline": training_ops._step20_adapter_tunnel_command(port),
            }
            for port in range(18200, 18208)
        ]
        with (
            patch.object(training_ops, "_download_adapter_file", side_effect=download),
            patch.object(training_ops, "_local_command", side_effect=command),
            patch.object(
                training_ops,
                "_download_glm_adapter_direct",
                side_effect=direct_download,
            ),
            patch.object(training_ops, "_remote_ok", side_effect=remote_ok),
            patch.object(
                training_ops,
                "_start_step20_adapter_tunnels",
                return_value=tunnel_records,
            ),
            patch.object(
                training_ops, "_ensure_glm_adapter_range_server"
            ) as ensure_server,
            patch.object(training_ops, "_stop_step20_adapter_tunnels") as stop_tunnels,
            patch.object(training_ops, "_cleanup_step20_adapter_candidate") as cleanup,
            patch.object(training_ops.secrets, "token_hex", return_value="1" * 16),
        ):
            self.assertEqual(
                training_ops._stage_step20_adapter_on_h200(
                    adapter, result_sha256
                ),
                training_ops.glm_step20_h200_gate["adapter"],
            )
        stop_tunnels.assert_called_once_with(
            training_ops.inference_hosts[0], tunnel_records
        )
        ensure_server.assert_called_once_with(adapter["path"])
        cleanup.assert_not_called()
        self.assertEqual(len(direct), 1)
        self.assertEqual(direct[0][1], adapter["path"])
        self.assertIn(".adapter-candidate-" + "1" * 16, direct[0][2])
        self.assertEqual(direct[0][3:5], (adapter["sha256"], adapter["size"]))
        self.assertEqual(len(uploads), 4)
        self.assertTrue(all(call[0][0] == "scp" and call[1] == 1800 for call in uploads))
        self.assertTrue(
            all(".adapter-candidate-" + "1" * 16 in call[0][-1] for call in uploads)
        )
        installer = remote[-1][0]
        self.assertIn("os.O_NOFOLLOW", installer)
        self.assertIn("os.rename(candidate_name,final_name", installer)
        self.assertNotIn("rm -rf", installer)
        embedded = installer.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        compile(embedded, "<step20-adapter-installer>", "exec")
        command = shlex.split(installer.split(" <<'PY'", 1)[0])
        uid = os.getuid()
        gid = os.getgid()
        executable = (
            embedded.replace("(1000,1000,0o755)", f"({uid},{gid},0o755)")
            .replace("(0,0,0o755)", f"({uid},{gid},0o755)")
            .replace("(0,0,0o644)", f"({uid},{gid},0o644)")
            .replace("os.fchown(fd,0,0)", f"os.fchown(fd,{uid},{gid})")
            .replace("os.fchown(child,0,0)", f"os.fchown(child,{uid},{gid})")
        )
        expected_files = {
            "adapter_model.safetensors": payloads["adapter_model.safetensors"],
            "adapter_config.json": payloads["adapter_config.json"],
            "STABLE": payloads["STABLE"],
            "adapter_model.safetensors.sha256": payloads[
                "adapter_model.safetensors.sha256"
            ],
            ".qualification-result.json": payloads["qualification-result.json"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.chmod(0o755)
            candidate_name, final_name, previous_name, encoded = command[4:8]
            candidate = root / candidate_name
            final = root / final_name
            candidate.mkdir(mode=0o750)
            final.mkdir(mode=0o755)
            for name, payload in expected_files.items():
                (candidate / name).write_bytes(payload)
                (final / name).write_bytes(b"old-" + payload)
                (final / name).chmod(0o644)
            result = subprocess.run(
                [
                    "python3",
                    "-",
                    str(root),
                    candidate_name,
                    final_name,
                    previous_name,
                    encoded,
                ],
                input=executable,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "replaced")
            self.assertFalse(candidate.exists())
            self.assertFalse((root / previous_name).exists())
            self.assertEqual(
                {path.name: path.read_bytes() for path in final.iterdir()},
                expected_files,
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.chmod(0o755)
            candidate = root / candidate_name
            final = root / final_name
            candidate.mkdir(mode=0o750)
            final.mkdir(mode=0o755)
            old_files = {}
            for name, payload in expected_files.items():
                (candidate / name).write_bytes(payload)
                old_files[name] = b"old-" + payload
                (final / name).write_bytes(old_files[name])
                (final / name).chmod(0o644)
            before, separator, after = executable.rpartition(
                "   validate(final_name,True)"
            )
            self.assertTrue(separator)
            faulted = before + '   raise RuntimeError("injected")' + after
            result = subprocess.run(
                [
                    "python3",
                    "-",
                    str(root),
                    candidate_name,
                    final_name,
                    previous_name,
                    encoded,
                ],
                input=faulted,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / previous_name).exists())
            self.assertEqual(
                {path.name: path.read_bytes() for path in final.iterdir()},
                old_files,
            )
            self.assertEqual(
                {path.name: path.read_bytes() for path in candidate.iterdir()},
                expected_files,
            )

        with (
            patch.object(training_ops, "_remote_ok", return_value=""),
            patch.object(
                training_ops,
                "_start_step20_adapter_tunnels",
                return_value=tunnel_records,
            ),
            patch.object(training_ops, "_ensure_glm_adapter_range_server"),
            patch.object(
                training_ops,
                "_download_glm_adapter_direct",
                side_effect=RuntimeError("injected transfer failure"),
            ),
            patch.object(
                training_ops, "_cleanup_step20_adapter_candidate"
            ) as cleanup,
            patch.object(
                training_ops, "_stop_step20_adapter_tunnels"
            ) as stop_tunnels,
            patch.object(training_ops.secrets, "token_hex", return_value="2" * 16),
            self.assertRaisesRegex(RuntimeError, "injected transfer failure"),
        ):
            training_ops._stage_step20_adapter_on_h200(adapter, result_sha256)
        cleanup.assert_called_once_with(
            training_ops.inference_hosts[0],
            PurePosixPath(training_ops.glm_step20_h200_gate["adapter"]).parent
            / (".adapter-candidate-" + "2" * 16),
        )
        stop_tunnels.assert_called_once_with(
            training_ops.inference_hosts[0], tunnel_records
        )

    def test_glm_adapter_server_is_deployed_supervised_and_exact(self):
        source = (
            "/data/glm52-rl/qualifications/test/adapter_model.safetensors"
        )
        digest = hashlib.sha256(
            (
                training_ops.repo_root
                / "scripts/ops/glm_adapter_range_server.py"
            ).read_bytes()
        ).hexdigest()
        reports = (
            json.dumps(
                {"state": "started", "raw_bytes": 1024, "script_sha256": digest}
            ),
            json.dumps(
                {"state": "checked", "raw_bytes": 1024, "script_sha256": digest}
            ),
        )
        with (
            patch.object(training_ops, "deploy", return_value=True) as deploy,
            patch.object(training_ops, "_remote_ok", side_effect=reports) as remote,
        ):
            started = training_ops._ensure_glm_adapter_range_server(source)
            checked = training_ops._check_glm_adapter_range_server(source)
        self.assertEqual(started["state"], "started")
        self.assertEqual(checked["state"], "checked")
        deploy.assert_called_once_with(
            "scripts/ops/glm_adapter_range_server.py",
            nodes=[training_ops.trainer_head],
        )
        ensure_arguments = shlex.split(remote.call_args_list[0].args[1])
        check_arguments = shlex.split(remote.call_args_list[1].args[1])
        self.assertEqual(ensure_arguments[-2], "ensure")
        self.assertEqual(check_arguments[-2], "check")
        compile(ensure_arguments[2], "<glm-adapter-server-manager>", "exec")
        expected = json.loads(
            base64.b64decode(ensure_arguments[-1], validate=True)
        )
        self.assertEqual(
            expected["name"], training_ops.glm_adapter_server_container
        )
        self.assertEqual(expected["port"], training_ops.glm_adapter_port)
        self.assertEqual(expected["root"], str(training_ops.glm_adapter_root))
        self.assertEqual(expected["script_sha256"], digest)
        manager = ensure_arguments[2]
        self.assertIn('"--restart","unless-stopped"', manager)
        self.assertIn('"--read-only"', manager)
        self.assertIn('"prime.glm-adapter-server=1"', manager)
        self.assertIn("refusing wrong GLM adapter server identity", manager)
        self.assertIn('"Content-Encoding"', manager)
        self.assertIn('"X-Prime-Raw-Length"', manager)
        tree = ast.parse(manager)
        inspect_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "inspect"
        )
        namespace = {
            "docker": lambda *_args, **_kwargs: SimpleNamespace(
                returncode=1,
                stderr="error: no such object: glm52-optims-adapter-gzip\n",
                stdout="",
            ),
            "json": json,
        }
        ast.fix_missing_locations(inspect_function)
        exec(
            compile(ast.Module(body=[inspect_function], type_ignores=[]), "<inspect>", "exec"),
            namespace,
        )
        self.assertIsNone(namespace["inspect"](training_ops.glm_adapter_server_container))

    def test_direct_glm_adapter_download_requires_gzip_ranges(self):
        digest = "a" * 64
        report = json.dumps(
            {
                "bytes": 123,
                "seconds": 9.5,
                "sha256": digest,
                "chunk_attempts": [1, 2, 1],
            }
        )
        with patch.object(training_ops, "_remote_ok", return_value=report) as remote:
            result = training_ops._download_glm_adapter_direct(
                training_ops.inference_hosts[0],
                "/data/glm52-rl/qualifications/test/adapter_model.safetensors",
                "/data/glm52-rl/qualifications/glm52-step20-h200/"
                ".adapter-candidate-1111111111111111/adapter_model.safetensors",
                digest,
                123,
            )
        self.assertEqual(result["seconds"], 9.5)
        self.assertEqual(result["chunk_attempts"], [1, 2, 1])
        arguments = shlex.split(remote.call_args.args[1])
        self.assertIn("PRIME_GLM_ADAPTER_GZIP=1", arguments)
        self.assertIn(
            "PRIME_ADAPTER_PORTS=18200,18201,18202,18203,18204,18205,18206,18207",
            arguments,
        )
        self.assertIn(
            "http://127.0.0.1:18200/qualifications/test/adapter_model.safetensors",
            arguments,
        )

    def test_step20_tunnels_record_and_clean_only_exact_processes(self):
        commands = [
            training_ops._step20_adapter_tunnel_command(port)
            for port in range(18200, 18208)
        ]
        records = [
            {"pid": 1000 + index, "cmdline": command, "port": 18200 + index}
            for index, command in enumerate(commands)
        ]
        replies = iter((json.dumps(records), "cleaned\n"))
        with patch.object(
            training_ops, "_remote_ok", side_effect=lambda *_args, **_kwargs: next(replies)
        ) as remote:
            observed = training_ops._start_step20_adapter_tunnels(
                training_ops.inference_hosts[0],
                "http://127.0.0.1:18200/qualifications/test/adapter_model.safetensors",
            )
            training_ops._stop_step20_adapter_tunnels(
                training_ops.inference_hosts[0], observed
            )
        self.assertEqual(observed, records)
        start_arguments = shlex.split(remote.call_args_list[0].args[1])
        self.assertIn(
            "required GLM gzip adapter server is not ready through temporary staging tunnels",
            start_arguments[2],
        )
        self.assertIn("for attempt in range(3)", start_arguments[2])
        self.assertIn("time.sleep(0.25)", start_arguments[2])
        self.assertIn("pending=launched", start_arguments[2])
        self.assertIn("if port in started: continue", start_arguments[2])
        self.assertIn("cleanup_processes(failed)", start_arguments[2])
        self.assertIn("except ProcessLookupError: continue", start_arguments[2])
        compile(start_arguments[2], "<step20-adapter-tunnel-start>", "exec")
        self.assertEqual(
            json.loads(base64.b64decode(start_arguments[-1], validate=True)),
            commands,
        )
        self.assertTrue(all(command[-1] == training_ops.trainer_ssh for command in commands))
        proven_max = (
            training_ops.step20_tunnel_attempts
            * (
                len(training_ops.glm_adapter_tunnel_ports)
                * training_ops.step20_tunnel_launch_stagger_seconds
                + training_ops.step20_tunnel_start_seconds
                + training_ops.step20_tunnel_cleanup_seconds
            )
            + training_ops.step20_tunnel_probe_seconds
        )
        self.assertEqual(proven_max, 186.0)
        self.assertGreaterEqual(
            training_ops.step20_tunnel_outer_timeout_seconds,
            proven_max + 60,
        )
        self.assertEqual(
            remote.call_args_list[0].kwargs["timeout"],
            training_ops.step20_tunnel_outer_timeout_seconds,
        )
        stop_arguments = shlex.split(remote.call_args_list[1].args[1])
        self.assertIn("current!=record[\"cmdline\"]", stop_arguments[2])
        self.assertIn("os.killpg", stop_arguments[2])
        compile(stop_arguments[2], "<step20-adapter-tunnel-stop>", "exec")

        script = start_arguments[2]
        first_command, second_command = commands[:2]
        first_port, second_port = 18200, 18201
        processes, killed, printed = self._run_tunnel_start_script(
            script,
            [first_command],
            {first_port: ["exit", "success"]},
        )
        self.assertEqual(len(processes), 2)
        self.assertEqual(killed, [])
        self.assertEqual(json.loads(printed[-1])[0]["pid"], 3002)

        processes, killed, printed = self._run_tunnel_start_script(
            script,
            [first_command],
            {first_port: [("slow", 8.0)]},
        )
        self.assertEqual(len(processes), 1)
        self.assertEqual(killed, [])
        self.assertEqual(json.loads(printed[-1])[0]["pid"], 3001)

        with self.assertRaisesRegex(RuntimeError, "failed after 3 exact attempts"):
            self._run_tunnel_start_script(
                script,
                [first_command, second_command],
                {
                    first_port: ["success"],
                    second_port: ["exit", "exit", "exit"],
                },
            )
        processes, killed, _printed = self._tunnel_simulation
        self.assertEqual(len(processes), 4)
        self.assertEqual(killed, [(3001, signal.SIGTERM)])

        processes, killed, printed = self._run_tunnel_start_script(
            script,
            [first_command],
            {first_port: ["wrong", "success"]},
        )
        self.assertNotIn(3001, {pid for pid, _action in killed})
        self.assertEqual(json.loads(printed[-1])[0]["pid"], 3002)

        with self.assertRaisesRegex(RuntimeError, "failed after 3 exact attempts"):
            self._run_tunnel_start_script(
                script,
                [first_command, second_command],
                {
                    first_port: ["success"],
                    second_port: ["exit", "exit", "exit"],
                },
                kill_race_pids={3001},
            )
        _processes, killed, _printed = self._tunnel_simulation
        self.assertEqual(killed, [])

    def test_step20_candidate_cleanup_is_limited_to_random_candidate(self):
        candidate = (
            PurePosixPath(training_ops.glm_step20_h200_gate["adapter"]).parent
            / ".adapter-candidate-1111111111111111"
        )
        with patch.object(training_ops, "_remote_ok", return_value="removed\n") as remote:
            training_ops._cleanup_step20_adapter_candidate(
                training_ops.inference_hosts[0], candidate
            )
        arguments = shlex.split(remote.call_args.args[1])
        compile(arguments[2], "<step20-adapter-candidate-cleanup>", "exec")
        self.assertIn("os.O_NOFOLLOW", arguments[2])
        self.assertIn('parts_name="adapter_model.safetensors.parts"', arguments[2])
        self.assertNotIn("adapter-step20", arguments[2])
        self.assertNotIn("rm -rf", arguments[2])

    def test_step20_trainer_parity_action_pins_exact_adapter_and_step(self):
        with patch.object(training_ops, "qualify") as qualify:
            training_ops._qualify_glm52_step20_parity_action([])
        qualify.assert_called_once_with(
            "glm52-policy-parity",
            resume_adapter=(
                "/data/glm52-rl/qualifications/glm52-native-stability20/"
                "run_trainer_only/broadcasts/step_20/adapter_model.safetensors"
            ),
            resume_step=0,
        )
        with self.assertRaisesRegex(training_ops.Refused, "takes no arguments"):
            training_ops._qualify_glm52_step20_parity_action(["unexpected"])

    def test_step20_h200_stage_action_binds_qualification_result(self):
        stability = {"adapter": {"sha256": "a" * 64}}
        with (
            patch.object(
                training_ops,
                "_step20_stability_evidence",
                return_value=(stability, "b" * 64),
            ),
            patch.object(
                training_ops,
                "_stage_step20_adapter_on_h200",
                return_value=training_ops.glm_step20_h200_gate["adapter"],
            ) as stage,
        ):
            result = training_ops._stage_glm52_step20_adapter_action([])
        stage.assert_called_once_with(stability["adapter"], "b" * 64)
        self.assertEqual(result["adapter_sha256"], "a" * 64)
        self.assertEqual(result["qualification_result_sha256"], "b" * 64)

    def test_trainer_qualification_session_publish_is_no_follow_and_atomic(self):
        profile_name = "glm52-native-b2048"
        path = (
            training_ops.repo_root
            / f"configs/qualification/{profile_name}.toml"
        )
        with path.open("rb") as stream:
            profile = tomllib.load(stream)
        hosts = [training_ops.trainer_head] + [
            host
            for host in training_ops.trainer_hosts
            if host != training_ops.trainer_head
        ]
        images = {host: "sha256:" + "a" * 64 for host in hosts}
        calls = []

        def ssh(host, command, timeout=60):
            calls.append((host, command, timeout))
            return 0, ""

        with (
            patch.object(training_ops, "_trainer_image_digest_map", return_value=images),
            patch.object(training_ops, "_ssh", side_effect=ssh),
            patch.object(training_ops.secrets, "token_hex", return_value="1" * 32),
        ):
            source_files = {
                source: hashlib.sha256(source.encode()).hexdigest()
                for source in training_ops._mounts_in(
                    training_ops.repo_root / "scripts/china/cluster.sh"
                )
            }
            source_files["scripts/china/cluster.sh"] = "c" * 64
            identity = training_ops._prepare_trainer_qualification_session(
                profile_name,
                profile,
                path,
                hosts,
                source_files,
            )

        self.assertCountEqual([call[0] for call in calls], hosts)
        self.assertTrue(all(call[2] == 60 for call in calls))
        self.assertTrue(all("os.O_EXCL | os.O_NOFOLLOW" in call[1] for call in calls))
        self.assertTrue(all("os.replace(temporary, destination" in call[1] for call in calls))
        self.assertTrue(all("os.fsync(directory)" in call[1] for call in calls))
        self.assertTrue(all(" tee " not in call[1] for call in calls))
        self.assertIn(
            "PRIME_QUALIFICATION_SESSION_ID=" + "1" * 32,
            identity.environment,
        )
        self.assertEqual(identity.profile, profile_name)
        self.assertEqual(identity.launcher_sha256, "c" * 64)
        self.assertEqual(dict(identity.image_digests), images)

    def test_trainer_qualification_source_manifest_includes_launcher(self):
        digest = hashlib.sha256(
            (training_ops.repo_root / "scripts/china/cluster.sh").read_bytes()
        ).hexdigest()
        hosts = ["n82", "n76"]
        with (
            patch.object(
                training_ops,
                "trainer_mount_preflight",
                return_value={"source/a.py": "a" * 64},
            ),
            patch.object(
                training_ops,
                "_qualification_source_paths",
                return_value={"source/a.py", "scripts/china/cluster.sh"},
            ),
            patch.object(
                training_ops,
                "_fan",
                return_value={host: (0, digest + "\n") for host in hosts},
            ),
        ):
            self.assertEqual(
                training_ops._trainer_qualification_source_files(hosts),
                {
                    "source/a.py": "a" * 64,
                    "scripts/china/cluster.sh": digest,
                },
            )

        with (
            patch.object(
                training_ops,
                "trainer_mount_preflight",
                return_value={"source/a.py": "a" * 64},
            ),
            patch.object(
                training_ops,
                "_qualification_source_paths",
                return_value={"source/a.py", "scripts/china/cluster.sh"},
            ),
            patch.object(
                training_ops,
                "_fan",
                return_value={"n82": (0, digest), "n76": (0, "0" * 64)},
            ),
            self.assertRaisesRegex(training_ops.Refused, "n76"),
        ):
            training_ops._trainer_qualification_source_files(hosts)

    def test_status_parser_keeps_metric_names_with_digits(self):
        self.assertEqual(training_ops._kv("FAILS5M 3\n"), {"FAILS5M": "3"})

    def test_blocked_host_is_refused_before_ssh(self):
        with self.assertRaises(training_ops.Refused):
            training_ops._ssh("n79", "true")

    def test_h200_startup_grace_comes_from_operations_config(self):
        self.assertEqual(
            training_ops._configured_h200_engine_startup_grace(), 7200
        )
        with patch.object(
            training_ops, "_profile", return_value={"kind": "glm52"}
        ), patch.object(training_ops, "_ssh") as ssh, self.assertRaisesRegex(
            training_ops.Refused, "diagnose-glm52-parity"
        ):
            training_ops._repair_engine("h200a")
        ssh.assert_not_called()

    def test_h200_startup_grace_rejects_missing_or_invalid_config(self):
        for value in (None, True, 0, -1, "7200"):
            with patch.dict(training_ops.limits, {"h200_engine_startup_grace_seconds": value}, clear=True):
                with self.assertRaisesRegex(
                    training_ops.Refused, "positive integer"
                ):
                    training_ops._configured_h200_engine_startup_grace()

    def test_glm_qualification_is_fake_data_in_the_isolated_deployment(self):
        profile = training_ops._profile("glm52-native-small")
        with (training_ops.repo_root / profile["trainer"]).open("rb") as stream:
            config = tomllib.load(stream)
        self.assertIn("fake", config["data"])
        self.assertNotIn("rollout_cache", config["data"])
        self.assertEqual(
            training_ops.remote_repo,
            "/data/glm52-rl/deployment-glm52-optims",
        )
        self.assertEqual(training_ops.trainer_container, "glm52-optims-trainer")

    def test_policy_parity_profile_and_stage_are_registered(self):
        profile = training_ops._profile("glm52-policy-parity")
        self.assertEqual(
            profile["trainer"],
            "configs/qualification/glm52-policy-parity-trainer.toml",
        )
        with (training_ops.repo_root / profile["trainer"]).open("rb") as stream:
            config = tomllib.load(stream)
        stage = training_ops.glm_parity_paths["trainer_stage"]
        self.assertEqual(
            config["data"]["fake"]["reference_path"],
            f"{stage}/trainer-reference.json",
        )
        self.assertEqual(
            config["env_vars"]["PRIME_PARITY_ROUTER_TRACE_DIR"],
            "/data/glm52-rl/qualifications/glm52-policy-parity/trainer-traces",
        )
        self.assertEqual(
            config["env_vars"]["PRIME_PARITY_ROUTER_TRACE_SOURCE"],
            f"{stage}/trainer-router-trace",
        )
        self.assertEqual(
            set(training_ops.trainer_internal_addresses),
            set(training_ops.trainer_hosts),
        )
        self.assertEqual(
            training_ops.trainer_internal_addresses["n82"], "10.92.20.82"
        )

    def test_parity_python_missing_numpy_fails_before_host_copy(self):
        failed = subprocess.CompletedProcess(
            [], 1, stdout="", stderr="No module named numpy"
        )
        with (
            patch.object(training_ops.subprocess, "run", return_value=failed),
            patch.object(training_ops.os, "access", return_value=True),
            self.assertRaisesRegex(RuntimeError, "no usable parity Python"),
        ):
            training_ops._parity_python()

        with (
            patch.object(
                training_ops,
                "_parity_python",
                side_effect=RuntimeError("no usable parity Python"),
            ),
            patch.object(training_ops, "_copy_parity_host_evidence") as copy_hosts,
            self.assertRaisesRegex(RuntimeError, "no usable parity Python"),
        ):
            training_ops.collect_glm52_policy_parity("test-run", nodes=["n82"])
        copy_hosts.assert_not_called()

    def test_parity_commands_use_the_selected_interpreter(self):
        calls = []

        def run(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0)

        selected = "/usr/bin/python3.12"
        with patch.object(training_ops.subprocess, "run", side_effect=run):
            for action in ("collect-trainer", "capture-trainer", "compare"):
                training_ops._run_parity_command(
                    selected,
                    Path("scripts/glm52_policy_parity.py"),
                    Path("configs/qualification/glm52-policy-parity.toml"),
                    action,
                    ["output"],
                    check=action != "compare",
                )

        self.assertEqual(len(calls), 3)
        self.assertEqual([call[0][0] for call in calls], [selected] * 3)
        self.assertEqual(
            [call[0][4] for call in calls],
            ["collect-trainer", "capture-trainer", "compare"],
        )

    def test_parity_host_copies_overlap_and_keep_configured_order(self):
        nodes = ["n82", "n76", "n77"]
        trace_root = training_ops.parity_trace_root
        token_root = training_ops.parity_token_root
        first_trace_copy = threading.Barrier(len(nodes))
        calls = []
        lock = threading.Lock()

        def copy_tree(host, remote, local):
            with lock:
                calls.append((host, remote))
            if remote.startswith(trace_root):
                first_trace_copy.wait(timeout=2)
            local.mkdir(parents=True, exist_ok=True)
            if remote.startswith(trace_root):
                (local / "session.json").write_text("{}\n")
                (local / f"trainer-runtime-pp-0-host-{host}.json").write_text("{}\n")
            else:
                (local / "rank_0.jsonl").write_text("{}\n")

        with tempfile.TemporaryDirectory() as temporary:
            copied = training_ops._copy_parity_host_evidence(
                nodes,
                "run-1",
                Path(temporary),
                trace_root,
                token_root,
                nodes[-1],
                copy_tree,
            )

        self.assertEqual([host for host, _, _ in copied], nodes)
        self.assertEqual(
            {host for host, remote in calls if remote.startswith(trace_root)},
            set(nodes),
        )
        self.assertEqual(
            {host for host, remote in calls if remote == token_root},
            {nodes[-1]},
        )

    def test_parity_host_copies_fail_when_final_tokens_are_missing(self):
        nodes = ["n82", "n76", "n77"]
        trace_root = training_ops.parity_trace_root
        token_root = training_ops.parity_token_root

        def copy_tree(host, remote, local):
            if host == nodes[-1] and remote == token_root:
                raise RuntimeError("final token exports are missing")
            local.mkdir(parents=True, exist_ok=True)
            if remote == f"{trace_root}/run-1":
                (local / "session.json").write_text("{}\n")
                (local / f"trainer-runtime-pp-0-host-{host}.json").write_text("{}\n")

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "final token exports"):
                training_ops._copy_parity_host_evidence(
                    nodes,
                    "run-1",
                    Path(temporary),
                    trace_root,
                    token_root,
                    nodes[-1],
                    copy_tree,
                )

    def test_parity_tar_stream_uses_safe_argv_and_accepts_success(self):
        class Stream:
            def close(self):
                return None

        class Process:
            def __init__(self, stdout=None):
                self.stdout = stdout
                self.returncode = 0

            def communicate(self, timeout=None):
                del timeout
                return b"", b""

            def kill(self):
                return None

        calls = []
        processes = [Process(Stream()), Process()]

        def popen(argv, **kwargs):
            calls.append((argv, kwargs))
            return processes.pop(0)

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "trace"
            with patch.object(training_ops.subprocess, "Popen", side_effect=popen):
                training_ops._copy_parity_tree_stream(
                    "n82",
                    "/data/glm52-rl/qualifications/glm52-policy-parity/trainer-traces/run-1",
                    destination,
                )

        self.assertEqual(
            calls[0][0],
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "n82",
                "tar",
                "-C",
                "/data/glm52-rl/qualifications/glm52-policy-parity/trainer-traces/run-1",
                "-cf",
                "-",
                ".",
            ],
        )
        self.assertEqual(calls[1][0][0:3], ["tar", "--extract", "--file=-"])
        self.assertNotIn("shell", calls[0][1])
        self.assertNotIn("shell", calls[1][1])

    def test_parity_tar_stream_fails_on_remote_or_local_error(self):
        class Stream:
            def close(self):
                return None

        class Process:
            def __init__(self, returncode, stderr=b"", has_stdout=False):
                self.stdout = Stream() if has_stdout else None
                self.returncode = returncode
                self.stderr = stderr

            def communicate(self, timeout=None):
                del timeout
                return b"", self.stderr

            def kill(self):
                return None

        for remote_code, local_code, message in (
            (1, 0, "remote tar"),
            (0, 1, "local tar"),
        ):
            processes = [
                Process(remote_code, b"remote failed", has_stdout=True),
                Process(local_code, b"local failed"),
            ]

            def popen(argv, **kwargs):
                del argv, kwargs
                return processes.pop(0)

            with tempfile.TemporaryDirectory() as temporary:
                with patch.object(training_ops.subprocess, "Popen", side_effect=popen):
                    with self.assertRaisesRegex(RuntimeError, message):
                        training_ops._copy_parity_tree_stream(
                            "n82",
                            f"{training_ops.parity_trace_root}/run-1",
                            Path(temporary) / "trace",
                        )

    def test_policy_parity_stage_validation_is_fail_closed(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            tempfile.TemporaryDirectory() as repository,
        ):
            root = Path(repository)
            profile = root / "configs/qualification/glm52-policy-parity.toml"
            profile.parent.mkdir(parents=True)
            profile.write_text(
                "context_length = 2\n"
                "hidden_layers = 2\n"
                "dense_layers = 1\n"
                "experts_per_token = 2\n"
                "experts = 4\n"
                "router_scale = 2.5\n"
                "max_router_weight_sum_error = 0.005\n"
                'trainer_image = "trainer-image"\n'
            )
            source = root / "scripts/glm52_policy_parity.py"
            source.parent.mkdir()
            source.write_text("source\n")
            stage = Path(temporary)
            stage.chmod(0o750)
            reference = stage / "trainer-reference.json"
            session = stage / "trainer-router-trace/session.json"
            session.parent.mkdir()
            session.parent.chmod(0o755)
            route_bytes = bytes((0, 0, 0, 1, 0, 0, 2, 3))
            routed = {
                "schema_version": 1,
                "encoding": "base64",
                "dtype": "uint8",
                "shape": [2, 2, 2],
                "dense_layers": 1,
                "experts": 4,
                "sha256": hashlib.sha256(route_bytes).hexdigest(),
                "data": base64.b64encode(route_bytes).decode(),
            }
            token_bytes = struct.pack("<2q", 1, 2)
            token_sha256 = hashlib.sha256(token_bytes).hexdigest()
            reference_payload = {
                "fixture_sha256": "1" * 64,
                "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
                "model_revision": "3" * 40,
                "adapter_sha256": "4" * 64,
                "image": "h200-image",
                "h200_capture_manifest_sha256": "2" * 64,
                "capture_router_ids_sha256": "7" * 64,
                "capture_full_logprobs_sha256": "8" * 64,
                "input_ids": [1, 2],
                "routed_experts": routed,
            }
            session_payload = {
                "schema_version": 1,
                "side": "trainer",
                "fixture_sha256": "1" * 64,
                "token_ids_sha256": token_sha256,
                "model_revision": "3" * 40,
                "adapter_sha256": "4" * 64,
                "image": "trainer-image",
                "expected_tokens": 2,
                "hidden_layers": 2,
                "dense_layers": 1,
                "topk": 2,
                "experts": 4,
                "route_output_scale": 2.5,
                "weight_sum_tolerance": 0.005,
            }
            reference.write_text(json.dumps(reference_payload) + "\n")
            session.write_text(json.dumps(session_payload) + "\n")
            reference.chmod(0o644)
            session.chmod(0o644)
            manifest = {
                "schema_version": 1,
                "kind": "glm52_policy_parity_trainer_stage",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
                "fixture_sha256": "1" * 64,
                "h200_capture_manifest_sha256": "2" * 64,
                "route_output_scale": 2.5,
                "router_weight_provenance": (
                    "selected_weights_before_route_output_scale"
                ),
                "reference_schema": {
                    **{
                        name: value
                        for name, value in routed.items()
                        if name != "data"
                    },
                    "routed_experts_sha256": routed["sha256"],
                    "sha256": "7" * 64,
                    "full_logprobs_sha256": "8" * 64,
                },
                "identity": {
                    "fixture_sha256": "1" * 64,
                    "profile_sha256": hashlib.sha256(
                        profile.read_bytes()
                    ).hexdigest(),
                    "model_revision": "3" * 40,
                    "adapter_sha256": "4" * 64,
                    "h200_image": "h200-image",
                    "trainer_image": "trainer-image",
                    "token_ids_sha256": token_sha256,
                },
                "files": {
                    "trainer-reference.json": hashlib.sha256(
                        reference.read_bytes()
                    ).hexdigest(),
                    "trainer-router-trace/session.json": hashlib.sha256(
                        session.read_bytes()
                    ).hexdigest(),
                },
            }
            manifest_path = stage / "manifest.json"

            def write_manifest(value):
                manifest_path.write_text(json.dumps(value) + "\n")
                manifest_path.chmod(0o644)

            write_manifest(manifest)
            with patch.object(training_ops, "repo_root", root):
                training_ops._validate_glm_parity_stage(stage)
            missing_schema = dict(manifest)
            missing_schema.pop("reference_schema")
            write_manifest(missing_schema)
            with patch.object(training_ops, "repo_root", root):
                with self.assertRaisesRegex(RuntimeError, "reference schema"):
                    training_ops._validate_glm_parity_stage(stage)
            wrong_source = dict(manifest)
            wrong_source["source_sha256"] = "0" * 64
            write_manifest(wrong_source)
            with patch.object(training_ops, "repo_root", root):
                with self.assertRaisesRegex(RuntimeError, "source checksum"):
                    training_ops._validate_glm_parity_stage(stage)
            reference.write_text("{}\n")
            missing_routes = dict(manifest)
            missing_routes["files"] = dict(manifest["files"])
            missing_routes["files"]["trainer-reference.json"] = hashlib.sha256(
                reference.read_bytes()
            ).hexdigest()
            write_manifest(missing_routes)
            with patch.object(training_ops, "repo_root", root):
                with self.assertRaisesRegex(RuntimeError, "route schema"):
                    training_ops._validate_glm_parity_stage(stage)
            reference.write_text(json.dumps(reference_payload) + "\n")
            write_manifest(manifest)
            mixed_reference = dict(reference_payload)
            mixed_reference["adapter_sha256"] = "5" * 64
            reference.write_text(json.dumps(mixed_reference) + "\n")
            mixed_manifest = dict(manifest)
            mixed_manifest["files"] = dict(manifest["files"])
            mixed_manifest["files"]["trainer-reference.json"] = hashlib.sha256(
                reference.read_bytes()
            ).hexdigest()
            write_manifest(mixed_manifest)
            with patch.object(training_ops, "repo_root", root):
                with self.assertRaisesRegex(RuntimeError, "reference identity"):
                    training_ops._validate_glm_parity_stage(stage)
            reference.write_text(json.dumps(reference_payload) + "\n")
            mixed_session = dict(session_payload)
            mixed_session["model_revision"] = "6" * 40
            session.write_text(json.dumps(mixed_session) + "\n")
            mixed_manifest = dict(manifest)
            mixed_manifest["files"] = dict(manifest["files"])
            mixed_manifest["files"][
                "trainer-router-trace/session.json"
            ] = hashlib.sha256(session.read_bytes()).hexdigest()
            write_manifest(mixed_manifest)
            with patch.object(training_ops, "repo_root", root):
                with self.assertRaisesRegex(RuntimeError, "session identity"):
                    training_ops._validate_glm_parity_stage(stage)
            session.write_text(json.dumps(session_payload) + "\n")
            write_manifest(manifest)
            extra = stage / "unexpected.bin"
            extra.write_bytes(b"x")
            with patch.object(training_ops, "repo_root", root):
                with self.assertRaisesRegex(RuntimeError, "unexpected entries"):
                    training_ops._validate_glm_parity_stage(stage)
            extra.unlink()
            session.write_text("changed\n")
            with patch.object(training_ops, "repo_root", root):
                with self.assertRaisesRegex(RuntimeError, "checksum differs"):
                    training_ops._validate_glm_parity_stage(stage)

    def test_policy_parity_stage_refuses_symlink_and_extra_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            link = root / "stage"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "regular directory"):
                training_ops._validate_glm_parity_stage(link)

    def test_parity_staging_deploys_inventory_before_script(self):
        events = []

        def deploy(relative, nodes=None):
            events.append(("deploy", relative, nodes))
            return True

        def remote(host, command, timeout=120):
            del timeout
            events.append(("verify", host, command))
            relative = (
                "configs/operations.toml"
                if "configs/operations.toml" in command
                else "scripts/glm52_policy_parity.py"
            )
            digest = hashlib.sha256(
                (training_ops.repo_root / relative).read_bytes()
            ).hexdigest()
            return 0, digest + "\n"

        with (
            patch.object(training_ops, "deploy", side_effect=deploy),
            patch.object(training_ops, "_ssh", side_effect=remote),
        ):
            training_ops._deploy_glm_parity_staging_sources()

        self.assertEqual(
            [event[1] for event in events if event[0] == "deploy"],
            ["configs/operations.toml", "scripts/glm52_policy_parity.py"],
        )
        self.assertEqual(events[0][0:2], ("deploy", "configs/operations.toml"))
        self.assertEqual(events[1][0], "verify")
        self.assertEqual(events[2][0:2], ("deploy", "scripts/glm52_policy_parity.py"))

    def test_parity_staging_refuses_missing_or_mismatched_inventory(self):
        with (
            patch.object(training_ops, "deploy", return_value=True),
            patch.object(training_ops, "_ssh", return_value=(1, "missing")),
        ):
            with self.assertRaisesRegex(RuntimeError, "remote source is missing"):
                training_ops._deploy_glm_parity_staging_sources()
        with (
            patch.object(training_ops, "deploy", return_value=True),
            patch.object(training_ops, "_ssh", return_value=(0, "0" * 64)),
        ):
            with self.assertRaisesRegex(RuntimeError, "checksum differs"):
                training_ops._deploy_glm_parity_staging_sources()

    def test_parity_staging_verifies_sources_before_docker(self):
        source = Path(training_ops.__file__).read_text()
        method = source[source.index("def stage_glm52_policy_parity"):]
        deploy = method.index("_deploy_glm_parity_staging_sources()")
        docker = method.index("_glm_parity_stage_build_command(")
        self.assertLess(deploy, docker)

    def test_parity_stage_build_uses_an_ordinary_inode_bound_mount_for_docker(self):
        candidate = (
            f"{training_ops.glm_parity_stage_candidate_root}/"
            "trainer-stage-candidate-0123456789abcdef"
        )
        command = training_ops._glm_parity_stage_build_command(
            candidate,
            "10:20",
            training_ops.glm_parity_paths["fixture"],
            training_ops.glm_parity_paths["h200_capture"],
        )
        self.assertIn("os.O_PATH|os.O_DIRECTORY|os.O_NOFOLLOW", command)
        self.assertIn('source=f"/proc/{os.getpid()}/fd/{candidate_descriptor}"', command)
        self.assertIn("libc.mount(source.encode(),mount_path.encode(),None,MS_BIND,None)", command)
        self.assertIn('mount_path=f"/run/{mount_name}"', command)
        self.assertIn('f"{mount_path}:/stage"', command)
        self.assertNotIn('f"{source}:/stage"', command)
        self.assertIn("libc.umount2(mount_path.encode(),0)", command)
        self.assertLess(command.index("os.umask(0o077)"), command.index('"docker","run"'))
        self.assertIn("subprocess.run(command,timeout=1800,check=True)", command)
        self.assertIn("cleanup_container()", command)
        self.assertIn("cleanup_candidate(root,candidate)", command)
        self.assertIn("except BaseException as error", command)
        self.assertNotIn(f"-v {candidate}:/stage", command)
        helper = command.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        compile(helper, "glm-parity-stage-helper", "exec")
        self.assertGreaterEqual(
            training_ops.glm_parity_stage_outer_timeout,
            training_ops.glm_parity_stage_docker_timeout
            + 3 * training_ops.glm_parity_stage_inspect_timeout
            + training_ops.glm_parity_stage_remove_timeout
            + 60,
        )

    def test_parity_stage_build_retains_replaced_candidate_and_cleans_name_only_container(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = root / ".stage-candidates"
            candidates.mkdir(mode=0o755)
            candidate = candidates / "trainer-stage-candidate-0123456789abcdef"
            candidate.mkdir(mode=0o750)
            identity = candidate.stat()
            fake_bin = root / "bin"
            fake_bin.mkdir()
            state = root / "container-state"
            removed = root / "container-removed"
            fake_docker = fake_bin / "docker"
            fake_docker.write_text(
                """#!/usr/bin/env python3
import os
import pathlib
import sys

args = sys.argv[1:]
state = pathlib.Path(os.environ["FAKE_DOCKER_STATE"])
removed = pathlib.Path(os.environ["FAKE_DOCKER_REMOVED"])
container = os.environ["FAKE_DOCKER_CONTAINER"]
container_id = "a" * 64
if args[:2] == ["container", "inspect"]:
    if not state.exists():
        print(f"Error: No such container: {args[-1]}", file=sys.stderr)
        raise SystemExit(1)
    if "--format" in args:
        print(f"{container_id} /{container}")
    raise SystemExit(0)
if args[0] == "run":
    state.write_text("running\\n")
    source = next(value[:-7] for value in args if value.endswith(":/stage"))
    opened = pathlib.Path(source)
    retained = opened.with_name(opened.name + "-retained")
    opened.rename(retained)
    opened.mkdir(mode=0o750)
    raise SystemExit(9)
if args[0] == "rm":
    if args[-1] != container_id:
        raise SystemExit(2)
    state.unlink()
    removed.write_text(container_id + "\\n")
    raise SystemExit(0)
raise SystemExit(3)
"""
            )
            fake_docker.chmod(0o755)
            with patch.object(
                training_ops,
                "glm_parity_stage_candidate_root",
                str(candidates),
            ):
                command = training_ops._glm_parity_stage_build_command(
                    str(candidate),
                    f"{identity.st_dev}:{identity.st_ino}",
                    "/fixture",
                    "/capture",
                )
            command = command.replace("sudo python3", "python3", 1).replace(
                "root_opened.st_uid!=0", "root_opened.st_uid!=os.geteuid()"
            ).replace(
                "root_opened.st_gid!=0", "root_opened.st_gid!=os.getegid()"
            )
            command = command.replace(
                "  try:\n"
                "   bridge=prepare_mount_bridge(runtime,candidate)\n"
                "  except FileExistsError as error:\n"
                "   bridge_cleanup_ok=False\n"
                '   raise RuntimeError("stage mount bridge path already exists") from error\n'
                "  bridge_mounted=True\n",
                "  bridge=os.dup(candidate)\n"
                f"  mount_path={str(candidate)!r}\n"
                "  bridge_mounted=False\n",
            ).replace(
                "  current=os.stat(mount_name,dir_fd=runtime,follow_symlinks=False)\n"
                "  if (current.st_dev,current.st_ino)!=(wanted_dev,wanted_ino):\n"
                '   raise RuntimeError("stage mount bridge does not identify the candidate")\n',
                "  pass\n",
                1,
            ).replace(
                "  cleanup_mount_bridge(runtime,bridge,bridge_mounted)\n"
                "  bridge_mounted=False\n"
                "  os.close(bridge)\n"
                "  bridge=-1\n",
                "  os.close(bridge)\n"
                "  bridge=-1\n",
                1,
            )
            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
            environment["FAKE_DOCKER_STATE"] = str(state)
            environment["FAKE_DOCKER_REMOVED"] = str(removed)
            environment["FAKE_DOCKER_CONTAINER"] = (
                "glm52-policy-parity-stage-0123456789abcdef"
            )
            result = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                check=False,
                env=environment,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("returned non-zero exit status 9", result.stderr)
            self.assertTrue(candidate.is_dir())
            self.assertTrue(
                candidate.with_name(candidate.name + "-retained").is_dir()
            )
            self.assertEqual(removed.read_text().strip(), "a" * 64)

    def test_parity_stage_build_retains_evidence_on_transient_container_inspect_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = root / ".stage-candidates"
            candidates.mkdir(mode=0o755)
            candidate = candidates / "trainer-stage-candidate-0123456789abcdef"
            candidate.mkdir(mode=0o750)
            identity = candidate.stat()
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_docker = fake_bin / "docker"
            fake_docker.write_text(
                "#!/bin/sh\necho 'permission denied' >&2\nexit 1\n"
            )
            fake_docker.chmod(0o755)
            with patch.object(
                training_ops,
                "glm_parity_stage_candidate_root",
                str(candidates),
            ):
                command = training_ops._glm_parity_stage_build_command(
                    str(candidate),
                    f"{identity.st_dev}:{identity.st_ino}",
                    "/fixture",
                    "/capture",
                )
            command = command.replace("sudo python3", "python3", 1).replace(
                "root_opened.st_uid!=0", "root_opened.st_uid!=os.geteuid()"
            ).replace(
                "root_opened.st_gid!=0", "root_opened.st_gid!=os.getegid()"
            )
            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
            result = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                check=False,
                env=environment,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("stage container inspect failed: permission denied", result.stderr)
            self.assertIn(
                "stage candidate and container ID evidence retained", result.stderr
            )
            self.assertTrue(candidate.is_dir())

    def test_parity_stage_build_retains_candidate_used_by_preexisting_container(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = root / ".stage-candidates"
            candidates.mkdir(mode=0o755)
            candidate = candidates / "trainer-stage-candidate-0123456789abcdef"
            candidate.mkdir(mode=0o750)
            identity = candidate.stat()
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_docker = fake_bin / "docker"
            fake_docker.write_text(
                "#!/bin/sh\n"
                "if [ \"$1 $2\" = \"container inspect\" ]; then\n"
                " printf '%s\\n' \"$(printf 'a%.0s' $(seq 1 64)) /$5\"\n"
                " exit 0\n"
                "fi\n"
                "exit 99\n"
            )
            fake_docker.chmod(0o755)
            with patch.object(
                training_ops,
                "glm_parity_stage_candidate_root",
                str(candidates),
            ):
                command = training_ops._glm_parity_stage_build_command(
                    str(candidate),
                    f"{identity.st_dev}:{identity.st_ino}",
                    "/fixture",
                    "/capture",
                )
            command = command.replace("sudo python3", "python3", 1).replace(
                "root_opened.st_uid!=0", "root_opened.st_uid!=os.geteuid()"
            ).replace(
                "root_opened.st_gid!=0", "root_opened.st_gid!=os.getegid()"
            )
            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
            result = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                check=False,
                env=environment,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("container name is already in use", result.stderr)
            self.assertIn(
                "stage candidate and container ID evidence retained", result.stderr
            )
            self.assertTrue(candidate.is_dir())

    def test_parity_stage_build_reconciles_valid_stale_cid_before_docker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "run"
            runtime.mkdir(mode=0o755)
            candidates = root / ".stage-candidates"
            candidates.mkdir(mode=0o755)
            candidate = candidates / "trainer-stage-candidate-0123456789abcdef"
            candidate.mkdir(mode=0o750)
            identity = candidate.stat()
            container = "glm52-policy-parity-stage-0123456789abcdef"
            cidfile = runtime / f"{container}.cid"
            cidfile.write_bytes(b"a" * 64)
            cidfile.chmod(0o600)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            marker = root / "stale-cid-was-absent"
            fake_docker = fake_bin / "docker"
            fake_docker.write_text(
                """#!/usr/bin/env python3
import os
import pathlib
import sys

args = sys.argv[1:]
if args[:2] == ["container", "inspect"]:
    print(f"Error: No such container: {args[-1]}", file=sys.stderr)
    raise SystemExit(1)
if args[0] == "run":
    cidfile = pathlib.Path(args[args.index("--cidfile") + 1])
    if cidfile.exists():
        raise SystemExit(98)
    pathlib.Path(os.environ["FAKE_DOCKER_MARKER"]).write_text("absent\\n")
    cidfile.write_bytes(b"b" * 64)
    raise SystemExit(9)
raise SystemExit(99)
"""
            )
            fake_docker.chmod(0o755)
            with patch.object(
                training_ops,
                "glm_parity_stage_candidate_root",
                str(candidates),
            ):
                command = training_ops._glm_parity_stage_build_command(
                    str(candidate),
                    f"{identity.st_dev}:{identity.st_ino}",
                    "/fixture",
                    "/capture",
                )
            command = command.replace("sudo python3", "python3", 1).replace(
                "root_opened.st_uid!=0", "root_opened.st_uid!=os.geteuid()"
            ).replace(
                "root_opened.st_gid!=0", "root_opened.st_gid!=os.getegid()"
            ).replace(
                "runtime_opened.st_uid!=0", "runtime_opened.st_uid!=os.geteuid()"
            ).replace(
                "runtime_opened.st_gid!=0", "runtime_opened.st_gid!=os.getegid()"
            ).replace(
                "opened.st_uid!=0", "opened.st_uid!=os.geteuid()"
            ).replace(
                "opened.st_gid!=0", "opened.st_gid!=os.getegid()"
            ).replace(
                "details.st_uid!=0", "details.st_uid!=os.geteuid()"
            ).replace(
                'cidfile=f"/run/{container}.cid"',
                f'cidfile=f"{runtime}/{{container}}.cid"',
            ).replace(
                'mount_path=f"/run/{mount_name}"',
                f'mount_path=f"{runtime}/{{mount_name}}"',
            ).replace(
                'runtime=os.open("/run",os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)',
                f"runtime=os.open({str(runtime)!r},os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)",
            ).replace(
                'runtime_current=os.stat("/run",follow_symlinks=False)',
                f"runtime_current=os.stat({str(runtime)!r},follow_symlinks=False)",
            ).replace(
                "  try:\n"
                "   bridge=prepare_mount_bridge(runtime,candidate)\n"
                "  except FileExistsError as error:\n"
                "   bridge_cleanup_ok=False\n"
                '   raise RuntimeError("stage mount bridge path already exists") from error\n'
                "  bridge_mounted=True\n",
                "  bridge=os.dup(candidate)\n"
                f"  mount_path={str(candidate)!r}\n"
                "  bridge_mounted=False\n",
            ).replace(
                "  current=os.stat(mount_name,dir_fd=runtime,follow_symlinks=False)\n"
                "  if (current.st_dev,current.st_ino)!=(wanted_dev,wanted_ino):\n"
                '   raise RuntimeError("stage mount bridge does not identify the candidate")\n',
                "  pass\n",
                1,
            ).replace(
                "    cleanup_mount_bridge(runtime,bridge,bridge_mounted)\n"
                "    bridge_mounted=False\n"
                "    os.close(bridge)\n"
                "    bridge=-1\n",
                "    os.close(bridge)\n"
                "    bridge=-1\n",
                1,
            )
            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
            environment["FAKE_DOCKER_MARKER"] = str(marker)
            result = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                check=False,
                env=environment,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("returned non-zero exit status 9", result.stderr)
            self.assertEqual(marker.read_text(), "absent\n")
            self.assertFalse(cidfile.exists())
            self.assertFalse(candidate.exists())

    def test_parity_stage_build_retains_candidate_when_mount_rollback_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "run"
            runtime.mkdir(mode=0o755)
            candidates = root / ".stage-candidates"
            candidates.mkdir(mode=0o755)
            candidate = candidates / "trainer-stage-candidate-0123456789abcdef"
            candidate.mkdir(mode=0o750)
            identity = candidate.stat()
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_docker = fake_bin / "docker"
            fake_docker.write_text(
                "#!/bin/sh\n"
                "echo \"Error: No such container: $5\" >&2\n"
                "exit 1\n"
            )
            fake_docker.chmod(0o755)
            with patch.object(
                training_ops,
                "glm_parity_stage_candidate_root",
                str(candidates),
            ):
                command = training_ops._glm_parity_stage_build_command(
                    str(candidate),
                    f"{identity.st_dev}:{identity.st_ino}",
                    "/fixture",
                    "/capture",
                )
            command = command.replace("sudo python3", "python3", 1).replace(
                "root_opened.st_uid!=0", "root_opened.st_uid!=os.geteuid()"
            ).replace(
                "root_opened.st_gid!=0", "root_opened.st_gid!=os.getegid()"
            ).replace(
                "runtime_opened.st_uid!=0", "runtime_opened.st_uid!=os.geteuid()"
            ).replace(
                "runtime_opened.st_gid!=0", "runtime_opened.st_gid!=os.getegid()"
            ).replace(
                "opened.st_uid!=0", "opened.st_uid!=os.geteuid()"
            ).replace(
                "opened.st_gid!=0", "opened.st_gid!=os.getegid()"
            ).replace(
                'cidfile=f"/run/{container}.cid"',
                f'cidfile=f"{runtime}/{{container}}.cid"',
            ).replace(
                'mount_path=f"/run/{mount_name}"',
                f'mount_path=f"{runtime}/{{mount_name}}"',
            ).replace(
                'runtime=os.open("/run",os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)',
                f"runtime=os.open({str(runtime)!r},os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)",
            ).replace(
                'runtime_current=os.stat("/run",follow_symlinks=False)',
                f"runtime_current=os.stat({str(runtime)!r},follow_symlinks=False)",
            ).replace(
                "libc=ctypes.CDLL(None,use_errno=True)",
                "class FakeLibc:\n"
                " def mount(self,*_args): return 0\n"
                " def umount2(self,*_args): return -1\n"
                "libc=FakeLibc()",
            )
            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
            result = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                check=False,
                env=environment,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mount bridge does not identify", result.stderr)
            self.assertIn("mount bridge cleanup failed", result.stderr)
            self.assertIn(
                "stage candidate and container ID evidence retained", result.stderr
            )
            self.assertTrue(candidate.is_dir())
            self.assertTrue(
                (runtime / "glm52-policy-parity-stage-0123456789abcdef.mount").is_dir()
            )

    def test_parity_stage_build_refuses_preexisting_bridge_or_invalid_cid(self):
        for artifact in ("bridge", "cid"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                runtime = root / "run"
                runtime.mkdir(mode=0o755)
                candidates = root / ".stage-candidates"
                candidates.mkdir(mode=0o755)
                candidate = candidates / "trainer-stage-candidate-0123456789abcdef"
                candidate.mkdir(mode=0o750)
                identity = candidate.stat()
                container = "glm52-policy-parity-stage-0123456789abcdef"
                bridge = runtime / f"{container}.mount"
                cidfile = runtime / f"{container}.cid"
                if artifact == "bridge":
                    bridge.mkdir(mode=0o700)
                else:
                    cidfile.write_text("invalid\n")
                    cidfile.chmod(0o600)
                fake_bin = root / "bin"
                fake_bin.mkdir()
                calls = root / "docker-calls"
                fake_docker = fake_bin / "docker"
                fake_docker.write_text(
                    "#!/bin/sh\n"
                    f"printf '%s\\n' \"$*\" >> {str(calls)!r}\n"
                    "echo \"Error: No such container: $5\" >&2\n"
                    "exit 1\n"
                )
                fake_docker.chmod(0o755)
                with patch.object(
                    training_ops,
                    "glm_parity_stage_candidate_root",
                    str(candidates),
                ):
                    command = training_ops._glm_parity_stage_build_command(
                        str(candidate),
                        f"{identity.st_dev}:{identity.st_ino}",
                        "/fixture",
                        "/capture",
                    )
                command = command.replace("sudo python3", "python3", 1).replace(
                    "root_opened.st_uid!=0", "root_opened.st_uid!=os.geteuid()"
                ).replace(
                    "root_opened.st_gid!=0", "root_opened.st_gid!=os.getegid()"
                ).replace(
                    "runtime_opened.st_uid!=0", "runtime_opened.st_uid!=os.geteuid()"
                ).replace(
                    "runtime_opened.st_gid!=0", "runtime_opened.st_gid!=os.getegid()"
                ).replace(
                    "opened.st_uid!=0", "opened.st_uid!=os.geteuid()"
                ).replace(
                    "opened.st_gid!=0", "opened.st_gid!=os.getegid()"
                ).replace(
                    'cidfile=f"/run/{container}.cid"',
                    f'cidfile=f"{runtime}/{{container}}.cid"',
                ).replace(
                    'mount_path=f"/run/{mount_name}"',
                    f'mount_path=f"{runtime}/{{mount_name}}"',
                ).replace(
                    'runtime=os.open("/run",os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)',
                    f"runtime=os.open({str(runtime)!r},os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)",
                ).replace(
                    'runtime_current=os.stat("/run",follow_symlinks=False)',
                    f"runtime_current=os.stat({str(runtime)!r},follow_symlinks=False)",
                )
                environment = os.environ.copy()
                environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
                result = subprocess.run(
                    command,
                    shell=True,
                    executable="/bin/bash",
                    capture_output=True,
                    text=True,
                    check=False,
                    env=environment,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(candidate.is_dir())
                self.assertEqual(len(calls.read_text().splitlines()), 1)
                if artifact == "bridge":
                    self.assertIn("mount bridge path already exists", result.stderr)
                    self.assertTrue(bridge.is_dir())
                else:
                    self.assertIn("ID file identity differs", result.stderr)
                    self.assertTrue(cidfile.is_file())

    def test_parity_stage_install_is_exact_and_transactional(self):
        stage = training_ops.glm_parity_paths["trainer_stage"]
        candidate = (
            f"{training_ops.glm_parity_stage_candidate_root}/"
            "trainer-stage-candidate-0123456789abcdef"
        )
        command = training_ops._glm_parity_stage_install_command(
            stage, candidate, "a" * 64, "b" * 64, "c" * 64
        )
        self.assertTrue(command.startswith("sudo python3 - "))
        self.assertIn(
            f"sudo python3 - {training_ops.glm_parity_root} "
            f"{training_ops.glm_parity_stage_candidate_root} ",
            command,
        )
        self.assertNotIn("rm -rf", command)
        self.assertNotIn("/tmp/", command)
        self.assertIn("os.O_NOFOLLOW", command)
        self.assertIn("fcntl.flock", command)
        self.assertIn("fcntl.LOCK_EX|fcntl.LOCK_NB", command)
        self.assertIn("src_dir_fd=candidates,dst_dir_fd=root", command)
        self.assertIn('print("replaced-retained:"', command)
        self.assertIn("libc.renameat2", command)
        self.assertIn("libc.renameat2(root,stage.encode(),candidates", command)
        with self.assertRaisesRegex(training_ops.Refused, "candidate"):
            training_ops._glm_parity_stage_install_command(
                stage,
                "/tmp/trainer-stage-candidate-0123456789abcdef",
                "a" * 64,
                "b" * 64,
                "c" * 64,
            )

    def test_parity_stage_candidate_cleanup_is_exact_and_bounded(self):
        root = training_ops.glm_parity_root
        candidate = (
            f"{training_ops.glm_parity_stage_candidate_root}/"
            "trainer-stage-candidate-0123456789abcdef"
        )
        command = training_ops._glm_parity_stage_candidate_cleanup_command(
            candidate, "10:20"
        )
        self.assertIn("os.O_NOFOLLOW", command)
        self.assertIn("fcntl.flock", command)
        self.assertIn("fcntl.LOCK_EX|fcntl.LOCK_NB", command)
        self.assertIn("entries<=allowed", command)
        self.assertNotIn("rm -rf", command)
        with self.assertRaisesRegex(training_ops.Refused, "cleanup target"):
            training_ops._glm_parity_stage_candidate_cleanup_command(
                f"{root}/trainer-stage-final", "10:20"
            )

    def test_parity_stage_transaction_installs_reuses_and_replaces(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.chmod(0o750)
            stage = root / "trainer-stage-final"
            candidates = root / ".stage-candidates"
            candidates.mkdir(mode=0o755)

            def make_candidate(nonce, marker):
                candidate = candidates / f"trainer-stage-candidate-{nonce}"
                trace = candidate / "trainer-router-trace"
                trace.mkdir(parents=True, mode=0o755)
                candidate.chmod(0o750)
                reference = f"reference-{marker}\n".encode()
                session = f"session-{marker}\n".encode()
                (candidate / "trainer-reference.json").write_bytes(reference)
                (trace / "session.json").write_bytes(session)
                (candidate / "trainer-reference.json").chmod(0o644)
                (trace / "session.json").chmod(0o644)
                reference_sha = hashlib.sha256(reference).hexdigest()
                session_sha = hashlib.sha256(session).hexdigest()
                manifest = {
                    "schema_version": 1,
                    "kind": "glm52_policy_parity_trainer_stage",
                    "files": {
                        "trainer-reference.json": reference_sha,
                        "trainer-router-trace/session.json": session_sha,
                    },
                }
                manifest_bytes = (json.dumps(manifest) + "\n").encode()
                (candidate / "manifest.json").write_bytes(manifest_bytes)
                (candidate / "manifest.json").chmod(0o644)
                return (
                    candidate,
                    hashlib.sha256(manifest_bytes).hexdigest(),
                    reference_sha,
                    session_sha,
                )

            def install_command(values):
                candidate, manifest_sha, reference_sha, session_sha = values
                with patch.object(training_ops, "glm_parity_root", str(root)):
                    with patch.object(
                        training_ops,
                        "glm_parity_stage_candidate_root",
                        str(candidates),
                    ):
                        command = training_ops._glm_parity_stage_install_command(
                            str(stage),
                            str(candidate),
                            manifest_sha,
                            reference_sha,
                            session_sha,
                        )
                command = command.replace("sudo python3 -", "python3 -", 1)
                command = command.replace(
                    "candidate_root_opened.st_uid!=0",
                    "candidate_root_opened.st_uid!=os.geteuid()",
                ).replace(
                    "candidate_root_opened.st_gid!=0",
                    "candidate_root_opened.st_gid!=os.getegid()",
                )
                return command

            def install(values):
                command = install_command(values)
                return subprocess.run(
                    command,
                    shell=True,
                    executable="/bin/bash",
                    capture_output=True,
                    text=True,
                    check=False,
                )

            absent_values = make_candidate("0" * 16, "zero")
            absent_candidate = absent_values[0]
            absent_fault_command = install_command(absent_values).replace(
                "    installed_identity,installed_hashes=validate(root,stage,required)",
                '    raise RuntimeError("injected absent-stage validation fault")',
                1,
            )
            absent_fault = subprocess.run(
                absent_fault_command,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(absent_fault.returncode, 0)
            self.assertIn("injected absent-stage validation fault", absent_fault.stderr)
            self.assertFalse(stage.exists())
            self.assertTrue(absent_candidate.is_dir())
            first = install(make_candidate("1" * 16, "one"))
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(first.stdout.strip(), "installed")
            first_manifest = hashlib.sha256(
                (stage / "manifest.json").read_bytes()
            ).hexdigest()
            first_reference = hashlib.sha256(
                (stage / "trainer-reference.json").read_bytes()
            ).hexdigest()
            first_session = hashlib.sha256(
                (stage / "trainer-router-trace/session.json").read_bytes()
            ).hexdigest()
            with patch.object(training_ops, "glm_parity_root", str(root)):
                current_command = training_ops._glm_parity_stage_current_command(
                    str(stage),
                    first_manifest,
                    first_reference,
                    first_session,
                )
            current_check = subprocess.run(
                current_command,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(current_check.returncode, 0, current_check.stderr)
            self.assertEqual(current_check.stdout.strip(), "current")
            stage.chmod(0o777)
            unsafe_stage = subprocess.run(
                current_command,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(unsafe_stage.returncode, 0, unsafe_stage.stderr)
            self.assertEqual(unsafe_stage.stdout.strip(), "stale")
            stage.chmod(0o750)
            (stage / "trainer-reference.json").chmod(0o755)
            unsafe_file = subprocess.run(
                current_command,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(unsafe_file.returncode, 0, unsafe_file.stderr)
            self.assertEqual(unsafe_file.stdout.strip(), "stale")
            (stage / "trainer-reference.json").chmod(0o644)
            (stage / "trainer-router-trace").chmod(0o777)
            unsafe_trace = subprocess.run(
                current_command,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(unsafe_trace.returncode, 0, unsafe_trace.stderr)
            self.assertEqual(unsafe_trace.stdout.strip(), "stale")
            (stage / "trainer-router-trace").chmod(0o755)
            current = install(make_candidate("2" * 16, "one"))
            self.assertEqual(current.returncode, 0, current.stderr)
            self.assertEqual(current.stdout.strip(), "current")
            (stage / "trainer-router-trace").chmod(0o500)
            replacement_values = make_candidate("3" * 16, "two")
            retained = replacement_values[0]
            replaced = install(replacement_values)
            self.assertEqual(replaced.returncode, 0, replaced.stderr)
            self.assertEqual(
                replaced.stdout.strip(), f"replaced-retained:{retained.name}"
            )
            self.assertEqual(
                (stage / "trainer-reference.json").read_text(),
                "reference-two\n",
            )
            self.assertEqual(
                (retained / "trainer-reference.json").read_text(),
                "reference-one\n",
            )
            self.assertEqual(
                (retained / "trainer-router-trace").stat().st_mode & 0o777,
                0o500,
            )
            fault_values = make_candidate("4" * 16, "three")
            candidate, manifest_sha, reference_sha, session_sha = fault_values
            with patch.object(training_ops, "glm_parity_root", str(root)):
                with patch.object(
                    training_ops,
                    "glm_parity_stage_candidate_root",
                    str(candidates),
                ):
                    fault_command = training_ops._glm_parity_stage_install_command(
                        str(stage),
                        str(candidate),
                        manifest_sha,
                        reference_sha,
                        session_sha,
                    )
            fault_command = fault_command.replace("sudo python3 -", "python3 -", 1)
            fault_command = fault_command.replace(
                "candidate_root_opened.st_uid!=0",
                "candidate_root_opened.st_uid!=os.geteuid()",
            ).replace(
                "candidate_root_opened.st_gid!=0",
                "candidate_root_opened.st_gid!=os.getegid()",
            )
            fault_command = fault_command.replace(
                "  installed_identity,_=validate(root,stage,required)",
                '  raise RuntimeError("injected postcommit validation fault")',
                1,
            )
            fault = subprocess.run(
                fault_command,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(fault.returncode, 0)
            self.assertIn("injected postcommit validation fault", fault.stderr)
            self.assertEqual(
                (stage / "trainer-reference.json").read_text(),
                "reference-two\n",
            )
            self.assertEqual(
                (candidate / "trainer-reference.json").read_text(),
                "reference-three\n",
            )
            self.assertFalse(
                any("previous" in path.name for path in root.iterdir())
            )
            (retained / "trainer-router-trace").chmod(0o755)

    def test_parity_stage_copy_timeout_is_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(
                training_ops.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired("scp", 600),
            ):
                with self.assertRaisesRegex(RuntimeError, "copy timed out"):
                    training_ops._copy_glm_parity_stage(
                        "n82", training_ops.glm_parity_paths["trainer_stage"], Path(temporary)
                    )

    def test_parity_stage_copy_and_fanout_are_compressed_and_bounded(self):
        copied = subprocess.CompletedProcess([], 0, "", "")
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            training_ops.subprocess, "run", return_value=copied
        ) as run:
            training_ops._copy_glm_parity_stage(
                "n82",
                training_ops.glm_parity_paths["trainer_stage"],
                Path(temporary),
            )
        command = run.call_args.args[0]
        self.assertEqual(command[:6], ["scp", "-C", "-q", "-o", "BatchMode=yes", "-r"])
        self.assertEqual(run.call_args.kwargs["timeout"], 600)
        candidate = (
            f"{training_ops.glm_parity_stage_candidate_root}/"
            "trainer-stage-candidate-0123456789abcdef"
        )
        command = training_ops._glm_parity_transfer_command(
            "/source", "10.92.20.76", candidate, "10:20"
        )
        self.assertEqual(command[:5], [
            "timeout", "--signal=TERM", "--kill-after=5s", "540s", "rsync",
        ])
        for option in ("--compress", "--partial", "--append-verify"):
            self.assertIn(option, command)
        remote = next(value for value in command if value.startswith("--rsync-path="))
        self.assertIn("stat -c '%d:%i'", remote)
        self.assertIn("exec timeout --signal=TERM --kill-after=5s 500s", remote)
        self.assertIn(f"flock -- {candidate} sh -c", remote)
        self.assertIn("shift 2 && exec rsync", remote)
        self.assertGreater(remote.count("stat -c"), 1)
        self.assertNotIn("flock -n", remote)
        transport = command[command.index("-e") + 1]
        for option in (
            "StrictHostKeyChecking=yes",
            "UpdateHostKeys=no",
            "ControlMaster=no",
            "ControlPath=none",
        ):
            self.assertEqual(transport.count(option), 1)
        self.assertNotIn("accept-new", transport)
        self.assertNotIn("ControlPersist", transport)
        self.assertNotIn("--foreground", remote)
        self.assertEqual(command[-2:], ["/source/.", f"10.92.20.76:{candidate}/"])
        internal = training_ops._glm_parity_internal_transfer_command(
            training_ops.glm_parity_paths["trainer_stage"],
            "10.92.20.76",
            candidate,
            "10:20",
        )
        self.assertIn("tar --create --gzip", internal)
        self.assertIn("manifest.json trainer-reference.json trainer-router-trace", internal)
        self.assertIn("os.O_DIRECTORY|os.O_NOFOLLOW", internal)
        self.assertIn("fcntl.flock(descriptor,fcntl.LOCK_EX)", internal)
        self.assertIn("candidate identity changed after lock", internal)
        self.assertIn("/proc/self/fd/", internal)
        self.assertIn("timeout --signal=TERM --kill-after=5s 500s", internal)
        self.assertIn("timeout --signal=TERM --kill-after=5s 540s", internal)
        self.assertNotIn("rsync", internal)
        for option in (
            "StrictHostKeyChecking=yes",
            "UpdateHostKeys=no",
            "ControlMaster=no",
            "ControlPath=none",
        ):
            self.assertEqual(internal.count(option), 1)

    def test_parity_stage_create_ack_loss_still_runs_cleanup(self):
        source = Path(training_ops.__file__).read_text()
        method = source[source.index("def stage_glm52_policy_parity"):]
        candidate_exists = method.index("candidate_exists = True")
        candidate_create = method.index('"GLM parity candidate directory"')
        cleanup = method.index("_cleanup_glm_parity_stage_candidate")
        self.assertLess(candidate_exists, candidate_create)
        self.assertLess(candidate_create, cleanup)
        self.assertIn("except BaseException as error", method)
        self.assertIn("transfer_started", method)
        self.assertIn("transfer started; exact candidate was retained", method)

    def test_parity_stage_flow_does_not_use_manifest_only_or_broad_delete(self):
        source = Path(training_ops.__file__).read_text()
        method = source[source.index("def stage_glm52_policy_parity"):]
        self.assertIn("_validate_glm_parity_stage_live_identity", method)
        self.assertIn("_glm_parity_stage_install_command", method)
        self.assertIn("_glm_parity_stage_current_command", method)
        self.assertIn("_glm_parity_stage_build_command", method)
        self.assertNotIn("rm -rf", method)
        self.assertNotIn("/tmp/glm52-parity-stage", method)
        install = method[method.index("def install("):]
        self.assertLess(
            install.index("_glm_parity_stage_current_command"),
            install.index("_transfer_glm_parity_stage"),
        )
        self.assertLess(install.index("seed = install"), install.index("_fan([host"))
        self.assertEqual(install.count("_ssh(trainer_head, current_command)"), 2)
        self.assertIn("state.splitlines()[-1]", install)
        self.assertNotIn("install failed: {state[:120]}", install)

    def test_parity_stage_transfer_retries_same_partial_candidate(self):
        candidate = (
            f"{training_ops.glm_parity_stage_candidate_root}/"
            "trainer-stage-candidate-0123456789abcdef"
        )
        failed = subprocess.CompletedProcess([], 23, "", "partial")
        copied = subprocess.CompletedProcess([], 0, "", "")
        with patch.object(
            training_ops.subprocess,
            "run",
            side_effect=[failed, failed, copied],
        ) as run:
            result = training_ops._transfer_glm_parity_stage(
                "/source", "n82", candidate, "10:20"
            )
        self.assertEqual(result, (True, "ok"))
        self.assertEqual(run.call_count, 3)
        self.assertTrue(
            all(call.args[0] == run.call_args_list[0].args[0] for call in run.call_args_list)
        )
        self.assertTrue(all(call.kwargs["timeout"] == 580 for call in run.call_args_list))

    def test_parity_stage_internal_tar_extracts_through_locked_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            candidates = root / ".stage-candidates"
            candidate = candidates / "trainer-stage-candidate-0123456789abcdef"
            source_trace = source / "trainer-router-trace"
            source_trace.mkdir(parents=True)
            candidates.mkdir()
            candidate.mkdir(mode=0o750)
            (source / "manifest.json").write_text("manifest\n")
            (source / "trainer-reference.json").write_text("reference\n")
            (source_trace / "session.json").write_text("session\n")
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_ssh = fake_bin / "ssh"
            fake_ssh.write_text(
                "#!/bin/bash\nfor command; do :; done\nexec bash -c \"$command\"\n"
            )
            fake_ssh.chmod(0o755)
            identity = f"{candidate.stat().st_dev}:{candidate.stat().st_ino}"
            with (
                patch.dict(
                    training_ops.glm_parity_paths,
                    {"trainer_stage": str(source)},
                ),
                patch.object(
                    training_ops,
                    "glm_parity_stage_candidate_root",
                    str(candidates),
                ),
            ):
                command = training_ops._glm_parity_internal_transfer_command(
                    str(source), "10.92.20.76", str(candidate), identity
                )
            result = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                (candidate / "trainer-reference.json").read_text(),
                "reference\n",
            )
            self.assertEqual(
                (candidate / "trainer-router-trace/session.json").read_text(),
                "session\n",
            )
            self.assertEqual(candidate.stat().st_mode & 0o777, 0o750)
            self.assertEqual(
                (candidate / "trainer-router-trace").stat().st_mode & 0o777,
                0o755,
            )
            for relative in (
                "manifest.json",
                "trainer-reference.json",
                "trainer-router-trace/session.json",
            ):
                self.assertEqual((candidate / relative).stat().st_mode & 0o777, 0o644)

    def test_parity_stage_transfer_timeout_is_uncertain_and_not_retried(self):
        candidate = (
            f"{training_ops.glm_parity_stage_candidate_root}/"
            "trainer-stage-candidate-0123456789abcdef"
        )
        with patch.object(
            training_ops.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired("rsync", 600),
        ) as run:
            copied, detail = training_ops._transfer_glm_parity_stage(
                "/source", "n82", candidate, "10:20"
            )
        self.assertFalse(copied)
        self.assertIn("process state uncertain", detail)
        run.assert_called_once()
        with patch.object(
            training_ops, "_ssh", return_value=(124, "TIMEOUT")
        ) as ssh:
            expected = training_ops._glm_parity_internal_transfer_command(
                training_ops.glm_parity_paths["trainer_stage"],
                "10.92.20.76",
                candidate,
                "10:20",
            )
            copied, detail = training_ops._transfer_glm_parity_stage(
                training_ops.glm_parity_paths["trainer_stage"],
                "10.92.20.76",
                candidate,
                "10:20",
                "n82",
            )
        self.assertFalse(copied)
        self.assertIn("controller timeout", detail)
        ssh.assert_called_once()
        self.assertEqual(ssh.call_args.args, ("n82", expected))
        self.assertEqual(ssh.call_args.kwargs, {"timeout": 580})

    def test_parity_stage_transfer_preflight_has_no_fallback(self):
        missing = subprocess.CompletedProcess([], 127, "", "")
        with patch.object(training_ops.subprocess, "run", return_value=missing):
            self.assertIn(
                "unavailable",
                training_ops._glm_parity_transfer_preflight("n82"),
            )
        with patch.object(
            training_ops, "_ssh", return_value=(127, "missing tar")
        ) as ssh:
            self.assertEqual(
                training_ops._glm_parity_transfer_preflight(
                    "10.92.20.76", "n82"
                ),
                "missing tar",
            )
        self.assertNotIn("rsync", ssh.call_args.args[1])
        self.assertIn("command -v tar", ssh.call_args.args[1])
        self.assertIn("StrictHostKeyChecking=yes", ssh.call_args.args[1])
        self.assertIn("UpdateHostKeys=no", ssh.call_args.args[1])
        self.assertIn("ControlMaster=no", ssh.call_args.args[1])
        self.assertIn("ControlPath=none", ssh.call_args.args[1])
        self.assertNotIn("accept-new", ssh.call_args.args[1])
        source = Path(training_ops.__file__).read_text()
        transfer = source[source.index("def _glm_parity_transfer_preflight"):]
        self.assertNotIn("transfer=scp", transfer)

    def test_parity_stage_requires_n82_seed(self):
        with self.assertRaisesRegex(training_ops.Refused, "the head"):
            training_ops.stage_glm52_policy_parity(["n76"])

    def test_parity_stage_rsync_append_verify_repairs_partial_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            target.mkdir()
            payload = bytes(range(256)) * 64
            (source / "trainer-reference.json").write_bytes(payload)
            (target / "trainer-reference.json").write_bytes(
                b"wrong!!" + payload[7:8192]
            )
            copied = subprocess.run(
                [
                    "rsync", "--archive", "--partial", "--append-verify",
                    f"{source}/.", f"{target}/",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(copied.returncode, 0, copied.stderr)
            self.assertEqual(
                (target / "trainer-reference.json").read_bytes(), payload
            )

    def test_parity_stage_directory_creates_only_exact_missing_path(self):
        commands = []

        def remote(host, command, timeout=120):
            del host, timeout
            commands.append(command)
            return 0, "missing\n"

        with patch.object(training_ops, "_ssh", side_effect=remote):
            self.assertEqual(
                training_ops._prepare_glm_parity_stage_directory(), "missing"
            )
        command = commands[0]
        exact = training_ops.glm_parity_paths["trainer_stage"]
        self.assertNotIn('sudo install -d -o 1000 -g 1000 -m 0750 "$stage"', command)
        self.assertIn(f"stage={exact}", command)
        self.assertNotIn("chmod -R", command)
        self.assertNotIn("chown -R", command)

    def test_parity_stage_candidate_root_is_exact_and_not_user_writable(self):
        command = training_ops._glm_parity_stage_candidate_root_prepare_command()
        self.assertIn(training_ops.glm_parity_stage_candidate_root, command)
        self.assertIn('candidate_stat.st_uid!=0', command)
        self.assertIn('candidate_stat.st_gid!=0', command)
        self.assertIn('stat.S_IMODE(candidate_stat.st_mode)!=0o755', command)
        self.assertIn('os.O_NOFOLLOW', command)
        self.assertNotIn("chmod -R", command)
        self.assertNotIn("chown -R", command)

    def test_parity_stage_directory_accepts_correct_existing_path(self):
        with patch.object(training_ops, "_ssh", return_value=(0, "existing\n")):
            self.assertEqual(
                training_ops._prepare_glm_parity_stage_directory(), "existing"
            )
        with patch.object(training_ops, "_ssh", return_value=(0, "stale\n")):
            self.assertEqual(
                training_ops._prepare_glm_parity_stage_directory(), "stale"
            )

    def test_parity_stage_directory_refuses_bad_existing_paths(self):
        for detail in (
            "stage path is not a directory",
            "stage path is a symlink",
            "candidate root owner or mode differs: 1000:1000:755",
        ):
            with patch.object(training_ops, "_ssh", return_value=(44, detail)):
                with self.assertRaisesRegex(RuntimeError, "directory refused"):
                    training_ops._prepare_glm_parity_stage_directory()

    def test_parity_capture_directory_creates_only_exact_missing_path(self):
        commands = []

        def remote(host, command, timeout=120):
            del host, timeout
            commands.append(command)
            return 0, "created\n"

        with patch.object(training_ops, "_ssh", side_effect=remote):
            self.assertEqual(
                training_ops._prepare_glm_parity_capture_directory(), "created"
            )
        command = commands[0]
        exact = training_ops.glm_parity_paths["h200_capture"]
        self.assertIn(
            'sudo install -d -o 1000 -g 1000 -m 0750 "$capture"', command
        )
        self.assertIn(f"capture={exact}", command)
        self.assertNotIn("chmod -R", command)
        self.assertNotIn("chown -R", command)

    def test_parity_capture_directory_accepts_correct_existing_path(self):
        with patch.object(training_ops, "_ssh", return_value=(0, "existing\n")):
            self.assertEqual(
                training_ops._prepare_glm_parity_capture_directory(), "existing"
            )

    def test_parity_capture_directory_refuses_bad_existing_paths(self):
        for detail in (
            "capture owner or mode differs: 0:0:755",
            "capture has content but no manifest",
            "capture path is a symlink",
        ):
            with patch.object(training_ops, "_ssh", return_value=(44, detail)):
                with self.assertRaisesRegex(RuntimeError, "directory refused"):
                    training_ops._prepare_glm_parity_capture_directory()

    def test_qualification_refuses_a_live_trainer(self):
        with patch.object(
            training_ops,
            "_ssh",
            return_value=(0, training_ops.protected_trainer + "\n"),
        ), patch.object(training_ops, "_fan") as fan:
            with self.assertRaises(training_ops.Refused):
                training_ops._assert_live_services_are_down(
                    ["n82"], include_inference=False
                )
        fan.assert_not_called()

    def test_live_service_check_retries_one_transient_banner_timeout(self):
        self.assertEqual(
            training_ops._configured_live_service_check(),
            (2, 45, 49, 1, 7, 210, 105),
        )
        with patch.object(training_ops.time, "sleep") as sleep, patch.object(
            training_ops,
            "_ssh",
            side_effect=[
                (255, "Connection timed out during banner exchange"),
                (0, ""),
            ],
        ) as ssh:
            training_ops._assert_live_services_are_down(
                ["n82"], include_inference=False
            )

        self.assertEqual(ssh.call_count, 2)
        sleep.assert_called_once_with(1)
        for call in ssh.call_args_list:
            self.assertEqual(call.kwargs["connect_timeout"], 45)
            self.assertEqual(call.kwargs["timeout"], 49)

    def test_live_service_check_does_not_retry_a_hard_failure(self):
        with patch.object(
            training_ops, "_ssh", return_value=(255, "Permission denied")
        ) as ssh, patch.object(training_ops, "_fan") as fan, self.assertRaisesRegex(
            RuntimeError, "cannot check the live trainer"
        ):
            training_ops._assert_live_services_are_down(
                ["n82"], include_inference=False
            )

        ssh.assert_called_once()
        fan.assert_not_called()

    def test_live_service_check_banner_retry_is_bounded(self):
        with patch.object(training_ops.time, "sleep") as sleep, patch.object(
            training_ops,
            "_ssh",
            return_value=(255, "Connection timed out during banner exchange"),
        ) as ssh, self.assertRaisesRegex(
            RuntimeError, "cannot check the live trainer"
        ):
            training_ops._assert_live_services_are_down(
                ["n82"], include_inference=False
            )

        self.assertEqual(ssh.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_live_service_check_still_refuses_active_service_after_retry(self):
        with patch.object(training_ops.time, "sleep") as sleep, patch.object(
            training_ops,
            "_ssh",
            side_effect=[
                (255, "Connection timed out during banner exchange"),
                (0, training_ops.protected_trainer + "\n"),
            ],
        ) as ssh, patch.object(training_ops, "_fan") as fan, self.assertRaises(
            training_ops.Refused
        ):
            training_ops._assert_live_services_are_down(
                ["n82"], include_inference=False
            )

        self.assertEqual(ssh.call_count, 2)
        sleep.assert_called_once_with(1)
        fan.assert_not_called()

    def test_live_service_check_uses_configured_worker_bound(self):
        observed = []
        order = []

        def fan(hosts, function, workers=None):
            observed.append((list(hosts), workers, list(order)))
            return {host: function(host) for host in hosts}

        def ssh(host, script, timeout=120, connect_timeout=None):
            del script, timeout, connect_timeout
            order.append(host)
            return 0, ""

        with patch.object(training_ops, "_fan", side_effect=fan), patch.object(
            training_ops, "_ssh", side_effect=ssh
        ):
            training_ops._assert_live_services_are_down(
                ["n76", "n82", "n77"], include_inference=False
            )

        self.assertEqual(observed, [(["n76", "n77"], 7, ["n82"])])
        self.assertEqual(order, ["n82", "n76", "n77"])

    def test_live_service_check_refuses_aggregate_config_over_budget(self):
        original = dict(training_ops.limits["live_service_check"])
        for change, message in (
            ({"workers": 1}, "trainer service check worst case"),
            ({"inference_budget_seconds": 98}, "inference service check worst case"),
            ({"trainer_budget_seconds": 3000}, "trainer service check budget"),
        ):
            settings = {**original, **change}
            with self.subTest(change=change), patch.dict(
                training_ops.limits, {"live_service_check": settings}
            ), self.assertRaisesRegex(training_ops.Refused, message):
                training_ops._configured_live_service_check()

    def test_live_service_check_enforces_runtime_aggregate_deadline(self):
        def fan(hosts, function, workers=None):
            del workers
            return {host: function(host) for host in hosts}

        with patch.object(
            training_ops.time, "monotonic", side_effect=[0.0, 301.0]
        ), patch.object(training_ops, "_fan", side_effect=fan), patch.object(
            training_ops, "_ssh"
        ) as ssh, self.assertRaisesRegex(
            RuntimeError, "aggregate deadline"
        ):
            training_ops._assert_live_services_are_down(
                ["n82"], include_inference=False
            )

        ssh.assert_not_called()

    def test_qualification_launch_uses_isolated_inventory(self):
        calls = []

        def fake_ssh(host, command, timeout=120):
            calls.append((host, command))
            return 0, ""

        def fake_fan(hosts, function, workers=None):
            return {host: function(host) for host in hosts}

        with (
            patch.object(training_ops, "_ssh", side_effect=fake_ssh),
            patch.object(training_ops, "_fan", side_effect=fake_fan),
            patch.object(training_ops, "deploy", return_value=True) as deploy_file,
            patch.object(
                training_ops, "_ssh_control_masters", side_effect=lambda _hosts: nullcontext()
            ),
            patch.object(training_ops, "_assert_live_services_are_down"),
            patch.object(training_ops, "trainer_mount_preflight"),
        ):
            training_ops.qualify("glm52-native-small", nodes=["n82"])

        deploy_file.assert_any_call(
            "source/src/prime_rl/trainer/rl/token_export.py", nodes=["n82"]
        )
        launch = next(command for host, command in calls if "cluster.sh trainer" in command)
        self.assertIn("prime_deployment=/data/glm52-rl/deployment-glm52-optims", launch)
        self.assertIn("prime_image=prime-rl:glm52-v080-qual", launch)
        self.assertIn("prime_trainer_container=glm52-optims-trainer", launch)
        self.assertIn(
            "TRAINER_CONFIG=/data/glm52-rl/deployment-glm52-optims/"
            "configs/qualification/glm52-native-small.toml",
            launch,
        )
        self.assertIn("PRIME_NCCL_CHANNELS=16", launch)

    def test_qualification_launch_reports_launcher_error(self):
        def fake_ssh(host, command, timeout=120):
            del host, timeout
            if "cluster.sh trainer" in command:
                return 17, "parity trace directory is not the pinned base directory"
            return 0, ""

        def fake_fan(hosts, function, workers=None):
            del workers
            return {host: function(host) for host in hosts}

        with self.assertRaisesRegex(
            RuntimeError,
            "qualification trainer launch failed on.*parity trace directory",
        ):
            with (
                patch.object(training_ops, "_ssh", side_effect=fake_ssh),
                patch.object(training_ops, "_fan", side_effect=fake_fan),
                patch.object(training_ops, "deploy", return_value=True),
                patch.object(
                    training_ops, "_ssh_control_masters", side_effect=lambda _hosts: nullcontext()
                ),
                patch.object(training_ops, "_assert_live_services_are_down"),
                patch.object(training_ops, "trainer_mount_preflight"),
            ):
                training_ops.qualify("glm52-native-small", nodes=["n82"])

    def test_cluster_config_parser_preserves_empty_optional_fields(self):
        launcher = (
            training_ops.repo_root / "scripts/china/cluster.sh"
        ).read_text()
        self.assertIn("IFS='|' read -r", launcher)
        self.assertIn('print("|".join(str(value) for value in values))', launcher)
        self.assertIn(
            'raise SystemExit("trainer config value contains the parser separator")',
            launcher,
        )

    def test_qualification_removes_stopped_exact_trainer_before_launch(self):
        calls = []

        def fake_ssh(host, command, timeout=120):
            del timeout
            calls.append((host, command))
            if "docker ps -a --filter" in command:
                return 0, "glm52-optims-trainer exited\n"
            return 0, ""

        def fake_fan(hosts, function, workers=None):
            del workers
            return {host: function(host) for host in hosts}

        with (
            patch.object(training_ops, "_ssh", side_effect=fake_ssh),
            patch.object(training_ops, "_fan", side_effect=fake_fan),
            patch.object(training_ops, "deploy", return_value=True),
            patch.object(
                training_ops, "_ssh_control_masters", side_effect=lambda _hosts: nullcontext()
            ),
            patch.object(training_ops, "_assert_live_services_are_down"),
            patch.object(training_ops, "trainer_mount_preflight"),
            patch.object(training_ops.time, "sleep"),
        ):
            training_ops.qualify("glm52-native-small", nodes=["n82"])

        self.assertTrue(
            any(
                "sudo docker rm -f glm52-optims-trainer" in command
                for _host, command in calls
            )
        )

    def test_strict_qualification_verifies_running_container_identity(self):
        nodes = [training_ops.trainer_head] + [
            host for host in training_ops.trainer_hosts
            if host != training_ops.trainer_head
        ]
        images = {host: "sha256:" + str(rank) * 64 for rank, host in enumerate(nodes)}
        identity = training_ops.TrainerQualificationIdentity(
            environment=(
                "PRIME_QUALIFICATION_SESSION_ID=" + "a" * 32 + " "
                "PRIME_QUALIFICATION_PROFILE=glm52-native-b2048 "
                "PRIME_QUALIFICATION_CONFIG_SHA256=" + "b" * 64 + " "
                "PRIME_QUALIFICATION_SOURCE_SHA256=" + "c" * 64 + " "
                "PRIME_QUALIFICATION_SESSION_SHA256=" + "d" * 64 + " "
            ),
            session_id="a" * 32,
            profile="glm52-native-b2048",
            config_sha256="b" * 64,
            source_sha256="c" * 64,
            session_sha256="d" * 64,
            launcher_sha256="e" * 64,
            image_digests=tuple(images.items()),
        )

        def record(host):
            rank = nodes.index(host)
            return {
                "name": "/glm52-optims-trainer",
                "status": "running",
                "image": training_ops.trainer_image,
                "image_digest": images[host],
                "path": "/app/.venv/bin/torchrun",
                "command": [
                    "--role=trainer",
                    "--nnodes=8",
                    "--nproc-per-node=8",
                    f"--node-rank={rank}",
                    f"--local-addr={training_ops.trainer_internal_addresses[host]}",
                    "--master-addr="
                    + training_ops.trainer_internal_addresses[training_ops.trainer_head],
                    "--master-port=29500",
                    "--log-dir=/data/glm52-rl/runs/glm52-native-b2048/"
                    "run_glm52-native-b2048/logs/trainer/torchrun",
                    "--tee=3",
                    "--redirects=3",
                    "--local-ranks-filter=0",
                    "-m",
                    "prime_rl.trainer.rl.train",
                    "@",
                    "/data/glm52-rl/deployment-glm52-optims/configs/qualification/"
                    "glm52-native-b2048.toml",
                ],
                "environment": {
                    "PRIME_QUALIFICATION_SESSION_ID": identity.session_id,
                    "PRIME_QUALIFICATION_PROFILE": identity.profile,
                    "PRIME_QUALIFICATION_CONFIG_SHA256": identity.config_sha256,
                    "PRIME_QUALIFICATION_SOURCE_SHA256": identity.source_sha256,
                    "PRIME_QUALIFICATION_SESSION_SHA256": identity.session_sha256,
                    "PRIME_QUALIFICATION_RUNTIME_SOURCE_SHA256": identity.source_sha256,
                    "PRIME_QUALIFICATION_LAUNCHER_SHA256": identity.launcher_sha256,
                    "PRIME_TRAINER_IMAGE_DIGEST": images[host],
                    "PRIME_TRAINER_HOST": host,
                    "PRIME_TRAINER_HEAD": training_ops.trainer_head,
                    "PRIME_QUALIFICATION_SESSION_DEADLINE_SECONDS": "3600",
                    "PRIME_QUALIFICATION_STARTUP_DEADLINE_SECONDS": "900",
                    "PRIME_QUALIFICATION_STEP_DEADLINE_SECONDS": "900",
                    "PRIME_QUALIFICATION_PROFILER_FINALIZE_DEADLINE_SECONDS": "120",
                },
                "nsys_environment": {},
            }

        def run(mutation=None):
            calls = []

            def fake_ssh(host, command, timeout=120):
                del timeout
                calls.append((host, command))
                if "python3 -c" in command:
                    observed = record(host)
                    if host == training_ops.trainer_head and mutation is not None:
                        section, key = mutation
                        if section == "record":
                            observed[key] = "wrong"
                        else:
                            observed["environment"][key] = "wrong"
                    return 0, json.dumps(observed)
                return 0, ""

            def fake_fan(hosts, function, workers=None):
                del workers
                return {host: function(host) for host in hosts}

            with (
                patch.object(training_ops, "_ssh", side_effect=fake_ssh),
                patch.object(training_ops, "_fan", side_effect=fake_fan),
                patch.object(training_ops, "deploy", return_value=True),
                patch.object(
                    training_ops, "_ssh_control_masters", side_effect=lambda _hosts: nullcontext()
                ),
                patch.object(training_ops, "_assert_live_services_are_down"),
                patch.object(
                    training_ops,
                    "_trainer_qualification_source_files",
                    return_value={},
                ),
                patch.object(
                    training_ops,
                    "_prepare_trainer_qualification_session",
                    return_value=identity,
                ),
            ):
                training_ops.qualify("glm52-native-b2048", nodes=nodes)
            return calls

        calls = run()
        self.assertEqual(
            sum("python3 -c" in command for _host, command in calls), len(nodes)
        )
        failures = (
            ("record", "image"),
            ("record", "image_digest"),
            ("record", "command"),
            ("environment", "PRIME_QUALIFICATION_RUNTIME_SOURCE_SHA256"),
            ("environment", "PRIME_QUALIFICATION_LAUNCHER_SHA256"),
            ("environment", "PRIME_TRAINER_IMAGE_DIGEST"),
            ("environment", "PRIME_TRAINER_HOST"),
            ("environment", "PRIME_TRAINER_HEAD"),
        )
        for mutation in failures:
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                RuntimeError, "qualification identity mismatch"
            ):
                run(mutation)

    def test_nsys_qualification_identity_is_pinned_to_rank24_and_n78(self):
        nodes = ["n82", "n76", "n77", "n78", "n80", "n83", "n84", "n85"]
        launch_env = {
            "PRIME_NSYS_PROFILE": "1",
            "PRIME_CUDA_PROFILE": "1",
            "PRIME_CUDA_PROFILE_RANK": "24",
            "PRIME_CUDA_PROFILE_STEPS": "2",
        }
        output_dir = "/data/glm52-rl/qualifications/glm52-native-b2048-nsys-20260827"
        capture = training_ops._qualification_nsys_capture(
            "glm52-native-b2048-nsys", launch_env, nodes, output_dir
        )
        self.assertEqual(capture.rank, 24)
        self.assertEqual(capture.host, "n78")
        self.assertEqual(capture.output, output_dir + "/nsys-rank-24")

        observed = {
            "PRIME_NSYS_BIN": (
                "/opt/nvidia/nsight-systems/2025.5.2/"
                "target-linux-x64/nsys"
            ),
            "PRIME_NSYS_VERSION": "NVIDIA Nsight Systems version 2025.5.2",
            "PRIME_NSYS_OUTPUT": capture.output,
            "PRIME_NSYS_SESSION": "glmkernel-r24-12345",
        }
        self.assertEqual(
            training_ops._qualification_nsys_environment_mismatches(
                observed, capture, "n78"
            ),
            [],
        )
        self.assertEqual(
            training_ops._qualification_nsys_environment_mismatches(
                observed, capture, "n76"
            ),
            ["Nsight environment keys"],
        )
        for key, value, mismatch in (
            ("PRIME_NSYS_BIN", "/tmp/nsys", "Nsight binary"),
            ("PRIME_NSYS_VERSION", "unknown", "Nsight version"),
            ("PRIME_NSYS_OUTPUT", "/wrong/output", "Nsight output"),
            ("PRIME_NSYS_SESSION", "bad session", "Nsight session"),
        ):
            with self.subTest(key=key):
                invalid = dict(observed)
                invalid[key] = value
                self.assertIn(
                    mismatch,
                    " ".join(
                        training_ops._qualification_nsys_environment_mismatches(
                            invalid, capture, "n78"
                        )
                    ),
                )

        with self.assertRaisesRegex(
            training_ops.Refused, "capture rank is not 24"
        ):
            training_ops._qualification_nsys_capture(
                "glm52-native-b2048-nsys",
                {**launch_env, "PRIME_CUDA_PROFILE_RANK": "25"},
                nodes,
                output_dir,
            )
        with self.assertRaisesRegex(
            training_ops.Refused, "capture step is not 2"
        ):
            training_ops._qualification_nsys_capture(
                "glm52-native-b2048-nsys",
                {**launch_env, "PRIME_CUDA_PROFILE_STEPS": "3"},
                nodes,
                output_dir,
            )
        with self.assertRaisesRegex(
            training_ops.Refused, "must run on n78"
        ):
            training_ops._qualification_nsys_capture(
                "glm52-native-b2048-nsys",
                launch_env,
                ["n82", "n76", "n77", "n85", "n80", "n83", "n84", "n78"],
                output_dir,
            )

    def test_nsys_phase_record_path_is_allowlisted(self):
        profile = training_ops._profile("glm52-native-b2048-nsys")
        launch_env = training_ops._qualification_launch_env(profile)
        self.assertEqual(
            launch_env["PRIME_PHASE_PROFILE_RECORD_PATH"],
            "/data/glm52-rl/qualifications/glm52-native-b2048-nsys-20260827/"
            "phase-profile.json",
        )
        invalid = dict(profile)
        invalid["launch_env"] = {
            **profile["launch_env"],
            "PRIME_PHASE_PROFILE_RECORD_PATH": "/tmp/phase-profile.json",
        }
        with self.assertRaisesRegex(
            training_ops.Refused, "PRIME_PHASE_PROFILE_RECORD_PATH"
        ):
            training_ops._qualification_launch_env(invalid)

    def test_nsys_environment_rejects_extra_or_missing_worker_metadata(self):
        launch_env = {
            "PRIME_NSYS_PROFILE": "1",
            "PRIME_CUDA_PROFILE": "1",
            "PRIME_CUDA_PROFILE_RANK": "24",
            "PRIME_CUDA_PROFILE_STEPS": "2",
        }
        capture = training_ops._qualification_nsys_capture(
            "glm52-native-b2048-nsys",
            launch_env,
            ["n82", "n76", "n77", "n78", "n80", "n83", "n84", "n85"],
            "/data/glm52-rl/qualifications/glm52-native-b2048-nsys-20260827",
        )
        valid = {
            "PRIME_NSYS_BIN": (
                "/opt/nvidia/nsight-systems/2025.5.2/"
                "target-linux-x64/nsys"
            ),
            "PRIME_NSYS_VERSION": "2025.5.2",
            "PRIME_NSYS_OUTPUT": capture.output,
            "PRIME_NSYS_SESSION": "glmkernel-r24-12345",
        }
        self.assertIn(
            "Nsight environment keys",
            training_ops._qualification_nsys_environment_mismatches(
                {**valid, "PRIME_NSYS_EXTRA": "1"}, capture, "n78"
            ),
        )
        self.assertIn(
            "Nsight environment keys",
            training_ops._qualification_nsys_environment_mismatches(
                {key: value for key, value in valid.items() if key != "PRIME_NSYS_BIN"},
                capture,
                "n78",
            ),
        )

    def test_current_run_reports_absent_container(self):
        with patch.object(
            training_ops,
            "_ssh",
            return_value=(1, "Error: No such object: glm52-optims-trainer"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "no active trainer container glm52-optims-trainer"
            ):
                training_ops.current_run()

    def test_current_run_reports_observed_daemon_absent_container(self):
        with patch.object(
            training_ops,
            "_ssh",
            return_value=(
                1,
                "[]\nError response from daemon: No such container: "
                "glm52-optims-trainer",
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "no active trainer container glm52-optims-trainer"
            ):
                training_ops.current_run()

    def test_trainer_absence_detection_rejects_ambiguous_errors(self):
        observed = (
            "[]\nError response from daemon: No such container: "
            "glm52-optims-trainer"
        )
        failures = (
            (2, observed),
            (1, observed + "\nwarning"),
            (1, "Error response from daemon: No such container: other"),
        )
        for rc, output in failures:
            with (
                self.subTest(rc=rc, output=output),
                patch.object(training_ops, "_ssh", return_value=(rc, output)),
                self.assertRaisesRegex(RuntimeError, "cannot inspect trainer"),
            ):
                training_ops.current_run()

    def test_trainer_inspect_shell_preserves_exact_absent_error(self):
        captured = []

        def fake_ssh(host, command, timeout=120):
            captured.append(command)
            return 0, _trainer_inspect_record(["RUN=glm52-7"], [])

        with patch.object(training_ops, "_ssh", side_effect=fake_ssh):
            training_ops._trainer_container_config()
        with tempfile.TemporaryDirectory() as temporary:
            sudo = Path(temporary) / "sudo"
            sudo.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' '[]'\n"
                "printf '%s\\n' 'Error response from daemon: No such container: "
                "glm52-optims-trainer' >&2\n"
                "exit 1\n",
                encoding="utf-8",
            )
            sudo.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{temporary}:{environment['PATH']}"
            result = subprocess.run(
                ["bash", "-c", captured[0]],
                capture_output=True,
                text=True,
                timeout=10,
                env=environment,
                check=False,
            )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(
            result.stdout.strip(),
            "[]\nError response from daemon: No such container: "
            "glm52-optims-trainer",
        )
        self.assertEqual(result.stderr, "")

    def test_status_reports_no_active_run_after_stop(self):
        absent = training_ops._TrainerContainerAbsent("trainer is absent")
        with (
            patch.object(
                training_ops, "_trainer_container_config", side_effect=absent
            ) as inspect_trainer,
            patch.object(training_ops, "_ssh", return_value=(0, "")),
            patch.object(training_ops, "_fan", return_value={}),
        ):
            result = training_ops.status(quiet=True)
        self.assertEqual(result.run, "none")
        self.assertFalse(result.trainer_up)
        self.assertEqual(inspect_trainer.call_count, 1)

    def test_status_resolves_configured_container_for_normal_and_qualification(self):
        container_values = {
            "trainer": "normal-trainer",
            "glm52_qualification_trainer": "qualification-trainer",
        }
        for profile_name in ("current", "glm52-native-b2048"):
            with self.subTest(profile=profile_name):
                profile = training_ops._profile(profile_name)
                expected = container_values[profile["trainer_container"]]
                with patch.dict(training_ops.containers, container_values):
                    with (
                        patch.object(
                            training_ops,
                            "_trainer_container_config",
                            side_effect=training_ops._TrainerContainerAbsent("absent"),
                        ) as inspect_trainer,
                        patch.object(training_ops, "_ssh", return_value=(0, "")),
                        patch.object(training_ops, "_fan", return_value={}),
                    ):
                        result = training_ops.status(profile=profile_name, quiet=True)
                self.assertEqual(result.run, "none")
                inspect_trainer.assert_called_once_with(expected)

    def test_profile_trainer_container_rejects_unconfigured_key(self):
        with self.assertRaisesRegex(
            training_ops.Refused, "not a configured container"
        ):
            training_ops._profile_trainer_container(
                {"trainer_container": "missing"}
            )

    def test_engines_do_not_require_an_active_trainer_after_stop(self):
        with (
            patch.object(training_ops, "_trainer_container_config") as inspect_trainer,
            patch.object(training_ops, "_fan", return_value={}),
        ):
            self.assertEqual(training_ops.engines(), {})
        inspect_trainer.assert_not_called()

    def test_current_run_reports_ssh_error(self):
        with patch.object(
            training_ops,
            "_ssh",
            return_value=(255, "ssh: connect to host n82: Connection timed out"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Connection timed out"):
                training_ops.current_run()

    def test_current_run_uses_valid_run_record(self):
        record = _trainer_inspect_record(
            ["RUN=glm52-7"],
            [
                "--role=trainer",
                "--log-dir=/data/glm52-rl/runs/glm52-7/run_glm52-7/logs/trainer/torchrun",
            ],
        )
        with patch.object(training_ops, "_ssh", return_value=(0, record)) as ssh:
            self.assertEqual(training_ops.current_run(), "glm52-7")
        self.assertEqual(ssh.call_count, 1)
        self.assertIn("docker inspect --type container", ssh.call_args.args[1])
        self.assertNotIn("2>/dev/null", ssh.call_args.args[1])

    def test_trainer_inspect_filters_other_environment_values_remotely(self):
        captured = []

        def fake_ssh(host, command, timeout=120):
            captured.append(command)
            return 0, _trainer_inspect_record(["RUN=glm52-7"], [])

        with patch.object(training_ops, "_ssh", side_effect=fake_ssh):
            training_ops._trainer_container_config()
        secret = "SECRET=must-not-cross-ssh"
        payload = json.dumps(
            [
                {
                    "Config": {
                        "Env": ["RUN=glm52-7", secret],
                        "Cmd": ["--role=trainer"],
                    }
                }
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            sudo = Path(temporary) / "sudo"
            sudo.write_text(
                "#!/usr/bin/env python3\n"
                f"print({payload!r})\n",
                encoding="utf-8",
            )
            sudo.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{temporary}:{environment['PATH']}"
            result = subprocess.run(
                ["bash", "-c", captured[0]],
                capture_output=True,
                text=True,
                timeout=10,
                env=environment,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(secret, captured[0])
        self.assertNotIn(secret, result.stdout)
        self.assertEqual(
            json.loads(result.stdout),
            {"run_records": ["RUN=glm52-7"], "command": ["--role=trainer"]},
        )

    def test_current_run_uses_trainer_log_path_when_run_env_is_absent(self):
        record = _trainer_inspect_record(
            [],
            [
                "--role=trainer",
                "--log-dir=/data/glm52-rl/runs/muse19/run_muse19/logs/trainer/torchrun",
            ],
        )
        with patch.object(training_ops, "_ssh", return_value=(0, record)):
            self.assertEqual(training_ops.current_run(), "muse19")

    def test_current_run_rejects_conflicting_records(self):
        records = (
            _trainer_inspect_record(["RUN=glm52-7", "RUN=glm52-8"], []),
            _trainer_inspect_record(
                ["RUN=glm52-7"],
                ["--log-dir=/data/glm52-rl/runs/glm52-8/run_glm52-8/logs"]
            ),
            _trainer_inspect_record(
                [],
                [
                    "--log-dir=/data/glm52-rl/runs/glm52-7/logs",
                    "--log-dir=/data/glm52-rl/runs/glm52-8/logs",
                ],
            ),
        )
        for record in records:
            with self.subTest(record=record), patch.object(
                training_ops, "_ssh", return_value=(0, record)
            ):
                with self.assertRaisesRegex(RuntimeError, "conflicting"):
                    training_ops.current_run()

    def test_current_run_rejects_invalid_records(self):
        records = (
            "not-json",
            json.dumps({"RUN": "glm52-7"}),
            _trainer_inspect_record(["RUN=../glm52-7"], []),
            _trainer_inspect_record([], ["--log-dir=/tmp/glm52-7/logs"]),
            _trainer_inspect_record(
                [],
                ["--log-dir=/data/glm52-rl/runs/glm52-7/../glm52-8/logs"]
            ),
            _trainer_inspect_record([], []),
        )
        for record in records:
            with self.subTest(record=record), patch.object(
                training_ops, "_ssh", return_value=(0, record)
            ):
                with self.assertRaises(RuntimeError):
                    training_ops.current_run()

    def test_trainer_mode_reports_absent_container(self):
        with patch.object(
            training_ops,
            "_ssh",
            return_value=(1, "Error: No such object: glm52-optims-trainer"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "no active trainer container glm52-optims-trainer"
            ):
                training_ops._trainer_mode()

    def test_trainer_mode_reports_inspect_ssh_error(self):
        with patch.object(
            training_ops,
            "_ssh",
            return_value=(255, "ssh: connect to host n82: Connection timed out"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Connection timed out"):
                training_ops._trainer_mode()

    def test_trainer_mode_reads_the_inspected_config(self):
        config = (
            "/data/glm52-rl/deployment-glm52-optims/"
            "configs/qualification/trainer.toml"
        )
        record = _trainer_inspect_record(
            [],
            ["-m", "prime_rl.trainer.rl.train", "@", config]
        )
        with patch.object(
            training_ops,
            "_ssh",
            side_effect=[(0, record), (0, "trainer-only\n")],
        ) as ssh:
            self.assertEqual(training_ops._trainer_mode(), "trainer-only")
        self.assertEqual(ssh.call_count, 2)
        self.assertIn(
            f"grep -q '^\\[data\\.fake\\]' -- {config}",
            ssh.call_args_list[1].args[1],
        )

    def test_trainer_mode_rejects_invalid_or_conflicting_configs(self):
        commands = (
            [],
            [
                "/data/glm52-rl/deployment-glm52-optims/configs/first.toml",
                "/data/glm52-rl/deployment-glm52-optims/configs/second.toml",
            ],
        )
        for command in commands:
            record = _trainer_inspect_record([], command)
            with self.subTest(command=command), patch.object(
                training_ops, "_ssh", return_value=(0, record)
            ) as ssh:
                with self.assertRaisesRegex(RuntimeError, "one TOML config"):
                    training_ops._trainer_mode()
                self.assertEqual(ssh.call_count, 1)

    def test_trainer_mode_rejects_malformed_environment_or_command(self):
        records = (
            json.dumps({"run_records": {"RUN": "glm52-7"}, "command": []}),
            _trainer_inspect_record(
                [],
                [
                    "/data/glm52-rl/deployment-glm52-optims/"
                    "configs/qualification/trainer.toml"
                ],
            ),
            _trainer_inspect_record([], ["relative/trainer.toml"]),
            _trainer_inspect_record(
                [],
                ["/data/glm52-rl/deployment-glm52-optims/configs/../bad.toml"],
            ),
            _trainer_inspect_record([], ["/tmp/trainer.toml"]),
        )
        for record in records:
            with self.subTest(record=record), patch.object(
                training_ops, "_ssh", return_value=(0, record)
            ) as ssh:
                with self.assertRaises(RuntimeError):
                    training_ops._trainer_mode()
                self.assertEqual(ssh.call_count, 1)

    def test_start_uses_edge_launcher_and_never_starts_china_orchestrator(self):
        calls = []

        def fake_ssh(host, command, timeout=120):
            calls.append((host, command))
            if command.startswith("test -f "):
                return 0, ""
            return 0, "done\n"

        with (
            patch.object(training_ops, "_ssh", side_effect=fake_ssh),
            patch.object(training_ops, "_fan", return_value={}),
            patch.object(training_ops, "deploy", return_value=True) as deploy_file,
            patch.object(training_ops, "_assert_live_services_are_down"),
            patch.object(training_ops, "trainer_mount_preflight"),
            patch.object(training_ops, "production_start_preflight"),
            patch.object(training_ops.time, "sleep"),
        ):
            training_ops.start("glm52-1", mode="cold", supervisors=False)

        ranked_trainers = [training_ops.trainer_head] + [
            host
            for host in training_ops.trainer_hosts
            if host != training_ops.trainer_head
        ]
        deploy_file.assert_any_call(
            "source/src/prime_rl/trainer/rl/token_export.py",
            nodes=ranked_trainers,
        )
        deploy_file.assert_any_call(
            "source/deps/renderers/renderers/base.py",
            nodes=[training_ops.orchestrator_host],
        )
        deploy_file.assert_any_call(
            "scripts/h200/glm_adapter_tunnels.py",
            nodes=[training_ops.orchestrator_host],
        )
        deploy_file.assert_any_call(
            "configs/operations.toml",
            nodes=[training_ops.orchestrator_host],
        )
        self.assertTrue(
            any(
                host == training_ops.orchestrator_host
                and "scripts/h200/start_orchestrator.sh" in command
                and (
                    "ORCH_CONFIG="
                    + training_ops.remote_repo
                    + "/configs/paused/orchestrator.glm52-production.toml"
                )
                in command
                for host, command in calls
            )
        )
        self.assertFalse(
            any(
                host == training_ops.trainer_head
                and "cluster.sh orchestrator" in command
                for host, command in calls
            )
        )

    def test_start_refuses_a_run_name_that_does_not_match_profile_paths(self):
        with self.assertRaises(training_ops.Refused):
            training_ops.start("wrong-run-name", supervisors=False)

    def test_stop_removes_edge_orchestrator(self):
        calls = []

        def fake_ssh(host, command, timeout=120):
            calls.append((host, command))
            return 0, ""

        with (
            patch.object(training_ops, "_ssh", side_effect=fake_ssh),
            patch.object(training_ops, "_fan", return_value={}),
        ):
            training_ops.stop()

        self.assertTrue(
            any(
                host == training_ops.orchestrator_host
                and training_ops.orchestrator_container in command
                and training_ops.environment_container in command
                for host, command in calls
            )
        )
        self.assertTrue(
            any(
                host == training_ops.orchestrator_host
                and "glm_adapter_tunnels.py --operations" in command
                and command.endswith(" stop; elif test -e /data/glm52-rl/glm-adapter-tunnels.json; then echo 'GLM adapter tunnel state exists without its exact manager or inventory' >&2; exit 44; fi")
                for host, command in calls
            )
        )

    def test_status_reports_glm_adapter_without_using_muse_health(self):
        status = training_ops.Status(
            kind="glm52",
            mode="rl",
            adapter="000",
            glm_adapter="200",
            glm_tunnels_recorded=8,
            glm_tunnels_healthy=8,
            glm_tunnels_committed=True,
        )
        warnings = training_ops._warnings(status)
        report = status.report()
        self.assertFalse(any("Muse adapter" in warning for warning in warnings))
        self.assertFalse(any("GLM gzip adapter" in warning for warning in warnings))
        self.assertFalse(any("GLM adapter tunnels" in warning for warning in warnings))
        self.assertIn("Muse/raw 8100 000", report)
        self.assertIn("GLM/gzip 8101 200", report)
        self.assertIn("GLM tunnels 8/8 recorded, committed=yes", report)
        self.assertIn("http://127.0.0.1:8101/", training_ops._head_script())
        self.assertIn(
            "glm_adapter_tunnels.py --operations "
            + training_ops.remote_repo
            + "/configs/operations.toml status",
            training_ops._orchestrator_script(),
        )


if __name__ == "__main__":
    unittest.main()
