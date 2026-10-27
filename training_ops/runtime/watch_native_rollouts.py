"""Read native rollout logs without importing operational modules."""

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import selectors
import shlex
import socket
import sys
import time


def read_json(path):
    with path.open() as stream:
        data = stream.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError(f"Process record exceeds 1 MiB: {path}")
    return json.loads(data)


def alive(state):
    try:
        pid = int(state["pid"])
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] != "Z" and fields[19] == str(state["start_ticks"])
    except (OSError, ValueError, KeyError, IndexError):
        return False


def training_state(root):
    state = read_json(root / "launcher/ops/process.json")
    manifest = read_json(root / "launcher/ops/manifest.json")
    if state.get("run") != root.name or manifest.get("run") != root.name:
        raise ValueError("run does not match its process record")
    if Path(manifest.get("output_dir", "")).resolve() != root.resolve():
        raise ValueError("run does not match its output directory")
    command = manifest.get("command", [])
    if not isinstance(command, list) or not all(isinstance(arg, str) for arg in command):
        raise ValueError("invalid training command record")
    # Native launchers use either uv argv or a shell command ending in rl @ CONFIG.
    words = [word for arg in command for word in shlex.split(arg)]
    if not any(Path(a).name == "rl" and b == "@" for a, b in zip(words, words[1:])):
        raise ValueError("process record is not a native RL training command")
    return state


def select_run(root, run=None):
    if run:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run):
            raise ValueError("run must be a single run name")
        selected = root / run
        return selected, training_state(selected)
    candidates = []
    with os.scandir(root) as entries:
        directories = []
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                directories.append(Path(entry.path))
            if len(directories) > 512:
                raise ValueError("More than 512 run directories. Select one with --run.")
    for selected in directories:
        try:
            state = training_state(selected)
            if alive(state):
                candidates.append((float(state["started_at"]), selected, state))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if not candidates:
        raise ValueError("No live training run was found. Use --run to inspect a saved run.")
    _, selected, state = max(candidates, key=lambda item: item[0])
    return selected, state


def log_path(root, role="orchestrator"):
    candidates = []
    for path in root.glob(f"logs/attempt_*/{role}.log"):
        match = re.fullmatch(r"attempt_(\d+)", path.parent.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise ValueError(f"No {role} log was found for {root.name}.")
    return max(candidates)[1]


def format_line(line):
    try:
        row = json.loads(line)
    except ValueError:
        row = None
    if isinstance(row, dict):
        record = row.get("record", row)
        if isinstance(record, dict) and "message" in record:
            stamp = record.get("time", record.get("timestamp", record.get("asctime", "")))
            if isinstance(stamp, dict):
                stamp = stamp.get("repr", "")
            level = record.get("level", record.get("levelname", ""))
            if isinstance(level, dict):
                level = level.get("name", "")
            line = " ".join(str(value) for value in (stamp, level, record["message"]) if value)
        elif isinstance(row.get("text"), str):
            line = row["text"]
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line).rstrip("\r\n")


def watch(root, state, lines=30, follow=True, seconds=600, role="orchestrator"):
    path = log_path(root, role)
    deadline = time.monotonic() + seconds
    print(f"Run {root.name}\nLog {path}", flush=True)
    with path.open("rb") as stream:
        # Bound the initial read even when the log contains millions of lines.
        stream.seek(0, 2)
        offset = max(0, stream.tell() - 1024 * 1024)
        stream.seek(offset)
        if offset:
            stream.readline()
        data = stream.read()
        chunks = data.split(b"\n")
        pending = chunks.pop()
        for line in deque(chunks, maxlen=lines):
            print(format_line(line.decode(errors="replace")), flush=True)
        if not follow:
            if pending:
                print(format_line(pending.decode(errors="replace")), flush=True)
            return 0
        while True:
            running = alive(state)
            data = pending + stream.read(1024 * 1024)
            chunks = data.split(b"\n")
            pending = chunks.pop()
            for line in chunks:
                print(format_line(line.decode(errors="replace")), flush=True)
            if not running:
                if pending:
                    print(format_line(pending.decode(errors="replace")), flush=True)
                try:
                    code = read_json(root / "launcher/ops/result.json")["returncode"]
                except (OSError, ValueError, KeyError):
                    print("Training process stopped without a result record.", flush=True)
                    return 1
                print(f"Training process stopped with exit code {code}.", flush=True)
                return int(code)
            if time.monotonic() >= deadline:
                print(f"Log watch reached its {seconds:g} second limit.", flush=True)
                return 0
            time.sleep(0.5)


def tail(path, limit):
    try:
        with path.open("rb") as stream:
            size = stream.seek(0, 2)
            start = max(0, size - limit)
            stream.seek(start)
            data = stream.read(limit)
        if start:
            data = data.partition(b"\n")[2]
        # A writer can still be in the middle of its last record.
        return data.rsplit(b"\n", 1)[0].decode(errors="replace"), size, start > 0
    except FileNotFoundError:
        return "", None, False


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def shown(value):
    return f"{value:.5g}" if finite(value) else "unknown"


def probe_health(ports, seconds=2):
    """Read local HTTP status lines with one deadline shared by all ports."""
    deadline = time.monotonic() + seconds
    results = {port: "deadline exceeded" for port in ports}
    pending = {}
    with selectors.DefaultSelector() as selector:
        try:
            for port in ports:
                connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    connection.setblocking(False)
                    connection.connect_ex(("127.0.0.1", port))
                    request = f"GET /health HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n".encode()
                    pending[connection] = {"port": port, "request": request, "response": b""}
                    selector.register(connection, selectors.EVENT_WRITE)
                except OSError as error:
                    results[port] = str(error)
                    pending.pop(connection, None)
                    connection.close()
            while pending and time.monotonic() < deadline:
                for key, events in selector.select(max(0, deadline - time.monotonic())):
                    connection = key.fileobj
                    state = pending[connection]
                    done = False
                    try:
                        if events & selectors.EVENT_WRITE:
                            error = connection.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                            if error:
                                raise OSError(error, os.strerror(error))
                            sent = connection.send(state["request"])
                            state["request"] = state["request"][sent:]
                            if not state["request"]:
                                selector.modify(connection, selectors.EVENT_READ)
                        else:
                            data = connection.recv(4096 - len(state["response"]))
                            state["response"] += data
                            if b"\n" in state["response"] or not data or len(state["response"]) >= 4096:
                                line = state["response"].split(b"\n", 1)[0]
                                match = re.match(rb"HTTP/1\.[01] ([0-9]{3})(?: |\r|$)", line)
                                code = int(match[1]) if match else None
                                results[state["port"]] = None if code is not None and 200 <= code < 300 else (
                                    f"HTTP {code}" if code is not None else "invalid HTTP response")
                                done = True
                    except BlockingIOError:
                        continue
                    except OSError as error:
                        results[state["port"]] = str(error)
                        done = True
                    if done:
                        selector.unregister(connection)
                        pending.pop(connection)
                        connection.close()
        finally:
            for connection in pending:
                connection.close()
    return results


def service_health(settings):
    configured = {}
    for role in ("policy", "judge"):
        ports = settings.get(role)
        if isinstance(ports, list) and 0 < len(ports) <= 32 and all(type(port) is int and 1 <= port <= 65535 for port in ports):
            configured[role] = list(dict.fromkeys(ports))
    ports = list(dict.fromkeys(port for values in configured.values() for port in values))
    results = probe_health(ports) if ports else {}
    print("Service health (local HTTP /health, 2 second total deadline):")
    for role in ("policy", "judge"):
        if role not in configured:
            print(f"  {role}: unknown (health ports are missing or invalid)")
            continue
        selected = configured[role]
        responding = sum(results[port] is None for port in selected)
        print(f"  {role}: {responding}/{len(selected)} responding")
        for port in selected:
            if results[port] is not None:
                print(f"    failed http://127.0.0.1:{port}/health ({results[port]})")


def reward_report(root):
    path = root / "monitors/file/traces/stream.index.jsonl"
    data, size, truncated = tail(path, 2 * 1024 * 1024)
    buckets = {}
    for line in data.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        key = (str(row.get("kind", "unknown")), str(row.get("env", "unknown")))
        bucket = buckets.setdefault(key, {"rewards": [], "failed": 0, "unknown": 0, "groups": set()})
        if row.get("ok") is False:
            bucket["failed"] += 1
            if row.get("group") is not None:
                bucket["groups"].add(str(row["group"]))
        elif row.get("ok") is True and finite(row.get("reward")):
            bucket["rewards"].append(row["reward"])
        else:
            bucket["unknown"] += 1
    print("Rewards by environment (episode sample window, at most 2 MiB):")
    for (kind, env), bucket in sorted(buckets.items()):
        rewards = bucket["rewards"]
        mean = sum(rewards) / len(rewards) if rewards else None
        print(f"  {kind}/{env}: mean {shown(mean)} from {len(rewards)} valid scores; "
              f"failed episodes {bucket['failed']}; groups with failed episodes {len(bucket['groups'])}; "
              f"unknown scores {bucket['unknown']}")
    if not buckets:
        print("  unknown (no readable episode records)")
    print("Learning trend: unknown. Startup policy v0 scores do not establish learning.")
    print(f"Reward source {path}; file bytes {size if size is not None else 'unknown'}; "
          f"older records omitted {'yes' if truncated else 'no'}.")


def status_report(root, state, health_ports=None):
    metrics_path = root / "monitors/file/metrics.jsonl"
    data, size, truncated = tail(metrics_path, 4 * 1024 * 1024)
    grouped = {"trainer": {}, "orch": {}}
    invalid = 0
    for line in data.splitlines():
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            actor, step = row.get("producer"), row.get("step")
            if actor in grouped and type(step) is int:
                grouped[actor].setdefault(step, {}).update(row)
        except ValueError:
            invalid += 1
    completed = sorted(step for step, row in grouped["trainer"].items()
                       if finite(row.get("time/forward_backward"))
                       and not row.get("diagnostic/parity_guard_rejected"))
    latest = {actor: rows[max(rows)] if rows else {} for actor, rows in grouped.items()}
    print(f"Run {root.name}\nProcess {'running' if alive(state) else 'stopped'} (PID {state['pid']})")
    service_health(health_ports or {})
    if size is None:
        count = "unknown (metrics file is absent)"
    elif not completed:
        count = "unknown (no completed trainer records in the read window)"
    elif truncated or invalid:
        count = f"at least {len(completed)} observed; total unknown (bounded or invalid records)"
    else:
        count = str(len(completed))
    print(f"Completed optimizer updates: {count}")
    print(f"Latest completed trainer step: {completed[-1] if completed else 'unknown'}")
    for actor, row in latest.items():
        stamp = row.get("time")
        stamp_text = datetime.fromtimestamp(stamp, timezone.utc).isoformat(timespec="seconds") if finite(stamp) else "unknown"
        print(f"Latest {actor} metrics: step {row.get('step', 'unknown')}, UTC {stamp_text}")
    logs = {}
    for role in ("trainer", "orchestrator"):
        try:
            path = log_path(root, role)
            raw, _, _ = tail(path, 256 * 1024)
            logs[role] = [format_line(line) for line in raw.splitlines()]
        except ValueError:
            logs[role] = []
    batch = "unknown"
    for line in reversed(logs["orchestrator"]):
        match = re.search(r"accepted_buffer=(\d+)\s+accepted_target=(\d+)", line)
        if not match:
            match = re.search(r"Train batch\s+(\d+)/(\d+)", line)
        if match:
            batch = f"{match[1]}/{match[2]} (latest orchestrator log record)"
            break
    print(f"Batch progress: {batch}")
    reward_report(root)
    print("Timings in seconds (latest metric step; values can overlap):")
    timing = {
        "trainer": {"Trainer step": "time/step", "Forward/backward and update": "time/forward_backward",
                    "Wait for batch": "time/wait_for_batch", "Load batch": "time/load_data",
                    "Weight broadcast": "time/broadcast_weights", "Save checkpoint": "time/save_ckpt"},
        "orch": {"Orchestrator pre-ship interval": "time/step", "Wait for policy": "time/wait_for_policy",
                 "Pack rollout": "time/pack", "Send rollout": "time/send_batch"},
    }
    for actor, fields in timing.items():
        for label, key in fields.items():
            print(f"  {label}: {shown(latest[actor].get(key))}")
    for key, value in sorted(latest["orch"].items()):
        if key.startswith("train/agg/") and "/timing/" in key and key.endswith("/mean"):
            print(f"  {key.removeprefix('train/agg/')}: {shown(value)}")
    print("Stability (latest trainer metric step):")
    for label, key in (("Gradient norm", "optim/grad_norm"), ("Masked KL", "masked_mismatch_kl/mean"),
                       ("All-token KL", "mismatch_kl/all/mean"),
                       ("Unmasked KL", "unmasked_mismatch_kl/mean"), ("Masked fraction", "is_masked/mean"),
                       ("Loss", "loss/mean")):
        print(f"  {label}: {shown(latest['trainer'].get(key))}")
    rejected = [step for step, row in grouped["trainer"].items() if row.get("diagnostic/parity_guard_rejected")]
    nonfinite = [f"{actor}/{key}" for actor, row in latest.items() for key, value in row.items()
                 if type(value) is float and not math.isfinite(value)]
    print(f"  Rejected parity steps in read window: {', '.join(map(str, rejected)) or 'none observed'}")
    print(f"  Nonfinite metrics: {', '.join(nonfinite) or 'none observed'}")
    failures = [line for lines in logs.values() for line in lines
                if re.search(r"\b(ERROR|CRITICAL|Traceback|OutOfMemoryError|CUDA error)\b", line)]
    print("Recent log errors:")
    print("\n".join(f"  {line[:500]}" for line in failures[-3:]) if failures else
          "  none observed in bounded log tails" if any(logs.values()) else "  unknown (no readable logs)")
    print(f"Source {metrics_path}; read at most 4 MiB; file bytes {size if size is not None else 'unknown'}.")
    print("Update count uses unique trainer steps with finite time/forward_backward, after the optimizer update.")
    print("Missing values are unknown; timings are not summed.")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run")
    parser.add_argument("--lines", type=int, default=30)
    parser.add_argument("--no-follow", action="store_true")
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--role", choices=("trainer", "orchestrator"), default="orchestrator")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--health-ports", type=json.loads, default={})
    args = parser.parse_args()
    try:
        root, state = select_run(args.root, args.run)
        if args.status:
            return status_report(root, state, args.health_ports)
        return watch(root, state, args.lines, not args.no_follow, args.seconds, args.role)
    except (OSError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
