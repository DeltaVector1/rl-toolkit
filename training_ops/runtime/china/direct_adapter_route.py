"""Own the configured direct adapter SSH connections on the trainer head."""

import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import tomllib


def settings(path):
    raw = Path(path).read_bytes()
    nodes = tomllib.loads(raw.decode())
    route = dict(nodes["adapter_transport"])
    if route["source_host"] != nodes["hosts"]["trainer_head"]:
        raise RuntimeError("direct adapter source must be the configured trainer head")
    if route["source_host"] in nodes["hosts"]["blocked"]:
        raise RuntimeError("direct adapter source is blocked")
    for key in ("setup_parallelism", "lanes_per_owner", "setup_timeout_seconds", "cleanup_timeout_seconds", "reconnect_attempts", "reconnect_backoff_seconds", "reconnect_timeout_seconds"):
        if type(route[key]) is not int or route[key] <= 0:
            raise RuntimeError(f"invalid {key}")
    if route["setup_parallelism"] > 4 or route["lanes_per_owner"] != 2:
        raise RuntimeError("direct route requires two lanes per owner and at most four setups")
    route["ports"] = list(range(nodes["ports"]["lora_source_first"], nodes["ports"]["lora_source_last"] + 1))
    if len(route["ports"]) != route["download_streams"] * route["lanes_per_owner"]:
        raise RuntimeError("source lanes do not match configured stream owners")
    route["adapter_port"] = nodes["ports"]["adapter"]
    route["digest"] = hashlib.sha256(raw).hexdigest()
    route["nodes_path"] = str(Path(path).resolve())
    return route


def commands(route):
    args = ["/usr/bin/ssh", "-v", "-NT", "-F", "/dev/null"]
    options = {
        "BatchMode": "yes", "IdentitiesOnly": "yes", "AddKeysToAgent": "no",
        "StrictHostKeyChecking": "yes", "UserKnownHostsFile": route["ssh_known_hosts_file"],
        "GlobalKnownHostsFile": "/dev/null", "UpdateHostKeys": "no",
        "ExitOnForwardFailure": "yes", "ConnectTimeout": "15",
        "ConnectionAttempts": "1", "ServerAliveInterval": "10", "ServerAliveCountMax": "3",
        "TCPKeepAlive": "yes", "Compression": "no", "ForwardAgent": "no",
        "ControlMaster": "no", "ControlPath": "none", "IdentityAgent": "none", "ProxyCommand": "none",
        "ProxyJump": "none", "PasswordAuthentication": "no", "KbdInteractiveAuthentication": "no",
    }
    for name, value in options.items():
        args.extend(["-o", f"{name}={value}"])
    args.extend(["-i", route["ssh_identity_file"], "-p", str(route["ssh_port"])])
    width = route["lanes_per_owner"]
    result = []
    for offset in range(0, len(route["ports"]), width):
        ports = route["ports"][offset:offset + width]
        forwards = [value for port in ports for value in ("-R", f"127.0.0.1:{port}:127.0.0.1:{route['adapter_port']}")]
        result.append((ports, args + forwards + [f"{route['ssh_user']}@{route['ssh_host']}"]))
    return result


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        return {"pid": pid, "start": fields[19], "argv": [part.decode() for part in argv]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def matches(record):
    return identity(record["pid"]) == record


def send(record, number):
    try:
        fd = os.pidfd_open(record["pid"])
    except ProcessLookupError:
        return
    try:
        if not matches(record):
            raise RuntimeError("process identity changed; refusing signal")
        signal.pidfd_send_signal(fd, number)
    finally:
        os.close(fd)


def child(parent_pid, parent_start, argv):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot set parent-death signal")
    parent = identity(parent_pid)
    if os.getppid() != parent_pid or parent is None or parent["start"] != parent_start:
        raise RuntimeError("manager exited before SSH start")
    os.execv(argv[0], argv)


def acquire(path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(fd)
        raise RuntimeError("direct route manager is already active")
    return fd


def publish(route, value):
    path = Path(route["state_dir"]) / "state.json"
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def read_state(route):
    path = Path(route["state_dir"]) / "state.json"
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    if not matches(state["manager"]):
        raise RuntimeError("stale manager identity; no process was signalled")
    if state["digest"] != route["digest"]:
        raise RuntimeError("running manager configuration differs")
    return state


def ready(route):
    state = read_state(route)
    expected = commands(route)
    if not state or state["status"] != "ready" or len(state["children"]) != len(expected):
        raise RuntimeError("direct route is not ready")
    for record, (_, argv) in zip(state["children"], expected):
        if record["argv"] != argv or not matches(record):
            raise RuntimeError("direct route child identity differs")
    return state


def forwards_ready(text, ports, adapter_port):
    return all(f"remote forward success for: listen 127.0.0.1:{port}, connect 127.0.0.1:{adapter_port}" in text for port in ports)


def cleanup(children, seconds):
    records = []
    for process, _, argv, wrapper, _ in children:
        if process.poll() is not None:
            continue
        record = identity(process.pid)
        if record is None or record["argv"] not in (argv, wrapper):
            raise RuntimeError("unidentified route child; refusing cleanup")
        records.append((process, record))
    for _, record in records:
        send(record, signal.SIGTERM)
    deadline = time.monotonic() + seconds
    graceful_deadline = deadline - min(2, seconds / 2)
    for process, record in records:
        try:
            process.wait(timeout=max(0, graceful_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            send(record, signal.SIGKILL)
    unfinished = []
    for process, _ in records:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            unfinished.append(process.pid)
    if unfinished:
        raise TimeoutError(f"route cleanup deadline exceeded for {unfinished}")


def serve(route):
    directory = Path(route["state_dir"])
    lock = acquire(directory / "manager.lock")
    manager = identity(os.getpid())
    children = []
    state = {"manager": manager, "digest": route["digest"], "status": "starting", "children": []}
    def stop(signum, frame):
        raise InterruptedError("direct route manager stopped")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        publish(route, state)
        pending = [(ports, argv, 0) for ports, argv in commands(route)]
        active = []
        active_deadlines = {}
        records = {}
        attempts = {}
        failed = set()
        established = False
        deadline = time.monotonic() + route["setup_timeout_seconds"]

        def reconnect(item):
            process, ports, argv, _, log = item
            children.remove(item)
            if item in active:
                active.remove(item)
            active_deadlines.pop(ports[0], None)
            records.pop(ports[0], None)
            if not established:
                raise RuntimeError(f"SSH owner {ports} failed: {log.read_text()[-2048:]}")
            count = attempts.get(ports[0], 0)
            if count >= route["reconnect_attempts"]:
                failed.add(ports[0])
                return
            attempts[ports[0]] = count + 1
            delay = route["reconnect_backoff_seconds"] * (2 ** count)
            pending.append((ports, argv, time.monotonic() + delay))

        while True:
            now = time.monotonic()
            if not established and now >= deadline:
                raise TimeoutError("direct route setup deadline exceeded")
            for item in list(children):
                if item[0].poll() is not None:
                    item[0].wait(timeout=0)
                    reconnect(item)
            for item in list(active):
                process, ports, argv, _, log = item
                if forwards_ready(log.read_text(), ports, route["adapter_port"]):
                    record = identity(process.pid)
                    if record is None or record["argv"] != argv:
                        raise RuntimeError("SSH owner argv differs")
                    records[ports[0]] = record
                    active.remove(item)
                    active_deadlines.pop(ports[0])
                elif now >= active_deadlines[ports[0]]:
                    cleanup([item], route["cleanup_timeout_seconds"])
                    reconnect(item)
            for item in children:
                if item not in active and not matches(records[item[1][0]]):
                    raise RuntimeError("SSH owner identity changed")
            pending.sort(key=lambda item: item[2])
            while pending and pending[0][2] <= now and len(active) < route["setup_parallelism"]:
                ports, argv, _ = pending.pop(0)
                wrapper = [sys.executable, str(Path(__file__).resolve()), "child", str(os.getpid()), manager["start"], *argv]
                log = directory / f"owner-{ports[0]}.log"
                with log.open("wb") as output:
                    process = subprocess.Popen(wrapper, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=output, close_fds=True)
                item = (process, ports, argv, wrapper, log)
                children.append(item)
                active.append(item)
                active_deadlines[ports[0]] = now + route["reconnect_timeout_seconds"]
            complete = not pending and not active and not failed
            if complete:
                established = True
            state.update(
                status="ready" if complete else "recovering" if established else "starting",
                children=[records[key] for key in sorted(records)],
                failed_owners=sorted(failed),
                reconnect_attempts=dict(attempts),
            )
            publish(route, state)
            time.sleep(0.1 if active or not established else 1)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        cleanup(children, route["cleanup_timeout_seconds"])
        state.update(status="stopped", children=[])
        publish(route, state)
        os.close(lock)


def operate(action, route):
    directory = Path(route["state_dir"])
    if action == "status":
        value = ready(route)
        print(f"direct route ready: owners={len(value['children'])} lanes={len(route['ports'])}")
        return
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    control = acquire(directory / "control.lock")
    try:
        path = directory / "state.json"
        state = json.loads(path.read_text()) if path.exists() else None
        if state and matches(state["manager"]):
            if action == "start":
                ready(route)
                return
            send(state["manager"], signal.SIGTERM)
            deadline = time.monotonic() + route["cleanup_timeout_seconds"] + 3
            while matches(state["manager"]):
                if time.monotonic() > deadline:
                    raise TimeoutError("manager cleanup deadline exceeded")
                time.sleep(0.05)
        elif state and state.get("status") != "stopped":
            # A killed manager's children have a kernel parent-death signal.
            if any(matches(record) for record in state.get("children", [])):
                raise RuntimeError("stale manager still has children; refusing restart")
        if action == "stop":
            return
        for key in ("ssh_identity_file", "ssh_known_hosts_file"):
            if not Path(route[key]).is_file():
                raise RuntimeError(f"configured {key} is missing")
        process = None
        manager_record = None
        spawning = True
        interrupted_during_spawn = False
        handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
        def interrupted(signum, frame):
            nonlocal interrupted_during_spawn
            if spawning:
                interrupted_during_spawn = True
                return
            raise InterruptedError("direct route start interrupted")
        for number in handlers:
            signal.signal(number, interrupted)
        try:
            with (directory / "manager.log").open("ab") as log:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "serve", route["nodes_path"]], stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, close_fds=True)
            manager_record = identity(process.pid)
            spawning = False
            if interrupted_during_spawn:
                raise InterruptedError("direct route start interrupted")
            deadline = time.monotonic() + route["setup_timeout_seconds"] + 2
            while process.poll() is None and time.monotonic() < deadline:
                try:
                    value = ready(route)
                    if value["manager"] != manager_record:
                        raise RuntimeError("started manager identity differs")
                    return
                except RuntimeError:
                    time.sleep(0.1)
            raise RuntimeError("direct route setup failed; see manager.log")
        except BaseException:
            for number in handlers:
                signal.signal(number, signal.SIG_IGN)
            if process is not None and process.poll() is None:
                if manager_record is None:
                    raise RuntimeError("started manager identity is unavailable")
                if not matches(manager_record):
                    raise RuntimeError("started manager identity changed; refusing cleanup")
                cleanup([(process, [], manager_record["argv"], [], None)], route["cleanup_timeout_seconds"] + 2)
            elif process is not None:
                process.wait(timeout=0)
            raise
        finally:
            for number, handler in handlers.items():
                signal.signal(number, handler)
    finally:
        os.close(control)


if __name__ == "__main__":
    if sys.argv[1] == "child":
        child(int(sys.argv[2]), sys.argv[3], sys.argv[4:])
    else:
        action, path = sys.argv[1:]
        if action not in {"start", "restart", "stop", "status", "serve"}:
            raise SystemExit("unknown direct route action")
        route = settings(path)
        serve(route) if action == "serve" else operate(action, route)
