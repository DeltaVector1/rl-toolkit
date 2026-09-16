"""Own the configured direct adapter SSH connections on the trainer head."""

import ctypes
import errno
import fcntl
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import signal
import stat
import socket
import subprocess
import sys
import time
import tomllib
from contextlib import contextmanager


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
    route["remote_script"] = nodes["paths"]["remote_repository"] + "/training_ops/runtime/china/direct_adapter_route.py"
    return route


def commands(route):
    args = ["/usr/bin/ssh", "-v", "-T", "-F", "/dev/null"]
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


def validate_record(record):
    if (not isinstance(record, dict) or type(record.get("pid")) is not int or record["pid"] <= 1
            or not isinstance(record.get("start"), str) or not record["start"].isdigit()
            or not isinstance(record.get("argv"), list) or not record["argv"]
            or not all(isinstance(value, str) and value for value in record["argv"])):
        raise RuntimeError("malformed process identity record")


def pidfd_signal(fd, number):
    native = getattr(signal, "pidfd_send_signal", None)
    if native is not None:
        return native(fd, number)
    library = ctypes.CDLL(None, use_errno=True)
    native = getattr(library, "pidfd_send_signal", None)
    if native is None:
        raise RuntimeError("safe pidfd signalling is unavailable on this platform")
    native.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    native.restype = ctypes.c_int
    if native(fd, number, None, 0) != 0:
        error = ctypes.get_errno()
        if error != errno.ESRCH:
            raise OSError(error, "pidfd signal failed")


def pidfd_open(pid):
    native = getattr(os, "pidfd_open", None)
    if native is not None:
        return native(pid)
    library = ctypes.CDLL(None, use_errno=True)
    native = getattr(library, "pidfd_open", None)
    if native is None:
        raise RuntimeError("safe pidfd opening is unavailable on this platform")
    native.argtypes = [ctypes.c_int, ctypes.c_uint]
    native.restype = ctypes.c_int
    fd = native(pid, 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "pidfd open failed")
    return fd


def check_pidfd_support():
    library = ctypes.CDLL(None, use_errno=True)
    if ((getattr(os, "pidfd_open", None) is None and getattr(library, "pidfd_open", None) is None)
            or (getattr(signal, "pidfd_send_signal", None) is None and getattr(library, "pidfd_send_signal", None) is None)):
        raise RuntimeError("safe pidfd operations are unavailable; no route state was changed")
    fd = pidfd_open(os.getpid())
    try:
        pidfd_signal(fd, 0)
    finally:
        os.close(fd)


def send(record, number):
    try:
        fd = pidfd_open(record["pid"])
    except ProcessLookupError:
        return
    try:
        current = identity(record["pid"])
        if current is None:
            return
        if current != record:
            raise RuntimeError("process identity changed; refusing signal")
        try:
            pidfd_signal(fd, number)
        except ProcessLookupError:
            pass
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


def private_stat(value, directory=False):
    expected = 0o700 if directory else 0o600
    kind = stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode)
    if not kind or value.st_uid != os.geteuid() or stat.S_IMODE(value.st_mode) != expected or (not directory and value.st_nlink != 1):
        raise RuntimeError("route state ownership, type, or mode is unsafe")


@contextmanager
def state_directory(route, create=False):
    path = Path(route["state_dir"])
    if not path.is_absolute() or len(path.parts) < 3 or ".." in path.parts:
        raise RuntimeError("unsafe route state directory")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, part in enumerate(path.parts[1:]):
            last = index == len(path.parts) - 2
            try:
                child_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            except FileNotFoundError:
                if not create or not last:
                    raise
                os.mkdir(part, 0o700, dir_fd=fd)
                child_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child_fd
        private_stat(os.fstat(fd), directory=True)
        yield fd
    finally:
        os.close(fd)


def state_open(route, name, flags, truncate=False):
    with state_directory(route) as directory:
        fd = os.open(name, flags | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=directory)
    try:
        private_stat(os.fstat(fd))
        if truncate:
            os.ftruncate(fd, 0)
        return fd
    except BaseException:
        os.close(fd)
        raise


def state_text(route, name):
    with os.fdopen(state_open(route, name, os.O_RDONLY)) as stream:
        return stream.read(4 * 1024 * 1024)


def acquire(path):
    route = {"state_dir": str(Path(path).parent)}
    fd = state_open(route, Path(path).name, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(fd)
        raise RuntimeError("direct route manager is already active")
    return fd


def publish(route, value):
    with state_directory(route) as directory:
        try:
            private_stat(os.stat("state.json", dir_fd=directory, follow_symlinks=False))
        except FileNotFoundError:
            pass
        fd = os.open("state.pending", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(json.dumps(value))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace("state.pending", "state.json", src_dir_fd=directory, dst_dir_fd=directory)
        finally:
            try:
                os.unlink("state.pending", dir_fd=directory)
            except FileNotFoundError:
                pass


def clear_pending(route):
    # The caller must hold manager.lock. A killed publisher can leave this file.
    with state_directory(route) as directory:
        try:
            fd = os.open("state.pending", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        except FileNotFoundError:
            return
        try:
            value = os.fstat(fd)
            private_stat(value)
            current = os.stat("state.pending", dir_fd=directory, follow_symlinks=False)
            if (value.st_dev, value.st_ino) != (current.st_dev, current.st_ino):
                raise RuntimeError("pending route state identity changed")
            os.unlink("state.pending", dir_fd=directory)
        finally:
            os.close(fd)


def manager_argv(route):
    return [sys.executable, str(Path(__file__).resolve()), "serve", route["nodes_path"]]


def session_argv(route, ports, token):
    return ["/usr/bin/python3", route["remote_script"], "session", route["nodes_path"], token, *map(str, ports)]


def session_command(route, ports, token):
    return "exec " + shlex.join(session_argv(route, ports, token))


def listener_owners(text, ports):
    owners = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 5 or not fields[3].startswith("127.0.0.1:"):
            continue
        port = int(fields[3].rsplit(":", 1)[1])
        if port not in ports:
            continue
        matches = re.findall(r'\("sshd(?:-session)?",pid=(\d+),fd=\d+\)', line)
        if len(matches) != 1 or port in owners:
            raise RuntimeError("direct listener ownership differs")
        owners[port] = int(matches[0])
    if set(owners) != set(ports):
        raise RuntimeError("direct listeners are incomplete")
    return owners


def ancestor(pid, expected):
    for _ in range(8):
        if pid == expected["pid"]:
            return matches(expected)
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            return False
        pid = int(fields[1])
        if pid <= 1:
            break
    return False


def tcp_sockets():
    result = []
    for name, family in (("tcp", socket.AF_INET), ("tcp6", socket.AF_INET6)):
        for line in Path(f"/proc/net/{name}").read_text().splitlines()[1:]:
            fields = line.split()
            def endpoint(value):
                address, port = value.split(":")
                raw = bytes.fromhex(address)
                raw = b"".join(raw[index:index + 4][::-1] for index in range(0, len(raw), 4))
                return str(ipaddress.ip_address(socket.inet_ntop(family, raw))), int(port, 16)
            result.append({"local": endpoint(fields[1]), "peer": endpoint(fields[2]), "state": fields[3], "inode": int(fields[9])})
    return result


def process_sockets(pid):
    result = set()
    for entry in Path(f"/proc/{pid}/fd").iterdir():
        try:
            target = os.readlink(entry)
        except FileNotFoundError:
            continue
        match = re.fullmatch(r"socket:\[(\d+)\]", target)
        if match:
            result.add(int(match[1]))
    return result


def transport_proof(owner, ports, connection):
    if len(connection) != 4:
        raise RuntimeError("invalid SSH connection tuple")
    client, client_port, server, server_port = connection
    peer = (str(ipaddress.ip_address(client)), int(client_port))
    local = (str(ipaddress.ip_address(server)), int(server_port))
    held = process_sockets(owner["pid"])
    sockets = [value for value in tcp_sockets() if value["inode"] in held]
    accepted = [value["inode"] for value in sockets if value["state"] == "01" and value["local"] == local and value["peer"] == peer]
    if len(accepted) != 1:
        raise RuntimeError("listener owner does not hold the exact SSH connection")
    listeners = {}
    for port in ports:
        found = [value["inode"] for value in sockets if value["state"] == "0A" and value["local"] == ("127.0.0.1", port)]
        if len(found) != 1:
            raise RuntimeError("listener socket is not held by the connection owner")
        listeners[str(port)] = found[0]
    return {"boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "connection": connection,
            "connection_inode": accepted[0], "listener_inodes": listeners}


def session(route, ports, token):
    if re.fullmatch(r"[a-f0-9]{32}", token) is None or ports not in [pair for pair, _ in commands(route)]:
        raise RuntimeError("invalid direct session identity")
    session_record = identity(os.getpid())
    if session_record["argv"] != session_argv(route, ports, token):
        raise RuntimeError("direct session argv differs")
    output = subprocess.check_output(["ss", "-H", "-ltnp"], text=True)
    owners = listener_owners(output, ports)
    if len(set(owners.values())) != 1:
        raise RuntimeError("direct session does not own both listeners")
    owner = identity(owners[ports[0]])
    if owner is None or not ancestor(os.getpid(), owner):
        raise RuntimeError("listeners do not belong to this authenticated session")
    transport = transport_proof(owner, ports, os.environ.get("SSH_CONNECTION", "").split())
    print(json.dumps({"token": token, "ports": ports, "digest": route["digest"], "session": session_record, "owner": owner, "transport": transport}), flush=True)
    # The manager retains this connection's input pipe. SSH disconnect closes it.
    sys.stdin.buffer.read(1)


def validate_receipt(route, ports, value, live=False):
    token = value.get("token", "")
    if re.fullmatch(r"[a-f0-9]{32}", token) is None or value.get("ports") != ports or value.get("digest") != route["digest"]:
        raise RuntimeError("direct session receipt identity differs")
    validate_record(value.get("session"))
    validate_record(value.get("owner"))
    if value["session"]["argv"] != session_argv(route, ports, token):
        raise RuntimeError("direct session receipt argv differs")
    if live and (not matches(value["session"]) or not matches(value["owner"]) or not ancestor(value["session"]["pid"], value["owner"])):
        raise RuntimeError("direct session or listener process identity changed")
    transport = value.get("transport", {})
    if (not transport.get("boot_id") or not isinstance(transport.get("connection"), list)
            or len(transport["connection"]) != 4 or type(transport.get("connection_inode")) is not int
            or set(transport.get("listener_inodes", {})) != set(map(str, ports))):
        raise RuntimeError("direct session transport proof is incomplete")
    if live and transport_proof(value["owner"], ports, transport["connection"]) != transport:
        raise RuntimeError("direct session transport proof changed")


def verify_snapshot(route, snapshot, listeners):
    if snapshot.get("digest") != route["digest"] or snapshot.get("status") != "ready":
        raise RuntimeError("source manager snapshot is not ready")
    owners = listener_owners(listeners, route["ports"])
    identities = []
    for ports, _ in commands(route):
        receipt = snapshot["receipts"][str(ports[0])]
        validate_receipt(route, ports, receipt, live=True)
        if any(owners[port] != receipt["owner"]["pid"] for port in ports):
            raise RuntimeError("listener is not bound to the managed SSH session")
        identities.append(receipt)
    if len({value["owner"]["pid"] for value in identities}) != len(identities):
        raise RuntimeError("duplicate direct session owner")
    proof = {"manager": snapshot["manager"], "children": snapshot["children"], "receipts": identities}
    return hashlib.sha256(json.dumps(proof, sort_keys=True).encode()).hexdigest()


def stored_state(route):
    try:
        fd = state_open(route, "state.json", os.O_RDONLY)
    except FileNotFoundError:
        return None
    with os.fdopen(fd) as stream:
        state = json.load(stream)
    validate_record(state.get("manager"))
    if state.get("digest") != route["digest"] or state.get("manager", {}).get("argv") != manager_argv(route):
        raise RuntimeError("stored manager argv or configuration differs; refusing signal")
    return state


def read_state(route):
    state = stored_state(route)
    if state is None:
        return None
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
    for record, (ports, argv) in zip(state["children"], expected):
        receipt = state["receipts"][str(ports[0])]
        validate_receipt(route, ports, receipt)
        if record["argv"] != argv + [session_command(route, ports, receipt["token"])] or not matches(record):
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
    try:
        clear_pending(route)
    except BaseException:
        os.close(lock)
        raise
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
        receipts = {}
        tokens = {}
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
            receipts.pop(str(ports[0]), None)
            if not established:
                raise RuntimeError(f"SSH owner {ports} failed: {state_text(route, log.name)[-2048:]}")
            count = attempts.get(ports[0], 0)
            if count >= route["reconnect_attempts"]:
                raise RuntimeError(f"owner {ports} exhausted reconnect attempts; start may retry")
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
                if now >= active_deadlines[ports[0]]:
                    cleanup([item], route["cleanup_timeout_seconds"])
                    reconnect(item)
                    continue
                if forwards_ready(state_text(route, log.name), ports, route["adapter_port"]):
                    try:
                        with os.fdopen(state_open(route, f"owner-{ports[0]}.receipt", os.O_RDONLY)) as stream:
                            receipt = json.load(stream)
                    except (FileNotFoundError, json.JSONDecodeError):
                        continue
                    validate_receipt(route, ports, receipt)
                    if receipt["token"] != tokens[ports[0]]:
                        raise RuntimeError("source session token differs")
                    record = identity(process.pid)
                    if record is None or record["argv"] != argv:
                        raise RuntimeError("SSH owner argv differs")
                    records[ports[0]] = record
                    receipts[str(ports[0])] = receipt
                    attempts.pop(ports[0], None)
                    active.remove(item)
                    active_deadlines.pop(ports[0])
            for item in children:
                if item not in active and not matches(records[item[1][0]]):
                    raise RuntimeError("SSH owner identity changed")
            pending.sort(key=lambda item: item[2])
            while pending and pending[0][2] <= now and len(active) < route["setup_parallelism"]:
                ports, argv, _ = pending.pop(0)
                token = secrets.token_hex(16)
                tokens[ports[0]] = token
                argv = argv[:len(commands(route)[0][1])] + [session_command(route, ports, token)]
                wrapper = [sys.executable, str(Path(__file__).resolve()), "child", str(os.getpid()), manager["start"], *argv]
                log = directory / f"owner-{ports[0]}.log"
                with os.fdopen(state_open(route, log.name, os.O_WRONLY | os.O_CREAT, truncate=True), "wb") as output, os.fdopen(state_open(route, f"owner-{ports[0]}.receipt", os.O_WRONLY | os.O_CREAT, truncate=True), "wb") as receipt_output:
                    process = subprocess.Popen(wrapper, stdin=subprocess.PIPE, stdout=receipt_output, stderr=output, close_fds=True)
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
                receipts=dict(receipts),
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


def pin_child(process):
    # Popen has not been polled/reaped: its PID cannot yet be reused.
    return pidfd_open(process.pid)


def cleanup_started(process, descriptor, seconds):
    # This descriptor is authority for our own spawned child, even while exec
    # changes its argv. Never recover this authority from a numeric PID or state.
    if process.poll() is not None:
        process.wait(timeout=0)
        return
    deadline = time.monotonic() + seconds
    try:
        pidfd_signal(descriptor, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=max(0, seconds - min(2, seconds / 2)))
        return
    except subprocess.TimeoutExpired:
        try:
            pidfd_signal(descriptor, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=max(0, deadline - time.monotonic()))


def operate(action, route):
    directory = Path(route["state_dir"])
    if action in {"status", "snapshot"}:
        value = ready(route)
        print(json.dumps(value) if action == "snapshot" else f"direct route ready: owners={len(value['children'])} lanes={len(route['ports'])}")
        return
    check_pidfd_support()
    with state_directory(route, create=True):
        pass
    control = acquire(directory / "control.lock")
    try:
        state = stored_state(route)
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
        descriptor = None
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
            deadline = time.monotonic() + route["setup_timeout_seconds"] + 2
            with os.fdopen(state_open(route, "manager.log", os.O_WRONLY | os.O_APPEND | os.O_CREAT), "ab") as log:
                process = subprocess.Popen(manager_argv(route), stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, close_fds=True)
            descriptor = pin_child(process)
            spawning = False
            if interrupted_during_spawn:
                raise InterruptedError("direct route start interrupted")
            def accept_ready():
                try:
                    value = ready(route)
                except RuntimeError:
                    return False
                # ready() proves the exact post-exec argv, config, and all
                # owner identities. A pre-exec cmdline is never retained.
                if value["manager"]["pid"] == process.pid:
                    if process.poll() is not None:
                        return False
                else:
                    cleanup_started(process, descriptor, route["cleanup_timeout_seconds"] + 2)
                    if ready(route)["manager"] != value["manager"]:
                        raise RuntimeError("ready manager changed during reconciliation")
                return True
            while True:
                if accept_ready():
                    return
                remaining = deadline - time.monotonic()
                if process.poll() is not None or remaining <= 0:
                    if accept_ready():
                        return
                    break
                time.sleep(min(0.1, remaining))
            raise RuntimeError("direct route setup failed; see manager.log")
        except BaseException:
            for number in handlers:
                signal.signal(number, signal.SIG_IGN)
            if process is not None:
                if descriptor is None:
                    if process.poll() is not None:
                        process.wait(timeout=0)
                    else:
                        raise RuntimeError("cannot pin spawned child identity; child may still be active; no signal was sent")
                else:
                    cleanup_started(process, descriptor, route["cleanup_timeout_seconds"] + 2)
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)
            for number, handler in handlers.items():
                signal.signal(number, handler)
    finally:
        os.close(control)


if __name__ == "__main__":
    if sys.argv[1] == "child":
        child(int(sys.argv[2]), sys.argv[3], sys.argv[4:])
    elif sys.argv[1] == "session":
        session(settings(sys.argv[2]), [int(value) for value in sys.argv[4:]], sys.argv[3])
    elif sys.argv[1] == "verify":
        route = settings(sys.argv[2])
        listeners = subprocess.check_output(["ss", "-H", "-ltnp"], text=True)
        print(verify_snapshot(route, json.load(sys.stdin), listeners))
    else:
        action, path = sys.argv[1:]
        if action not in {"start", "restart", "stop", "status", "snapshot", "serve"}:
            raise SystemExit("unknown direct route action")
        route = settings(path)
        serve(route) if action == "serve" else operate(action, route)
