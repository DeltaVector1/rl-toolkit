import argparse
import hashlib
import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path


state_path = Path("/data/glm52-rl/glm-adapter-tunnels.json")
ports = tuple(range(18200, 18212))
target = "127.0.0.1:8101"
ssh_target = "china-trainer"


def _supervisor_command(port: int) -> list[str]:
    if port not in ports:
        raise RuntimeError("GLM adapter tunnel port is invalid")
    return [sys.executable, str(Path(__file__).resolve()), "supervise", str(port)]


def _ssh_command(port: int) -> list[str]:
    return [
        "ssh",
        "-NT",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=30",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "TCPKeepAlive=yes",
        "-o",
        "IPQoS=none",
        "-o",
        "Compression=no",
        "-L",
        f"127.0.0.1:{port}:{target}",
        ssh_target,
    ]


def _script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _process_identity(pid: int) -> dict | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        tail = raw.rsplit(")", 1)[1].split()
        cmdline = [
            value.decode()
            for value in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if value
        ]
    except (FileNotFoundError, ProcessLookupError):
        return None
    return {"pgid": int(tail[2]), "starttime": int(tail[19]), "cmdline": cmdline}


def _listening(port: int) -> bool:
    with socket.socket() as client:
        client.settimeout(1)
        return client.connect_ex(("127.0.0.1", port)) == 0


def _spawn_supervisor(port: int) -> int:
    process = subprocess.Popen(
        _supervisor_command(port),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process.pid


def _kill_group(pid: int, action: signal.Signals) -> None:
    os.killpg(pid, action)


def _record_matches(record: dict, identity: dict | None) -> bool:
    return bool(
        identity is not None
        and identity["pgid"] == record["pid"]
        and identity["starttime"] == record["starttime"]
        and identity["cmdline"] == record["cmdline"]
    )


def _validate_state(value: object) -> dict:
    if not isinstance(value, dict):
        raise RuntimeError("GLM adapter tunnel state is not an object")
    expected_keys = {
        "schema_version",
        "script_sha256",
        "launch_id",
        "committed",
        "ports",
        "target",
        "ssh_target",
        "records",
    }
    if set(value) != expected_keys:
        raise RuntimeError("GLM adapter tunnel state fields differ")
    if (
        value["schema_version"] != 1
        or re.fullmatch(r"[0-9a-f]{64}", value["script_sha256"] or "") is None
        or re.fullmatch(r"[0-9a-f]{32}", value["launch_id"] or "") is None
        or not isinstance(value["committed"], bool)
        or value["ports"] != list(ports)
        or value["target"] != target
        or value["ssh_target"] != ssh_target
        or not isinstance(value["records"], list)
        or len(value["records"]) != len(ports)
    ):
        raise RuntimeError("GLM adapter tunnel state identity differs")
    seen = set()
    for record in value["records"]:
        if not isinstance(record, dict) or set(record) != {
            "pid",
            "starttime",
            "port",
            "target",
            "cmdline",
        }:
            raise RuntimeError("GLM adapter tunnel record fields differ")
        port = record["port"]
        if (
            isinstance(record["pid"], bool)
            or not isinstance(record["pid"], int)
            or record["pid"] <= 1
            or isinstance(record["starttime"], bool)
            or not isinstance(record["starttime"], int)
            or record["starttime"] <= 0
            or port not in ports
            or port in seen
            or record["target"] != target
            or record["cmdline"] != _supervisor_command(port)
        ):
            raise RuntimeError("GLM adapter tunnel record identity differs")
        seen.add(port)
    if seen != set(ports):
        raise RuntimeError("GLM adapter tunnel record ports differ")
    return value


def _read_state() -> dict | None:
    try:
        details = state_path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or stat.S_IMODE(details.st_mode) != 0o600
        or details.st_uid != os.getuid()
    ):
        raise RuntimeError("GLM adapter tunnel state file identity differs")
    return _validate_state(json.loads(state_path.read_text()))


def _write_state(value: dict) -> None:
    _validate_state(value)
    state_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    temporary = state_path.with_name(f".{state_path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    try:
        os.replace(temporary, state_path)
    finally:
        temporary.unlink(missing_ok=True)
    directory = os.open(state_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _unlink_state() -> None:
    try:
        details = state_path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise RuntimeError("refusing unsafe GLM adapter tunnel state cleanup")
    state_path.unlink()


def _stop_records(records: list[dict]) -> set[int]:
    unmatched = set()
    exact = []
    for record in records:
        identity = _process_identity(record["pid"])
        if identity is None:
            continue
        if not _record_matches(record, identity):
            unmatched.add(record["port"])
            continue
        _kill_group(record["pid"], signal.SIGTERM)
        exact.append(record)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if not any(
            _record_matches(record, _process_identity(record["pid"]))
            for record in exact
        ):
            break
        time.sleep(0.1)
    for record in exact:
        if _record_matches(record, _process_identity(record["pid"])):
            _kill_group(record["pid"], signal.SIGKILL)
    return unmatched


def _wait_ports_free() -> None:
    deadline = time.monotonic() + 10
    occupied = []
    while time.monotonic() < deadline:
        occupied = [port for port in ports if _listening(port)]
        if not occupied:
            return
        time.sleep(0.1)
    raise RuntimeError(f"refusing occupied GLM adapter tunnel ports: {occupied}")


def start() -> str:
    old = _read_state()
    if old is not None:
        _stop_records(old["records"])
        _wait_ports_free()
        _unlink_state()
    else:
        _wait_ports_free()
    records = []
    try:
        for port in ports:
            pid = _spawn_supervisor(port)
            deadline = time.monotonic() + 5
            identity = None
            while time.monotonic() < deadline:
                identity = _process_identity(pid)
                if identity is not None:
                    break
                time.sleep(0.05)
            if identity is None or identity["pgid"] != pid or identity["cmdline"] != _supervisor_command(port):
                raise RuntimeError(f"GLM adapter tunnel supervisor {port} has wrong identity")
            records.append(
                {
                    "pid": pid,
                    "starttime": identity["starttime"],
                    "port": port,
                    "target": target,
                    "cmdline": _supervisor_command(port),
                }
            )
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if all(_listening(port) for port in ports) and all(
                _record_matches(record, _process_identity(record["pid"]))
                for record in records
            ):
                break
            time.sleep(0.2)
        else:
            raise RuntimeError("GLM adapter tunnels did not bind every intended port")
        launch_id = uuid.uuid4().hex
        _write_state(
            {
                "schema_version": 1,
                "script_sha256": _script_sha256(),
                "launch_id": launch_id,
                "committed": False,
                "ports": list(ports),
                "target": target,
                "ssh_target": ssh_target,
                "records": records,
            }
        )
        return launch_id
    except BaseException:
        _stop_records(records)
        raise


def commit(launch_id: str) -> None:
    state = _read_state()
    if state is None or state["launch_id"] != launch_id:
        raise RuntimeError("GLM adapter tunnel launch identity changed before commit")
    if not all(
        _record_matches(record, _process_identity(record["pid"]))
        and _listening(record["port"])
        for record in state["records"]
    ):
        raise RuntimeError("GLM adapter tunnels are not complete at commit")
    state["committed"] = True
    _write_state(state)


def cleanup(launch_id: str | None = None) -> None:
    state = _read_state()
    if state is None:
        return
    if launch_id is not None and state["launch_id"] != launch_id:
        raise RuntimeError("refusing cleanup of another GLM tunnel launch")
    unmatched = _stop_records(state["records"])
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and any(_listening(port) for port in ports):
        time.sleep(0.1)
    occupied = [port for port in ports if _listening(port)]
    if unmatched or occupied:
        raise RuntimeError(
            f"GLM adapter tunnel cleanup is incomplete: unmatched={sorted(unmatched)}, occupied={occupied}"
        )
    _unlink_state()


def supervise(port: int) -> None:
    if port not in ports:
        raise RuntimeError("GLM adapter tunnel supervisor port is invalid")
    stopping = False
    child = None

    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping:
        child = subprocess.Popen(_ssh_command(port))
        while child.poll() is None and not stopping:
            time.sleep(0.2)
        if stopping:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=10)
            return
        time.sleep(3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "commit", "cleanup", "stop", "supervise"))
    parser.add_argument("value", nargs="?")
    args = parser.parse_args()
    if args.action == "start":
        if args.value is not None:
            raise RuntimeError("start takes no value")
        print(start())
    elif args.action == "commit":
        if re.fullmatch(r"[0-9a-f]{32}", args.value or "") is None:
            raise RuntimeError("commit launch ID is invalid")
        commit(args.value)
    elif args.action == "cleanup":
        if re.fullmatch(r"[0-9a-f]{32}", args.value or "") is None:
            raise RuntimeError("cleanup launch ID is invalid")
        cleanup(args.value)
    elif args.action == "stop":
        if args.value is not None:
            raise RuntimeError("stop takes no value")
        cleanup()
    else:
        if args.value is None or not args.value.isdigit():
            raise RuntimeError("supervisor port is invalid")
        supervise(int(args.value))


if __name__ == "__main__":
    main()
