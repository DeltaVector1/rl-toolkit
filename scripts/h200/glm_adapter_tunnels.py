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
import tomllib
import uuid
from pathlib import Path


operations_path: Path | None = None
state_path: Path | None = None
ports: tuple[int, ...] = ()
target = ""
ssh_target = ""


def _configure(path: Path) -> None:
    global operations_path, state_path, ports, target, ssh_target
    try:
        details = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(details.st_mode):
            raise RuntimeError("operations config is not a regular file")
        with path.open("rb") as stream:
            config = tomllib.load(stream)
        config_hosts = config["hosts"]
        config_ports = config["ports"]
        config_paths = config["paths"]
        first = config_ports["glm_adapter_tunnel_first"]
        last = config_ports["glm_adapter_tunnel_last"]
        backend = config_ports["glm_adapter"]
        shared = config_ports["adapter_tunnel"]
        root = Path(config_paths["glm_adapter_root"])
        configured_state = Path(config_paths["glm_adapter_tunnel_state"])
        configured_ssh = config_hosts["trainer_ssh"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(f"operations config is invalid: {error}") from error
    values = (first, last, backend, shared)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise RuntimeError("GLM adapter port inventory is invalid")
    configured_ports = tuple(range(first, last + 1))
    if (
        len(configured_ports) != 12
        or any(not 1 <= port <= 65535 for port in (*configured_ports, backend, shared))
        or shared in configured_ports
        or not root.is_absolute()
        or not configured_state.is_absolute()
        or configured_state.parent != root
        or not isinstance(configured_ssh, str)
        or re.fullmatch(r"[A-Za-z0-9_.-]+", configured_ssh) is None
    ):
        raise RuntimeError("GLM adapter tunnel inventory is invalid")
    operations_path = path.resolve()
    state_path = configured_state
    ports = configured_ports
    target = f"127.0.0.1:{backend}"
    ssh_target = configured_ssh


def _require_configuration() -> tuple[Path, Path]:
    if operations_path is None or state_path is None:
        raise RuntimeError("operations config was not loaded")
    return operations_path, state_path


def _supervisor_command(port: int) -> list[str]:
    configured_operations, _ = _require_configuration()
    if port not in ports:
        raise RuntimeError("GLM adapter tunnel port is invalid")
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--operations",
        str(configured_operations),
        "supervise",
        str(port),
    ]


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
    _require_configuration()
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
        or len(value["records"]) > len(ports)
        or (value["committed"] and len(value["records"]) != len(ports))
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
    if [record["port"] for record in value["records"]] != list(
        ports[: len(value["records"])]
    ):
        raise RuntimeError("GLM adapter tunnel record ports differ")
    return value


def _read_state() -> dict | None:
    _, configured_state = _require_configuration()
    try:
        details = configured_state.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or stat.S_IMODE(details.st_mode) != 0o600
        or details.st_uid != os.getuid()
    ):
        raise RuntimeError("GLM adapter tunnel state file identity differs")
    return _validate_state(json.loads(configured_state.read_text()))


def _write_state(value: dict) -> None:
    _, configured_state = _require_configuration()
    _validate_state(value)
    configured_state.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    temporary = configured_state.with_name(
        f".{configured_state.name}.{uuid.uuid4().hex}.tmp"
    )
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
        os.replace(temporary, configured_state)
    finally:
        temporary.unlink(missing_ok=True)
    directory = os.open(configured_state.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _unlink_state() -> None:
    _, configured_state = _require_configuration()
    try:
        details = configured_state.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise RuntimeError("refusing unsafe GLM adapter tunnel state cleanup")
    configured_state.unlink()


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
    launch_id = uuid.uuid4().hex
    state = {
        "schema_version": 1,
        "script_sha256": _script_sha256(),
        "launch_id": launch_id,
        "committed": False,
        "ports": list(ports),
        "target": target,
        "ssh_target": ssh_target,
        "records": [],
    }
    _write_state(state)
    try:
        for port in ports:
            blocked = {signal.SIGTERM, signal.SIGHUP, signal.SIGINT}
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
            try:
                pid = _spawn_supervisor(port)
                deadline = time.monotonic() + 5
                identity = None
                while time.monotonic() < deadline:
                    identity = _process_identity(pid)
                    if identity is not None:
                        break
                    time.sleep(0.05)
                if (
                    identity is None
                    or identity["pgid"] != pid
                    or identity["cmdline"] != _supervisor_command(port)
                ):
                    raise RuntimeError(
                        f"GLM adapter tunnel supervisor {port} has wrong identity"
                    )
                state["records"].append(
                    {
                        "pid": pid,
                        "starttime": identity["starttime"],
                        "port": port,
                        "target": target,
                        "cmdline": _supervisor_command(port),
                    }
                )
                _write_state(state)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if all(_listening(port) for port in ports) and all(
                _record_matches(record, _process_identity(record["pid"]))
                for record in state["records"]
            ):
                break
            time.sleep(0.2)
        else:
            raise RuntimeError("GLM adapter tunnels did not bind every intended port")
        return launch_id
    except BaseException as error:
        try:
            cleanup(launch_id)
        except Exception as cleanup_error:
            error.add_note(f"GLM adapter tunnel cleanup failed: {cleanup_error}")
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


def status() -> dict[str, object]:
    state = _read_state()
    if state is None:
        return {"recorded": 0, "healthy": 0, "committed": False}
    healthy = sum(
        _record_matches(record, _process_identity(record["pid"]))
        and _listening(record["port"])
        for record in state["records"]
    )
    return {
        "recorded": len(state["records"]),
        "healthy": healthy,
        "committed": state["committed"],
    }


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
    parser.add_argument("--operations", type=Path, required=True)
    parser.add_argument(
        "action", choices=("start", "commit", "cleanup", "stop", "status", "supervise")
    )
    parser.add_argument("value", nargs="?")
    args = parser.parse_args()
    _configure(args.operations)
    if args.action == "start":
        if args.value is not None:
            raise RuntimeError("start takes no value")

        def stop_uncommitted(received, _frame):
            try:
                current = _read_state()
                if current is not None and not current["committed"]:
                    cleanup(current["launch_id"])
            except Exception as error:
                print(f"GLM adapter tunnel signal cleanup failed: {error}", file=sys.stderr)
            raise SystemExit(128 + received)

        for received in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(received, stop_uncommitted)
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
    elif args.action == "status":
        if args.value is not None:
            raise RuntimeError("status takes no value")
        print(json.dumps(status(), sort_keys=True, separators=(",", ":")))
    else:
        if args.value is None or not args.value.isdigit():
            raise RuntimeError("supervisor port is invalid")
        supervise(int(args.value))


if __name__ == "__main__":
    main()
