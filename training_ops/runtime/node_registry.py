"""Manage local single-node profiles in configs/nodes.toml."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import tomllib


default_path = Path(__file__).resolve().parents[2] / "configs" / "nodes.toml"
name_pattern = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
required_fields = (
    "ssh_host",
    "ssh_port",
    "ssh_user",
    "ssh_identity_file",
    "runs",
    "python",
)


class NodeRegistryError(ValueError):
    """Raised when a node profile or its local config is invalid."""


def active_host(nodes: dict) -> str | None:
    """Return the selected node name, or None when no node is selected."""
    selection = nodes.get("selection", {})
    if not isinstance(selection, dict):
        raise NodeRegistryError("selection must be a TOML table")
    selected = selection.get("active_node")
    if selected is not None and not isinstance(selected, str):
        raise NodeRegistryError("selection.active_node must be a string")
    return selected


def profile(nodes: dict, host: str) -> dict | None:
    """Return one single-node profile by name."""
    profiles = nodes.get("single_node", {})
    if not isinstance(profiles, dict):
        raise NodeRegistryError("single_node must be a TOML table")
    value = profiles.get(host)
    if value is not None and not isinstance(value, dict):
        raise NodeRegistryError(f"single_node.{host} must be a TOML table")
    return value


def _validate_name(name: str) -> str:
    if not name_pattern.fullmatch(name):
        raise NodeRegistryError(
            "node name must start with a letter or digit and contain only letters, digits, _ or -"
        )
    return name


def _blocked_hosts(nodes: dict) -> set[str]:
    hosts = nodes.get("hosts", {})
    if not isinstance(hosts, dict):
        return set()
    blocked = hosts.get("blocked", [])
    if not isinstance(blocked, list) or any(not isinstance(host, str) for host in blocked):
        raise NodeRegistryError("hosts.blocked must be a list of host names")
    return set(blocked)


def _check_not_blocked(nodes: dict, name: str, ssh_host: str | None = None) -> None:
    blocked = _blocked_hosts(nodes)
    for candidate in (name, ssh_host):
        if candidate in blocked:
            raise NodeRegistryError(f"host {candidate} is blocked")


def _read(path: Path) -> tuple[str, dict]:
    try:
        text = path.read_text(encoding="utf-8")
        nodes = tomllib.loads(text)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise NodeRegistryError(f"cannot read nodes config {path}: {error}") from error
    return text, nodes


def _header_for_table(header: str) -> re.Pattern[str]:
    return re.compile(rf"^\s*\[{re.escape(header)}\]\s*(?:#.*)?(?:\r?\n)?$")


def _table_bounds(lines: list[str], header: str) -> tuple[int, int] | None:
    pattern = _header_for_table(header)
    start = next((index for index, line in enumerate(lines) if pattern.fullmatch(line)), None)
    if start is None:
        return None
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if re.match(r"^\s*\[", lines[index]):
            end = index
            break
    return start, end


def _replace_table_values(text: str, header: str, values: dict[str, str | int]) -> str:
    """Replace selected keys in one table and keep all other source text."""
    try:
        expected = deepcopy(tomllib.loads(text))
    except tomllib.TOMLDecodeError as error:
        raise NodeRegistryError(f"cannot update invalid nodes TOML: {error}") from error
    target = expected
    for component in header.split("."):
        if component not in target:
            target[component] = {}
        if not isinstance(target[component], dict):
            raise NodeRegistryError(f"TOML table {header} cannot be updated")
        target = target[component]
    target.update(values)

    lines = text.splitlines(keepends=True)
    bounds = _table_bounds(lines, header)
    if bounds is None:
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines[-1] += "\n"
        if lines and lines[-1].strip():
            lines.append("\n")
        lines.append(f"[{header}]\n")
        start, end = len(lines) - 1, len(lines)
    else:
        start, end = bounds

    for key, value in values.items():
        encoded = json.dumps(value, ensure_ascii=False) if isinstance(value, str) else str(value)
        assignment = f"{key} = {encoded}\n"
        key_pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
        match_index = next(
            (index for index in range(start + 1, end) if key_pattern.match(lines[index])),
            None,
        )
        if match_index is not None:
            lines[match_index] = assignment
        else:
            lines.insert(end, assignment)
            end += 1

    result = "".join(lines)
    try:
        actual = tomllib.loads(result)
    except tomllib.TOMLDecodeError as error:
        raise NodeRegistryError(f"node config update would make invalid TOML: {error}") from error
    if actual != expected:
        raise NodeRegistryError("node config update changed values outside the requested table fields")
    return result


def _write_atomic(path: Path, text: str) -> None:
    try:
        mode = path.stat().st_mode & 0o777
    except OSError as error:
        raise NodeRegistryError(f"cannot inspect nodes config {path}: {error}") from error
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_name = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_name, mode)
        os.replace(temp_name, path)
        temp_name = None
    except OSError as error:
        raise NodeRegistryError(f"cannot write nodes config {path}: {error}") from error
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except OSError:
                pass


def _reload_training_ops(path: Path) -> None:
    """Refresh the package's parsed node config after a default-file update."""
    if path.resolve() != default_path.resolve():
        return
    import training_ops

    loaded = training_ops._read_nodes(path)
    training_ops.nodes.clear()
    training_ops.nodes.update(loaded)


def _read_validate(path: Path) -> tuple[str, dict]:
    text, nodes = _read(path)
    _blocked_hosts(nodes)
    return text, nodes


def _non_empty(label: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not value.strip() or any(ord(char) < 32 for char in value):
        raise NodeRegistryError(f"{label} must be a non-empty value without control characters")
    return value


def _validate_profile(name: str, value: dict) -> None:
    for key in ("ssh_host", "ssh_user", "ssh_identity_file", "runs", "python"):
        field_value = value.get(key)
        if not isinstance(field_value, str):
            raise NodeRegistryError(f"profile {name} needs a string {key}")
        _non_empty(key, field_value)
    port = value.get("ssh_port")
    if type(port) is not int or not 1 <= port <= 65535:
        raise NodeRegistryError("ssh_port must be an integer from 1 to 65535")
    if "role" in value:
        role = value["role"]
        if not isinstance(role, str):
            raise NodeRegistryError(f"profile {name} role must be a string")
        _non_empty("role", role)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="training-ops nodes")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="list registered single-node profiles")

    add = commands.add_parser("add", help="add or update one node profile")
    add.add_argument("name")
    add.add_argument("--ssh-host", "--host", dest="ssh_host")
    add.add_argument("--ssh-port", "--port", dest="ssh_port", type=int)
    add.add_argument("--ssh-user", "--user", dest="ssh_user")
    add.add_argument(
        "--identity", "--ssh-identity", "--ssh-identity-file",
        dest="ssh_identity_file",
    )
    add.add_argument("--runs", "--runs-path")
    add.add_argument("--python", "--python-path")
    add.add_argument("--role")

    use = commands.add_parser("use", help="select one registered node profile")
    use.add_argument("name")
    return parser


def _list(nodes: dict) -> int:
    profiles = nodes.get("single_node", {})
    if not isinstance(profiles, dict):
        raise NodeRegistryError("single_node must be a TOML table")
    selected = active_host(nodes)
    if not profiles:
        print("No node profiles are registered.")
        return 0
    for name in sorted(profiles):
        marker = "*" if name == selected else " "
        item = profile(nodes, name) or {}
        address = item.get("ssh_host", "")
        role = item.get("role", "")
        details = f" {address}" if address else ""
        if role:
            details += f" ({role})"
        print(f"{marker} {name}{details}")
    return 0


def _add(path: Path, text: str, nodes: dict, args: argparse.Namespace) -> int:
    name = _validate_name(args.name)
    existing = profile(nodes, name)
    values = {
        key: getattr(args, key)
        for key in (*required_fields, "role")
        if getattr(args, key) is not None
    }
    for key in ("ssh_host", "ssh_user", "ssh_identity_file", "runs", "python", "role"):
        if key in values:
            values[key] = _non_empty(key, values[key])
    if "ssh_port" in values and not 1 <= values["ssh_port"] <= 65535:
        raise NodeRegistryError("ssh_port must be an integer from 1 to 65535")
    old_host = (existing or {}).get("ssh_host")
    _check_not_blocked(nodes, name, values.get("ssh_host", old_host))
    merged = dict(existing or {})
    merged.update(values)
    missing = [key for key in required_fields if key not in merged]
    if missing:
        required = ", ".join(missing)
        raise NodeRegistryError(f"profile {name} needs these fields: {required}")
    _validate_profile(name, merged)
    updates = {key: merged[key] for key in values}
    updated = _replace_table_values(text, f"single_node.{name}", updates)
    _write_atomic(path, updated)
    _reload_training_ops(path)
    print(f"Registered node profile {name}.")
    return 0


def _use(path: Path, text: str, nodes: dict, name: str) -> int:
    name = _validate_name(name)
    selected = profile(nodes, name)
    if selected is None:
        raise NodeRegistryError(f"node profile {name} is not registered")
    _validate_profile(name, selected)
    _check_not_blocked(nodes, name, selected.get("ssh_host"))
    updated = _replace_table_values(text, "selection", {"active_node": name})
    _write_atomic(path, updated)
    _reload_training_ops(path)
    print(f"Selected node profile {name}.")
    return 0


def cli(arguments: list[str] | tuple[str, ...], path: str | Path | None = None) -> int:
    """Run the local nodes command. Arguments may start with the word nodes."""
    args = list(arguments)
    if args[:1] == ["nodes"]:
        args = args[1:]
    parser = _parser()
    if not args or args[0] in {"-h", "--help", "help"}:
        parser.print_help()
        return 0
    parsed = parser.parse_args(args)
    config_path = Path(path) if path is not None else default_path
    try:
        text, nodes = _read_validate(config_path)
        if parsed.command == "list":
            return _list(nodes)
        if parsed.command == "add":
            return _add(config_path, text, nodes, parsed)
        if parsed.command == "use":
            return _use(config_path, text, nodes, parsed.name)
    except NodeRegistryError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    raise AssertionError(f"unknown nodes command: {parsed.command}")


__all__ = ["NodeRegistryError", "active_host", "cli", "profile"]
