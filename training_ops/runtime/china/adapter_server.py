import argparse
import fcntl
import hashlib
import json
import os
import re
import socket
import shutil
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from threading import Lock
from urllib.parse import unquote, urlsplit


adapter_config_name = "adapter_config.json"
adapter_weights_name = "adapter_model.safetensors"
adapter_manifest_name = "adapter_manifest.json"
adapter_digest_name = "adapter_model.safetensors.sha256"
stable_name = "STABLE"
cache_name = ".adapter-predictive-cache"
lock_name = ".adapter-predictive-upgrade.lock"
digest_pattern = re.compile(r"[0-9a-f]{64}\Z")
step_pattern = re.compile(r"step_(\d+)\Z")
run_pattern = re.compile(r"[A-Za-z0-9_-]+\Z")
transfer_write_bytes = 256 * 1024
# Keep the server idle read bound above the worker's 45-second chunk deadline.
connection_timeout_seconds = 60


class UpgradeError(RuntimeError):
    pass


_identity_cache: dict[
    tuple[Path, bool, bool, tuple[tuple[int, int, int, int, int], ...]],
    dict[str, object],
] = {}
_identity_lock = Lock()


def _json_object(path: Path, description: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise UpgradeError(f"{description} is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise UpgradeError(f"{description} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise UpgradeError(f"{description} is not an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(16 * 1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise UpgradeError(f"cannot read {path.name}") from exc
    return digest.hexdigest()


def _sha256_fd(fd: int, name: str) -> str:
    digest = hashlib.sha256()
    offset = 0
    try:
        while chunk := os.pread(fd, 16 * 1024 * 1024, offset):
            digest.update(chunk)
            offset += len(chunk)
    except OSError as exc:
        raise UpgradeError(f"cannot read {name}") from exc
    return digest.hexdigest()


def _stat_identity_from_stat(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise UpgradeError(f"cannot stat {path.name}") from exc
    return _stat_identity_from_stat(stat)


def _fd_stat_identity(fd: int) -> tuple[int, int, int, int, int]:
    try:
        return _stat_identity_from_stat(os.fstat(fd))
    except OSError as exc:
        raise UpgradeError("cannot stat served adapter file") from exc


def _file_records(manifest: dict[str, object], description: str) -> dict[str, dict[str, object]]:
    records = manifest.get("files")
    if not isinstance(records, list):
        raise UpgradeError(f"{description} has no file records")
    result: dict[str, dict[str, object]] = {}
    for item in records:
        if not isinstance(item, dict):
            raise UpgradeError(f"{description} has an invalid file record")
        name = item.get("name")
        size = item.get("size")
        digest = item.get("sha256")
        if (
            not isinstance(name, str)
            or name in result
            or name not in {adapter_config_name, adapter_weights_name}
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or digest_pattern.fullmatch(digest) is None
        ):
            raise UpgradeError(f"{description} has an invalid file identity")
        result[name] = {"size": size, "sha256": digest}
    if set(result) != {adapter_config_name, adapter_weights_name}:
        raise UpgradeError(f"{description} has an incomplete file set")
    return result


def _identity_from_published_directory(
    directory: Path,
    step: int,
    *,
    cache: bool,
    weight_fd: int | None = None,
    verify_weights: bool = True,
) -> dict[str, object]:
    if directory.is_symlink() or not directory.is_dir():
        raise UpgradeError(f"adapter step {step} directory is missing")
    if not cache and (directory / stable_name).is_symlink():
        raise UpgradeError(f"adapter step {step} marker is a symlink")
    if not cache and not (directory / stable_name).is_file():
        raise UpgradeError(f"adapter step {step} is not stable")

    if cache:
        identity = _json_object(directory / ".identity.json", f"adapter cache step {step} identity")
        if identity.get("step") != step:
            raise UpgradeError(f"adapter cache step identity differs for step {step}")
        records = {
            adapter_config_name: {
                "size": identity.get("config_size"),
                "sha256": identity.get("config_sha256"),
            },
            adapter_weights_name: {
                "size": identity.get("weights_size"),
                "sha256": identity.get("weights_sha256"),
            },
        }
        for name, record in records.items():
            if (
                type(record["size"]) is not int
                or record["size"] <= 0
                or not isinstance(record["sha256"], str)
                or digest_pattern.fullmatch(record["sha256"]) is None
            ):
                raise UpgradeError(f"adapter cache step {step} has an invalid {name} identity")
        description = f"adapter cache step {step}"
    else:
        manifest = _json_object(directory / adapter_manifest_name, f"adapter step {step} manifest")
        if manifest.get("kind") != "lora-adapter" or manifest.get("target_step") != step:
            raise UpgradeError(f"adapter step {step} manifest has the wrong identity")
        records = _file_records(manifest, f"adapter step {step} manifest")
        description = f"adapter step {step}"

    if records[adapter_weights_name]["size"] <= 0:
        raise UpgradeError(f"{description} weights are empty")

    for name, record in records.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != record["size"]:
            raise UpgradeError(f"{description} file is incomplete: {name}")

    digest_path = directory / adapter_digest_name
    if digest_path.is_symlink() or not digest_path.is_file():
        raise UpgradeError(f"{description} weight checksum is missing")
    try:
        sidecar_digest = digest_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise UpgradeError(f"{description} weight checksum cannot be read") from exc
    if sidecar_digest != records[adapter_weights_name]["sha256"]:
        raise UpgradeError(f"{description} weight checksum differs")
    opened_weight_stat = _fd_stat_identity(weight_fd) if weight_fd is not None else None
    identity_path = directory / (".identity.json" if cache else adapter_manifest_name)
    stat_paths = [
        directory / adapter_config_name,
        directory / adapter_weights_name,
        identity_path,
        digest_path,
    ]
    if not cache:
        stat_paths.append(directory / stable_name)
    cache_key = (
        directory.resolve(),
        cache,
        verify_weights,
        tuple(_stat_identity(path) for path in stat_paths),
    )
    if opened_weight_stat is not None and opened_weight_stat != cache_key[3][1]:
        raise UpgradeError(f"{description} weights changed before validation")
    with _identity_lock:
        cached = _identity_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        actual_config_digest = _sha256(directory / adapter_config_name)
        if actual_config_digest != records[adapter_config_name]["sha256"]:
            raise UpgradeError(f"{description} configuration checksum differs")
        if verify_weights:
            actual_weights_digest = (
                _sha256_fd(weight_fd, adapter_weights_name)
                if weight_fd is not None
                else _sha256(directory / adapter_weights_name)
            )
            if actual_weights_digest != records[adapter_weights_name]["sha256"]:
                raise UpgradeError(f"{description} weight checksum differs")
        if weight_fd is not None and _fd_stat_identity(weight_fd) != opened_weight_stat:
            raise UpgradeError(f"{description} weights changed during validation")
        validated_identity = {
            "step": step,
            "config_size": records[adapter_config_name]["size"],
            "config_sha256": records[adapter_config_name]["sha256"],
            "weights_size": records[adapter_weights_name]["size"],
            "weights_sha256": records[adapter_weights_name]["sha256"],
        }
        _identity_cache[cache_key] = validated_identity
        return validated_identity.copy()


def _cache_directory(active_run: Path, step: int) -> Path:
    return active_run / cache_name / f"step_{step}"


def _find_base(active_run: Path, step: int) -> tuple[Path, dict[str, object]]:
    errors: list[str] = []
    published = active_run / f"step_{step}"
    try:
        return published, _identity_from_published_directory(published, step, cache=False)
    except UpgradeError as exc:
        errors.append(str(exc))
    cached = _cache_directory(active_run, step)
    try:
        return cached, _identity_from_published_directory(cached, step, cache=True)
    except UpgradeError as exc:
        errors.append(str(exc))
    raise UpgradeError(f"cannot find exact adapter base step {step}: {'; '.join(errors)}")


def _link_cache_entry(source: Path, identity: dict[str, object], destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        try:
            existing = _identity_from_published_directory(destination, int(identity["step"]), cache=True)
        except UpgradeError:
            shutil.rmtree(destination) if destination.is_dir() and not destination.is_symlink() else destination.unlink()
        else:
            if existing != identity:
                raise UpgradeError(f"adapter cache step {identity['step']} identity differs")
            return
    destination.mkdir(parents=True)
    for name in (adapter_config_name, adapter_weights_name):
        source_path = source / name
        target_path = destination / name
        try:
            os.link(source_path, target_path)
        except OSError as exc:
            raise UpgradeError(f"cannot retain adapter cache step {identity['step']}") from exc
    (destination / adapter_digest_name).write_text(f"{identity['weights_sha256']}\n", encoding="ascii")
    (destination / ".identity.json").write_text(
        json.dumps(identity, separators=(",", ":")) + "\n", encoding="utf-8"
    )


def _prepare_cache(
    active_run: Path,
    entries: list[tuple[Path, dict[str, object]]],
    pending: Path,
) -> None:
    pending.mkdir(parents=True)
    for source, identity in entries:
        step = int(identity["step"])
        _link_cache_entry(source, identity, pending / f"step_{step}")


def _promote_cache(active_run: Path, pending: Path, newest_step: int) -> None:
    root = active_run / cache_name
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise UpgradeError("adapter cache root is not a directory")
    root.mkdir(exist_ok=True)

    prepared: dict[int, tuple[Path, Path, dict[str, object]]] = {}
    for step in (newest_step, newest_step - 1):
        source = pending / f"step_{step}"
        destination = root / f"step_{step}"
        identity = _identity_from_published_directory(source, step, cache=True)
        if destination.exists() or destination.is_symlink():
            existing = _identity_from_published_directory(destination, step, cache=True)
            if existing != identity:
                raise UpgradeError(f"adapter cache step {step} identity differs")
        prepared[step] = source, destination, identity

    for source, destination, _identity in prepared.values():
        if destination.exists():
            shutil.rmtree(source)
        else:
            os.replace(source, destination)

    for step, (_source, destination, identity) in prepared.items():
        if _identity_from_published_directory(destination, step, cache=True) != identity:
            raise UpgradeError(f"published adapter cache step {step} identity differs")


def _prune_cache(active_run: Path, newest_step: int) -> None:
    root = active_run / cache_name
    if root.is_symlink() or not root.is_dir():
        raise UpgradeError("adapter cache root is not a directory")
    for step in (newest_step, newest_step - 1):
        _identity_from_published_directory(root / f"step_{step}", step, cache=True)
    for candidate in root.iterdir():
        match = step_pattern.fullmatch(candidate.name)
        if match is None or int(match.group(1)) >= newest_step - 1:
            continue
        if candidate.is_dir() and not candidate.is_symlink():
            shutil.rmtree(candidate)
        else:
            candidate.unlink()


def _install_predictive_manifest(
    target: Path,
    build: Path,
    manifest: dict[str, object],
    step: int,
) -> None:
    if (
        manifest.get("version") != 2
        or manifest.get("mode") != "delta"
        or manifest.get("target_step") != step
        or manifest.get("base_step") != step - 1
        or manifest.get("base2_step") != step - 2
        or not isinstance(manifest.get("chunks"), list)
        or not manifest["chunks"]
    ):
        raise UpgradeError("predictive manifest build returned an invalid manifest")
    chunk_names: list[str] = []
    seen_chunks: set[str] = set()
    for item in manifest["chunks"]:
        if not isinstance(item, dict) or not isinstance(item.get("file"), str):
            raise UpgradeError("predictive manifest contains an invalid chunk")
        name = item["file"]
        if (
            Path(name).name != name
            or name in seen_chunks
            or not name.startswith(f"{adapter_weights_name}.transport-predictive-")
        ):
            raise UpgradeError("predictive manifest contains an unsafe chunk name")
        chunk = build / name
        if chunk.is_symlink() or not chunk.is_file():
            raise UpgradeError(f"predictive manifest chunk is missing: {name}")
        if chunk.stat().st_size != item.get("wire_size"):
            raise UpgradeError(f"predictive manifest chunk size differs: {name}")
        seen_chunks.add(name)
        chunk_names.append(name)
    target_manifest = target / adapter_manifest_name
    target_digest = target / adapter_digest_name
    build_manifest = build / adapter_manifest_name
    build_digest = build / adapter_digest_name
    if target_manifest.is_symlink() or target_digest.is_symlink() or not build_manifest.is_file() or not build_digest.is_file():
        raise UpgradeError("predictive manifest files are not regular files")
    for name in chunk_names:
        os.replace(build / name, target / name)
    os.replace(build_digest, target_digest)
    os.replace(build_manifest, target_manifest)
    for name in (adapter_config_name, adapter_weights_name):
        for old_chunk in target.glob(f"{name}.transport-delta-*"):
            if old_chunk.is_file() and not old_chunk.is_symlink():
                old_chunk.unlink()


def _predictive_step_url(base_url: str, step: int) -> str:
    return f"{base_url.rstrip('/')}/step_{step}"


def _has_newer_stable_step(active_run: Path, step: int) -> bool:
    """Keep the public manifest for an older stable step unchanged."""
    try:
        candidates = active_run.iterdir()
        for candidate in candidates:
            match = step_pattern.fullmatch(candidate.name)
            if match is None or int(match.group(1)) <= step:
                continue
            stable = candidate / stable_name
            if stable.is_file() and not stable.is_symlink():
                return True
    except OSError as exc:
        raise UpgradeError("cannot inspect active adapter steps") from exc
    return False


def _upgrade_v1_step(active_run: Path, target: Path, step: int, source_base_url: str) -> None:
    lock_path = active_run / lock_name
    if lock_path.is_symlink():
        raise UpgradeError("predictive upgrade lock is a symlink")
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = lock_path.open("a+")
    except OSError as exc:
        raise UpgradeError("cannot open predictive upgrade lock") from exc
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            stable = target / stable_name
            if stable.is_symlink() or not stable.is_file():
                raise UpgradeError(f"adapter step {step} is not stable")
            manifest_path = target / adapter_manifest_name
            manifest = _json_object(manifest_path, f"adapter step {step} manifest")
            if manifest.get("version") == 2:
                return
            if manifest.get("version") != 1 or manifest.get("kind") != "lora-adapter":
                raise UpgradeError(f"adapter step {step} has an unsupported manifest")
            if manifest.get("target_step") != step:
                raise UpgradeError(f"adapter step {step} manifest has the wrong target")
            target_identity = _identity_from_published_directory(target, step, cache=False)
            base1, base1_identity = _find_base(active_run, step - 1)
            base2, base2_identity = _find_base(active_run, step - 2)
            identities = (target_identity, base1_identity, base2_identity)
            if len({item["config_sha256"] for item in identities}) != 1:
                raise UpgradeError(f"adapter step {step} predecessor configurations differ")
            if len({item["weights_size"] for item in identities}) != 1:
                raise UpgradeError(f"adapter step {step} predecessor sizes differ")

            cache_root = active_run / cache_name
            pending = cache_root / f".pending-{os.getpid()}-{time.time_ns()}"
            build = active_run / f".predictive-build-{os.getpid()}-{time.time_ns()}"
            try:
                _prepare_cache(
                    active_run,
                    [
                        (target, target_identity),
                        (base1, base1_identity),
                        (base2, base2_identity),
                    ],
                    pending,
                )
                build.mkdir(parents=True)
                for name in (adapter_config_name, adapter_weights_name):
                    os.link(target / name, build / name)
                from prime_rl.utils.adapter_delta import publish_adapter_manifest

                manifest = publish_adapter_manifest(
                    build,
                    pending / f"step_{step - 1}",
                    base2_dir=pending / f"step_{step - 2}",
                    target_url=_predictive_step_url(source_base_url, step),
                    base_url=_predictive_step_url(source_base_url, step - 1),
                    base2_url=_predictive_step_url(source_base_url, step - 2),
                    target_step=step,
                    base_step=step - 1,
                    base2_step=step - 2,
                )
                _promote_cache(active_run, pending, step)
                _install_predictive_manifest(target, build, manifest, step)
                _prune_cache(active_run, step)
            finally:
                if build.exists():
                    shutil.rmtree(build)
                if pending.exists():
                    shutil.rmtree(pending)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _safe_request_path(handler: SimpleHTTPRequestHandler) -> tuple[Path, tuple[str, ...]]:
    raw_path = unquote(urlsplit(handler.path).path)
    if "\\" in raw_path:
        raise UpgradeError("request path contains a backslash")
    pieces = tuple(part for part in PurePosixPath(raw_path).parts if part not in {"/", ""})
    if any(part in {".", ".."} for part in pieces):
        raise UpgradeError("request path contains traversal")
    root = Path(handler.directory).resolve()
    path = Path(handler.translate_path(handler.path)).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise UpgradeError("request path is outside the adapter root") from exc
    relative = path.relative_to(root)
    if cache_name in relative.parts or any(
        part.startswith(".predictive-") for part in relative.parts
    ):
        raise UpgradeError("private adapter state is not served")
    return path, pieces


class RangeHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, upgrade_path=None, source_base_url=None, **kwargs):
        self.upgrade_path = upgrade_path
        self.source_base_url = source_base_url
        super().__init__(*args, **kwargs)

    def setup(self):
        super().setup()
        self.connection.settimeout(connection_timeout_seconds)

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            self.close_connection = True

    def _maybe_upgrade(self):
        path, pieces = _safe_request_path(self)
        if self.upgrade_path is None:
            return path
        if cache_name in pieces or any(part.startswith(".predictive-") for part in pieces):
            raise UpgradeError("private adapter state is not served")
        prefix_length = len(self.upgrade_path)
        if len(pieces) < prefix_length + 2 or pieces[:prefix_length] != self.upgrade_path:
            return path
        match = step_pattern.fullmatch(pieces[prefix_length])
        if match is None:
            return path
        step = int(match.group(1))
        if step < 2:
            return path
        if pieces[prefix_length + 1] == stable_name:
            return path
        active_run = Path(self.directory).resolve().joinpath(*self.upgrade_path)
        if _has_newer_stable_step(active_run, step):
            return path
        target = active_run / pieces[prefix_length]
        _upgrade_v1_step(active_run, target, step, self.source_base_url)
        return path

    def send_head(self):
        self.__dict__.pop("remaining", None)
        source = None
        try:
            _request_path, pieces = _safe_request_path(self)
            path = self._maybe_upgrade()
            prefix = self.upgrade_path
            step_path = None
            step = None
            if prefix is not None:
                prefix_length = len(prefix)
                if len(pieces) > prefix_length and pieces[:prefix_length] == prefix:
                    match = step_pattern.fullmatch(pieces[prefix_length])
                    if match is not None:
                        step = int(match.group(1))
                        step_path = Path(self.directory).resolve().joinpath(
                            *pieces[: prefix_length + 1]
                        )
            if not os.path.isdir(path):
                try:
                    source = os.fdopen(
                        os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)),
                        "rb",
                    )
                except OSError:
                    self.send_error(404)
                    return None
            if step_path is not None and step is not None:
                weight_path = step_path / adapter_weights_name
                weight_fd = source.fileno() if source is not None and path == weight_path else None
                _identity_from_published_directory(
                    step_path,
                    step,
                    cache=False,
                    weight_fd=weight_fd,
                    verify_weights=weight_fd is not None,
                )
        except UpgradeError as exc:
            if source is not None:
                source.close()
            self.send_error(500, str(exc))
            return None
        if os.path.isdir(path):
            return super().send_head()
        size = os.fstat(source.fileno()).st_size
        start, end = 0, size - 1
        if value := self.headers.get("Range"):
            try:
                start_text, end_text = value.removeprefix("bytes=").split("-", 1)
                start = int(start_text)
                end = min(int(end_text), end) if end_text else end
                if start > end:
                    raise ValueError
            except ValueError:
                source.close()
                self.send_error(416)
                return None
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Last-Modified", self.date_time_string(os.fstat(source.fileno()).st_mtime))
        self.end_headers()
        source.seek(start)
        self.remaining = end - start + 1
        return source

    def copyfile(self, source, output):
        if not hasattr(self, "remaining"):
            return super().copyfile(source, output)
        while self.remaining:
            chunk = source.read(min(self.remaining, transfer_write_bytes))
            if not chunk:
                self.close_connection = True
                source.close()
                break
            try:
                output.write(chunk)
            except OSError:
                self.close_connection = True
                source.close()
                return
            self.remaining -= len(chunk)
        source.close()


class AdapterServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--upgrade-run")
    parser.add_argument("--upgrade-path")
    parser.add_argument("--source-base-url")
    args = parser.parse_args()

    upgrade_path = None
    upgrade_values = (args.upgrade_run, args.upgrade_path, args.source_base_url)
    if any(value is not None for value in upgrade_values):
        if not all(value is not None for value in upgrade_values):
            parser.error("predictive upgrade requires run, path, and source URL")
        if run_pattern.fullmatch(args.upgrade_run) is None:
            parser.error("--upgrade-run contains an unsafe run name")
        expected_path = f"{args.upgrade_run}/run_{args.upgrade_run}/broadcasts"
        if args.upgrade_path != expected_path:
            parser.error("--upgrade-path is not the active run broadcast path")
        if not args.source_base_url.startswith(("http://", "https://")):
            parser.error("--source-base-url is required for predictive upgrade")
        parsed_source = urlsplit(args.source_base_url.rstrip("/"))
        if (
            parsed_source.hostname not in {"127.0.0.1", "localhost"}
            or parsed_source.username
            or parsed_source.password
            or parsed_source.query
            or parsed_source.fragment
            or parsed_source.path.rstrip("/") != f"/{expected_path}"
        ):
            parser.error("--source-base-url is not bound to --upgrade-path")
        upgrade_path = tuple(expected_path.split("/"))

    AdapterServer(
        ("127.0.0.1", args.port),
        partial(
            RangeHandler,
            directory=args.directory,
            upgrade_path=upgrade_path,
            source_base_url=args.source_base_url.rstrip("/") if args.source_base_url else None,
        ),
    ).serve_forever()


if __name__ == "__main__":
    main()
