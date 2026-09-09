from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import mimetypes
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


relay_manifest_name = "relay_manifest.json"
ready_marker_name = "READY"
relaying_marker_name = "RELAYING"
relay_transport = "chunked-shuffled-xor-zstd"
predictive_transport = "chunked-bf16-linear-zigzag-sparse-zstd"
full_transport = "full"
digest_pattern = re.compile(r"[0-9a-f]{64}\Z")


class RelayRequestError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class CanonicalSource:
    origin: str
    prefix: str


@dataclass(frozen=True)
class RelayFile:
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True)
class RelayIndex:
    transport: str
    files: dict[str, dict[str, object]]


_relay_index_cache_lock = threading.Lock()
_relay_index_cache: dict[str, tuple[tuple[object, ...], RelayIndex]] = {}
_MAX_RELAY_INDEX_CACHE_ENTRIES = 8


class RelayConfig:
    def __init__(
        self,
        cache_dir: Path,
        canonical_sources: list[str],
        source_port_first: int,
        source_port_last: int,
    ):
        if not cache_dir.is_absolute():
            raise ValueError("relay cache directory must be absolute")
        if not 1 <= source_port_first <= source_port_last <= 65535:
            raise ValueError("relay source port range is invalid")
        self.cache_dir = cache_dir
        self.source_port_first = source_port_first
        self.source_port_last = source_port_last
        self.sources = self._parse_sources(
            canonical_sources, source_port_first, source_port_last
        )
        if not self.sources:
            raise ValueError("relay needs at least one canonical source URL")

    @staticmethod
    def _parse_sources(
        values: list[str], source_port_first: int, source_port_last: int
    ) -> list[CanonicalSource]:
        sources: list[CanonicalSource] = []
        for value in values:
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"}:
                raise ValueError("relay source URL must use HTTP or HTTPS")
            if parsed.hostname not in {"127.0.0.1", "localhost"}:
                raise ValueError("relay source URL must use a loopback host")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("relay source URL cannot contain credentials, query, or fragment")
            if parsed.port is None:
                raise ValueError("relay source URL must include a port")
            if not source_port_first <= parsed.port <= source_port_last:
                raise ValueError("relay source URL port is outside the source range")
            prefix = parsed.path.rstrip("/") or "/"
            if any(part in {".", "..", ""} for part in prefix.split("/")[1:]):
                raise ValueError("relay source URL has an unsafe path")
            origin = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
            source = CanonicalSource(origin=origin, prefix=prefix)
            if source not in sources:
                sources.append(source)
        return sources

    def source_url(self, request_path: str, host: str) -> tuple[str, str]:
        parsed_host = urlsplit(f"//{host}")
        if parsed_host.hostname not in {"127.0.0.1", "localhost"}:
            raise RelayRequestError(400, "relay requests must use a loopback host")
        try:
            source_port = parsed_host.port
        except ValueError as exc:
            raise RelayRequestError(400, "relay request host port is invalid") from exc
        if source_port is None or not self.source_port_first <= source_port <= self.source_port_last:
            raise RelayRequestError(400, "relay request host port is not an adapter source port")
        for source in self.sources:
            if source.prefix == "/":
                matches = request_path.startswith("/")
            else:
                matches = request_path == source.prefix or request_path.startswith(f"{source.prefix}/")
            if not matches:
                continue
            directory, separator, filename = request_path.rpartition("/")
            if not separator or not filename or directory == "":
                break
            return f"{source.origin}{directory}", filename
        raise RelayRequestError(404, "source URL is not configured")

    def cache_path(self, source_url: str) -> Path:
        name = hashlib.sha256(source_url.encode("utf-8")).hexdigest()
        return self.cache_dir / name


@contextmanager
def reader_lock(cache_path: Path):
    lock_path = cache_path.parent / f".{cache_path.name}.readers.lock"
    if lock_path.is_symlink() or not lock_path.is_file():
        raise RelayRequestError(404, "adapter reader lock is missing")
    try:
        lock = lock_path.open("rb")
    except OSError as exc:
        raise RelayRequestError(404, "adapter reader lock is unavailable") from exc
    with lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _safe_relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RelayRequestError(500, "relay manifest has an invalid file path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RelayRequestError(500, "relay manifest has an unsafe file path")
    return Path(*path.parts)


def _predictive_step_url(source_url: str, step: int) -> str:
    parsed = urlsplit(source_url)
    path = Path(parsed.path.rstrip("/")).parent / f"step_{step}"
    return parsed._replace(path=str(path)).geturl()


def _validate_predictive_source_manifest(manifest: dict[str, object], source_url: str) -> None:
    """Validate the second-order adapter identity before serving any file."""
    if (
        manifest.get("version") != 2
        or manifest.get("kind") != "lora-adapter"
        or manifest.get("mode") != "delta"
        or manifest.get("transport") != predictive_transport
    ):
        raise RelayRequestError(500, "relay source manifest selects an unsupported predictive transport")
    target_step = manifest.get("target_step")
    base_step = manifest.get("base_step")
    base2_step = manifest.get("base2_step")
    if (
        type(target_step) is not int
        or target_step < 1
        or base_step != target_step - 1
        or base2_step != (0 if target_step == 1 else target_step - 2)
    ):
        raise RelayRequestError(500, "relay source manifest steps are not adjacent")
    parsed_source = urlsplit(source_url)
    if Path(parsed_source.path.rstrip("/")).name != f"step_{target_step}":
        raise RelayRequestError(500, "relay source URL does not identify the target step")

    def valid_digest(value: object) -> bool:
        return isinstance(value, str) and digest_pattern.fullmatch(value) is not None

    for prefix in ("target", "base", "base2"):
        if (
            type(manifest.get(f"{prefix}_size")) is not int
            or manifest[f"{prefix}_size"] <= 0
            or not valid_digest(manifest.get(f"{prefix}_sha256"))
        ):
            raise RelayRequestError(500, f"relay source manifest has an invalid {prefix} identity")
    for name in ("config_sha256", "base_config_sha256", "base2_config_sha256"):
        if not valid_digest(manifest.get(name)):
            raise RelayRequestError(500, f"relay source manifest has an invalid {name}")
    if target_step == 1 and (
        manifest.get("base_url") != manifest.get("base2_url")
        or manifest.get("base_sha256") != manifest.get("base2_sha256")
        or manifest.get("base_size") != manifest.get("base2_size")
        or manifest.get("base_config_sha256") != manifest.get("base2_config_sha256")
    ):
        raise RelayRequestError(500, "step-one predictive adapter bases are not the same identity")
    expected_predictor = (
        "bf16-first-order-exact"
        if target_step == 1
        else "bf16-float32-second-order-rne"
    )
    if manifest.get("predictor") != expected_predictor:
        raise RelayRequestError(500, "relay source manifest predictor does not match its step")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise RelayRequestError(500, "relay source manifest has no file records")
    file_map = {
        item.get("name"): item
        for item in files
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if (
        len(file_map) != len(files)
        or set(file_map) != {"adapter_config.json", "adapter_model.safetensors"}
    ):
        raise RelayRequestError(500, "relay source manifest has an incomplete file set")
    target_file = file_map["adapter_model.safetensors"]
    config_file = file_map["adapter_config.json"]
    target_size = manifest["target_size"]
    if manifest["base_size"] != target_size or manifest["base2_size"] != target_size:
        raise RelayRequestError(500, "relay source predecessor sizes differ from the target")
    if (
        manifest.get("target_size") != target_file.get("size")
        or manifest.get("target_sha256") != target_file.get("sha256")
        or manifest.get("config_size") != config_file.get("size")
        or manifest.get("config_sha256") != config_file.get("sha256")
    ):
        raise RelayRequestError(500, "relay source manifest target identity differs from its files")
    for name, expected in (
        ("target_url", source_url),
        ("base_url", _predictive_step_url(source_url, target_step - 1)),
        ("base2_url", _predictive_step_url(source_url, 0 if target_step == 1 else target_step - 2)),
    ):
        value = manifest.get(name)
        if not isinstance(value, str) or value != expected:
            raise RelayRequestError(500, f"relay source manifest {name} is not adjacent")


def _stat_identity(path: Path) -> tuple[int, int, int, int] | None:
    try:
        stat = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RelayRequestError(500, "relay cache metadata cannot be read") from exc
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _relay_index_signature(
    cache_path: Path,
    published_path: Path,
    source_url: str,
    *,
    ready: Path,
    relaying_path: Path,
) -> tuple[object, ...]:
    return (
        cache_path.as_posix(),
        published_path.as_posix(),
        source_url,
        "ready" if published_path == cache_path else "relaying",
        _stat_identity(published_path / relay_manifest_name),
        _stat_identity(published_path / "adapter_manifest.json"),
        _stat_identity(ready),
        _stat_identity(relaying_path / relaying_marker_name),
    )


def _build_relay_index(
    published_path: Path,
    source_url: str,
) -> RelayIndex:
    manifest_path = published_path / relay_manifest_name
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RelayRequestError(500, "relay manifest cannot be read") from exc
    if not isinstance(manifest, dict):
        raise RelayRequestError(500, "relay manifest is invalid")
    transport = manifest.get("transport")
    required_names = {"adapter_manifest.json", "adapter_config.json", "adapter_model.safetensors"}
    if (
        manifest.get("version") not in {1, 2}
        or manifest.get("source_url") != source_url
        or transport not in {relay_transport, predictive_transport, full_transport}
        or not isinstance(manifest.get("files"), dict)
    ):
        raise RelayRequestError(500, "relay manifest is invalid")
    if transport == predictive_transport:
        source_manifest_path = published_path / "adapter_manifest.json"
        try:
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RelayRequestError(500, "relay source manifest cannot be read") from exc
        if not isinstance(source_manifest, dict):
            raise RelayRequestError(500, "relay source manifest is invalid")
        _validate_predictive_source_manifest(source_manifest, source_url)
        for name in (
            "target_step",
            "base_step",
            "base2_step",
            "target_sha256",
            "base_sha256",
            "base2_sha256",
            "target_size",
            "base_size",
            "base2_size",
        ):
            if name in manifest and manifest.get(name) != source_manifest.get(name):
                raise RelayRequestError(500, f"relay manifest {name} differs from its source manifest")
    files = manifest["files"]
    for name, raw_spec in files.items():
        if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
            raise RelayRequestError(500, "relay manifest has an invalid file name")
        if not isinstance(raw_spec, dict):
            raise RelayRequestError(500, "relay manifest file entry is invalid")
        relative = _safe_relative_path(raw_spec.get("path"))
        if transport in {relay_transport, predictive_transport} and name not in required_names:
            if relative != Path("transport-chunks") / name:
                raise RelayRequestError(500, "relay manifest has an invalid chunk path")
        size = raw_spec.get("size")
        digest = raw_spec.get("sha256")
        if type(size) is not int or size < 0 or not isinstance(digest, str) or not digest_pattern.fullmatch(digest):
            raise RelayRequestError(500, "relay manifest file identity is invalid")
    if not required_names.issubset(files):
        raise RelayRequestError(500, "relay manifest is missing an adapter file")
    if transport == full_transport and set(files) != required_names:
        raise RelayRequestError(500, "full relay manifest contains transport chunks")
    return RelayIndex(transport=transport, files=files)


def _cached_relay_index(
    cache_path: Path,
    published_path: Path,
    source_url: str,
    *,
    ready: Path,
    relaying_path: Path,
) -> RelayIndex:
    signature = _relay_index_signature(
        cache_path, published_path, source_url, ready=ready, relaying_path=relaying_path
    )
    cache_key = cache_path.as_posix()
    with _relay_index_cache_lock:
        cached = _relay_index_cache.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[1]
    index = _build_relay_index(published_path, source_url)
    if _relay_index_signature(
        cache_path, published_path, source_url, ready=ready, relaying_path=relaying_path
    ) != signature:
        raise RelayRequestError(503, "relay manifest changed during validation")
    with _relay_index_cache_lock:
        _relay_index_cache[cache_key] = (signature, index)
        while len(_relay_index_cache) > _MAX_RELAY_INDEX_CACHE_ENTRIES:
            _relay_index_cache.pop(next(iter(_relay_index_cache)))
    return index


def _load_file(cache_path: Path, source_url: str, requested_name: str) -> RelayFile:
    ready = cache_path / ready_marker_name
    relaying_path = cache_path.with_name(f"{cache_path.name}.partial")
    ready_file = ready.is_file()
    relaying = (
        not ready_file
        and relaying_path.is_dir()
        and (relaying_path / relaying_marker_name).is_file()
        and not (relaying_path / relaying_marker_name).is_symlink()
    )
    if not ready_file and not relaying:
        raise RelayRequestError(404, "adapter is not ready")
    published_path = cache_path if ready_file else relaying_path
    manifest_path = published_path / relay_manifest_name
    if (
        published_path.is_symlink()
        or not published_path.is_dir()
        or (ready_file and ready.is_symlink())
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
    ):
        raise RelayRequestError(404, "adapter is not ready")
    index = _cached_relay_index(
        cache_path,
        published_path,
        source_url,
        ready=ready,
        relaying_path=relaying_path,
    )
    transport = index.transport
    files = index.files
    if relaying and transport == predictive_transport and requested_name == "adapter_model.safetensors":
        raise RelayRequestError(404, "adapter model is not published until staging is complete")
    raw_spec = files.get(requested_name)
    if not isinstance(raw_spec, dict):
        raise RelayRequestError(404, "file is not published by the relay")
    relative = _safe_relative_path(raw_spec["path"])
    size = raw_spec["size"]
    digest = raw_spec["sha256"]
    raw_candidate = published_path / relative
    if raw_candidate.is_symlink():
        raise RelayRequestError(500, "relay manifest points to a symbolic link")
    cursor = published_path
    for part in relative.parts[:-1]:
        cursor /= part
        if cursor.is_symlink():
            raise RelayRequestError(500, "relay manifest traverses a symbolic link")
    candidate = raw_candidate.resolve()
    try:
        candidate.relative_to(published_path.resolve())
    except ValueError as exc:
        raise RelayRequestError(500, "relay manifest escapes the adapter cache") from exc
    if not candidate.is_file() or candidate.stat().st_size != size:
        raise RelayRequestError(503, "relay file is incomplete")
    if relaying and transport == predictive_transport and requested_name not in {
        "adapter_manifest.json",
        "adapter_config.json",
    }:
        # A chunk is visible only after the writer has validated its wire
        # checksum and atomically published a wire or completed proof.
        proofs = (
            ("wire-ready", candidate.with_name(f"{candidate.name}.wire-ready")),
            ("completed", candidate.with_name(f"{candidate.name}.done")),
        )
        proof_valid = False
        for proof_type, proof_path in proofs:
            if proof_path.is_symlink() or not proof_path.is_file():
                continue
            try:
                proof = json.loads(proof_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(proof, dict):
                continue
            if (
                proof.get("file") != requested_name
                or proof.get("wire_size") != size
                or proof.get("wire_sha256") != digest
            ):
                continue
            if proof_type == "wire-ready" and proof.get("type") == "wire-ready":
                proof_valid = True
                break
            if (
                proof_type == "completed"
                and isinstance(proof.get("output_sha256"), str)
                and digest_pattern.fullmatch(proof["output_sha256"]) is not None
            ):
                proof_valid = True
                break
        if not proof_valid:
            raise RelayRequestError(503, "relay chunk proof is unavailable or invalid")
    return RelayFile(path=candidate, size=size, sha256=digest)


def _byte_range(value: str | None, size: int) -> tuple[int, int, int]:
    if value is None:
        return 200, 0, max(0, size - 1)
    if not value.startswith("bytes=") or "," in value:
        raise RelayRequestError(416, "only one byte range is supported")
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
    if match is None or size == 0:
        raise RelayRequestError(416, "byte range is invalid")
    first, last = match.groups()
    if not first and not last:
        raise RelayRequestError(416, "byte range is invalid")
    if not first:
        try:
            count = int(last)
        except ValueError as exc:
            raise RelayRequestError(416, "byte range is invalid") from exc
        if count <= 0:
            raise RelayRequestError(416, "byte range is invalid")
        start = max(0, size - count)
        end = size - 1
    else:
        try:
            start = int(first)
            requested_last = int(last) if last else None
        except ValueError as exc:
            raise RelayRequestError(416, "byte range is invalid") from exc
        if start >= size:
            raise RelayRequestError(416, "byte range starts after the file")
        end = min(size - 1, requested_last) if requested_last is not None else size - 1
        if end < start:
            raise RelayRequestError(416, "byte range ends before it starts")
    return 206, start, end


class RelayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _error(self, error: RelayRequestError) -> None:
        body = f"{error.message}\n".encode("utf-8")
        self.send_response(error.status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command == "GET":
            self.wfile.write(body)

    def _serve(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            raise RelayRequestError(400, "relay request URL is invalid")
        raw_path = parsed.path
        path = unquote(raw_path)
        if not path.startswith("/") or "\x00" in path or "\\" in path:
            raise RelayRequestError(400, "relay request path is invalid")
        if any(part in {".", ".."} for part in path.split("/")):
            raise RelayRequestError(400, "relay request path is unsafe")
        config: RelayConfig = self.server.relay_config  # type: ignore[attr-defined]
        source_url, filename = config.source_url(path, self.headers.get("Host", ""))
        cache_path = config.cache_path(source_url)
        with reader_lock(cache_path):
            selected = _load_file(cache_path, source_url, filename)
            status, start, end = _byte_range(self.headers.get("Range"), selected.size)
            length = end - start + 1 if selected.size else 0
            self.send_response(status)
            self.send_header("Content-Type", mimetypes.guess_type(filename)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{selected.size}")
            self.end_headers()
            if self.command != "GET" or length == 0:
                return
            with selected.path.open("rb") as source:
                source.seek(start)
                remaining = length
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise RelayRequestError(503, "relay file ended before its declared size")
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    def do_GET(self) -> None:
        try:
            self._serve()
        except RelayRequestError as error:
            self._error(error)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_HEAD(self) -> None:
        try:
            self._serve()
        except RelayRequestError as error:
            self._error(error)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class RelayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128

    def __init__(self, address, handler, relay_config: RelayConfig):
        super().__init__(address, handler)
        self.relay_config = relay_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve validated adapter cache files")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--source-port-first", type=int, required=True)
    parser.add_argument("--source-port-last", type=int, required=True)
    parser.add_argument("--canonical-source-url", action="append", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("relay port is invalid")
    if args.source_port_first <= args.port <= args.source_port_last:
        raise SystemExit("relay port overlaps the adapter source range")
    config = RelayConfig(
        cache_dir=args.cache_dir,
        canonical_sources=args.canonical_source_url,
        source_port_first=args.source_port_first,
        source_port_last=args.source_port_last,
    )
    with RelayServer((args.host, args.port), RelayHandler, config) as server:
        server.serve_forever(poll_interval=0.5)


if __name__ == "__main__":
    main()
