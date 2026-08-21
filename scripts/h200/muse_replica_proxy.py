#!/usr/bin/env python3
"""Least-connections HTTP proxy for independent Muse vLLM replicas."""

from __future__ import annotations

import http.client
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


backends = tuple(
    backend.strip()
    for backend in os.environ.get(
        "MUSE_REPLICA_BACKENDS",
        "127.0.0.1:8001,127.0.0.1:8002,127.0.0.1:8003,127.0.0.1:8004",
    ).split(",")
    if backend.strip()
)
listen_port = int(os.environ.get("MUSE_PROXY_PORT", "8000"))
hop_headers = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

_lock = threading.Lock()
_active = {backend: 0 for backend in backends}
_cursor = 0


def _ordered_backends() -> list[str]:
    global _cursor
    with _lock:
        start = _cursor
        _cursor = (_cursor + 1) % len(backends)
        rotated = backends[start:] + backends[:start]
        return sorted(rotated, key=lambda backend: _active[backend])


def _change_active(backend: str, delta: int) -> None:
    with _lock:
        _active[backend] += delta


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self._proxy()

    def do_POST(self) -> None:
        self._proxy()

    def do_PUT(self) -> None:
        self._proxy()

    def do_DELETE(self) -> None:
        self._proxy()

    def _proxy(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else None
        last_error = "no backend"
        for backend in _ordered_backends():
            host, port_text = backend.rsplit(":", 1)
            connection = http.client.HTTPConnection(host, int(port_text), timeout=3700)
            _change_active(backend, 1)
            started = time.monotonic()
            try:
                headers = {
                    name: value
                    for name, value in self.headers.items()
                    if name.lower() not in hop_headers and name.lower() != "host"
                }
                connection.request(self.command, self.path, body=body, headers=headers)
                response = connection.getresponse()
                self.send_response(response.status, response.reason)
                for name, value in response.getheaders():
                    if name.lower() not in hop_headers:
                        self.send_header(name, value)
                self.end_headers()
                while chunk := response.read(1024 * 1024):
                    self.wfile.write(chunk)
                self.wfile.flush()
                elapsed = time.monotonic() - started
                print(
                    f"backend={backend} method={self.command} path={self.path} "
                    f"status={response.status} seconds={elapsed:.3f}",
                    flush=True,
                )
                return
            except (ConnectionError, OSError, http.client.HTTPException) as error:
                last_error = f"{backend}: {type(error).__name__}: {error}"
            finally:
                connection.close()
                _change_active(backend, -1)
        message = last_error.encode()
        self.send_response(502)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(message)))
        self.end_headers()
        self.wfile.write(message)

    def log_message(self, format: str, *args: object) -> None:
        return


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    if not backends:
        raise SystemExit("MUSE_REPLICA_BACKENDS is empty")
    print(f"Muse replica proxy listening on :{listen_port}; backends={backends}", flush=True)
    ProxyServer(("0.0.0.0", listen_port), ProxyHandler).serve_forever()


if __name__ == "__main__":
    main()
