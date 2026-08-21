#!/usr/bin/env python3
"""Download one verified file over the Muse reverse-tunnel pool."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import os
import queue
import threading
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("size", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--ports", required=True)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    chunk_size = min(args.chunk_mib * 1024 * 1024, args.size)
    chunk_count = (args.size + chunk_size - 1) // chunk_size
    partial = args.output.with_name(args.output.name + ".partial")
    state = args.output.with_name(args.output.name + ".partial.done")
    state_header = f"size={args.size} chunk={chunk_size}\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.output.is_file() and args.output.stat().st_size == args.size:
        if args.sha256 is None or sha256(args.output) == args.sha256:
            print(f"{args.filename} {args.size} bytes already complete", flush=True)
            return

    ports = [int(value) for value in args.ports.split(",")]

    def port_is_healthy(port: int) -> bool:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            connection.request(
                "GET", "/muse-files/config.json", headers={"Range": "bytes=0-0"}
            )
            response = connection.getresponse()
            return response.status == 206 and response.read() == b"{"
        except Exception:
            return False
        finally:
            connection.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ports)) as executor:
        ports = [
            port
            for port, healthy in zip(ports, executor.map(port_is_healthy, ports))
            if healthy
        ]
    if not ports:
        raise RuntimeError("no Muse checkpoint tunnel passed the range probe")
    print(f"using {len(ports)} healthy checkpoint tunnels", flush=True)

    completed: set[int] = set()
    if state.is_file():
        lines = state.read_text().splitlines()
        if lines and lines[0] + "\n" == state_header:
            completed = {
                int(value)
                for value in lines[1:]
                if value.isdigit() and int(value) < chunk_count
            }
        else:
            state.unlink()

    file_fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(file_fd, args.size)
    state_fd = os.open(state, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
    if os.fstat(state_fd).st_size == 0:
        os.write(state_fd, state_header.encode())
        os.fsync(state_fd)

    tasks: queue.Queue[int | None] = queue.Queue()
    for index in range(chunk_count):
        if index not in completed:
            tasks.put(index)

    lock = threading.Lock()
    started = time.monotonic()
    last_report = 0.0
    initial_done_bytes = sum(
        min(chunk_size, args.size - index * chunk_size) for index in completed
    )

    def report(force: bool = False) -> None:
        nonlocal last_report
        now = time.monotonic()
        if not force and now - last_report < 30:
            return
        with lock:
            done_bytes = sum(
                min(chunk_size, args.size - index * chunk_size)
                for index in completed
            )
            elapsed = max(now - started, 0.001)
            rate = (done_bytes - initial_done_bytes) / elapsed / (1024 * 1024)
            print(
                f"{args.filename}: {done_bytes}/{args.size} bytes "
                f"({100 * done_bytes / args.size:.2f}%), {rate:.2f} MiB/s",
                flush=True,
            )
            last_report = now

    def worker(port: int) -> None:
        consecutive_failures = 0
        while True:
            index = tasks.get()
            if index is None:
                tasks.task_done()
                return
            start = index * chunk_size
            end = min(start + chunk_size, args.size) - 1
            expected = end - start + 1
            connection: http.client.HTTPConnection | None = None
            try:
                connection = http.client.HTTPConnection(
                    "127.0.0.1", port, timeout=args.timeout
                )
                connection.request(
                    "GET",
                    f"/muse-files/{args.filename}",
                    headers={"Range": f"bytes={start}-{end}"},
                )
                response = connection.getresponse()
                if response.status != 206:
                    raise RuntimeError(f"port {port} returned HTTP {response.status}")
                if int(response.getheader("Content-Length", "-1")) != expected:
                    raise RuntimeError(f"port {port} returned an invalid content length")
                offset = start
                remaining = expected
                while remaining:
                    block = response.read(min(1024 * 1024, remaining))
                    if not block:
                        raise RuntimeError(f"port {port} closed the range early")
                    os.pwrite(file_fd, block, offset)
                    offset += len(block)
                    remaining -= len(block)
                with lock:
                    if index not in completed:
                        completed.add(index)
                        os.write(state_fd, f"{index}\n".encode())
                consecutive_failures = 0
                report()
            except Exception as exc:
                consecutive_failures += 1
                print(
                    f"{args.filename}: chunk {index} failed on port {port}: {exc}",
                    flush=True,
                )
                tasks.put(index)
                if consecutive_failures >= 3:
                    print(
                        f"{args.filename}: disabling port {port} after "
                        f"{consecutive_failures} consecutive failures",
                        flush=True,
                    )
                    return
                time.sleep(0.25)
            finally:
                if connection is not None:
                    connection.close()
                tasks.task_done()

    threads = [threading.Thread(target=worker, args=(port,), daemon=True) for port in ports]
    for thread in threads:
        thread.start()
    report(force=True)
    tasks.join()
    for _ in threads:
        tasks.put(None)
    tasks.join()
    for thread in threads:
        thread.join()

    os.fsync(file_fd)
    os.close(file_fd)
    os.fsync(state_fd)
    os.close(state_fd)
    actual_hash = sha256(partial) if args.sha256 else None
    if args.sha256 and actual_hash != args.sha256:
        raise RuntimeError(
            f"{args.filename} SHA-256 is {actual_hash}, expected {args.sha256}"
        )
    os.replace(partial, args.output)
    state.unlink(missing_ok=True)
    report(force=True)
    print(f"{args.filename} {args.size} bytes complete", flush=True)


if __name__ == "__main__":
    main()
