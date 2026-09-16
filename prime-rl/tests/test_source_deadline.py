import ast
import asyncio
import fcntl
import functools
import json
import math
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import httpx

import test_adapter_server as server_tests
from test_adapter_server import adapter_server, QuietAdapterHandler
from test_predictive_adapter_codec import adapter_delta, valid_safetensor_bytes


def manifest_readers():
    path = Path(__file__).parents[1] / "src/prime_rl/orchestrator/watcher.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name in {"_read_adapter_manifest", "_read_remote_adapter_manifest"}]
    namespace = dict(asyncio=asyncio, httpx=httpx, time=time, json=json, math=math,
                     adapter_manifest_max_bytes=2 * 1024 * 1024,
                     adapter_deadline_header="X-Adapter-Deadline-Seconds")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class ManifestDeadlineTest(unittest.IsolatedAsyncioTestCase):
    async def probe(self, parts, *, timeout=0.15, error=TimeoutError):
        closed = asyncio.Event()
        tasks = set()

        async def serve(reader, writer):
            task = asyncio.current_task()
            tasks.add(task)
            try:
                await reader.readuntil(b"\r\n\r\n")
                for part, delay in parts:
                    writer.write(part)
                    await writer.drain()
                    if delay:
                        await asyncio.sleep(delay)
                await reader.read()
            except (ConnectionError, asyncio.IncompleteReadError):
                pass
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except ConnectionError:
                    pass
                tasks.discard(task)
                closed.set()

        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        thread_ids = {thread.ident for thread in threading.enumerate()}
        started = time.monotonic()
        try:
            with self.assertRaises(error):
                await manifest_readers()["_read_remote_adapter_manifest"](
                    f"http://127.0.0.1:{port}", started + timeout)
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, timeout + 0.25)
            self.assertEqual(thread_ids, {thread.ident for thread in threading.enumerate()})
            print(f"manifest failure: {elapsed:.3f}s, deadline={timeout:.3f}s")
        finally:
            server.close()
            await server.wait_closed()
            for task in list(tasks):
                task.cancel()
            await asyncio.gather(*list(tasks), return_exceptions=True)

    async def test_header_trickle_is_cancelled_without_reader_thread(self):
        await self.probe([(b"HTTP/1.1 200 OK\r\nX-Slow: ", 0)] + [(b"x", 0.03)] * 30)

    async def test_body_trickle_is_cancelled_without_reader_thread(self):
        await self.probe([(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n", 0)]
                         + [(b" ", 0.03)] * 30)

    async def test_declared_and_streamed_oversize_are_rejected(self):
        await self.probe([(b"HTTP/1.1 200 OK\r\nContent-Length: 2097153\r\n\r\n", 0)],
                         error=RuntimeError)
        await self.probe([(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n" + b" " * 2097153, 0)],
                         timeout=1, error=RuntimeError)

    async def test_connect_is_within_absolute_deadline(self):
        async def stalled_connect(*_args, **_kwargs):
            await asyncio.sleep(5)
        with mock.patch("httpcore._backends.anyio.AnyIOBackend.connect_tcp", stalled_connect):
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                await manifest_readers()["_read_remote_adapter_manifest"](
                    "http://127.0.0.1:1", started + 0.1)
            self.assertLess(time.monotonic() - started, 0.35)


class SourceDeadlineTest(unittest.TestCase):
    def predictive_steps(self, root):
        for step in range(3):
            target = root / f"step_{step}"
            target.mkdir()
            (target / "STABLE").touch()
            (target / "adapter_config.json").write_text('{}')
            (target / "adapter_model.safetensors").write_bytes(
                valid_safetensor_bytes([16000 + step] * 64))
            adapter_delta.publish_adapter_manifest(target, target_step=step)
        return target

    def target(self, root):
        target = root / "step_2"
        target.mkdir()
        (target / "STABLE").touch()
        (target / "adapter_manifest.json").write_text('{"version":1}')
        (target / "adapter_config.json").write_text('{}')
        (target / "adapter_model.safetensors").write_bytes(b'weights')
        return target

    def test_header_validation_and_clamp(self):
        for value in ("nan", "inf", "-inf", "0", "-1", "bad", ""):
            with self.subTest(value=value), self.assertRaises(adapter_server.UpgradeError):
                adapter_server._request_upgrade_deadline(value)
        for value in (None, "10000", "600"):
            before = time.monotonic()
            deadline = adapter_server._request_upgrade_deadline(value)
            self.assertGreaterEqual(deadline, before + 600)
            self.assertLess(deadline, before + 600.1)

    def test_held_lock_expires_and_retry_acquires(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = self.target(root)
            (target / "adapter_manifest.json").write_text('{"version":2}')
            with (root / adapter_server.lock_name).open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                started = time.monotonic()
                with self.assertRaisesRegex(adapter_server.UpgradeError, "deadline"):
                    adapter_server._upgrade_v1_step(root, target, 2, "http://source",
                                                   deadline=started + 0.1)
                self.assertLess(time.monotonic() - started, 0.3)
            adapter_server._upgrade_v1_step(root, target, 2, "http://source")

    def test_stalled_worker_is_reaped_cleans_and_retry_publishes(self):
        real_popen = subprocess.Popen
        processes = []

        def launch(command, **kwargs):
            # Use the real exec boundary. The child creates both private dirs,
            # then stalls as a build or filesystem hash can stall.
            script = (
                "import json,os,pathlib,sys,time; "
                "r=json.load(os.fdopen(int(sys.argv[1]))); "
                "pathlib.Path(r['pending']).mkdir(parents=True); "
                "pathlib.Path(r['build']).mkdir(); time.sleep(30)"
            )
            process = real_popen([sys.executable, "-c", script, command[-2]], **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = self.target(root)
            old_manifest = (target / "adapter_manifest.json").read_bytes()
            descriptors = len(os.listdir("/proc/self/fd"))
            with mock.patch.object(adapter_server.subprocess, "Popen", launch):
                started = time.monotonic()
                with self.assertRaisesRegex(adapter_server.UpgradeError, "deadline"):
                    adapter_server._upgrade_v1_step(root, target, 2, "http://source",
                                                   deadline=started + 0.15)
                self.assertLess(time.monotonic() - started, 0.4)
            self.assertTrue(processes and all(p.poll() is not None for p in processes))
            self.assertEqual(len(os.listdir("/proc/self/fd")), descriptors)
            self.assertEqual((target / "adapter_manifest.json").read_bytes(), old_manifest)
            # Expired requests leave only private work; the next worker
            # scavenges it before it builds.
            self.assertTrue(list(root.glob(".predictive-build-*")))

            def complete(_active, _target, _step, _url, _pending, build, *_args, **kwargs):
                if kwargs.get("cleanup_only"):
                    adapter_server._scavenge_upgrade_leftovers(_active)
                    return
                adapter_server._scavenge_upgrade_leftovers(_active)
                build.mkdir()
                (build / "adapter_manifest.json").write_text('{"version":2}')
                return {name: list(adapter_server._stat_identity(_target / name)) for name in (
                    adapter_server.adapter_config_name, adapter_server.adapter_weights_name,
                    adapter_server.adapter_manifest_name, adapter_server.stable_name)}
            with mock.patch.object(adapter_server, "_run_upgrade_worker", complete):
                adapter_server._upgrade_v1_step(root, target, 2, "http://source")
            self.assertEqual(json.loads((target / "adapter_manifest.json").read_text())["version"], 2)
            self.assertFalse(list(root.glob(".predictive-build-*")))
            self.assertFalse(list((root / adapter_server.cache_name).glob(".pending-*")))

    def test_real_exec_retry_keeps_cache_only_second_base_until_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = self.predictive_steps(root)
            base2 = root / "step_0"
            private_base2 = root / adapter_server.cache_name / "step_0"
            private_base2.parent.mkdir()
            adapter_server._link_cache_entry(base2,
                adapter_server._identity_from_published_directory(base2, 0, cache=False), private_base2)
            shutil.rmtree(base2)
            replace = adapter_server._replace_from_dirs

            def expire_commit(source, destination, *args, **kwargs):
                if destination.name == adapter_server.adapter_manifest_name:
                    raise adapter_server.UpgradeError("injected deadline before commit")
                return replace(source, destination, *args, **kwargs)

            env = {"PYTHONPATH": str(Path(__file__).parents[1] / "src")}
            with mock.patch.dict(os.environ, env):
                with mock.patch.object(adapter_server, "_replace_from_dirs", expire_commit):
                    with self.assertRaisesRegex(adapter_server.UpgradeError, "injected deadline"):
                        adapter_server._upgrade_v1_step(root, target, 2, "http://source")
                self.assertEqual(json.loads((target / "adapter_manifest.json").read_text())["version"], 1)
                self.assertTrue(private_base2.is_dir())
                adapter_server._upgrade_v1_step(root, target, 2, "http://source")
            self.assertEqual(json.loads((target / "adapter_manifest.json").read_text())["version"], 2)
            self.assertFalse(list(root.glob(".predictive-build-*")))
            self.assertFalse(list((root / adapter_server.cache_name).glob(".pending-*")))

    def test_mutation_after_final_checksum_keeps_old_manifest_and_retry_succeeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = self.predictive_steps(root)
            manifest_path = target / adapter_server.adapter_manifest_name
            weights_path = target / adapter_server.adapter_weights_name
            old_manifest, old_weights = manifest_path.read_bytes(), weights_path.read_bytes()
            real_popen = subprocess.Popen
            processes = []

            def launch(command, **kwargs):
                script = """
import runpy, sys
from pathlib import Path
namespace = runpy.run_path(sys.argv[1])
globals_ = namespace['_upgrade_worker_main'].__globals__
sha256 = globals_['_sha256']
weights = Path(sys.argv[4])
checks = 0
def mutate_after_checksum(path):
    global checks
    digest = sha256(path)
    if path == weights:
        checks += 1
        if checks == 2:
            payload = weights.read_bytes()
            weights.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    return digest
globals_['_sha256'] = mutate_after_checksum
sys.exit(namespace['_upgrade_worker_main'](int(sys.argv[2]), int(sys.argv[3])))
"""
                process = real_popen([sys.executable, "-c", script, command[1],
                                      command[-2], command[-1], str(weights_path)], **kwargs)
                processes.append(process)
                return process

            env = {"PYTHONPATH": str(Path(__file__).parents[1] / "src")}
            with mock.patch.dict(os.environ, env):
                with mock.patch.object(adapter_server.subprocess, "Popen", launch):
                    with self.assertRaisesRegex(adapter_server.UpgradeError, "changed during final validation"):
                        adapter_server._upgrade_v1_step(root, target, 2, "http://source")
                self.assertNotEqual(weights_path.read_bytes(), old_weights)
                self.assertEqual(manifest_path.read_bytes(), old_manifest)
                self.assertTrue(all(process.poll() is not None for process in processes))
                weights_path.write_bytes(old_weights)
                adapter_server._upgrade_v1_step(root, target, 2, "http://source")
            self.assertEqual(json.loads(manifest_path.read_text())["version"], 2)

    def test_final_stat_expires_budget_before_commit_and_retry_succeeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = self.predictive_steps(root)
            manifest_path = target / adapter_server.adapter_manifest_name
            old_manifest = manifest_path.read_bytes()
            deadline = time.monotonic() + 600
            monotonic = time.monotonic
            stat_identity = adapter_server._stat_identity
            replace = adapter_server._replace_from_dirs
            final_check = False
            expired = False

            def delayed_stat(path):
                nonlocal expired
                result = stat_identity(path)
                if final_check and path == target / adapter_server.stable_name:
                    # Advance the clock at the last stat, after the callback's
                    # initial deadline check, without a 600-second test sleep.
                    expired = True
                return result

            def final_replace(source, destination, *args, **kwargs):
                nonlocal final_check
                if destination == manifest_path:
                    final_check = True
                try:
                    return replace(source, destination, *args, **kwargs)
                finally:
                    final_check = False

            env = {"PYTHONPATH": str(Path(__file__).parents[1] / "src")}
            with mock.patch.dict(os.environ, env):
                with mock.patch.object(adapter_server, "_stat_identity", delayed_stat), \
                        mock.patch.object(adapter_server, "_replace_from_dirs", final_replace), \
                        mock.patch.object(adapter_server.time, "monotonic",
                                          lambda: deadline + 0.01 if expired else monotonic()):
                    with self.assertRaisesRegex(adapter_server.UpgradeError, "deadline"):
                        adapter_server._upgrade_v1_step(root, target, 2, "http://source", deadline=deadline)
                self.assertTrue(expired)
                self.assertEqual(manifest_path.read_bytes(), old_manifest)
                adapter_server._upgrade_v1_step(root, target, 2, "http://source")
            self.assertEqual(json.loads(manifest_path.read_text())["version"], 2)

    def test_slow_hash_and_scavenge_are_killed_in_exec_worker(self):
        real_popen = subprocess.Popen
        for stalled_name in ("_sha256", "_scavenge_upgrade_leftovers"):
            with self.subTest(stalled_name=stalled_name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                target = self.target(root)
                # A valid source gets as far as checksum validation.
                adapter_delta.publish_adapter_manifest(target, target_step=2)
                processes = []

                def launch(command, **kwargs):
                    script = (
                        "import runpy,sys,time; n=runpy.run_path(sys.argv[1]); "
                        "g=n['_upgrade_worker_main'].__globals__; "
                        "g[sys.argv[4]]=lambda *a,**k: time.sleep(30); "
                        "sys.exit(n['_upgrade_worker_main'](int(sys.argv[2]),int(sys.argv[3])))"
                    )
                    process = real_popen([sys.executable, "-c", script, command[1],
                                          command[-2], command[-1], stalled_name], **kwargs)
                    processes.append(process)
                    return process

                with mock.patch.object(adapter_server.subprocess, "Popen", launch):
                    started = time.monotonic()
                    with self.assertRaisesRegex(adapter_server.UpgradeError, "deadline"):
                        adapter_server._upgrade_v1_step(root, target, 2, "http://source",
                                                       deadline=started + 0.2)
                    self.assertLess(time.monotonic() - started, 0.45)
                self.assertTrue(processes and all(process.poll() is not None for process in processes))
                self.assertEqual(json.loads((target / "adapter_manifest.json").read_text())["version"], 1)

    def test_manifest_validation_skips_other_wire_hashes_and_waiter_expires(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target, _ = server_tests.AdapterServerTest()._write_predictive_step(root)
            entered = threading.Event()
            release = threading.Event()
            original_sha = adapter_server._sha256
            errors = []

            def slow_config(path):
                entered.set()
                release.wait(1)
                return original_sha(path)

            def owner():
                try:
                    adapter_server._identity_from_published_directory(target, 0, cache=False,
                                                                      verify_weights=False)
                except Exception as exc:
                    errors.append(exc)

            with mock.patch.object(adapter_server, "_sha256", slow_config), mock.patch.object(
                adapter_server, "_validate_predictive_chunks", side_effect=AssertionError("whole wire scan")
            ):
                thread = threading.Thread(target=owner)
                thread.start()
                try:
                    self.assertTrue(entered.wait(1))
                    started = time.monotonic()
                    with self.assertRaisesRegex(adapter_server.UpgradeError, "deadline"):
                        adapter_server._identity_from_published_directory(target, 0, cache=False,
                            verify_weights=False, deadline=started + 0.1)
                    self.assertLess(time.monotonic() - started, 0.3)
                finally:
                    release.set()
                    thread.join(1)
                self.assertFalse(thread.is_alive())
                self.assertFalse(errors)
                adapter_server._identity_from_published_directory(target, 0, cache=False,
                    verify_weights=False, deadline=time.monotonic() + 0.1)

    def test_disconnect_aborts_and_http_passes_socket_and_deadline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active = root / "active"
            active.mkdir()
            self.target(active)
            server = adapter_server.AdapterServer(("127.0.0.1", 0), functools.partial(
                QuietAdapterHandler, directory=str(root), upgrade_path=("active",),
                source_base_url="http://source"))
            started = threading.Event()
            finished = threading.Event()

            def wait_for_disconnect(*_args, deadline, connection):
                self.assertLess(deadline - time.monotonic(), 0.5)
                started.set()
                try:
                    while True:
                        adapter_server._check_upgrade_deadline(deadline, connection)
                        time.sleep(0.01)
                finally:
                    finished.set()

            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                with mock.patch.object(adapter_server, "_upgrade_v1_step", wait_for_disconnect):
                    client = socket.create_connection(server.server_address)
                    client.sendall(b"GET /active/step_2/adapter_manifest.json HTTP/1.1\r\n"
                                   b"Host: localhost\r\nX-Adapter-Deadline-Seconds: 0.5\r\n\r\n")
                    self.assertTrue(started.wait(1))
                    client.close()
                    self.assertTrue(finished.wait(0.3))
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_deadline_rechecked_after_manifest_hash_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = self.target(root)
            source = root / "new.json"
            source.write_text('{"version":2}')
            deadline = time.monotonic() + 0.05
            sha = adapter_server._sha256_fd

            def slow_sha(*args):
                time.sleep(0.08)
                return sha(*args)

            with mock.patch.object(adapter_server, "_sha256_fd", slow_sha):
                with self.assertRaisesRegex(adapter_server.UpgradeError, "deadline"):
                    adapter_server._replace_from_dirs(
                        source, target / "adapter_manifest.json",
                        before_publish=lambda: adapter_server._check_upgrade_deadline(deadline))
            self.assertEqual(json.loads((target / "adapter_manifest.json").read_text())["version"], 1)


if __name__ == "__main__":
    unittest.main()
