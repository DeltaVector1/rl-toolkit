import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import training_ops
from training_ops.runtime.h200 import b2_rollout as tool
from training_ops.tests.test_rollout_pull import RolloutPullTest, puller


class Missing(Exception):
    response = {"ResponseMetadata": {"HTTPStatusCode": 404}}


class ClientError(Exception):
    def __init__(self, code, status, message="provider detail https://secret.invalid/key"):
        super().__init__(message)
        self.response = {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.uploads = 0
        self.deletes = []
        self.signed = None
        self.mutate = lambda: None

    def head_object(self, Bucket, Key, VersionId="version-1"):
        if (Key, VersionId) not in self.objects:
            raise Missing()
        return self.objects[Key, VersionId]

    def upload_fileobj(self, stream, bucket, key, ExtraArgs, Config):
        self.uploads += 1
        body = stream.read()
        assert ExtraArgs["ACL"] == "private"
        self.objects[key, "version-1"] = dict(
            ContentLength=len(body),
            Metadata=ExtraArgs["Metadata"],
            VersionId="version-1",
        )
        self.mutate()

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        self.signed = Params
        assert ExpiresIn == 600
        return "https://example.invalid/" + Params["Key"] + "?signature=temporary"

    def delete_object(self, **kwargs):
        self.deletes.append(kwargs)
        if "VersionId" in kwargs:
            self.objects.pop((kwargs["Key"], kwargs["VersionId"]), None)


class TransientPublishS3(FakeS3):
    def __init__(self, *, committed=False, failures=1):
        super().__init__()
        self.committed = committed
        self.failures = failures
        self.attempts = 0

    def upload_fileobj(self, stream, bucket, key, ExtraArgs, Config):
        self.attempts += 1
        if self.failures:
            self.failures -= 1
            if self.committed:
                super().upload_fileobj(stream, bucket, key, ExtraArgs, Config)
            raise ClientError("SlowDown", 503)
        super().upload_fileobj(stream, bucket, key, ExtraArgs, Config)


class MultipartS3(FakeS3):
    def __init__(
        self,
        *,
        complete_error=False,
        create_error=False,
        abort_failures=0,
        retain_on_complete_error=False,
    ):
        super().__init__()
        self.uploads_by_id = {}
        self.aborts = []
        self.multipart_creates = 0
        self.multipart_completes = 0
        self.complete_error = complete_error
        self.create_error = create_error
        self.abort_failures = abort_failures
        self.retain_on_complete_error = retain_on_complete_error
        self.list_response = None

    def create_multipart_upload(self, **kwargs):
        self.multipart_creates += 1
        upload_id = f"upload-{self.multipart_creates}"
        self.uploads_by_id[upload_id] = {"kwargs": kwargs, "parts": {}}
        if self.create_error:
            raise ClientError("ServiceUnavailable", 503)
        return {"UploadId": upload_id}

    def upload_part(self, **kwargs):
        upload = self.uploads_by_id[kwargs["UploadId"]]
        upload["parts"][kwargs["PartNumber"]] = kwargs["Body"]
        return {"ETag": f'"part-{kwargs["PartNumber"]}"'}

    def complete_multipart_upload(self, **kwargs):
        self.multipart_completes += 1
        if self.complete_error and self.retain_on_complete_error:
            upload = self.uploads_by_id[kwargs["UploadId"]]
        else:
            upload = self.uploads_by_id.pop(kwargs["UploadId"])
        body = b"".join(upload["parts"][part["PartNumber"]] for part in kwargs["MultipartUpload"]["Parts"])
        key = kwargs["Key"]
        self.objects[key, "version-1"] = {
            "ContentLength": len(body),
            "Metadata": upload["kwargs"]["Metadata"],
            "VersionId": "version-1",
        }
        if self.complete_error:
            raise ClientError("ServiceUnavailable", 503)
        return {"VersionId": "version-1"}

    def abort_multipart_upload(self, **kwargs):
        self.aborts.append(kwargs)
        if self.abort_failures:
            self.abort_failures -= 1
            raise ClientError("SlowDown", 503)
        self.uploads_by_id.pop(kwargs["UploadId"], None)

    def list_multipart_uploads(self, **kwargs):
        if self.list_response is not None:
            return self.list_response
        return {
            "Bucket": kwargs["Bucket"],
            "Prefix": kwargs["Prefix"],
            "IsTruncated": False,
            "Uploads": [
                {"Key": item["kwargs"]["Key"], "UploadId": upload_id}
                for upload_id, item in self.uploads_by_id.items()
            ]
        }


class DeleteErrorS3(FakeS3):
    def __init__(self, *, remove=False, failures=1, code="SlowDown", status=503):
        super().__init__()
        self.remove = remove
        self.failures = failures
        self.code = code
        self.status = status

    def delete_object(self, **kwargs):
        self.deletes.append(kwargs)
        if self.remove:
            self.objects.pop((kwargs["Key"], kwargs["VersionId"]), None)
        if self.failures:
            self.failures -= 1
            raise ClientError(self.code, self.status)
        self.objects.pop((kwargs["Key"], kwargs["VersionId"]), None)


class B2RolloutTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.runs = self.root / "runs"
        self.source = self.runs / "alpha/run_alpha/rollouts/step_2/train_rollouts.bin"
        self.source.parent.mkdir(parents=True)
        self.body = bytes(range(256)) * 16
        self.source.write_bytes(self.body)
        self.client = FakeS3()
        self.partial = self.root / "destination/.incoming/step_2/train_rollouts.bin"
        self.final = self.root / "destination/rollouts/step_2/train_rollouts.bin"

    def publish(self):
        return tool.publish(self.client, "fixture", self.source, self.runs)

    def curl(self, command, **kwargs):
        self.assertEqual(command[:2], ["curl", "--disable"])
        self.assertNotIn("temporary", " ".join(command))
        self.assertIn('"https://', kwargs["input"])
        self.assertEqual(command[command.index("--retry") + 1], "0")
        self.assertNotIn("--location", command)
        self.assertEqual(kwargs["timeout"], 300)
        self.assertEqual(os.fstat(kwargs["pass_fds"][0]).st_mode & 0o777, 0o600)
        os.write(kwargs["pass_fds"][0], self.body)
        return SimpleNamespace(returncode=0, stdout="200")

    def test_publish_reuse_receive_and_exact_version_delete(self):
        value = self.publish()
        self.assertEqual(self.publish()["key"], value["key"])
        self.assertEqual(self.client.uploads, 1)
        self.assertEqual(self.client.signed["VersionId"], "version-1")
        tool.receive(value, str(self.source), self.partial, self.final, self.curl)
        pending = Path(str(self.final) + ".b2-ack.json")
        receipt = tool.parse_json(pending.read_text())
        self.assertNotIn("url", receipt)
        self.assertEqual(pending.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.final.read_bytes(), self.body)
        self.assertEqual(self.final.stat().st_uid, self.source.stat().st_uid)
        self.assertEqual(self.final.stat().st_gid, self.source.stat().st_gid)
        self.assertEqual(
            self.final.stat().st_mode & 0o777, self.source.stat().st_mode & 0o777
        )
        with self.assertRaises(ValueError):
            tool.acknowledge(self.client, "fixture", receipt, self.runs)
        marker = Path(str(self.source) + ".delivered")
        marker.write_text(value["sha256"] + "\n")
        self.client.objects[value["key"], "unrelated-version"] = {"preserve": True}
        tool.acknowledge(self.client, "fixture", receipt, self.runs)
        tool.acknowledge(self.client, "fixture", receipt, self.runs)
        self.assertEqual(
            self.client.deletes,
            [dict(Bucket="fixture", Key=value["key"], VersionId="version-1")],
        )
        self.assertIn((value["key"], "unrelated-version"), self.client.objects)

    def test_transient_publish_after_commit_reuses_exact_object_once(self):
        client = TransientPublishS3(committed=True)
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            value = tool.publish(client, "fixture", self.source, self.runs)
        self.assertEqual(client.attempts, 1)
        self.assertEqual(client.uploads, 1)
        self.assertEqual(value["version_id"], "version-1")
        self.assertEqual(client.signed["VersionId"], value["version_id"])
        telemetry = output.getvalue()
        self.assertIn('"operation": "publish.upload"', telemetry)
        self.assertIn('"error_code": "SlowDown"', telemetry)
        self.assertIn('"retry": true', telemetry)
        self.assertNotIn(self.source.name, telemetry)
        self.assertNotIn(value["key"], telemetry)
        self.assertNotIn("https://secret.invalid", telemetry)

    def test_transient_publish_without_commit_retries_once(self):
        client = TransientPublishS3(committed=False)
        with contextlib.redirect_stderr(io.StringIO()):
            value = tool.publish(client, "fixture", self.source, self.runs)
        self.assertEqual(client.attempts, 2)
        self.assertEqual(client.uploads, 1)
        self.assertEqual(value["version_id"], "version-1")

    def test_permanent_publish_error_is_terminal_and_sanitized(self):
        client = TransientPublishS3(committed=False, failures=2)
        client.error_code = "AccessDenied"

        def denied(stream, bucket, key, ExtraArgs, Config):
            client.attempts += 1
            raise ClientError("AccessDenied", 403)

        client.upload_fileobj = denied
        output = io.StringIO()
        with contextlib.redirect_stderr(output), self.assertRaises(ClientError):
            tool.publish(client, "fixture", self.source, self.runs)
        self.assertEqual(client.attempts, 1)
        telemetry = output.getvalue()
        self.assertIn('"error_code": "AccessDenied"', telemetry)
        self.assertIn('"retry": false', telemetry)
        self.assertNotIn("fixture", telemetry)
        self.assertNotIn("https://secret.invalid", telemetry)

    def test_terminal_identity_codes_override_server_error_retry(self):
        for code in (
            "AccessDenied",
            "SignatureDoesNotMatch",
            "InvalidAccessKeyId",
            "NoSuchKey",
            "NoSuchUpload",
            "AuthorizationHeaderMalformed",
            "InvalidBucketName",
            "AccessDeniedException",
            "NotFoundException",
            "PermissionDenied",
        ):
            with self.subTest(code=code):
                details = tool._client_error_details(ClientError(code, 503))
                self.assertFalse(details["retryable"])
                expected = code if code in tool._KNOWN_ERROR_CODES else "http_503"
                self.assertEqual(details["code"], expected)

    def test_second_transient_publish_error_is_terminal(self):
        client = TransientPublishS3(committed=False, failures=2)
        output = io.StringIO()
        with contextlib.redirect_stderr(output), self.assertRaises(ClientError):
            tool.publish(client, "fixture", self.source, self.runs)
        self.assertEqual(client.attempts, 2)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([record["retry"] for record in records], [True, False])

    def test_multipart_complete_transient_does_not_duplicate_object(self):
        client = MultipartS3(complete_error=True)
        config = SimpleNamespace(multipart_threshold=1024, multipart_chunksize=1024)
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            value = tool.publish(
                client, "fixture", self.source, self.runs, transfer_config=config
            )
        self.assertEqual(client.multipart_creates, 1)
        self.assertEqual(client.multipart_completes, 1)
        self.assertEqual(client.aborts, [
            {"Bucket": "fixture", "Key": value["key"], "UploadId": "upload-1"}
        ])
        self.assertEqual(value["version_id"], "version-1")
        self.assertEqual(client.signed["VersionId"], "version-1")
        self.assertNotIn("https://secret.invalid", output.getvalue())

    def test_multipart_create_response_loss_does_not_retry_unknown_upload(self):
        client = MultipartS3(create_error=True)
        config = SimpleNamespace(multipart_threshold=1024, multipart_chunksize=1024)
        output = io.StringIO()
        with contextlib.redirect_stderr(output), self.assertRaises(ClientError):
            tool.publish(
                client, "fixture", self.source, self.runs, transfer_config=config
            )
        self.assertEqual(client.multipart_creates, 1)
        self.assertEqual(client.aborts, [])
        self.assertEqual(set(client.uploads_by_id), {"upload-1"})
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(records[-1]["operation"], "publish.multipart.create")
        self.assertFalse(records[-1]["retry"])
        self.assertNotIn("https://secret.invalid", output.getvalue())

    def test_multipart_malformed_empty_upload_entry_fails_closed(self):
        client = MultipartS3(complete_error=True)
        key = tool.source_info(self.source, self.runs)[0]["key"]
        client.list_response = {
            "Bucket": "fixture",
            "Prefix": key,
            "IsTruncated": False,
            "Uploads": [{}],
        }
        config = SimpleNamespace(multipart_threshold=1024, multipart_chunksize=1024)
        with self.assertRaises(tool._MultipartCleanupFailed):
            tool.publish(
                client, "fixture", self.source, self.runs, transfer_config=config
            )
        self.assertEqual(client.multipart_creates, 1)
        self.assertEqual(len(client.aborts), 1)

    def test_multipart_listing_malformed_types_duplicates_and_pagination_fail(self):
        key = tool.source_info(self.source, self.runs)[0]["key"]
        upload = {"Key": key, "UploadId": "upload-1"}

        def page(**changes):
            value = {
                "Bucket": "fixture",
                "Prefix": key,
                "IsTruncated": False,
                "Uploads": [],
            }
            value.update(changes)
            return value

        cases = [
            page(Uploads=[None]),
            page(Uploads=[{"Key": key}]),
            page(Uploads=[{"Key": key, "UploadId": 1}]),
            page(Uploads=[upload, {"Key": key, "UploadId": "upload-2"}]),
            page(Uploads=[upload, dict(upload)]),
            page(Uploads=None),
            page(IsTruncated="false"),
            page(IsTruncated=True, NextKeyMarker="next-key"),
            page(NextKeyMarker=1, NextUploadIdMarker="next-upload"),
            page(NextKeyMarker="next-key", NextUploadIdMarker=""),
            page(NextKeyMarker="next-key", NextUploadIdMarker="next-upload"),
            page(KeyMarker="unexpected"),
            page(CommonPrefixes=[{"Prefix": "other"}]),
            page(Delimiter="/"),
            page(Bucket="other"),
            page(Prefix="other"),
        ]

        class ListingClient:
            def __init__(self, value):
                self.value = value

            def list_multipart_uploads(self, **kwargs):
                return self.value

        omitted_uploads = page()
        del omitted_uploads["Uploads"]
        tool._verify_multipart_absent(
            ListingClient(omitted_uploads), "fixture", key, "upload-1"
        )
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                tool._verify_multipart_absent(
                    ListingClient(value), "fixture", key, "upload-1"
                )

    def test_multipart_abort_retries_and_verifies_exact_absence(self):
        client = MultipartS3(complete_error=True, abort_failures=1)
        config = SimpleNamespace(multipart_threshold=1024, multipart_chunksize=1024)
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            value = tool.publish(
                client, "fixture", self.source, self.runs, transfer_config=config
            )
        self.assertEqual(value["version_id"], "version-1")
        self.assertEqual(len(client.aborts), 2)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(
            [(record["operation"], record["retry"]) for record in records],
            [("publish.multipart.abort", True), ("publish.upload", True)],
        )

    def test_multipart_abort_failure_is_fail_closed_without_replacement(self):
        client = MultipartS3(
            complete_error=True,
            abort_failures=2,
            retain_on_complete_error=True,
        )
        config = SimpleNamespace(multipart_threshold=1024, multipart_chunksize=1024)
        output = io.StringIO()
        with contextlib.redirect_stderr(output), self.assertRaises(
            tool._MultipartCleanupFailed
        ):
            tool.publish(
                client, "fixture", self.source, self.runs, transfer_config=config
            )
        self.assertEqual(client.multipart_creates, 1)
        self.assertEqual(len(client.aborts), 2)
        self.assertEqual(set(client.uploads_by_id), {"upload-1"})
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(
            [(record["operation"], record["retry"]) for record in records],
            [("publish.multipart.abort", True), ("publish.multipart.abort", False)],
        )

    def test_transient_delete_retries_exact_version_and_confirms_absence(self):
        client = DeleteErrorS3(remove=True)
        self.client = client
        value = self.publish()
        receipt = {key: item for key, item in value.items() if key not in {"url", "expires_at"}}
        Path(str(self.source) + ".delivered").write_text(receipt["sha256"])
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            tool.acknowledge(client, "fixture", receipt, self.runs)
        self.assertEqual(client.deletes, [
            {"Bucket": "fixture", "Key": value["key"], "VersionId": "version-1"},
            {"Bucket": "fixture", "Key": value["key"], "VersionId": "version-1"},
        ])
        self.assertNotIn("https://secret.invalid", output.getvalue())

    def test_permanent_delete_error_stays_terminal_when_object_remains(self):
        client = DeleteErrorS3(remove=False, code="AccessDenied", status=403)
        value = tool.publish(client, "fixture", self.source, self.runs)
        receipt = {key: item for key, item in value.items() if key not in {"url", "expires_at"}}
        Path(str(self.source) + ".delivered").write_text(receipt["sha256"])
        output = io.StringIO()
        with contextlib.redirect_stderr(output), self.assertRaises(ClientError):
            tool.acknowledge(client, "fixture", receipt, self.runs)
        self.assertEqual(len(client.deletes), 1)
        self.assertIn('"error_code": "AccessDenied"', output.getvalue())
        self.assertIn('"retry": false', output.getvalue())

    def test_root_receiver_publishes_directories_readable_by_trainer(self):
        # A normal root Docker container can exercise the actual UID transition;
        # an unprivileged local run still checks every new directory's other bits.
        self.root.chmod(0o755)
        self.source.chmod(0o600)
        if os.geteuid() == 0:
            os.chown(self.source, 1000, 1000)
        value = self.publish()
        previous_mask = os.umask(0o077)
        try:
            tool.receive(value, str(self.source), self.partial, self.final, self.curl)
        finally:
            os.umask(previous_mask)
        for directory in (
            self.final.parent,
            self.final.parent.parent,
            self.root / "destination",
        ):
            self.assertEqual(directory.stat().st_mode & 0o777, 0o755)
        self.assertEqual(self.partial.parent.stat().st_mode & 0o777, 0o700)
        if os.geteuid() == 0:
            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os,sys; from pathlib import Path; os.setgroups([]); os.setgid(1000); os.setuid(1000); print(len(Path(sys.argv[1]).read_bytes()))",
                    str(self.final),
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertEqual(int(child.stdout), len(self.body))

    def test_source_paths_and_mutation_fail_closed(self):
        link = self.source.parent / "link"
        link.symlink_to(self.source)
        for path in (
            link,
            self.root / "outside/train_rollouts.bin",
            self.source.parent / "wrong.bin",
        ):
            with self.subTest(path=path), self.assertRaises(
                (ValueError, FileNotFoundError)
            ):
                tool.source_info(path, self.runs)
        self.client.mutate = lambda: self.source.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.publish()
        self.assertIsNone(self.client.signed)

    def test_wrong_hash_size_auth_timeout_redirect_never_publish(self):
        value = self.publish()
        for result in (
            SimpleNamespace(returncode=22, stdout="403"),
            SimpleNamespace(returncode=28, stdout="000"),
            SimpleNamespace(returncode=0, stdout="302"),
        ):
            with self.subTest(result=result), self.assertRaises(ValueError):
                tool.receive(
                    value,
                    str(self.source),
                    self.partial,
                    self.final,
                    lambda *a, **k: result,
                )
            self.assertFalse(self.final.exists())

        def wrong(command, **kwargs):
            os.write(kwargs["pass_fds"][0], b"bad")
            return SimpleNamespace(returncode=0, stdout="200")

        with self.assertRaises(ValueError):
            tool.receive(value, str(self.source), self.partial, self.final, wrong)
        self.assertFalse(self.final.exists())
        self.assertFalse(self.client.deletes)

    def test_strict_json_key_expiry_and_no_replace(self):
        with self.assertRaises(ValueError):
            tool.parse_json('{"size":1,"size":2}')
        value = self.publish()
        for field, item in (
            ("key", "../other"),
            ("size", True),
            ("version_id", "null"),
            ("expires_at", 0),
            ("url", "http://example.invalid/file"),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                tool.receive(
                    dict(value, **{field: item}),
                    str(self.source),
                    self.partial,
                    self.final,
                    self.curl,
                )
        self.final.parent.mkdir(parents=True)
        self.final.write_bytes(b"preserve")
        with self.assertRaises(ValueError):
            tool.receive(value, str(self.source), self.partial, self.final, self.curl)
        self.assertEqual(self.final.read_bytes(), b"preserve")

    def test_pending_install_failure_recovers_after_publication(self):
        value = self.publish()
        real_link = os.link

        def fail_receipt(source, dest, **kwargs):
            if str(dest).endswith(".b2-ack.json"):
                raise OSError("fixture interruption")
            return real_link(source, dest, **kwargs)

        with patch.object(tool.os, "link", side_effect=fail_receipt), self.assertRaises(
            OSError
        ):
            tool.receive(value, str(self.source), self.partial, self.final, self.curl)
        self.assertTrue(self.final.exists())
        self.assertFalse(Path(str(self.final) + ".b2-ack.json").exists())
        self.assertEqual(list(self.final.parent.glob(".b2-ack-*")), [])
        tool.receive(
            value,
            str(self.source),
            self.partial,
            self.final,
            lambda *a, **k: self.fail("must reuse verified final"),
        )

    def test_credentials_required_even_with_ambient_aws(self):
        fields = self.root / "fixture.env"
        fields.write_text("OTHER=value\n")
        fields.chmod(0o600)
        with patch.dict(
            os.environ,
            {"AWS_ACCESS_KEY_ID": "ambient", "AWS_SECRET_ACCESS_KEY": "ambient"},
        ):
            with self.assertRaises(ValueError):
                tool.credentials(fields)

    def test_sdk_has_explicit_credentials_classic_manager_and_no_retries(self):
        boto = ModuleType("boto3")
        boto.client = Mock()
        transfer = ModuleType("boto3.s3.transfer")
        transfer.TransferConfig = Mock()
        config = ModuleType("botocore.config")
        config.Config = Mock()
        fields = {
            name: "fixture"
            for name in (
                "B2_S3_KEY_ID",
                "B2_S3_APPLICATION_KEY",
                "B2_S3_BUCKET",
                "B2_S3_ENDPOINT",
                "B2_S3_REGION",
            )
        }
        with patch.dict(
            sys.modules,
            {"boto3": boto, "boto3.s3.transfer": transfer, "botocore.config": config},
        ), patch.object(tool, "credentials", return_value=fields):
            tool.client_settings()
        self.assertEqual(
            config.Config.call_args.kwargs["retries"], {"total_max_attempts": 1}
        )
        self.assertEqual(
            transfer.TransferConfig.call_args.kwargs["preferred_transfer_client"],
            "classic",
        )
        self.assertEqual(transfer.TransferConfig.call_args.kwargs["max_concurrency"], 1)
        self.assertFalse(transfer.TransferConfig.call_args.kwargs["use_threads"])
        self.assertTrue(boto.client.call_args.kwargs["verify"])
        self.assertIsNone(boto.client.call_args.kwargs["aws_session_token"])
        self.assertEqual(boto.client.call_args.kwargs["aws_access_key_id"], "fixture")

    def test_server_timeout_kills_a_swallowed_exception(self):
        code = "import signal,time\ndef alarm(*a): raise TimeoutError()\nsignal.signal(signal.SIGALRM,alarm)\nsignal.alarm(1)\ntry: time.sleep(10)\nexcept TimeoutError: time.sleep(10)\n"
        started = time.monotonic()
        result = subprocess.run(
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=1s",
                "2s",
                sys.executable,
                "-c",
                code,
            ],
            capture_output=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 124)
        self.assertLess(time.monotonic() - started, 4)
        script = puller.read_text()
        self.assertIn(
            "printf -v command '%q ' timeout --signal=TERM --kill-after=5s 300s", script
        )

    def test_config_environment_and_inventory(self):
        with patch.object(training_ops, "rollout_transfer_backend", "ssh"):
            env = training_ops._rollout_pull_environment(
                str(self.runs / "alpha/run_alpha")
            )
            self.assertEqual(env["ROLLOUT_TRANSFER_BACKEND"], "ssh")
        with patch.object(training_ops, "rollout_transfer_backend", "b2_s3"):
            env = training_ops._rollout_pull_environment(
                str(self.runs / "alpha/run_alpha")
            )
            self.assertEqual(env["ROLLOUT_TRANSFER_BACKEND"], "b2_s3")
            self.assertFalse(any("KEY" in key for key in env))
            with patch.object(training_ops, "_remote_ok") as remote:
                training_ops._b2_rollout_preflight()
                self.assertIn("preflight", remote.call_args_list[0].args[1])
        source = Path(training_ops.__file__).read_text()
        self.assertIn(
            'add("training_ops/runtime/h200/b2_rollout.py", [orchestrator_host, trainer_head])',
            source,
        )


class B2PullIntegrationTest(unittest.TestCase):
    def test_binary_only_uses_b2_and_never_logs_url(self):
        fixture = RolloutPullTest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            binary = source / "rollouts/step_0/train_rollouts.bin"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"test payload")
            control = source / "control/orch.toml"
            control.parent.mkdir()
            control.write_text("control")
            os.utime(control, (time.time() - 5, time.time() - 5))
            env = fixture.environment(source, destination, fixture.tools(directory))
            interpreter = root / "tools/b2-python"
            interpreter.write_text("""#!/usr/bin/env python3
import hashlib,json,os,sys,time
from pathlib import Path
operation=sys.argv[2]
source=Path(os.environ["B2_TEST_SOURCE"])
digest=hashlib.sha256(source.read_bytes()).hexdigest()
if operation=="publish":
 print(json.dumps(dict(source=str(source),size=source.stat().st_size,sha256=digest,uid=source.stat().st_uid,gid=source.stat().st_gid,mode=source.stat().st_mode&0o777,key="rollout-transfer/v1/alpha/"+"a"*32+"/step_0/"+digest+".bin",version_id="version-1",url="https://example.invalid/secret-signed-url",expires_at=int(time.time())+600)))
elif operation=="ack":
 assert Path(str(source)+".delivered").read_text().strip()==digest
 Path(os.environ["B2_TEST_ACK"]).write_text("deleted exact version")
elif operation=="preflight": pass
else: raise SystemExit(1)
""")
            interpreter.chmod(0o755)
            curl = root / "tools/curl"
            curl.write_text("""#!/usr/bin/env python3
import os,sys
from pathlib import Path
assert sys.argv[1]=="--disable"
assert "secret-signed-url" in sys.stdin.read()
body=Path(os.environ["B2_TEST_SOURCE"]).read_bytes()
if "--range" in sys.argv:
 start,end=map(int,sys.argv[sys.argv.index("--range")+1].split("-"))
 Path(sys.argv[sys.argv.index("--dump-header")+1]).write_text(f"HTTP/1.1 206 OK\\nContent-Range: bytes {start}-{end}/{len(body)}\\n\\n")
 body=body[start:end+1]
 print(f"206\\t192.0.2.1\\t1.1\\t0.01\\t0.02\\t0.03\\t{len(body)}",end="")
else: print("200",end="")
Path(sys.argv[sys.argv.index("--output")+1]).write_bytes(body)
""")
            curl.chmod(0o755)
            env.update(
                ROLLOUT_TRANSFER_BACKEND="b2_s3",
                B2_DOWNLOAD_WORKERS="8",
                B2_DOWNLOAD_THRESHOLD_BYTES="1",
                B2_ROLLOUT_PYTHON=str(interpreter),
                B2_TEST_SOURCE=str(binary),
                B2_TEST_ACK=str(root / "acked"),
                ONCE="1",
                RSYNC_ATTEMPTS_LOG=str(root / "rsync"),
            )
            result = subprocess.run(
                ["bash", str(puller), "run"],
                env=env,
                text=True,
                capture_output=True,
                timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                (destination / "rollouts/step_0/train_rollouts.bin").read_bytes(),
                binary.read_bytes(),
            )
            self.assertEqual((destination / "control/orch.toml").read_text(), "control")
            self.assertTrue((root / "acked").exists())
            self.assertNotIn("train_rollouts.bin", (root / "rsync").read_text())
            self.assertNotIn("secret-signed-url", result.stdout + result.stderr)
            self.assertFalse(list(destination.rglob("*.b2-ack.json")))


if __name__ == "__main__":
    unittest.main()
