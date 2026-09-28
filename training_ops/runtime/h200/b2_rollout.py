"""Host-only B2 rollout publication and credential-free HTTPS reception.

Provision the configured host venv from b2-rollout-requirements.txt explicitly.
Only publisher commands read /data/trinity/training.env. Never log publish JSON.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import time
import tomllib
from urllib.parse import urlsplit


def safe_path(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("absolute non-symlink path required")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError("symlink path refused")
    return path


def parse_json(raw):
    if len(raw) > 16384:
        raise ValueError("JSON message is too large")

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON field")
            value[key] = item
        return value

    return json.loads(raw, object_pairs_hook=unique)


def digest_file(path):
    path = safe_path(path)
    with os.fdopen(
        os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb"
    ) as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("regular file required")
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
    if identity(before) != identity(after) or identity(after) != identity(path.stat()):
        raise ValueError("source changed during hashing")
    return before.st_size, digest, identity(before)


def identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def source_info(source, run_root):
    source = safe_path(source)
    relative = source.relative_to(safe_path(run_root)).as_posix()
    match = re.fullmatch(
        r"([A-Za-z0-9_-]+)/run_\1/rollouts/step_([0-9]+)/train_rollouts\.bin", relative
    )
    if not match:
        raise ValueError(
            "source must be a rollout binary below the configured run root"
        )
    size, digest, stamp = digest_file(source)
    if size <= 0:
        raise ValueError("empty rollout refused")
    generation = hashlib.sha256(repr(stamp).encode()).hexdigest()[:32]
    key = f"rollout-transfer/v1/{match[1]}/{generation}/step_{match[2]}/{digest}.bin"
    status = source.stat()
    if identity(status) != stamp:
        raise ValueError("source changed during identity capture")
    return dict(
        source=str(source),
        key=key,
        size=size,
        sha256=digest,
        uid=status.st_uid,
        gid=status.st_gid,
        mode=stat.S_IMODE(status.st_mode),
    ), stamp


def validate_receipt(value):
    if not isinstance(value, dict) or set(value) != {
        "source",
        "key",
        "size",
        "sha256",
        "version_id",
        "uid",
        "gid",
        "mode",
    }:
        raise ValueError("invalid receipt schema")
    version = value["version_id"]
    if (
        not isinstance(version, str)
        or not re.fullmatch(r"[A-Za-z0-9_.~+/-]{1,1024}", version)
        or version == "null"
    ):
        raise ValueError("exact object version required")
    if type(value["size"]) is not int or value["size"] <= 0:
        raise ValueError("invalid size")
    if any(
        type(value[name]) is not int or not 0 <= value[name] < 2**31
        for name in ("uid", "gid")
    ):
        raise ValueError("invalid file ownership")
    if (
        type(value["mode"]) is not int
        or not 0 <= value["mode"] <= 0o777
        or not value["mode"] & 0o400
    ):
        raise ValueError("invalid file mode")
    if not isinstance(value["sha256"], str) or not re.fullmatch(
        r"[a-f0-9]{64}", value["sha256"]
    ):
        raise ValueError("invalid hash")
    if not isinstance(value["source"], str) or not re.fullmatch(
        r"/[A-Za-z0-9_./-]+/train_rollouts\.bin", value["source"]
    ):
        raise ValueError("invalid source")
    key = value["key"]
    if not isinstance(key, str) or not re.fullmatch(
        r"rollout-transfer/v1/[A-Za-z0-9_-]+/[a-f0-9]{32}/step_[0-9]+/"
        + value["sha256"]
        + r"\.bin",
        key,
    ):
        raise ValueError("invalid object key")
    return value


def matching_object(client, bucket, receipt):
    arguments = dict(Bucket=bucket, Key=receipt["key"])
    if "version_id" in receipt:
        arguments["VersionId"] = receipt["version_id"]
    try:
        head = client.head_object(**arguments)
    except Exception as error:
        response = getattr(error, "response", {})
        if response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
            return False
        raise
    if (
        type(head.get("ContentLength")) is not int
        or head.get("ContentLength") != receipt["size"]
        or head.get("Metadata")
        != {
            "sha256": receipt["sha256"],
            "size": str(receipt["size"]),
        }
    ):
        raise ValueError("existing object identity differs")
    if "version_id" in receipt and head.get("VersionId") != receipt["version_id"]:
        raise ValueError("object version differs")
    return head


def publish(client, bucket, source, run_root, transfer_config=None):
    receipt, stamp = source_info(source, run_root)
    if not matching_object(client, bucket, receipt):
        with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
            if identity(os.fstat(stream.fileno())) != stamp:
                raise ValueError("source changed before upload")
            client.upload_fileobj(
                stream,
                bucket,
                receipt["key"],
                ExtraArgs={
                    "ACL": "private",
                    "Metadata": {
                        "sha256": receipt["sha256"],
                        "size": str(receipt["size"]),
                    },
                },
                Config=transfer_config,
            )
            if identity(os.fstat(stream.fileno())) != stamp:
                raise ValueError("source changed during upload")
    current, current_stamp = source_info(source, run_root)
    if current != receipt or current_stamp != stamp:
        raise ValueError("source changed after upload")
    head = matching_object(client, bucket, receipt)
    if not head:
        raise ValueError("uploaded object is absent")
    receipt["version_id"] = head.get("VersionId")
    validate_receipt(receipt)
    url = client.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": bucket,
            "Key": receipt["key"],
            "VersionId": receipt["version_id"],
        },
        ExpiresIn=600,
    )
    return dict(receipt, url=url, expires_at=int(time.time()) + 600)


def acknowledge(client, bucket, receipt, run_root):
    validate_receipt(receipt)
    current, _ = source_info(receipt["source"], run_root)
    if current != {key: value for key, value in receipt.items() if key != "version_id"}:
        raise ValueError("acknowledgement source identity differs")
    marker = safe_path(receipt["source"] + ".delivered")
    if not marker.is_file() or marker.read_text().strip() != receipt["sha256"]:
        raise ValueError("verified delivery marker required before deletion")
    if matching_object(client, bucket, receipt):
        client.delete_object(
            Bucket=bucket, Key=receipt["key"], VersionId=receipt["version_id"]
        )
    if matching_object(client, bucket, receipt):
        raise ValueError("object deletion was not confirmed")


def checked_final(final, receipt):
    size, digest, _ = digest_file(final)
    if (size, digest) != (receipt["size"], receipt["sha256"]):
        raise ValueError("delivered file size or hash differs")


def publication_directory(directory):
    """Match SSH's readable publication tree, even with a private host umask."""
    directory = safe_path(directory)
    missing = []
    current = directory
    while not current.exists():
        missing.append(current)
        current = current.parent
    for current in reversed(missing):
        try:
            current.mkdir(mode=0o755)
        except FileExistsError:
            if current.is_symlink() or not current.is_dir():
                raise ValueError("publication directory changed")
        else:
            # Only adjust directories created by this call, not existing paths.
            current.chmod(0o755)


def receive(value, source, partial, final, run_curl=subprocess.run):
    if not isinstance(value, dict) or set(value) != {
        "source",
        "key",
        "size",
        "sha256",
        "version_id",
        "uid",
        "gid",
        "mode",
        "url",
        "expires_at",
    }:
        raise ValueError("invalid publish schema")
    receipt = validate_receipt(
        {k: v for k, v in value.items() if k not in {"url", "expires_at"}}
    )
    if receipt["source"] != source:
        raise ValueError("source binding differs")
    url = value["url"]
    if not isinstance(url, str) or any(ord(c) < 33 or ord(c) > 126 for c in url):
        raise ValueError("invalid download URL")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise ValueError("verified HTTPS required")
    if (
        type(value["expires_at"]) is not int
        or not time.time() < value["expires_at"] <= time.time() + 600
    ):
        raise ValueError("invalid URL expiry")
    partial, final = safe_path(partial), safe_path(final)
    # Create public traversal first: partial parents may share the run root.
    publication_directory(final.parent)
    partial.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not final.exists():
        fd = os.open(
            partial, os.O_CREAT | os.O_TRUNC | os.O_WRONLY | os.O_NOFOLLOW, 0o600
        )
        try:
            os.fchmod(fd, 0o600)
            result = run_curl(
                [
                    "curl",
                    "--disable",
                    "--config",
                    "-",
                    "--proto",
                    "=https",
                    "--retry",
                    "0",
                    "--max-redirs",
                    "0",
                    "--max-time",
                    "300",
                    "--speed-limit",
                    "1",
                    "--speed-time",
                    "60",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--output",
                    f"/proc/self/fd/{fd}",
                    "--write-out",
                    "%{http_code}",
                ],
                input="url = " + json.dumps(url) + "\n",
                text=True,
                capture_output=True,
                timeout=300,
                pass_fds=(fd,),
            )
        finally:
            os.close(fd)
        if result.returncode != 0 or result.stdout != "200":
            raise ValueError("HTTPS download failed")
        checked_final(partial, receipt)
        os.chown(partial, receipt["uid"], receipt["gid"], follow_symlinks=False)
        os.chmod(partial, receipt["mode"])
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(partial, final, follow_symlinks=False)
        except FileExistsError:
            checked_final(final, receipt)
        partial.unlink()
    checked_final(final, receipt)
    pending = safe_path(str(final) + ".b2-ack.json")
    payload = json.dumps(receipt, sort_keys=True) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=final.parent, prefix=".b2-ack-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, pending, follow_symlinks=False)
        except FileExistsError:
            if pending.read_text() != payload:
                raise ValueError("pending acknowledgement differs")
        directory_fd = os.open(final.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink()


def credentials(path=Path("/data/trinity/training.env")):
    allowed = (
        "B2_S3_KEY_ID",
        "B2_S3_APPLICATION_KEY",
        "B2_S3_BUCKET",
        "B2_S3_ENDPOINT",
        "B2_S3_REGION",
    )
    path = safe_path(path)
    status = path.stat()
    if not stat.S_ISREG(status.st_mode) or stat.S_IMODE(status.st_mode) & 0o077:
        raise ValueError("credential file permissions are too broad")
    values = {}
    for line in path.read_text().splitlines():
        line = line.removeprefix("export ")
        name, separator, raw = line.partition("=")
        if separator and name in allowed:
            parts = shlex.split(raw, comments=True)
            if len(parts) != 1 or name in values:
                raise ValueError("invalid credential field")
            values[name] = parts[0]
    if set(values) != set(allowed) or not all(values.values()):
        raise ValueError("required B2 credential fields are missing")
    parsed = urlsplit(values["B2_S3_ENDPOINT"])
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid B2 endpoint")
    return values


def client_settings():
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config

    fields = credentials()
    client = boto3.client(
        "s3",
        endpoint_url=fields["B2_S3_ENDPOINT"],
        region_name=fields["B2_S3_REGION"],
        aws_access_key_id=fields["B2_S3_KEY_ID"],
        aws_secret_access_key=fields["B2_S3_APPLICATION_KEY"],
        aws_session_token=None,
        verify=True,
        config=Config(
            signature_version="s3v4",
            retries={"total_max_attempts": 1},
            connect_timeout=30,
            read_timeout=60,
        ),
    )
    transfer = TransferConfig(
        preferred_transfer_client="classic",
        multipart_threshold=8 * 1024 * 1024,
        multipart_chunksize=8 * 1024 * 1024,
        max_concurrency=1,
        use_threads=False,
    )
    return client, fields["B2_S3_BUCKET"], transfer


@contextmanager
def deadline():
    def expired(*_):
        raise TimeoutError("operation deadline")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(300)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "configs/nodes.toml",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    pub = sub.add_parser("publish")
    pub.add_argument("--source", required=True)
    ack = sub.add_parser("ack")
    ack.add_argument("--receipt", required=True)
    receiver = sub.add_parser("receive")
    receiver.add_argument("--source", required=True)
    receiver.add_argument("--partial", required=True)
    receiver.add_argument("--final", required=True)
    pending = sub.add_parser("ack-json")
    pending.add_argument("--receipt", required=True)
    pending.add_argument("--final", required=True)
    args = parser.parse_args(argv)
    try:
        with deadline():
            if args.command == "receive":
                receive(
                    parse_json(sys.stdin.read(16385)),
                    args.source,
                    args.partial,
                    args.final,
                )
            elif args.command == "ack-json":
                receipt = validate_receipt(
                    parse_json(safe_path(args.receipt).read_text())
                )
                checked_final(args.final, receipt)
                print(json.dumps(receipt))
            else:
                config = tomllib.loads(args.config.read_text())
                client, bucket, transfer = client_settings()
                if args.command == "publish":
                    print(
                        json.dumps(
                            publish(
                                client,
                                bucket,
                                args.source,
                                config["paths"]["runs"],
                                transfer,
                            )
                        )
                    )
                elif args.command == "ack":
                    acknowledge(
                        client,
                        bucket,
                        parse_json(args.receipt),
                        config["paths"]["runs"],
                    )
                else:
                    print('{"ready":true}')
    except Exception as error:
        # SDK/curl error strings can contain URLs or credential identifiers.
        print(f"B2 rollout operation failed ({type(error).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
