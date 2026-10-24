"""Host-only B2 rollout publication and credential-free HTTPS reception.

Provision the configured host venv from b2-rollout-requirements.txt explicitly.
Only publisher commands read /data/trinity/training.env. Never log publish JSON.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import fcntl
import hashlib
import ipaddress
import json
import math
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import time
import tomllib
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

try:
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover - the host venv provides botocore.
    ClientError = None


_TRANSIENT_ERROR_CODES = frozenset(
    {
        "InternalError",
        "InternalFailure",
        "PriorRequestNotComplete",
        "RequestLimitExceeded",
        "RequestTimeout",
        "RequestTimeoutException",
        "ServiceUnavailable",
        "SlowDown",
        "Throttling",
        "ThrottlingException",
        "TooManyRequestsException",
    }
)
_TERMINAL_ERROR_CODES = frozenset(
    {
        "AccessDenied",
        "AccountProblem",
        "AllAccessDisabled",
        "AuthorizationHeaderMalformed",
        "BucketAlreadyExists",
        "BucketNotEmpty",
        "Conflict",
        "EntityTooSmall",
        "ExpiredToken",
        "InvalidAccessKeyId",
        "InvalidBucketName",
        "InvalidClientTokenId",
        "InvalidDigest",
        "InvalidObjectState",
        "InvalidPart",
        "InvalidPartOrder",
        "InvalidRequest",
        "InvalidSecurityToken",
        "InvalidToken",
        "InvalidRange",
        "MethodNotAllowed",
        "MissingAuthenticationToken",
        "NoSuchBucket",
        "NoSuchKey",
        "NoSuchUpload",
        "NoSuchVersion",
        "NotFound",
        "PermanentRedirect",
        "RequestTimeTooSkewed",
        "RequestExpired",
        "SignatureDoesNotMatch",
        "TokenRefreshRequired",
        "UnrecognizedClientException",
        "PreconditionFailed",
    }
)
_TERMINAL_ERROR_MARKERS = (
    "access",
    "auth",
    "credential",
    "forbidden",
    "identity",
    "notfound",
    "nosuch",
    "permission",
    "signature",
    "unauthorized",
)
_KNOWN_ERROR_CODES = _TRANSIENT_ERROR_CODES | frozenset(
    {
        "AccessDenied",
        "AuthorizationHeaderMalformed",
        "Conflict",
        "EntityTooSmall",
        "ExpiredToken",
        "InvalidAccessKeyId",
        "InvalidBucketName",
        "InvalidPart",
        "InvalidPartOrder",
        "InvalidRequest",
        "MethodNotAllowed",
        "NoSuchBucket",
        "NoSuchKey",
        "NoSuchUpload",
        "NotFound",
        "PermanentRedirect",
        "PreconditionFailed",
        "RequestTimeTooSkewed",
        "SignatureDoesNotMatch",
    }
)
_API_OPERATIONS = frozenset(
    {
        "publish.head",
        "publish.upload",
        "publish.multipart.create",
        "publish.multipart.part",
        "publish.multipart.complete",
        "publish.multipart.abort",
        "publish.sign",
        "ack.head",
        "ack.delete",
        "publish.multipart.list",
    }
)


class _MultipartCreateUnconfirmed(Exception):
    """The create request may have succeeded, but its upload ID was lost."""

    def __init__(self, error):
        super().__init__("multipart upload identity was not confirmed")
        self.error = error


class _MultipartCleanupFailed(Exception):
    """Abort or exact absence verification failed; never retry the upload."""

    def __init__(self, error):
        super().__init__("multipart cleanup was not confirmed")
        self.error = error


def _is_terminal_error_code(provider_code):
    if provider_code in _TERMINAL_ERROR_CODES:
        return True
    lowered = provider_code.casefold() if isinstance(provider_code, str) else ""
    return any(marker in lowered for marker in _TERMINAL_ERROR_MARKERS)


def _client_error_details(error):
    """Return safe SDK error facts, or None for local/non-SDK errors."""
    candidate = ClientError
    is_client_error = candidate is not None and isinstance(error, candidate)
    # Tests and the host preflight can use a botocore-shaped error without
    # importing botocore in the repository test environment.
    if not is_client_error and error.__class__.__name__ == "ClientError":
        is_client_error = isinstance(getattr(error, "response", None), dict)
    if not is_client_error:
        return None
    response = getattr(error, "response", {})
    if not isinstance(response, Mapping):
        response = {}
    metadata = response.get("ResponseMetadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    status = metadata.get("HTTPStatusCode")
    if type(status) is not int or not 100 <= status <= 599:
        status = None
    provider_error = response.get("Error", {})
    provider_code = (
        provider_error.get("Code") if isinstance(provider_error, dict) else None
    )
    if not isinstance(provider_code, str):
        provider_code = None
    if provider_code in _KNOWN_ERROR_CODES or provider_code in _TERMINAL_ERROR_CODES:
        code = provider_code
    elif status is not None and 500 <= status <= 599:
        code = f"http_{status}"
    else:
        code = "unknown"
    return dict(
        code=code,
        status=status,
        retryable=(
            not _is_terminal_error_code(provider_code)
            and (
                provider_code in _TRANSIENT_ERROR_CODES
                or (status is not None and 500 <= status <= 599)
            )
        ),
    )


def _api_telemetry(operation, details, attempt, retry):
    operation = operation if operation in _API_OPERATIONS else "unknown"
    print(
        json.dumps(
            {
                "b2_api_error": True,
                "operation": operation,
                "error_code": details["code"],
                "status": details["status"],
                "attempt": attempt,
                "retry": retry,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )


def _api_call(operation, call):
    """Call one API operation with at most one explicit transient retry."""
    for attempt in (1, 2):
        try:
            return call()
        except Exception as error:
            details = _client_error_details(error)
            if details is None:
                raise
            retry = bool(details["retryable"] and attempt == 1)
            _api_telemetry(operation, details, attempt, retry)
            if not retry:
                raise



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


def _open_regular_fd(path, mode=None):
    path = safe_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        status = os.fstat(fd)
        if not stat.S_ISREG(status.st_mode):
            raise ValueError("regular file required")
        if mode is not None and (
            status.st_uid != os.geteuid() or stat.S_IMODE(status.st_mode) != mode
        ):
            raise ValueError("regular file identity differs")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _read_fd(fd, maximum):
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
        raise ValueError("file size or type differs")
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    total = 0
    while True:
        chunk = os.read(fd, min(1024 * 1024, maximum - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > maximum:
            raise ValueError("file size exceeds bound")
        chunks.append(chunk)
    after = os.fstat(fd)
    if identity(before) != identity(after) or total != before.st_size:
        raise ValueError("file changed during read")
    return b"".join(chunks)


def _digest_fd(fd):
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("regular file required")
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(fd, 1024 * 1024):
        digest.update(chunk)
    after = os.fstat(fd)
    if identity(before) != identity(after):
        raise ValueError("file changed during hashing")
    return before.st_size, digest.hexdigest(), identity(before)


def digest_file(path):
    path = safe_path(path)
    fd = _open_regular_fd(path)
    try:
        size, digest, stamp = _digest_fd(fd)
    finally:
        os.close(fd)
    if identity(path.stat()) != stamp:
        raise ValueError("source changed during hashing")
    return size, digest, stamp


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


def matching_object(client, bucket, receipt, *, operation="publish.head"):
    arguments = dict(Bucket=bucket, Key=receipt["key"])
    if "version_id" in receipt:
        arguments["VersionId"] = receipt["version_id"]
    try:
        head = _api_call(operation, lambda: client.head_object(**arguments))
    except Exception as error:
        response = getattr(error, "response", {})
        metadata = response.get("ResponseMetadata", {}) if isinstance(response, dict) else {}
        if isinstance(metadata, dict) and metadata.get("HTTPStatusCode") == 404:
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


def _metadata(receipt):
    return {"sha256": receipt["sha256"], "size": str(receipt["size"])}


def _multipart_settings(transfer_config):
    threshold = getattr(transfer_config, "multipart_threshold", 8 * 1024 * 1024)
    chunksize = getattr(transfer_config, "multipart_chunksize", 8 * 1024 * 1024)
    if type(threshold) is not int or threshold <= 0:
        threshold = 8 * 1024 * 1024
    if type(chunksize) is not int or chunksize <= 0:
        chunksize = 8 * 1024 * 1024
    return threshold, chunksize


def _open_source(source):
    return os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW), "rb")


def _verify_multipart_absent(client, bucket, key, upload_id):
    if any(
        type(value) is not str or not value
        for value in (bucket, key, upload_id)
    ):
        raise ValueError("multipart identity is invalid")
    response = _api_call(
        "publish.multipart.list",
        lambda: client.list_multipart_uploads(Bucket=bucket, Prefix=key),
    )
    if not isinstance(response, dict):
        raise ValueError("multipart absence is not proven")
    if response.get("Bucket") != bucket or response.get("Prefix") != key:
        raise ValueError("multipart listing identity differs")
    if type(response.get("IsTruncated")) is not bool:
        raise ValueError("multipart pagination flag is invalid")
    uploads = response.get("Uploads", [])
    if "Uploads" in response and type(uploads) is not list:
        raise ValueError("multipart listing is invalid")
    for marker in ("KeyMarker", "UploadIdMarker"):
        value = response.get(marker, "")
        if type(value) is not str or value:
            raise ValueError("multipart pagination marker is invalid")
    if "Delimiter" in response and (
        type(response["Delimiter"]) is not str or response["Delimiter"]
    ):
        raise ValueError("multipart delimiter is invalid")
    if "CommonPrefixes" in response:
        common = response["CommonPrefixes"]
        if type(common) is not list or common:
            raise ValueError("multipart common prefixes are invalid")
    next_key = response.get("NextKeyMarker", "")
    next_upload = response.get("NextUploadIdMarker", "")
    if type(next_key) is not str or type(next_upload) is not str:
        raise ValueError("multipart next marker is invalid")
    if response["IsTruncated"]:
        if not next_key or not next_upload:
            raise ValueError("multipart pagination is incomplete")
        raise ValueError("multipart listing is truncated")
    if bool(next_key) != bool(next_upload):
        raise ValueError("multipart next markers are inconsistent")
    if next_key or next_upload:
        raise ValueError("multipart next markers are unexpected")
    if "MaxUploads" in response and (
        type(response["MaxUploads"]) is not int or response["MaxUploads"] <= 0
    ):
        raise ValueError("multipart page size is invalid")
    seen_keys, seen_ids = set(), set()
    for upload in uploads:
        if not isinstance(upload, Mapping):
            raise ValueError("multipart listing is invalid")
        item_key, item_id = upload.get("Key"), upload.get("UploadId")
        if (
            type(item_key) is not str
            or not item_key
            or type(item_id) is not str
            or not item_id
        ):
            raise ValueError("multipart upload identity is invalid")
        if item_key != key or item_key in seen_keys or item_id in seen_ids:
            raise ValueError("multipart listing is ambiguous")
        seen_keys.add(item_key)
        seen_ids.add(item_id)
        if item_key == key and item_id == upload_id:
            raise ValueError("multipart upload was not aborted")


def _abort_multipart(client, bucket, key, upload_id):
    try:
        _api_call(
            "publish.multipart.abort",
            lambda: client.abort_multipart_upload(
                Bucket=bucket, Key=key, UploadId=upload_id
            ),
        )
    except Exception as error:
        # A failed abort is not safe to ignore. A fresh exact listing can
        # prove that the attempt-owned upload is already gone; otherwise the
        # operation fails closed and never starts a replacement upload.
        try:
            _verify_multipart_absent(client, bucket, key, upload_id)
        except Exception as verification_error:
            raise _MultipartCleanupFailed(error) from verification_error
        return
    try:
        _verify_multipart_absent(client, bucket, key, upload_id)
    except Exception as verification_error:
        raise _MultipartCleanupFailed(verification_error) from verification_error


def _create_multipart(client, bucket, receipt):
    try:
        return client.create_multipart_upload(
            Bucket=bucket,
            Key=receipt["key"],
            ACL="private",
            Metadata=_metadata(receipt),
        )
    except Exception as error:
        if _client_error_details(error) is not None:
            # The server may have created an upload whose ID was lost. There
            # is no safe identity to abort or retry, so fail closed instead
            # of creating a second attempt that could orphan more state.
            raise _MultipartCreateUnconfirmed(error) from error
        raise


def _multipart_upload_once(client, bucket, receipt, source, stamp, chunksize):
    upload_id = None
    parts = []
    try:
        created = _create_multipart(client, bucket, receipt)
        upload_id = created.get("UploadId") if isinstance(created, dict) else None
        if not isinstance(upload_id, str) or not upload_id:
            raise ValueError("multipart upload identity missing")
        with _open_source(source) as stream:
            if identity(os.fstat(stream.fileno())) != stamp:
                raise ValueError("source changed before multipart upload")
            part_number = 1
            while True:
                body = stream.read(chunksize)
                if not body:
                    break
                if identity(os.fstat(stream.fileno())) != stamp:
                    raise ValueError("source changed during multipart upload")
                response = client.upload_part(
                    Bucket=bucket,
                    Key=receipt["key"],
                    PartNumber=part_number,
                    UploadId=upload_id,
                    Body=body,
                )
                etag = response.get("ETag") if isinstance(response, dict) else None
                if not isinstance(etag, str) or not etag:
                    raise ValueError("multipart part identity missing")
                parts.append({"PartNumber": part_number, "ETag": etag})
                part_number += 1
            if not parts:
                raise ValueError("empty multipart upload")
            if identity(os.fstat(stream.fileno())) != stamp:
                raise ValueError("source changed after multipart upload")
        client.complete_multipart_upload(
            Bucket=bucket,
            Key=receipt["key"],
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
        # A successful completion consumes the upload id. Do not issue an
        # abort for this accepted object if a later local check fails.
        upload_id = None
    except Exception:
        if upload_id is not None:
            _abort_multipart(client, bucket, receipt["key"], upload_id)
        raise


def _upload_fileobj_once(client, bucket, receipt, source, stamp, transfer_config):
    with _open_source(source) as stream:
        if identity(os.fstat(stream.fileno())) != stamp:
            raise ValueError("source changed before upload")
        client.upload_fileobj(
            stream,
            bucket,
            receipt["key"],
            ExtraArgs={"ACL": "private", "Metadata": _metadata(receipt)},
            Config=transfer_config,
        )
        if identity(os.fstat(stream.fileno())) != stamp:
            raise ValueError("source changed during upload")


def _upload_once(client, bucket, receipt, source, stamp, transfer_config):
    threshold, chunksize = _multipart_settings(transfer_config)
    if receipt["size"] >= threshold:
        _multipart_upload_once(
            client, bucket, receipt, source, stamp, chunksize
        )
    else:
        _upload_fileobj_once(
            client, bucket, receipt, source, stamp, transfer_config
        )


def _revalidate_source(source, run_root, receipt, stamp):
    current, current_stamp = source_info(source, run_root)
    if current != receipt or current_stamp != stamp:
        raise ValueError("source changed during publish")


def _upload_with_retry(
    client, bucket, receipt, source, run_root, stamp, transfer_config
):
    for attempt in (1, 2):
        try:
            _upload_once(
                client, bucket, receipt, source, stamp, transfer_config
            )
            return
        except _MultipartCleanupFailed:
            raise
        except _MultipartCreateUnconfirmed as error:
            details = _client_error_details(error.error)
            if details is not None:
                _api_telemetry("publish.multipart.create", details, attempt, False)
            raise error.error
        except Exception as error:
            details = _client_error_details(error)
            if details is None:
                raise
            retry = bool(details["retryable"] and attempt == 1)
            _api_telemetry("publish.upload", details, attempt, retry)
            if not retry:
                raise
            # A completed multipart upload or a successful small upload can
            # be hidden by a lost response. Accept only the exact object;
            # never publish a second accepted version for the same receipt.
            _revalidate_source(source, run_root, receipt, stamp)
            if matching_object(client, bucket, receipt):
                return
    raise AssertionError("bounded publish retry loop did not return")


def publish(client, bucket, source, run_root, transfer_config=None):
    receipt, stamp = source_info(source, run_root)
    if not matching_object(client, bucket, receipt):
        _upload_with_retry(
            client,
            bucket,
            receipt,
            source,
            run_root,
            stamp,
            transfer_config,
        )
    current, current_stamp = source_info(source, run_root)
    if current != receipt or current_stamp != stamp:
        raise ValueError("source changed after upload")
    head = matching_object(client, bucket, receipt)
    if not head:
        raise ValueError("uploaded object is absent")
    receipt["version_id"] = head.get("VersionId")
    validate_receipt(receipt)
    url = _api_call(
        "publish.sign",
        lambda: client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": bucket,
                "Key": receipt["key"],
                "VersionId": receipt["version_id"],
            },
            ExpiresIn=600,
        ),
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
    delete_attempted = False
    try:
        if matching_object(client, bucket, receipt, operation="ack.head"):
            delete_attempted = True
            _api_call(
                "ack.delete",
                lambda: client.delete_object(
                    Bucket=bucket,
                    Key=receipt["key"],
                    VersionId=receipt["version_id"],
                ),
            )
    except Exception as error:
        # Delete of an exact version is idempotent. If a response was lost or
        # the provider reports that the version is already gone, accept only
        # after a fresh exact-version absence check. Other faults remain
        # terminal and retain their original error.
        if delete_attempted and _client_error_details(error) is not None:
            try:
                if not matching_object(client, bucket, receipt, operation="ack.head"):
                    return
            except Exception:
                pass
        raise
    if matching_object(client, bucket, receipt, operation="ack.head"):
        raise ValueError("object deletion was not confirmed")


def checked_final(final, receipt):
    size, digest, _ = digest_file(final)
    if (size, digest) != (receipt["size"], receipt["sha256"]):
        raise ValueError("delivered file size or hash differs")


def checked_published(final, receipt):
    final = safe_path(final)
    fd = _open_regular_fd(final)
    try:
        size, digest, stamp = _digest_fd(fd)
        status = os.fstat(fd)
        if (
            (size, digest) != (receipt["size"], receipt["sha256"])
            or status.st_uid != receipt["uid"]
            or status.st_gid != receipt["gid"]
            or stat.S_IMODE(status.st_mode) != receipt["mode"]
            or identity(status) != stamp
            or identity(final.stat()) != stamp
        ):
            raise ValueError("published file identity differs")
    finally:
        os.close(fd)


def _publish_partial(partial, final, receipt):
    fd = _open_regular_fd(partial, 0o600)
    try:
        size, digest, _ = _digest_fd(fd)
        if (size, digest) != (receipt["size"], receipt["sha256"]):
            raise ValueError("delivered file size or hash differs")
        status = os.fstat(fd)
        if identity(partial.stat()) != identity(status):
            raise ValueError("partial file changed before publication")
        os.fchown(fd, receipt["uid"], receipt["gid"])
        os.fchmod(fd, receipt["mode"])
        os.fsync(fd)
        status = os.fstat(fd)
        if identity(partial.stat()) != identity(status):
            raise ValueError("partial file changed during publication")
        try:
            os.link(partial, final, follow_symlinks=False)
        except FileExistsError:
            checked_published(final, receipt)
        checked_published(final, receipt)
        if identity(partial.stat()) != identity(os.fstat(fd)):
            raise ValueError("partial file changed after publication")
        partial.unlink()
        _sync_directory(final.parent)
    finally:
        os.close(fd)


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


def download_settings(workers=1, threshold=64 * 1024 * 1024):
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("download_workers must be an integer from 1 to 32")
    if type(threshold) is not int or not 1 <= threshold <= 2**40:
        raise ValueError("download_threshold_bytes must be a positive bounded integer")
    return workers, threshold


def validate_ranges(ranges, size):
    if type(size) is not int or size <= 0 or not ranges:
        raise ValueError("positive download size and nonempty ranges required")
    position = 0
    for start, end in ranges:
        if (
            type(start) is not int
            or type(end) is not int
            or start != position
            or end < start
        ):
            raise ValueError("download ranges overlap or have a gap")
        position = end + 1
    if position != size:
        raise ValueError("download ranges do not cover the object")


def download_ranges(size, workers):
    download_settings(workers)
    if type(size) is not int or size <= 0:
        raise ValueError("positive download size required")
    width, extra = divmod(size, min(size, workers))
    ranges, position = [], 0
    for index in range(min(size, workers)):
        end = position + width + (index < extra) - 1
        ranges.append((position, end))
        position = end + 1
    validate_ranges(ranges, size)
    return ranges


def _range_result_bytes(headers_raw, metrics_raw, start, end, size):
    # No server strings are included in logs. Accept only the final HTTP block.
    if len(headers_raw) > 128 * 1024 or len(metrics_raw) > 4096:
        raise ValueError("range response metadata too large")
    headers = headers_raw.decode("iso-8859-1")
    metrics = metrics_raw.decode("ascii")
    blocks = headers.strip().split("\n\n")
    lines = blocks[-1].splitlines()
    if not lines or not re.fullmatch(r"HTTP/(?:1\.[01]|2|3) 206(?: .*)?", lines[0]):
        raise ValueError("range HTTP status differs")
    values = [
        line.split(":", 1)[1].strip()
        for line in lines[1:]
        if line.lower().startswith("content-range:")
    ]
    if values != [f"bytes {start}-{end}/{size}"]:
        raise ValueError("Content-Range differs")
    lengths = [
        line.split(":", 1)[1].strip()
        for line in lines[1:]
        if line.lower().startswith("content-length:")
    ]
    if lengths != [str(end - start + 1)]:
        raise ValueError("Content-Length differs")
    fields = metrics.split("\t")
    if (
        len(fields) != 7
        or fields[0] != "206"
        or fields[2] not in {"1.0", "1.1", "2", "3"}
    ):
        raise ValueError("range metrics differ")
    remote_ip = str(ipaddress.ip_address(fields[1]))
    times = [float(value) for value in fields[3:6]]
    if any(not math.isfinite(value) or value < 0 for value in times):
        raise ValueError("invalid range timing")
    if not re.fullmatch(r"[0-9]+", fields[6]) or int(fields[6]) != end - start + 1:
        raise ValueError("range byte count differs")
    return dict(
        remote_ip=remote_ip,
        protocol=fields[2],
        connect_s=times[0],
        first_byte_s=times[1],
        total_s=times[2],
        bytes=int(fields[6]),
    )


def range_result(headers, metrics, start, end, size):
    header_fd = _open_regular_fd(headers, 0o600)
    metrics_fd = _open_regular_fd(metrics, 0o600)
    try:
        headers_raw = _read_fd(header_fd, 128 * 1024)
        metrics_raw = _read_fd(metrics_fd, 4096)
    finally:
        os.close(header_fd)
        os.close(metrics_fd)
    return _range_result_bytes(headers_raw, metrics_raw, start, end, size)


def stop_downloads(children):
    # Each curl owns a new process group; stop only groups created here.
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    until = time.monotonic() + 1
    for child in children:
        try:
            child.wait(timeout=max(0, until - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        finally:
            if child.stdin is not None and not child.stdin.closed:
                try:
                    child.stdin.close()
                except OSError:
                    pass


@contextmanager
def defer_download_signals():
    previous = signal.pthread_sigmask(
        signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT, signal.SIGALRM}
    )
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


_PARTS_SCHEMA = 1
_RANGE_RECORD_KEYS = {
    "schema",
    "index",
    "start",
    "end",
    "size",
    "sha256",
    "http_status",
    "content_range",
    "report",
}
_REPORT_KEYS = {
    "remote_ip",
    "protocol",
    "connect_s",
    "first_byte_s",
    "total_s",
    "bytes",
}
_MAX_QUARANTINE_FILES = 64
_MAX_QUARANTINE_BYTES = 64 * 1024 * 1024


def _sync_directory(directory):
    directory = safe_path(directory)
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _owned_regular(path, mode=0o600):
    path = safe_path(path)
    status = path.lstat()
    if (
        not stat.S_ISREG(status.st_mode)
        or status.st_uid != os.geteuid()
        or stat.S_IMODE(status.st_mode) != mode
    ):
        raise ValueError("range evidence file identity differs")
    return status


def _part_path(parts, index, suffix):
    return parts / f"{index}.{suffix}"


def _range_paths(parts, index):
    return tuple(
        _part_path(parts, index, suffix)
        for suffix in ("part", "headers", "metrics", "record")
    )


def _quarantine_entry(parts, path):
    """Move bounded regular evidence or discard it under the receiver lock."""
    try:
        source_status = path.lstat()
    except FileNotFoundError:
        return
    quarantine = parts / "quarantine"
    try:
        status = quarantine.lstat()
    except FileNotFoundError:
        quarantine.mkdir(mode=0o700)
        status = quarantine.lstat()
        _sync_directory(parts)
    if (
        not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.geteuid()
        or stat.S_IMODE(status.st_mode) != 0o700
    ):
        raise ValueError("range quarantine identity differs")
    if not stat.S_ISREG(source_status.st_mode):
        if stat.S_ISDIR(source_status.st_mode):
            raise ValueError("range quarantine directory refused")
        path.unlink()
        _sync_directory(parts)
        return
    if source_status.st_uid != os.geteuid() or stat.S_IMODE(
        source_status.st_mode
    ) != 0o600:
        raise ValueError("range quarantine file identity differs")
    try:
        existing = sorted(quarantine.iterdir(), key=lambda item: item.name)
        usage = []
        for item in existing:
            status = item.lstat()
            if (
                not stat.S_ISREG(status.st_mode)
                or status.st_uid != os.geteuid()
                or stat.S_IMODE(status.st_mode) != 0o600
            ):
                raise ValueError("range quarantine entry identity differs")
            usage.append((item, status.st_size))
        total = sum(size for _, size in usage)
        while usage and (
            len(usage) >= _MAX_QUARANTINE_FILES
            or total + source_status.st_size > _MAX_QUARANTINE_BYTES
        ):
            old, old_size = usage.pop(0)
            old.unlink()
            total -= old_size
        if (
            source_status.st_size > _MAX_QUARANTINE_BYTES
            or len(usage) >= _MAX_QUARANTINE_FILES
            or total + source_status.st_size > _MAX_QUARANTINE_BYTES
        ):
            path.unlink()
            _sync_directory(parts)
            return
        for sequence in range(_MAX_QUARANTINE_FILES):
            destination = quarantine / f"q-{sequence:08d}-{path.name}"
            if not destination.exists() and not destination.is_symlink():
                break
        else:
            path.unlink()
            _sync_directory(parts)
            return
        os.rename(path, destination)
    finally:
        _sync_directory(quarantine)
    _sync_directory(parts)


def _quarantine_range(parts, index):
    for path in _range_paths(parts, index):
        _quarantine_entry(parts, path)


def _quarantine_all_ranges(parts, count):
    for index in range(count):
        _quarantine_range(parts, index)


def _purge_quarantine(quarantine):
    try:
        entries = sorted(quarantine.iterdir(), key=lambda item: item.name)
    except FileNotFoundError:
        return
    usage = []
    for path in entries:
        status = path.lstat()
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_uid != os.geteuid()
            or stat.S_IMODE(status.st_mode) != 0o600
        ):
            raise ValueError("range quarantine entry identity differs")
        usage.append((path, status.st_size))
    total = sum(size for _, size in usage)
    while usage and (
        len(usage) > _MAX_QUARANTINE_FILES or total > _MAX_QUARANTINE_BYTES
    ):
        path, size = usage.pop(0)
        path.unlink()
        total -= size
    _sync_directory(quarantine)


def _atomic_json(path, value):
    path = safe_path(path)
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError("durable range record differs")
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _parts_lock(parts):
    """Return a process-safe lock for one final path's range directory."""
    lock_path = parts / "receiver.lock"
    try:
        status = lock_path.lstat()
    except FileNotFoundError:
        status = None
    if status is not None and (
        not stat.S_ISREG(status.st_mode) or status.st_uid != os.geteuid()
    ):
        raise ValueError("range lock identity differs")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    locked = False
    try:
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        locked = True
        _sync_directory(parts)
    except BaseException:
        if locked:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        raise
    return fd


def _release_parts_lock(fd):
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _range_identity(receipt, size):
    if receipt is None:
        # Direct unit callers do not have a publish receipt. Production calls
        # always pass the validated receipt from receive().
        return {"size": size}
    validated = validate_receipt(receipt)
    if validated["size"] != size:
        raise ValueError("range receipt size differs")
    return dict(validated)


def _range_manifest(receipt, size, workers, ranges):
    return {
        "schema": _PARTS_SCHEMA,
        "identity": _range_identity(receipt, size),
        "workers": workers,
        "ranges": [
            {
                "index": index,
                "start": start,
                "end": end,
                "size": end - start + 1,
                "total_size": size,
            }
            for index, (start, end) in enumerate(ranges)
        ],
    }


def _read_json(path):
    fd = _open_regular_fd(path, 0o600)
    try:
        return parse_json(_read_fd(fd, 16384).decode("utf-8"))
    finally:
        os.close(fd)


def _prepare_parts(parts, expected):
    try:
        status = parts.lstat()
    except FileNotFoundError:
        try:
            parts.mkdir(mode=0o700)
        except FileExistsError:
            status = parts.lstat()
        else:
            status = parts.lstat()
            _sync_directory(parts.parent)
    if (
        not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.geteuid()
        or stat.S_IMODE(status.st_mode) != 0o700
    ):
        raise ValueError("range directory identity differs")
    lock_fd = _parts_lock(parts)
    try:
        known = {"manifest.json", "receiver.lock", "quarantine"}
        for index in range(len(expected["ranges"])):
            known.update(path.name for path in _range_paths(parts, index))
        for path in list(parts.iterdir()):
            if path.name not in known:
                _quarantine_entry(parts, path)
        quarantine = parts / "quarantine"
        try:
            quarantine_status = quarantine.lstat()
        except FileNotFoundError:
            quarantine_status = None
        if quarantine_status is not None:
            if (
                not stat.S_ISDIR(quarantine_status.st_mode)
                or quarantine_status.st_uid != os.geteuid()
                or stat.S_IMODE(quarantine_status.st_mode) != 0o700
            ):
                raise ValueError("range quarantine identity differs")
            _purge_quarantine(quarantine)
        manifest_path = parts / "manifest.json"
        try:
            manifest_path.lstat()
        except FileNotFoundError:
            _atomic_json(manifest_path, expected)
        else:
            if _read_json(manifest_path) != expected:
                # The old receipt or range layout is never safe to reuse. Keep it
                # intact for diagnosis and force the caller to choose a new path.
                raise ValueError("durable range identity differs")
    except BaseException:
        _release_parts_lock(lock_fd)
        raise
    return lock_fd


def _record_report(report, expected_size):
    if (
        not isinstance(report, dict)
        or set(report) != _REPORT_KEYS
        or type(report["bytes"]) is not int
        or report["bytes"] != expected_size
    ):
        raise ValueError("range report differs")
    if not isinstance(report["remote_ip"], str):
        raise ValueError("range report differs")
    if report["protocol"] not in {"1.0", "1.1", "2", "3"}:
        raise ValueError("range report differs")
    if any(
        not isinstance(report[name], (int, float))
        or isinstance(report[name], bool)
        or not math.isfinite(report[name])
        or report[name] < 0
        for name in ("connect_s", "first_byte_s", "total_s")
    ):
        raise ValueError("range report differs")


def _valid_range_record(parts, spec):
    index, start, end, expected_size, total_size = (
        spec["index"],
        spec["start"],
        spec["end"],
        spec["size"],
        spec["total_size"],
    )
    part, headers, metrics, record = _range_paths(parts, index)
    present = []
    for path in (part, headers, metrics, record):
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        present.append(path)
    if not present:
        return False
    fds = []
    try:
        for path in (part, headers, metrics, record):
            fds.append(_open_regular_fd(path, 0o600))
        part_fd, headers_fd, metrics_fd, record_fd = fds
        value = parse_json(_read_fd(record_fd, 16384).decode("utf-8"))
        if set(value) != _RANGE_RECORD_KEYS:
            raise ValueError("range record schema differs")
        if value["schema"] != _PARTS_SCHEMA or value["index"] != index:
            raise ValueError("range record identity differs")
        if (value["start"], value["end"], value["size"]) != (start, end, expected_size):
            raise ValueError("range record layout differs")
        if (
            value["http_status"] != 206
            or value["content_range"] != f"bytes {start}-{end}/{total_size}"
        ):
            raise ValueError("range record response differs")
        if not isinstance(value["sha256"], str) or not re.fullmatch(
            r"[a-f0-9]{64}", value["sha256"]
        ):
            raise ValueError("range record hash differs")
        _record_report(value["report"], expected_size)
        report = _range_result_bytes(
            _read_fd(headers_fd, 128 * 1024),
            _read_fd(metrics_fd, 4096),
            start,
            end,
            total_size,
        )
        if report != value["report"]:
            raise ValueError("saved range response differs")
        actual_size, actual_digest, _ = _digest_fd(part_fd)
        if (actual_size, actual_digest) != (expected_size, value["sha256"]):
            raise ValueError("range part hash differs")
    except (OSError, ValueError, TypeError, KeyError):
        _quarantine_range(parts, index)
        return False
    finally:
        for fd in fds:
            os.close(fd)
    return value


def _publish_range_record(parts, spec, report):
    index, start, end, expected_size, total_size = (
        spec["index"],
        spec["start"],
        spec["end"],
        spec["size"],
        spec["total_size"],
    )
    _record_report(report, expected_size)
    part, headers, metrics, record = _range_paths(parts, index)
    fds = []
    try:
        for path in (part, headers, metrics):
            fds.append(_open_regular_fd(path, 0o600))
        for fd in fds:
            os.fsync(fd)
        current_report = _range_result_bytes(
            _read_fd(fds[1], 128 * 1024),
            _read_fd(fds[2], 4096),
            start,
            end,
            total_size,
        )
        if current_report != report:
            raise ValueError("range response changed before record")
        actual_size, actual_digest, _ = _digest_fd(fds[0])
    finally:
        for fd in fds:
            os.close(fd)
    if actual_size != expected_size:
        raise ValueError("range file size differs")
    value = {
        "schema": _PARTS_SCHEMA,
        "index": index,
        "start": start,
        "end": end,
        "size": expected_size,
        "sha256": actual_digest,
        "http_status": 206,
        "content_range": f"bytes {start}-{end}/{total_size}",
        "report": report,
    }
    _atomic_json(record, value)
    return value


def receive_ranges(
    url,
    size,
    partial,
    final,
    workers,
    *,
    receipt=None,
    hold_lock=False,
    lock_fd=None,
    clock=time.monotonic,
):
    until = clock() + 300
    ranges = download_ranges(size, workers)
    parts = safe_path(str(final) + ".b2-parts")
    expected = _range_manifest(receipt, size, workers, ranges)
    owns_lock = lock_fd is None
    if owns_lock:
        lock_fd = _prepare_parts(parts, expected)
    children, records = [], [None] * len(ranges)
    attempts = [0] * len(ranges)
    max_attempts = 2
    handed_off = False

    def interrupted(*_):
        raise InterruptedError("range download interrupted")

    previous = {
        number: signal.signal(number, interrupted)
        for number in (signal.SIGTERM, signal.SIGINT)
    }

    try:
        if final.exists():
            checked_published(final, receipt)
            if hold_lock:
                handed_off = True
                return lock_fd, True
            return None
        completed_records = {}
        for spec in expected["ranges"]:
            value = _valid_range_record(parts, spec)
            if value is not False:
                completed_records[spec["index"]] = value

        def start_range(index, start, end):
            """Start one range with a fresh connection and clean evidence files."""
            remaining = until - clock()
            if remaining <= 0:
                raise TimeoutError("range download deadline")
            _quarantine_range(parts, index)
            part, headers, metrics, _ = _range_paths(parts, index)
            opened = []
            try:
                for path in (part, headers, metrics):
                    fd = os.open(
                        path,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                    )
                    opened.append(os.fdopen(fd, "wb"))
                body, head, output = opened
                for stream in opened:
                    os.fchmod(stream.fileno(), 0o600)
                command = [
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
                    f"{remaining:.6f}",
                    "--speed-limit",
                    "1",
                    "--speed-time",
                    "60",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--range",
                    f"{start}-{end}",
                    "--output",
                    f"/proc/self/fd/{body.fileno()}",
                    "--dump-header",
                    f"/proc/self/fd/{head.fileno()}",
                    "--write-out",
                    "%{http_code}\t%{remote_ip}\t%{http_version}\t%{time_connect}\t%{time_starttransfer}\t%{time_total}\t%{size_download}",
                ]
                with defer_download_signals():
                    child = subprocess.Popen(
                        command,
                        stdin=subprocess.PIPE,
                        stdout=output,
                        stderr=subprocess.DEVNULL,
                        pass_fds=(body.fileno(), head.fileno()),
                        start_new_session=True,
                        # Unmask in the new process; parent masks only registration.
                        restore_signals=True,
                        preexec_fn=lambda: signal.pthread_sigmask(
                            signal.SIG_SETMASK, set()
                        ),
                    )
                    children.append(child)
                records[index] = (part, headers, metrics, start, end, child)
                child.stdin.write(("url = " + json.dumps(url) + "\n").encode())
                child.stdin.close()
            finally:
                for stream in opened:
                    if not stream.closed:
                        stream.close()

        pending = set(range(len(ranges))) - set(completed_records)
        for index, (start, end) in enumerate(ranges):
            if index not in completed_records:
                start_range(index, start, end)
        while pending:
            if clock() >= until:
                raise TimeoutError("range download deadline")
            for index in sorted(pending):
                part, headers, metrics, start, end, child = records[index]
                code = child.poll()
                if code is None:
                    continue
                error = None
                if code != 0:
                    # Retry only connection/partial-transfer failures. Do not
                    # repeat HTTP authentication or local-output failures.
                    if code not in {6, 7, 18, 28, 52, 55, 56}:
                        _quarantine_range(parts, index)
                        print(
                            json.dumps(
                                dict(
                                    range_index=index,
                                    attempt=attempts[index] + 1,
                                    failed=True,
                                )
                            ),
                            file=sys.stderr,
                        )
                        raise ValueError("HTTPS range download failed")
                    error = "curl incomplete"
                else:
                    try:
                        report = range_result(headers, metrics, start, end, size)
                        if part.stat().st_size != end - start + 1:
                            raise ValueError("range file size differs")
                    except (OSError, ValueError):
                        error = "range validation"
                if error is not None:
                    _quarantine_range(parts, index)
                    if attempts[index] + 1 >= max_attempts:
                        print(
                            json.dumps(
                                dict(
                                    range_index=index,
                                    attempt=attempts[index] + 1,
                                    failed=True,
                                )
                            ),
                            file=sys.stderr,
                        )
                        if code == 28:
                            raise TimeoutError("HTTPS range download timeout")
                        raise ValueError("HTTPS range download failed")
                    if clock() >= until:
                        raise TimeoutError("range retry deadline")
                    attempts[index] += 1
                    print(
                        json.dumps(
                            dict(
                                range_index=index,
                                attempt=attempts[index],
                                retry=True,
                            )
                        ),
                        file=sys.stderr,
                    )
                    start_range(index, start, end)
                    continue
                completed_records[index] = _publish_range_record(
                    parts,
                    expected["ranges"][index],
                    report,
                )
                print(json.dumps(dict(range_index=index, **report)), file=sys.stderr)
                pending.remove(index)
            if pending:
                time.sleep(min(0.02, max(0, until - clock())))
        part_fds = []
        try:
            for spec in expected["ranges"]:
                fd = _open_regular_fd(
                    _range_paths(parts, spec["index"])[0], 0o600
                )
                part_fds.append(fd)
                actual_size, actual_digest, _ = _digest_fd(fd)
                saved = completed_records[spec["index"]]
                if (actual_size, actual_digest) != (
                    spec["size"],
                    saved["sha256"],
                ):
                    raise ValueError("range part changed during assembly")
                os.lseek(fd, 0, os.SEEK_SET)
            partial_fd = os.open(
                partial,
                os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
                0o600,
            )
            try:
                partial_status = os.fstat(partial_fd)
                if (
                    not stat.S_ISREG(partial_status.st_mode)
                    or partial_status.st_uid != os.geteuid()
                    or stat.S_IMODE(partial_status.st_mode) != 0o600
                ):
                    raise ValueError("partial file identity differs")
                os.ftruncate(partial_fd, 0)
                os.lseek(partial_fd, 0, os.SEEK_SET)
                for fd in part_fds:
                    while chunk := os.read(fd, 1024 * 1024):
                        if clock() >= until:
                            raise TimeoutError("range assembly deadline")
                        view = memoryview(chunk)
                        while view:
                            view = view[os.write(partial_fd, view) :]
                os.fsync(partial_fd)
                if receipt is not None:
                    actual_size, actual_digest, _ = _digest_fd(partial_fd)
                    if (actual_size, actual_digest) != (
                        receipt["size"],
                        receipt["sha256"],
                    ):
                        raise ValueError("assembled range hash differs")
            finally:
                os.close(partial_fd)
        except (OSError, ValueError):
            _quarantine_all_ranges(parts, len(expected["ranges"]))
            raise
        finally:
            for fd in part_fds:
                os.close(fd)
        _sync_directory(partial.parent)
        if hold_lock:
            handed_off = True
            return lock_fd, False
    finally:
        try:
            with defer_download_signals():
                cleanup_started = time.monotonic()
                timer = signal.setitimer(signal.ITIMER_REAL, 0)
                try:
                    stop_downloads(children)
                finally:
                    if timer[0]:
                        signal.setitimer(
                            signal.ITIMER_REAL,
                            max(0.001, timer[0] - (time.monotonic() - cleanup_started)),
                            timer[1],
                        )
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
            if owns_lock and not handed_off:
                _release_parts_lock(lock_fd)


def cleanup_parts(final, receipt):
    receipt = validate_receipt(receipt)
    checked_published(final, receipt)
    parts = safe_path(str(final) + ".b2-parts")
    try:
        parts.lstat()
    except FileNotFoundError:
        return
    status = parts.lstat()
    if (
        not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.geteuid()
        or stat.S_IMODE(status.st_mode) != 0o700
    ):
        raise ValueError("range directory ownership differs")
    lock_fd = _parts_lock(parts)
    try:
        manifest = _read_json(parts / "manifest.json")
        ranges = manifest.get("ranges")
        layout = [(item["start"], item["end"]) for item in ranges]
        expected = _range_manifest(
            receipt, receipt["size"], manifest.get("workers"), layout
        )
        if manifest != expected:
            raise ValueError("durable range identity differs")
        allowed = {"manifest.json", "receiver.lock", "quarantine"}
        for index in range(len(ranges)):
            allowed.update(path.name for path in _range_paths(parts, index))
        for path in parts.iterdir():
            if path.name not in allowed:
                raise ValueError("unexpected range evidence file")
            if path.name == "quarantine":
                status = path.lstat()
                if (
                    not stat.S_ISDIR(status.st_mode)
                    or status.st_uid != os.geteuid()
                    or stat.S_IMODE(status.st_mode) != 0o700
                ):
                    raise ValueError("range quarantine identity differs")
                for child in path.iterdir():
                    _owned_regular(child)
                    child.unlink()
                path.rmdir()
            elif path.name != "receiver.lock":
                _owned_regular(path)
                path.unlink()
        (parts / "receiver.lock").unlink()
        parts.rmdir()
        _sync_directory(parts.parent)
    finally:
        _release_parts_lock(lock_fd)


def receive(
    value,
    source,
    partial,
    final,
    run_curl=subprocess.run,
    *,
    download_workers=1,
    download_threshold_bytes=64 * 1024 * 1024,
):
    download_settings(download_workers, download_threshold_bytes)
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
    parallel = (
        download_workers > 1
        and receipt["size"] >= download_threshold_bytes
    )
    parts = safe_path(str(final) + ".b2-parts")
    expected = _range_manifest(
        receipt,
        receipt["size"],
        download_workers,
        download_ranges(receipt["size"], download_workers),
    )
    publication_lock = None
    try:
        final_ready = final.exists()
        if not final_ready:
            publication_lock = _prepare_parts(parts, expected)
            # A concurrent receiver may have published while this receiver
            # waited for the lock. Re-check before touching the partial file.
            final_ready = final.exists()
            if not final_ready and parallel:
                receive_ranges(
                    url,
                    receipt["size"],
                    partial,
                    final,
                    download_workers,
                    receipt=receipt,
                    lock_fd=publication_lock,
                )
                final_ready = final.exists()
            if not final_ready:
                fd = os.open(
                    partial,
                    os.O_WRONLY
                    | os.O_NOFOLLOW
                    | (0 if parallel else os.O_CREAT | os.O_TRUNC),
                    0o600,
                )
                try:
                    os.fchmod(fd, 0o600)
                    result = (
                        None
                        if parallel
                        else run_curl(
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
                    )
                finally:
                    os.close(fd)
                if result is not None and result.returncode == 28:
                    raise TimeoutError("HTTPS download timeout")
                if result is not None and (
                    result.returncode != 0 or result.stdout != "200"
                ):
                    raise ValueError("HTTPS download failed")
                _publish_partial(partial, final, receipt)
        checked_published(final, receipt)
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
    finally:
        if publication_lock is not None:
            _release_parts_lock(publication_lock)


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
    receiver.add_argument("--download-workers", type=int, default=1)
    receiver.add_argument(
        "--download-threshold-bytes", type=int, default=64 * 1024 * 1024
    )
    cleaner = sub.add_parser("cleanup-parts")
    cleaner.add_argument("--receipt", required=True)
    cleaner.add_argument("--final", required=True)
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
                    download_workers=args.download_workers,
                    download_threshold_bytes=args.download_threshold_bytes,
                )
            elif args.command == "cleanup-parts":
                cleanup_parts(
                    args.final, parse_json(safe_path(args.receipt).read_text())
                )
            elif args.command == "ack-json":
                receipt = validate_receipt(
                    parse_json(safe_path(args.receipt).read_text())
                )
                checked_published(args.final, receipt)
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
    except (TimeoutError, subprocess.TimeoutExpired) as error:
        # Timeout is a bounded transport failure. The shell caller owns the
        # finite retry budget; do not print exception text because it may
        # contain a presigned URL or credential detail.
        print(
            json.dumps({"b2_transient_failure": True, "error": type(error).__name__}),
            file=sys.stderr,
        )
        return 75
    except Exception as error:
        # SDK/curl error strings can contain URLs or credential identifiers.
        print(f"B2 rollout operation failed ({type(error).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
