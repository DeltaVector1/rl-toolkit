from __future__ import annotations

import os
import re


MODEL_REPO_ID_ENV = "PRIME_LORA_BASE_MODEL_REPO_ID"
MODEL_REVISION_ENV = "PRIME_LORA_BASE_MODEL_REVISION"
MODEL_VARIANT_ENV = "PRIME_LORA_BASE_MODEL_VARIANT"
MODEL_CONTENT_SHA256_ENV = "PRIME_LORA_BASE_MODEL_CONTENT_SHA256"
MODEL_MANIFEST_SHA256_ENV = "PRIME_LORA_BASE_MODEL_MANIFEST_SHA256"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_VARIANT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class ModelIdentityError(ValueError):
    pass


def expected_model_identity() -> dict[str, str]:
    values = {
        "repo_id": os.environ.get(MODEL_REPO_ID_ENV, ""),
        "revision": os.environ.get(MODEL_REVISION_ENV, ""),
        "variant": os.environ.get(MODEL_VARIANT_ENV, ""),
        "content_sha256": os.environ.get(MODEL_CONTENT_SHA256_ENV, ""),
        "manifest_sha256": os.environ.get(MODEL_MANIFEST_SHA256_ENV, ""),
    }
    return validate_model_identity(values)


def validate_model_identity(values: dict[str, str]) -> dict[str, str]:
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ModelIdentityError(
            "canonical model identity is incomplete: " + ", ".join(missing)
        )
    if _REPO_ID.fullmatch(values["repo_id"]) is None:
        raise ModelIdentityError("canonical model repository ID is invalid")
    if _REVISION.fullmatch(values["revision"]) is None:
        raise ModelIdentityError("canonical model revision is invalid")
    if _VARIANT.fullmatch(values["variant"]) is None:
        raise ModelIdentityError("canonical model variant is invalid")
    for name in ("content_sha256", "manifest_sha256"):
        if _SHA256.fullmatch(values[name]) is None:
            raise ModelIdentityError(f"canonical model {name} is invalid")
    return values
