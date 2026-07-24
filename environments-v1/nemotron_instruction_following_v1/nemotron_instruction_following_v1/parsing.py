"""Deterministic parsing and scoring for the non-judge instruction-following subtasks.

Ported verbatim from the v0 env (`nemotron_instruction_following.py`): JSON extraction,
structured-payload parsing/validation (json/yaml/toml/csv/xml), the format-verification
grader (regex / inline_prose / string_match), the native tool-call helpers, and the
deterministic instruction scorer used by the ultra-judge subtasks. These are pure functions
with no audit findings against them, so they parse and grade exactly as v0 did.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from typing import Any

import jsonschema
import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py310 fallback
    import tomli as tomllib

from nemotron_instruction_following_v1.checkers import CHECKERS, _evaluate_constraints


def coerce(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            try:
                return ast.literal_eval(value)
            except Exception:
                return value
    return value


# --- Native tool-call helpers ------------------------------------------------


def _object_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_args(args: Any) -> Any:
    args = coerce(args)
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            return {"raw": args}
    return args


def _tool_call_signature(tool_call: Any) -> tuple[str, Any]:
    tool_call = coerce(tool_call)
    if hasattr(tool_call, "model_dump"):
        try:
            tool_call = tool_call.model_dump()
        except Exception:
            pass
    if not isinstance(tool_call, dict):
        name = _object_get(tool_call, "name", "")
        args = _object_get(tool_call, "arguments", {})
        return str(name or ""), _normalize_args(args)
    function = tool_call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        args = function.get("arguments", {})
    else:
        name = tool_call.get("name")
        args = tool_call.get("arguments", {})
    return str(name or ""), _normalize_args(args)


def completion_tool_calls(assistant_messages: list[Any]) -> list[dict[str, Any]]:
    """Every assistant tool call as ``{name, arguments}`` (arguments JSON-decoded).

    v0 read these off the raw completion; v1 reads the trace's ``assistant_messages``
    (each carrying ``.tool_calls`` with a raw-JSON ``arguments`` string).
    """
    calls: list[dict[str, Any]] = []
    for message in assistant_messages:
        for tool_call in getattr(message, "tool_calls", None) or []:
            name, arguments = _tool_call_signature(tool_call)
            if name:
                calls.append({"name": name, "arguments": arguments})
    return calls


def tool_payload_candidates(calls: list[dict[str, Any]], tool_name: str, payload_key: str) -> list[Any]:
    expected_tool_name = str(tool_name or "").strip()
    payload_key = str(payload_key or "").strip()
    if payload_key.lower() in {"", "none", "null"}:
        payload_key = ""

    candidates: list[Any] = []
    for call in calls:
        if expected_tool_name and call.get("name") != expected_tool_name:
            continue
        arguments = _normalize_args(call.get("arguments"))
        if payload_key and isinstance(arguments, dict) and payload_key in arguments:
            candidates.append(arguments[payload_key])
        else:
            candidates.append(arguments)
    return candidates


# --- JSON extraction ---------------------------------------------------------

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
JSON_OBJECT_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def extract_json(text: str) -> Any | None:
    if not text:
        return None
    m = JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = JSON_OBJECT_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    try:
        return json.loads(text.strip())
    except Exception:
        return None


# --- Structured-outputs scoring ---------------------------------------------


def schema_from_str(schema_str: str) -> dict[str, Any]:
    try:
        schema = json.loads(schema_str)
    except Exception:
        return {}
    if isinstance(schema, dict) and isinstance(schema.get("schema"), dict):
        return schema["schema"]
    return schema if isinstance(schema, dict) else {}


def _extract_structured_text(text: str) -> str:
    text = (text or "").strip()
    fence = re.search(r"```(?:json|yaml|yml|xml|toml|csv)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return fence.group(1).strip() if fence else text


def parse_structured_payload(text: str, schema_type: str) -> Any | None:
    content = _extract_structured_text(text)
    kind = (schema_type or "json").strip().lower()
    try:
        if kind == "json":
            return json.loads(content)
        if kind in {"yaml", "yml"}:
            return yaml.safe_load(content)
        if kind == "toml":
            return tomllib.loads(content)
        if kind == "csv":
            return list(csv.DictReader(io.StringIO(content)))
        if kind == "xml":
            ET.fromstring(content)
            return {"_xml_well_formed": True}
    except Exception:
        return None
    return None


def structured_payload_valid(payload: Any, schema_str: str, schema_type: str) -> float:
    if payload is None:
        return 0.0
    # XML rows carry no XSD, so well-formedness (checked by the parser above) is the only
    # thing we can assert. Every other type is graded against its schema.
    if schema_type == "xml":
        return 1.0
    schema = schema_from_str(schema_str)
    if not schema:
        return 0.0
    try:
        jsonschema.validate(payload, schema)
        return 1.0
    except Exception:
        return 0.0


# --- Format verification -----------------------------------------------------


def format_verification_score(text: str, verifier: dict[str, Any]) -> float:
    verifier_type = str(verifier.get("type", ""))
    if verifier_type in {"regex", "inline_prose"}:
        patterns = verifier.get("verify_regex") or []
        min_matches = int(verifier.get("verify_min_matches", 1) or 1)
        compiled = [re.compile(str(pattern)) for pattern in patterns]
        matching_lines = 0
        for line in text.splitlines():
            if any(rx.search(line) for rx in compiled):
                matching_lines += 1
        return 1.0 if matching_lines >= min_matches else 0.0
    if verifier_type == "string_match":
        expected = [str(x) for x in (verifier.get("expected_markers") or [])]
        patterns = [str(x) for x in (verifier.get("patterns") or [])]
        missing = [marker for marker in expected if marker not in text]
        spurious: list[str] = []
        expected_set = set(expected)
        for pattern in patterns:
            for found in re.finditer(pattern, text):
                if found.group(0) not in expected_set:
                    spurious.append(found.group(0))
        return 1.0 if not missing and not spurious else 0.0
    return 0.0


# --- Deterministic instruction scoring (ultra-judge subtasks) ----------------


def deterministic_instruction_score(text: str, instructions: list[dict[str, Any]]) -> float | None:
    ids: list[str] = []
    kwargs_list: list[dict[str, Any]] = []
    for item in instructions:
        if not isinstance(item, dict):
            continue
        instruction_id = str(item.get("instruction_id", "")).strip()
        if instruction_id not in CHECKERS:
            continue
        ids.append(instruction_id)
        kwargs_list.append({k: v for k, v in item.items() if k not in {"instruction_id", "source", "uid"}})
    if not ids:
        return None
    return float(_evaluate_constraints(text, ids, kwargs_list)["pass_fraction"])


def ifeval_pass_fraction(text: str, instruction_ids: list[str], kwargs_list: list[dict[str, Any]]) -> float:
    return float(_evaluate_constraints(text, instruction_ids, kwargs_list)["pass_fraction"])


def ifeval_fully_passed(text: str, instruction_ids: list[str], kwargs_list: list[dict[str, Any]]) -> float:
    return float(_evaluate_constraints(text, instruction_ids, kwargs_list)["fully_passed"])
