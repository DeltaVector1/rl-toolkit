"""Tool-call extraction and structural scoring.

Copied verbatim from the v0 env (`rl_environments/nemotron-tool-calling/nemotron_tool_calling.py`)
so the ported rewards extract predicted calls and score them exactly as v0 did. These are pure
functions with no audit findings against them.

The only adaptation from v0: `predicted_calls` reads the model's tool calls off v1
`AssistantMessage` objects (`content` + `tool_calls`) instead of a raw dict/list completion,
reproducing v0's `_completion_function_calls` semantics — native `tool_calls` are authoritative,
and a turn with none is scraped from its visible prose.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from nemotron_guard_v1.text import strip_think_tags

TOOL_CALL_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
ARRAY_JSON_RE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
FUNCTION_BLOCK_RE = re.compile(r"<function[^>]*>(.*?)</function>", re.DOTALL | re.IGNORECASE)
TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call[^>]*>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
VLLM_TOOL_CALL_RE = re.compile(r"<\|tool_call\|>\s*(.*?)\s*<\|/?tool_call\|>", re.DOTALL | re.IGNORECASE)
# Anchored to the start of a line and requiring the argument object, because a
# compact call is always emitted as a standalone structured line. Matching loosely
# meant ordinary prose ("I will call:search_web to look this up") registered as a
# tool call, and a stray phantom call makes the exactly-one-call check fail a
# response that was in fact correct.
COMPACT_CALL_RE = re.compile(
    r"^[ \t]*_?call(?::[A-Za-z0-9_.-]+)*:([A-Za-z_][A-Za-z0-9_.-]*)[ \t]*(\{[^\n`]*\})",
    re.IGNORECASE | re.MULTILINE,
)


def _parse_loose_args(args_text: str) -> dict[str, Any]:
    args_text = (args_text or "").strip()
    if not args_text:
        return {}
    for candidate in (
        args_text,
        re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)", r"\1'\2'\3", args_text),
    ):
        try:
            parsed = json.loads(candidate)
        except Exception:
            try:
                parsed = ast.literal_eval(candidate)
            except Exception:
                continue
        return parsed if isinstance(parsed, dict) else {"raw": args_text}
    return {"raw": args_text}


def _object_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _coerce_call_obj(obj: Any) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            dumped = obj.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    name = _object_get(obj, "name")
    function = _object_get(obj, "function")
    arguments = _object_get(obj, "arguments")
    if name is not None or function is not None or arguments is not None:
        out: dict[str, Any] = {}
        if name is not None:
            out["name"] = name
        if function is not None:
            out["function"] = function
        if arguments is not None:
            out["arguments"] = arguments
        return out
    return None


def _append_call_from_obj(calls: list[dict[str, Any]], obj: Any) -> None:
    if isinstance(obj, list):
        for item in obj:
            _append_call_from_obj(calls, item)
        return
    obj = _coerce_call_obj(obj)
    if obj is None:
        return
    if isinstance(obj.get("tool_calls"), list):
        _append_call_from_obj(calls, obj["tool_calls"])
    function = obj.get("function") if isinstance(obj.get("function"), dict) else {}
    name = obj.get("name") or obj.get("tool_name") or obj.get("tool") or function.get("name")
    if not name and obj.get("type") not in {"function_call", "tool_call", "function"}:
        return
    args = (
        obj.get("arguments")
        if obj.get("arguments") is not None
        else obj.get("args")
        if obj.get("args") is not None
        else obj.get("parameters")
        if obj.get("parameters") is not None
        else function.get("arguments")
    )
    calls.append({"name": str(name or ""), "arguments": _normalize_args(args)})


def _extract_function_calls(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    calls: list[dict[str, Any]] = []
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            _append_call_from_obj(calls, json.loads(stripped))
        except Exception:
            pass
    for m in ARRAY_JSON_RE.finditer(text):
        try:
            arr = json.loads(m.group(1))
        except Exception:
            continue
        _append_call_from_obj(calls, arr)
    for m in TOOL_CALL_JSON_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except Exception:
            continue
        _append_call_from_obj(calls, obj)
    for m in FUNCTION_BLOCK_RE.finditer(text):
        inner = m.group(0)
        name_m = re.search(r'name="([^"]+)"', inner)
        if name_m:
            args_text = m.group(1).strip()
            calls.append({"name": name_m.group(1), "arguments": _parse_loose_args(args_text)})
    for block_re in (TOOL_CALL_BLOCK_RE, VLLM_TOOL_CALL_RE):
        for m in block_re.finditer(text):
            inner = m.group(1).strip()
            for loader in (json.loads, ast.literal_eval):
                try:
                    _append_call_from_obj(calls, loader(inner))
                    break
                except Exception:
                    pass
            else:
                calls.extend(_extract_function_calls(inner))
    for m in COMPACT_CALL_RE.finditer(text):
        args_text = (m.group(2) or "").strip()
        args = _parse_loose_args(args_text)
        calls.append({"name": m.group(1), "arguments": args})
    return calls


def predicted_calls(assistant_messages: list[Any]) -> list[dict[str, Any]]:
    """Predicted calls off v1 `AssistantMessage`s — v0's `_completion_function_calls`.

    Native `tool_calls` are authoritative (also scraping the same message's prose would
    double-count the action and trip the exactly-one-call check); a turn with no native
    call is scraped from its visible content. The content is think-stripped first, matching
    v0 (its `GuardedRubric` scored the think-stripped completion).
    """
    calls: list[dict[str, Any]] = []
    for message in assistant_messages:
        native = getattr(message, "tool_calls", None) or []
        if native:
            for call in native:
                _append_call_from_obj(calls, {"name": call.name, "arguments": call.arguments})
            continue
        calls.extend(_extract_function_calls(strip_think_tags(getattr(message, "content", None) or "")))
    return calls


def _call_names(calls: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for call in calls:
        name, _args = _call_signature(call)
        if name:
            names.append(name)
    return names


def _normalize_args(args: Any) -> dict[str, Any]:
    if isinstance(args, str):
        return _parse_loose_args(args)
    return args if isinstance(args, dict) else {}


def _args_overlap(expected: dict[str, Any], predicted: dict[str, Any]) -> float:
    if not expected:
        return 1.0 if not predicted else 0.5
    matched = 0
    for k, v in expected.items():
        if k in predicted:
            if str(predicted[k]).strip() == str(v).strip():
                matched += 1
            elif (
                str(v).strip()
                and str(predicted[k]).strip()
                and (str(v).strip() in str(predicted[k]).strip() or str(predicted[k]).strip() in str(v).strip())
            ):
                matched += 0.5
    return matched / max(1, len(expected))


def _strict_arg_match(expected: Any, predicted: Any) -> bool:
    if not isinstance(predicted, type(expected)):
        return False
    if isinstance(expected, dict):
        if set(expected.keys()) != set(predicted.keys()):
            return False
        return all(_strict_arg_match(value, predicted[key]) for key, value in expected.items())
    if isinstance(expected, list):
        if len(expected) != len(predicted):
            return False
        return all(_strict_arg_match(exp, pred) for exp, pred in zip(expected, predicted))
    if isinstance(expected, float):
        return abs(predicted - expected) < 1e-6
    if isinstance(expected, str):
        return expected == predicted
    return expected == predicted


def _call_signature(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    call = _coerce_call_obj(call) or {}
    name = call.get("name") or (call.get("function") or {}).get("name") or ""
    args = call.get("arguments") if call.get("arguments") is not None else (call.get("function") or {}).get("arguments")
    return str(name), _normalize_args(args)


def structural_call_score(predicted: list[dict[str, Any]], expected: dict[str, Any]) -> float:
    name = expected.get("name") or (expected.get("function") or {}).get("name")
    if not name or len(predicted) != 1:
        return 0.0
    exp_args = _normalize_args(expected.get("arguments") or (expected.get("function") or {}).get("arguments"))
    call_name, call_args = _call_signature(predicted[0])
    if call_name != name:
        return 0.0
    return 1.0 if _strict_arg_match(exp_args, call_args) else 0.0


def sequence_score(expected_calls: list[dict[str, Any]], predicted: list[dict[str, Any]]) -> float:
    if not expected_calls:
        return 1.0 if not predicted else 0.5
    total = 0.0
    used: set[int] = set()
    for exp in expected_calls:
        exp_name, exp_args = _call_signature(exp)
        best = 0.0
        best_idx: int | None = None
        for i, pred in enumerate(predicted):
            if i in used:
                continue
            pred_name, pred_args = _call_signature(pred)
            if pred_name != exp_name:
                continue
            score = 0.5 + 0.5 * _args_overlap(exp_args, pred_args)
            if score > best:
                best = score
                best_idx = i
        if best_idx is not None:
            used.add(best_idx)
        total += best
    return total / max(1, len(expected_calls))


def _name_similarity(predicted: str, expected: str) -> float:
    if not predicted or not expected:
        return 0.0
    if predicted == expected:
        return 1.0
    pred_parts = predicted.split("_")
    exp_parts = expected.split("_")
    common = 0
    for pred_part, exp_part in zip(pred_parts, exp_parts):
        if pred_part != exp_part:
            break
        common += 1
    if common:
        return max(0.25, 0.7 * common / max(len(pred_parts), len(exp_parts)))
    return 0.15 if pred_parts[0] == exp_parts[0] else 0.0


def _expected_calls_from_info(info: dict[str, Any] | None) -> list[dict[str, Any]]:
    info = info or {}
    for key in ("ground_truth_json", "expected_action_json"):
        try:
            parsed = json.loads(info.get(key, "[]"))
        except Exception:
            parsed = []
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    try:
        required_tools = json.loads(info.get("required_tools_json", "[]"))
    except Exception:
        required_tools = []
    if isinstance(required_tools, list) and required_tools:
        return [{"name": str(name)} for name in required_tools if name]
    target_tool = str(info.get("target_tool", "")).strip()
    return [{"name": target_tool}] if target_tool else []


def _valid_tool_names_from_info(info: dict[str, Any] | None) -> set[str]:
    info = info or {}
    try:
        tool_defs = json.loads(info.get("tool_defs_json", "[]"))
    except Exception:
        tool_defs = []
    if not isinstance(tool_defs, list):
        return set()
    return {str(tool.get("name", "")).strip() for tool in tool_defs if isinstance(tool, dict) and tool.get("name")}


def tool_call_shape_score(predicted: list[dict[str, Any]], info: dict[str, Any]) -> float:
    if not predicted:
        return 0.0
    expected = _expected_calls_from_info(info)
    valid_names = _valid_tool_names_from_info(info)
    expected_names = [name for call in expected if (name := _call_signature(call)[0])]
    expected_arg_sets = [_call_signature(call)[1] for call in expected]

    scores: list[float] = []
    for call in predicted:
        name, args = _call_signature(call)
        valid_score = 1.0 if not valid_names or name in valid_names else 0.0
        if expected_names:
            relevance = max(_name_similarity(name, expected_name) for expected_name in expected_names)
        else:
            relevance = 0.5 if name else 0.0
        if expected_arg_sets:
            arg_overlap = max(_args_overlap(exp_args, args) for exp_args in expected_arg_sets)
            expected_arg_count = max(1, max(len(exp_args) for exp_args in expected_arg_sets))
        else:
            arg_overlap = 0.5 if args else 0.0
            expected_arg_count = 3
        arg_count_score = min(1.0, len(args) / expected_arg_count) if args else 0.0
        scores.append(0.35 * valid_score + 0.25 * arg_count_score + 0.25 * relevance + 0.15 * arg_overlap)
    call_count_score = min(1.0, len(predicted) / max(1, len(expected))) if expected else 1.0
    return float(0.9 * max(scores) + 0.1 * call_count_score)
