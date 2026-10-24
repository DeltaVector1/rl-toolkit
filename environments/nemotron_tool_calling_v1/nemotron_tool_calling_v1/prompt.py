"""Prompt and tool-definition construction for the pivot transcripts.

Ported from v0 (`nemotron_tool_calling.py` + its shared guardrail): rebuild the OpenAI
Responses-style transcript in `responses_create_params.input` as native chat messages, and
advertise the source tool schemas.

Tools are advertised in-prompt (v0's `enable_native_tool_calls=False` path): the null chat
harness this env runs under does not stand up an MCP tool server, so the tool defs are
formatted into the system prompt and the model emits a visible ```json call, which the reward
scrapes. Grading is identical either way — `calls.predicted_calls` reads native trace
`tool_calls` and scraped prose the same way v0's `_completion_function_calls` did.
"""

from __future__ import annotations

import ast
import json
from typing import Any


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


def _response_content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text is None:
                    text = item.get("content")
                if text is None:
                    text = item.get("summary")
                if text is not None:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        for key in ("text", "content", "summary", "output"):
            if content.get(key) is not None:
                return _response_content_text(content[key])
    return str(content)


def _reasoning_summary_text(item: dict[str, Any]) -> str:
    summary = item.get("summary")
    if isinstance(summary, list):
        return "\n".join(_response_content_text(part) for part in summary if _response_content_text(part)).strip()
    return _response_content_text(summary)


def input_to_prompt(payload: Any) -> list[dict[str, Any]]:
    payload = coerce(payload)
    items = payload.get("input") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []
    msgs: list[dict[str, Any]] = []
    pending_reasoning = ""
    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip().lower()
        role = str(item.get("role") or "").strip().lower()

        if item_type == "reasoning":
            pending_reasoning = _reasoning_summary_text(item)
            continue

        if item_type == "function_call":
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            arguments = item.get("arguments")
            tool_call_id = str(item.get("call_id") or item.get("id") or f"call_{len(msgs)}")
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "name": name,
                        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {}),
                    }
                ],
            }
            if pending_reasoning:
                msg["reasoning_content"] = pending_reasoning
                pending_reasoning = ""
            msgs.append(msg)
            continue

        if item_type == "function_call_output":
            output = _response_content_text(item.get("output"))
            call_id = str(item.get("call_id") or f"call_{len(msgs)}")
            msgs.append({"role": "tool", "tool_call_id": call_id, "content": output})
            continue

        if role in {"system", "user"}:
            content = _response_content_text(item.get("content"))
            if content.strip():
                msgs.append({"role": role, "content": content})
            continue

        if role == "assistant":
            content = _response_content_text(item.get("content"))
            msg = {"role": "assistant", "content": content}
            if pending_reasoning:
                msg["reasoning_content"] = pending_reasoning
                pending_reasoning = ""
            if content.strip() or msg.get("reasoning_content"):
                msgs.append(msg)

    return msgs


def normalize_tool_def(raw_tool: Any) -> dict[str, Any] | None:
    raw_tool = coerce(raw_tool)
    if not isinstance(raw_tool, dict):
        return None
    spec = raw_tool.get("function") if isinstance(raw_tool.get("function"), dict) else raw_tool
    name = str(spec.get("name") or "").strip()
    if not name:
        return None
    parameters = coerce(spec.get("parameters"))
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}
    tool: dict[str, Any] = {
        "name": name,
        "description": str(spec.get("description") or ""),
        "parameters": parameters,
    }
    strict = spec.get("strict", raw_tool.get("strict"))
    if strict is not None:
        tool["strict"] = bool(strict)
    return tool


def extract_tool_defs(payload: Any) -> list[dict[str, Any]]:
    payload = coerce(payload)
    tools = payload.get("tools") if isinstance(payload, dict) else None
    if not isinstance(tools, list):
        return []
    return [tool for tool in (normalize_tool_def(raw) for raw in tools) if tool is not None]


def format_tool_defs_for_prompt(tool_defs: list[dict[str, Any]]) -> str:
    if not tool_defs:
        return ""
    lines = [
        "Available external actions are listed below. Use these exact action names and argument fields.",
        "When an external action is needed, output only a visible JSON object or JSON array in a ```json code block:",
        "```json",
        '{"name":"action_name","arguments":{...}}',
        "```",
        "<available_actions>",
    ]
    for tool in tool_defs:
        params = json.dumps(tool.get("parameters") or {}, ensure_ascii=False, sort_keys=True)
        desc = str(tool.get("description") or "").strip()
        if desc:
            lines.append(f"- {tool['name']}: {desc}")
        else:
            lines.append(f"- {tool['name']}")
        lines.append(f"  parameters: {params}")
    lines.append("</available_actions>")
    return "\n".join(lines)


def _compose_system(existing: str, addition: str) -> str:
    if not addition:
        return existing
    if not existing:
        return addition
    if addition in existing:
        return existing
    return f"{addition.strip()}\n\n{existing.rstrip()}"


def merge_system_prompt(prompt: list[dict[str, Any]], system_prompt: str | None) -> list[dict[str, Any]]:
    """Merge a system prompt into a dict-based chat prompt (v0's `merge_system_prompt`)."""
    text = (system_prompt or "").strip()
    if not text:
        return prompt
    messages = [dict(message) for message in prompt]
    for message in messages:
        if str(message.get("role", "")).lower() == "system":
            message["content"] = _compose_system(str(message.get("content") or "").strip(), text)
            break
    else:
        messages = [{"role": "system", "content": text}, *messages]
    return messages


def prepare_prompt(
    payload: Any, system_prompt: str | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rebuild the chat prompt and advertise tool defs in-prompt; return (prompt, tool_defs)."""
    prompt = input_to_prompt(payload)
    tool_defs = extract_tool_defs(payload)
    prompt = merge_system_prompt(prompt, format_tool_defs_for_prompt(tool_defs))
    prompt = merge_system_prompt(prompt, system_prompt)
    return prompt, tool_defs
