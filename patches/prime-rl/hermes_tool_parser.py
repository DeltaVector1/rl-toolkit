"""Text-level Hermes tool-call parser, registered as ``"hermes"``.

The parsers shipped in ``renderers`` locate ``<tool_call>`` by token id, which works only
when the tokenizer carries it as a single special token. Arcee's Trinity family does not —
``<tool_call>`` encodes as four ordinary tokens — so every id-based parser returns "no tool
calls" for every completion, no matter what the model emitted. Trinity's own model card
serves it with vLLM's ``--tool-call-parser hermes``, which is text-based; this is the
renderer-side equivalent so the orchestrator sees the same calls the inference engine does.

Registering from here rather than editing ``deps/renderers`` (a submodule) keeps the change
in this repo. ``DefaultRendererConfig.tool_parser`` is a free-form string resolved against
the registry at renderer-construction time, so importing this module before
``create_renderer`` is all that is required.
"""

from __future__ import annotations

import json
import re
from typing import Any

from renderers.base import ParsedToolCall, ToolCallParseStatus
from renderers.parsers import TOOL_PARSERS

#: Tolerates a missing closing tag so a completion truncated mid-call is still reported
#: (as UNCLOSED_BLOCK) instead of silently vanishing.
_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*(</tool_call>|$)", re.DOTALL)


def _token_index_for_char(tokenizer, ids: list[int], char_offset: int) -> int:
    """Smallest token count whose decoded text reaches `char_offset`.

    Decoded length is monotonic in the token count, so this binary-searches instead of
    decoding every prefix — cheap enough to keep real `token_span`s, which the trainer uses
    to mask malformed calls out of the loss.
    """
    low, high = 0, len(ids)
    while low < high:
        mid = (low + high) // 2
        if len(tokenizer.decode(ids[:mid], skip_special_tokens=False)) < char_offset:
            low = mid + 1
        else:
            high = mid
    return low


class HermesToolParser:
    """Hermes/ChatML JSON tool calls: ``<tool_call>{"name": .., "arguments": {..}}</tool_call>``."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def extract(self, ids: list[int]) -> tuple[list[int], list[ParsedToolCall]]:
        text = self._tokenizer.decode(ids, skip_special_tokens=False)
        if "<tool_call>" not in text:
            return ids, []

        tool_calls: list[ParsedToolCall] = []
        for match in _BLOCK_RE.finditer(text):
            raw = match.group(1).strip()
            span = (
                _token_index_for_char(self._tokenizer, ids, match.start()),
                _token_index_for_char(self._tokenizer, ids, match.end()),
            )
            if not match.group(2):
                tool_calls.append(
                    ParsedToolCall(raw=raw, token_span=span, status=ToolCallParseStatus.UNCLOSED_BLOCK)
                )
                continue
            tool_calls.append(_parse_block(raw, span))

        if not tool_calls:
            return ids, []
        content_ids = ids[: _token_index_for_char(self._tokenizer, ids, text.index("<tool_call>"))]
        return content_ids, tool_calls


def _parse_block(raw: str, span: tuple[int, int]) -> ParsedToolCall:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ParsedToolCall(raw=raw, token_span=span, status=ToolCallParseStatus.INVALID_JSON)
    if not isinstance(payload, dict) or not payload.get("name"):
        return ParsedToolCall(raw=raw, token_span=span, status=ToolCallParseStatus.MALFORMED_STRUCTURE)

    arguments: Any = payload.get("arguments", {})
    if isinstance(arguments, str):
        # Some emissions double-encode the arguments object; a string that isn't JSON is
        # left as-is for the tool to reject rather than being guessed at here.
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    return ParsedToolCall(raw=raw, name=str(payload["name"]), arguments=arguments, token_span=span)


TOOL_PARSERS.setdefault("hermes", HermesToolParser)
