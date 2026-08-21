#!/usr/bin/env python3
"""Test Muse Glimmer ATEM tool and reasoning parser registration."""

from renderers.base import (
    MuseGlimmerReasoningParser,
    MuseGlimmerToolParser,
    ToolCallParseStatus,
    _populate_registry,
)


class CharacterTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


def encode(text: str) -> list[int]:
    return [ord(character) for character in text]


def main() -> None:
    tokenizer = CharacterTokenizer()

    response = (
        " to=self<|message|>reasoning trace<|eom|>"
        "<|start|>assistant to=user<|message|>public answer<|eot|>"
    )
    reasoning, content = MuseGlimmerReasoningParser(tokenizer).extract(response)
    assert reasoning == "reasoning trace"
    assert content == "public answer"

    tool_response = (
        " to=self<|message|>ignore "
        '<atem:invoke name="ignored"></atem:invoke><|eom|>'
        "<|start|>assistant to=tools.search<|message|>"
        "<atem:function_calls>"
        '<atem:invoke name="tools.search">'
        '<atem:parameter name="limit">2</atem:parameter>'
        '<atem:parameter name="query">muse</atem:parameter>'
        "</atem:invoke>"
        "</atem:function_calls><|eom|>"
    )
    remaining_ids, calls = MuseGlimmerToolParser(tokenizer).extract(
        encode(tool_response)
    )
    assert remaining_ids == encode(tool_response)
    assert len(calls) == 1
    assert calls[0].status == ToolCallParseStatus.OK
    assert calls[0].name == "tools.search"
    assert calls[0].arguments == {"limit": 2, "query": "muse"}

    _populate_registry()
    from renderers.parsers import REASONING_PARSERS, TOOL_PARSERS

    assert TOOL_PARSERS["muse_glimmer"] is MuseGlimmerToolParser
    assert REASONING_PARSERS["muse_glimmer"] is MuseGlimmerReasoningParser
    print("Muse Glimmer renderer parser test passed")


if __name__ == "__main__":
    main()
