import json
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RENDERERS = ROOT / "prime-rl/deps/renderers"
sys.path.insert(0, str(RENDERERS))

# Keep this focused test runnable in the repository's lightweight test image.
# The renderer client only needs these names at import time.
openai_stub = types.ModuleType("openai")
openai_stub.APIConnectionError = type("APIConnectionError", (Exception,), {})
openai_stub.AsyncOpenAI = type("AsyncOpenAI", (), {})
sys.modules.setdefault("openai", openai_stub)

from renderers.client import generate  # noqa: E402


class _Renderer:
    supports_tools = True

    def get_stop_token_ids(self):
        return [2]

    def parse_response(self, completion_ids, *, tools=None):
        return types.SimpleNamespace(
            content="",
            reasoning_content="",
            tool_calls=[],
        )


class _Response:
    def __init__(self):
        self.content = json.dumps(
            {
                "choices": [
                    {
                        "token_ids": [9],
                        "logprobs": {
                            "content": [
                                {"token": "token_id:9", "logprob": -0.25}
                            ]
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        ).encode()


class _Client:
    base_url = "http://engine:8000/v1"

    def __init__(self):
        self.calls = []

    async def post(self, endpoint, **kwargs):
        self.calls.append((endpoint, kwargs))
        return _Response()


class RendererCacheSaltTest(unittest.IsolatedAsyncioTestCase):
    async def test_nested_salt_is_promoted_and_ordinary_sampling_fields_remain(self):
        client = _Client()
        await generate(
            client=client,
            renderer=_Renderer(),
            messages=[],
            model="policy-0",
            prompt_ids=[101, 102],
            sampling_params={"cache_salt": "nested", "top_k": 32, "max_tokens": 1},
            max_prompt_len=10,
        )
        body = client.calls[0][1]["body"]
        self.assertEqual(body["cache_salt"], "nested")
        self.assertNotIn("cache_salt", body["sampling_params"])
        self.assertEqual(body["sampling_params"]["top_k"], 32)
        self.assertEqual(body["sampling_params"]["max_tokens"], 1)

    async def test_explicit_salt_wins(self):
        client = _Client()
        await generate(
            client=client,
            renderer=_Renderer(),
            messages=[],
            model="policy-0",
            prompt_ids=[101],
            sampling_params={"cache_salt": 7, "top_p": 1.0},
            cache_salt="explicit",
            max_prompt_len=10,
        )
        body = client.calls[0][1]["body"]
        self.assertEqual(body["cache_salt"], "explicit")
        self.assertNotIn("cache_salt", body["sampling_params"])
        self.assertEqual(body["sampling_params"]["top_p"], 1.0)

    async def test_empty_or_non_string_salt_is_rejected(self):
        for sampling_params, explicit in (
            ({"cache_salt": ""}, None),
            ({"cache_salt": 7}, None),
            ({}, ""),
            ({}, 7),
        ):
            with self.subTest(sampling_params=sampling_params, explicit=explicit):
                with self.assertRaisesRegex(ValueError, "non-empty string"):
                    await generate(
                        client=_Client(),
                        renderer=_Renderer(),
                        messages=[],
                        model="policy-0",
                        prompt_ids=[101],
                        sampling_params=sampling_params,
                        cache_salt=explicit,
                        max_prompt_len=10,
                    )


if __name__ == "__main__":
    unittest.main()
