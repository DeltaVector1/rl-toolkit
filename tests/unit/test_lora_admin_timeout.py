"""Regression for a vLLM LoRA load that applies but never returns HTTP."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from prime_rl.utils.client import load_lora_adapter


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"data": [{"id": "adapter-slot1"}]}


class _AppliedButHungClient:
    base_url = "http://engine.test"

    async def post(self, *args, **kwargs):
        await asyncio.Event().wait()

    async def get(self, *args, **kwargs):
        return _Response()


class LoRAAdminTimeoutTest(unittest.IsolatedAsyncioTestCase):
    async def test_visible_adapter_completes_after_hung_load_response(self):
        with patch("prime_rl.utils.client.LORA_LOAD_READ_TIMEOUT_S", 0.01):
            await asyncio.wait_for(
                load_lora_adapter(
                    [_AppliedButHungClient()],
                    "adapter-slot1",
                    "http://adapter.test/step_1",
                ),
                timeout=1.0,
            )


if __name__ == "__main__":
    unittest.main()
