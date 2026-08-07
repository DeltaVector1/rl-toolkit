"""Client configs: describe an OpenAI-compatible endpoint and resolve it to a Client.

A `BaseClientConfig` is an OpenAI-compatible endpoint (base_url + API-key env var
+ extra headers) that `resolve_client` turns into a `Client`. The default Prime
endpoint, API key, and team fall back to the active Prime CLI config, so direct
`uv run eval` calls behave like `prime eval`. Both the eval entrypoint (its model
client) and in-env LLM calls (e.g. a judge reward) build clients from these.
`ClientConfig` is the CLI-selectable discriminated union (eval | train).
"""

import os
from typing import Annotated, Literal
from urllib.parse import urlparse

import httpx
from openai import AsyncOpenAI
from pydantic import Field, model_validator
from pydantic_config import BaseConfig
from renderers import RendererConfig

from verifiers.utils.client_utils import load_prime_config
from verifiers.v1.clients.client import Client
from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.clients.train import TrainClient

DEFAULT_PRIME_INFERENCE_URL = "https://api.pinference.ai/api/v1"
PRIME_INFERENCE_HOST = "pinference.ai"
PRIME_TEAM_ID_HEADER = "X-Prime-Team-ID"


class BaseClientConfig(BaseConfig):
    """An OpenAI-compatible endpoint. The API key is read from an env var."""

    base_url: str = DEFAULT_PRIME_INFERENCE_URL
    api_key_var: str = "PRIME_API_KEY"
    headers: dict[str, str] = Field(default_factory=dict)
    """Extra HTTP headers sent on every request."""

    @model_validator(mode="after")
    def apply_prime_config(self) -> "BaseClientConfig":
        if self.api_key_var != "PRIME_API_KEY":
            return self
        prime_config = load_prime_config()
        prime_base_url = (
            os.environ.get("PRIME_INFERENCE_URL")
            or prime_config.get("inference_url")
            or DEFAULT_PRIME_INFERENCE_URL
        )
        if "base_url" not in self.model_fields_set:
            self.base_url = prime_base_url
        host = urlparse(self.base_url).hostname or ""
        if host != PRIME_INFERENCE_HOST and not host.endswith(
            f".{PRIME_INFERENCE_HOST}"
        ):
            return self
        team_id = os.environ.get("PRIME_TEAM_ID") or prime_config.get("team_id")
        if team_id:
            self.headers.setdefault(PRIME_TEAM_ID_HEADER, team_id)
        return self


class EvalClientConfig(BaseClientConfig):
    """The default (eval): forward each request to a matching endpoint via `EvalClient`."""

    type: Literal["eval"] = "eval"


class TrainClientConfig(BaseClientConfig):
    """Training: a vLLM `/inference/v1/generate` endpoint with client-side tokenization (via
    `TrainClient`), so responses carry token ids + logprobs. Needs a running vLLM engine."""

    type: Literal["train"] = "train"
    renderer: RendererConfig | None = None
    """The `renderers.RendererConfig` to use (the same shared type prime-rl configures).
    `None` auto-resolves from the model — which falls back to the default renderer (no
    tool support) for models not in the renderer map, so set it explicitly for
    fine-tunes / tool-using envs."""
    pool_size: int = 1
    """Renderer slots shared across concurrent rollouts (client-side tokenization)."""
    renderer_model_name: str | None = None
    """Model the tokenizer/renderer pool is built for. Pin to the base model so a LoRA
    adapter name (served only for sampling) never drives tokenizer loading. Falls back to
    the per-request model when None."""


# Discriminated union for a CLI-selectable client (`--client.type eval|train`).
ClientConfig = Annotated[
    EvalClientConfig | TrainClientConfig, Field(discriminator="type")
]


def resolve_api_key(config: BaseClientConfig) -> str:
    """The API key for `config`: its env var, falling back to the Prime CLI config for a
    `PRIME_API_KEY`-keyed pinference endpoint. `"EMPTY"` when unset."""
    api_key = os.environ.get(config.api_key_var)
    host = urlparse(config.base_url).hostname or ""
    if (
        not api_key
        and config.api_key_var == "PRIME_API_KEY"
        and (host == PRIME_INFERENCE_HOST or host.endswith(f".{PRIME_INFERENCE_HOST}"))
    ):
        api_key = load_prime_config().get("api_key")
    return api_key or "EMPTY"


# Local change. Upstream leaves the OpenAI SDK defaults in place:
# connect=5.0s, read=600s. Both are wrong for the China link, which has 217 ms
# of round trip and 10-33% packet loss.
#
# connect=5s: a TCP handshake that loses a SYN waits about 1s, then 2s, then 4s
# for the retransmits, so a connect through a busy tunnel blows the 5s budget
# while the path is fine.
#
# read=600s is the expensive one. The orchestrator runs a fixed number of
# rollouts in flight, so a wedged request holds one of those slots for ten
# minutes. Measured 2026-08-07: 121 rollouts in flight while the two engines
# ran 3 between them, and 78.4% of a step's rollouts ended in ReadTimeout. Most
# of the "in flight" count was dead requests waiting out this timeout, and the
# engines starved behind them.
#
# 300s still leaves a wide margin: generation averages 54.6s and p90 is 87.8s.
# Override with PRIME_ROLLOUT_* if a longer generation is ever configured.
_CONNECT_TIMEOUT_S = float(os.environ.get("PRIME_ROLLOUT_CONNECT_TIMEOUT_S", "30"))
_READ_TIMEOUT_S = float(os.environ.get("PRIME_ROLLOUT_TIMEOUT_S", "300"))


def build_async_openai(config: BaseClientConfig) -> AsyncOpenAI:
    """An `AsyncOpenAI` for `config` (resolved key + extra headers) — for in-env model calls
    (e.g. a judge) and the training client's engine connection."""
    return AsyncOpenAI(
        base_url=config.base_url,
        api_key=resolve_api_key(config),
        default_headers=config.headers or None,
        timeout=httpx.Timeout(
            connect=_CONNECT_TIMEOUT_S,
            read=_READ_TIMEOUT_S,
            write=_READ_TIMEOUT_S,
            pool=_CONNECT_TIMEOUT_S,
        ),
    )


def resolve_client(config: BaseClientConfig) -> Client:
    if isinstance(config, TrainClientConfig):
        # The renderer calls a vLLM `/inference/v1/generate` engine through the OpenAI SDK.
        return TrainClient(
            build_async_openai(config),
            pool_size=config.pool_size,
            config=config.renderer,
            renderer_model_name=config.renderer_model_name,
        )
    # The proxy is a raw httpx forwarder; the dialect supplies the auth scheme + upstream path.
    return EvalClient(
        config.base_url, resolve_api_key(config), headers=config.headers or None
    )
