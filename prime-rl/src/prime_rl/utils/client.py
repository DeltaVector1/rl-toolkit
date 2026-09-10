from __future__ import annotations

import asyncio
import math
import os
import re
import time
from collections.abc import Mapping
from itertools import cycle
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit

import httpx
import verifiers.v1 as vf
from httpx import AsyncClient
from openai import AsyncOpenAI, NotFoundError
from renderers import RendererConfig
from verifiers.v1.clients.config import EvalClientConfig, TrainClientConfig

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.logger import get_logger

# Identity tuple used by ``select_train_client`` to key load counts. ``base_url``
# distinguishes servers; ``X-data-parallel-rank`` distinguishes DP shards within a
# server, since the router uses that header to route to specific GPU ranks.
ClientIdentity = tuple[str, str | None]
AdapterIdentity = dict[str, object]


async def _cancel_and_reap(tasks) -> None:
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def client_identity(client: vf.ClientConfig) -> ClientIdentity:
    """Stable identity for load balancing across inference clients."""
    return (client.base_url, client.headers.get("X-data-parallel-rank"))


def _validate_adapter_identity(value: object, source_url: str | None = None) -> AdapterIdentity:
    if not isinstance(value, dict):
        raise RuntimeError("adapter identity is missing")
    actual_source = value.get("source_url")
    target_sha256 = value.get("target_sha256")
    target_size = value.get("target_size")
    if (
        not isinstance(actual_source, str)
        or (source_url is not None and actual_source != source_url)
        or not isinstance(target_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", target_sha256) is None
        or type(target_size) is not int
        or target_size <= 0
    ):
        raise RuntimeError("adapter identity is invalid")
    config_sha256 = value.get("config_sha256")
    config_size = value.get("config_size")
    if (
        not isinstance(config_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", config_sha256) is None
        or type(config_size) is not int
        or config_size < 0
    ):
        raise RuntimeError("adapter identity configuration is invalid")
    for name in ("target_step", "base_step", "base2_step"):
        step = value.get(name)
        if step is not None and (type(step) is not int or step < 0):
            raise RuntimeError(f"adapter identity {name} is invalid")
    for name in ("base_url", "base_sha256", "base2_url", "base2_sha256"):
        identity = value.get(name)
        if identity is not None and not isinstance(identity, str):
            raise RuntimeError(f"adapter identity {name} is invalid")
    return dict(value)


def _validate_loaded_receipt(
    value: object,
    *,
    source_url: str,
    lora_name: str,
    expected: Mapping[str, object] | None,
) -> AdapterIdentity:
    receipt = _validate_adapter_identity(value, source_url)
    if receipt.get("lora_name") != lora_name:
        raise RuntimeError("loaded adapter receipt has the wrong LoRA name")
    if not isinstance(receipt.get("server_instance"), str) or not receipt["server_instance"]:
        raise RuntimeError("loaded adapter receipt has no server instance")
    if expected is not None:
        expected_identity = _validate_adapter_identity(dict(expected), source_url)
        for name, expected_value in expected_identity.items():
            if name in {"lora_name", "server_instance"}:
                continue
            if expected_value is not None and receipt.get(name) != expected_value:
                raise RuntimeError(f"loaded adapter receipt differs in {name}")
    return receipt


@runtime_checkable
class InferencePool(Protocol):
    """Protocol for inference pools (static or elastic)."""

    @property
    def model_name(self) -> str:
        """Get current model name for inference requests."""
        ...

    @property
    def train_clients(self) -> list[vf.ClientConfig]:
        """Get inference clients."""
        ...

    @property
    def admin_clients(self) -> list[AsyncClient]:
        """Get admin clients."""
        ...

    def update_model_name(self, model_name: str) -> None:
        """Update the model name."""
        ...

    async def get_eval_client(self) -> vf.ClientConfig:
        """Get next eval client in round-robin fashion."""
        ...

    async def select_train_client(self, load: Mapping[ClientIdentity, int]) -> vf.ClientConfig:
        """Pick the train client with lowest in-flight load.

        Waits for at least one train client to be available, then returns
        the one with the smallest ``load[client_identity(client)]``. The
        caller owns the in-flight counter; the pool just picks against it.
        """
        ...

    async def wait_for_ready(self, model_name: str, timeout: int | None = None) -> None:
        """Wait for inference pool to be ready."""
        ...

    async def update_weights(
        self,
        weight_dir: Path | str | None,
        lora_name: str | None = None,
        step: int = 0,
        require_delta: bool = False,
        prefetched: bool = False,
        expected_adapter_identity: Mapping[str, object] | None = None,
    ) -> list[AdapterIdentity] | None:
        """Update weights on all inference servers."""
        ...

    @property
    def adapter_origin_base_urls(self) -> list[str] | None:
        """Return the configured admin URLs used as adapter origins."""
        ...

    async def score(self, token_ids: list[int]) -> list[float]:
        """Prefill-score ``token_ids`` under the pool's model — one logprob per token."""
        ...

    async def stop(self) -> None:
        """Stop the inference pool."""
        ...


class PrefillScorer:
    """Prefill-scores token ids against a pool's *current* endpoints. Resolves one
    client per endpoint, cached by endpoint identity — so it fills once for a
    static pool and tolerates churn for an elastic one (a departed endpoint is
    simply never selected again; its client is closed at stop). Round-robins over
    the live endpoints."""

    def __init__(self) -> None:
        self._clients: dict = {}  # client_identity -> AsyncOpenAI, one per endpoint
        self._rr = 0

    async def score(self, configs: list[vf.ClientConfig], model: str, token_ids: list[int]) -> list[float]:
        if not configs:
            raise RuntimeError("no inference endpoints available to prefill-score")
        cfg = configs[self._rr % len(configs)]
        self._rr += 1
        key = client_identity(cfg)
        openai = self._clients.get(key)
        if openai is None:
            # Build the OpenAI client straight from the config fields — works for any
            # ClientConfig type; resolve_client would hand back an EvalClient (no `.openai`)
            # for these chat-completions teacher configs.
            openai = self._clients[key] = AsyncOpenAI(
                base_url=cfg.base_url,
                api_key=os.environ.get(cfg.api_key_var) or "EMPTY",
                default_headers=cfg.headers or None,
            )
        return await prefill_logprobs(openai, model, token_ids)

    async def aclose(self) -> None:
        await asyncio.gather(*(c.close() for c in self._clients.values()))


class StaticInferencePool:
    """Static inference pool with fixed client list."""

    def __init__(
        self,
        client_config: ClientConfig,
        model_name: str,
        train_client_type: str = "openai_chat_completions",
        eval_client_type: str = "openai_chat_completions",
        renderer_config: RendererConfig | None = None,
        pool_size: int | None = None,
    ):
        renderer_model_name = model_name if train_client_type == "renderer" else None
        self._train_clients = setup_clients(
            client_config,
            client_type=train_client_type,
            renderer_config=renderer_config,
            renderer_model_name=renderer_model_name,
            pool_size=pool_size,
        )
        self._eval_clients = setup_clients(client_config, client_type=eval_client_type)
        self._admin_clients = setup_admin_clients(client_config)
        self._skip_model_check = client_config.skip_model_check
        self._wait_for_ready_timeout = client_config.wait_for_ready_timeout
        self._adapter_origin_base_urls = client_config.adapter_origin_base_urls
        admin_origins = {
            str(client.base_url).rstrip("/").removesuffix("/v1")
            for client in self._admin_clients
        }
        if len(admin_origins) > 1 and not self._adapter_origin_base_urls:
            raise ValueError("adapter origin base URLs are required for multiple admin endpoints")
        self._eval_cycle = cycle(self._eval_clients)
        self._scorer = PrefillScorer()
        self.model_name = model_name

    @property
    def train_clients(self) -> list[vf.ClientConfig]:
        return self._train_clients

    @property
    def admin_clients(self) -> list[AsyncClient]:
        return self._admin_clients

    @property
    def adapter_origin_base_urls(self) -> list[str] | None:
        return self._adapter_origin_base_urls

    def update_model_name(self, model_name: str) -> None:
        self.model_name = model_name

    @property
    def eval_clients(self) -> list[vf.ClientConfig]:
        return self._eval_clients

    async def get_eval_client(self) -> vf.ClientConfig:
        return next(self._eval_cycle)

    async def select_train_client(self, load: Mapping[ClientIdentity, int]) -> vf.ClientConfig:
        while not self.train_clients:
            await asyncio.sleep(0.5)
        return min(self.train_clients, key=lambda c: load[client_identity(c)])

    async def wait_for_ready(self, model_name: str, timeout: int | None = None) -> None:
        await check_health(
            self._admin_clients, timeout=timeout if timeout is not None else self._wait_for_ready_timeout
        )
        await maybe_check_has_model(self._admin_clients, model_name, skip_model_check=self._skip_model_check)

    async def update_weights(
        self,
        weight_dir: Path | str | None,
        lora_name: str | None = None,
        step: int = 0,
        require_delta: bool = False,
        prefetched: bool = False,
        expected_adapter_identity: Mapping[str, object] | None = None,
    ) -> list[AdapterIdentity] | None:
        return await update_weights(
            self._admin_clients,
            weight_dir,
            lora_name=lora_name,
            step=step,
            require_delta=require_delta,
            prefetched=prefetched,
            expected_adapter_identity=expected_adapter_identity,
        )

    async def score(self, token_ids: list[int]) -> list[float]:
        """Prefill-score ``token_ids`` under this pool's model (one logprob per
        token, 0.0 for the leading token). Delegates to the shared scorer."""
        return await self._scorer.score(self.train_clients, self.model_name, token_ids)

    async def stop(self) -> None:
        await self._scorer.aclose()


async def setup_inference_pool(
    client_config: ClientConfig,
    model_name: str,
    train_client_type: str = "openai_chat_completions",
    eval_client_type: str = "openai_chat_completions",
    renderer_config: RendererConfig | None = None,
    pool_size: int | None = None,
) -> InferencePool:
    """Create an inference pool from config (static or elastic)."""
    if client_config.is_elastic:
        from prime_rl.utils.elastic import ElasticInferencePool

        return await ElasticInferencePool.from_config(
            client_config,
            model_name=model_name,
            train_client_type=train_client_type,
            eval_client_type=eval_client_type,
            renderer_config=renderer_config,
            pool_size=pool_size,
        )

    return StaticInferencePool(
        client_config,
        model_name=model_name,
        train_client_type=train_client_type,
        eval_client_type=eval_client_type,
        renderer_config=renderer_config,
        pool_size=pool_size,
    )


def setup_clients(
    client_config: ClientConfig,
    client_type: str = "openai_chat_completions",
    renderer_config: RendererConfig | None = None,
    renderer_model_name: str | None = None,
    pool_size: int | None = None,
) -> list[vf.ClientConfig]:
    """Build v1 client configs (one per base_url × DP rank). ``client_type``
    ``renderer`` → token-in/out (``TrainClientConfig``, with the renderer the env
    server should use forwarded as a serialized config so it doesn't fall back to the
    default renderer); otherwise plain chat-completions (``EvalClientConfig``)."""
    is_renderer = client_type == "renderer"
    config_cls = TrainClientConfig if is_renderer else EvalClientConfig
    renderer_extra: dict = {}
    if is_renderer:
        renderer_extra = {
            "renderer": renderer_config,
            "pool_size": pool_size or 1,
            "renderer_model_name": renderer_model_name,
        }
    env_headers = {
        k: v for k, v in ((k, os.getenv(v)) for k, v in client_config.headers_from_env.items()) if v is not None
    }
    clients: list[vf.ClientConfig] = []
    for base_url in client_config.base_url:
        for dp_rank in range(client_config.dp_rank_count):
            headers = {**client_config.headers, **env_headers}
            if client_config.dp_rank_count > 1:
                headers["X-data-parallel-rank"] = str(dp_rank)
            clients.append(
                config_cls(base_url=base_url, api_key_var=client_config.api_key_var, headers=headers, **renderer_extra)
            )
    return clients


def setup_admin_clients(client_config: ClientConfig) -> list[AsyncClient]:
    """Create dedicated admin clients for weight update operations.

    Uses a separate connection pool to avoid queueing behind streaming requests.
    When admin_base_url is set, uses those URLs instead of base_url, allowing
    weight updates to bypass routers in disaggregated P/D deployments.
    """
    urls = client_config.admin_base_url if client_config.admin_base_url else client_config.base_url

    def _setup_admin_client(base_url: str) -> httpx.AsyncClient:
        env_headers = {
            k: v for k, v in ((k, os.getenv(v)) for k, v in client_config.headers_from_env.items()) if v is not None
        }
        headers = {**client_config.headers, **env_headers}
        api_key = os.getenv(client_config.api_key_var, "EMPTY")
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"

        # Strip /v1 suffix since admin endpoints are at root level
        base_url = base_url.rstrip("/").removesuffix("/v1")

        return AsyncClient(
            base_url=base_url,
            headers=headers,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=1),
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0),
        )

    return [_setup_admin_client(base_url) for base_url in urls]


async def maybe_check_has_model(
    admin_clients: list[AsyncClient], model_name: str, skip_model_check: bool = False
) -> None:
    if skip_model_check:
        return
    logger = get_logger()
    logger.debug(f"Checking if model {model_name} is in the inference pool")
    async def _models(client: AsyncClient) -> httpx.Response:
        response = await client.get(
            "/v1/models",
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0),
        )
        response.raise_for_status()
        return response

    tasks = [asyncio.create_task(_models(admin_client)) for admin_client in admin_clients]
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        await _cancel_and_reap(tasks)
        raise
    for admin_client, result in zip(admin_clients, results):
        payload = result.json()
        models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models, list) or not all(isinstance(model, dict) for model in models):
            raise RuntimeError(f"model response from {admin_client.base_url} has invalid data")
        if not any(model.get("id") == model_name for model in models):
            raise ValueError(f"Model {model_name} was not found in the inference pool on {admin_client.base_url}")
    logger.debug(f"Model {model_name} was found in the inference pool")


async def model_is_visible(admin_client: AsyncClient, model_name: str) -> bool:
    """Return whether one endpoint currently exposes ``model_name``.

    Engine health and model visibility are separate. A restarted vLLM server
    can answer health and base-model requests while its dynamic LoRA is absent.
    Keep this probe bounded so recovery cannot wedge on an open HTTP response.
    """
    try:
        response = await admin_client.get(
            "/v1/models",
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0),
        )
        response.raise_for_status()
        return any(model.get("id") == model_name for model in response.json().get("data", []))
    except Exception:
        return False


async def check_health(
    admin_clients: list[AsyncClient], interval: int = 1, log_interval: int = 10, timeout: int = 1800
) -> None:
    logger = get_logger()

    async def _check_health(admin_client: AsyncClient) -> None:
        started = time.monotonic()
        deadline = started + max(0.0, float(timeout))
        logger.debug("Starting pinging /health to check health")
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                request_timeout = min(30.0, remaining)
                response = await asyncio.wait_for(
                    admin_client.get(
                        "/health",
                        timeout=httpx.Timeout(
                            connect=min(10.0, request_timeout),
                            read=request_timeout,
                            write=request_timeout,
                            pool=min(10.0, request_timeout),
                        ),
                    ),
                    timeout=request_timeout,
                )
                if response.status_code == 404:
                    logger.warning("The route /health does not exist. Skipping health check.")
                    return
                response.raise_for_status()
                logger.debug(f"Inference pool is ready after {time.monotonic() - started:.3f} seconds")
                return
            except NotFoundError:
                logger.warning("The route /health does not exist. Skipping health check.")
                return
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.warning("The route /health does not exist. Skipping health check.")
                    return
                elapsed = time.monotonic() - started
                if elapsed >= log_interval:
                    logger.warning(
                        f"Inference server was not reached after {elapsed:.1f} seconds (Error: {e}) on {admin_client.base_url}"
                    )
            except Exception as e:
                elapsed = time.monotonic() - started
                if elapsed >= log_interval:
                    logger.warning(
                        f"Inference server was not reached after {elapsed:.1f} seconds (Error: {e}) on {admin_client.base_url}"
                    )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(max(0.0, interval), remaining))
        elapsed = time.monotonic() - started
        msg = f"Inference server is not ready after {elapsed:.3f} seconds (>{timeout}) seconds. Aborting..."
        logger.error(msg)
        raise TimeoutError(msg)

    tasks = [asyncio.create_task(_check_health(admin_client)) for admin_client in admin_clients]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        await _cancel_and_reap(tasks)
        raise


NCCL_READY_MARKER = "NCCL_READY"


def _is_retryable_admin_error(exception: BaseException) -> bool:
    """Check if an exception should trigger a retry for an admin op (pause/resume/update_weights)."""
    if isinstance(exception, httpx.HTTPStatusError):
        # Retry on transient server errors (5xx, e.g. engine briefly unresponsive);
        # client errors (4xx) won't fix themselves on retry.
        return exception.response.status_code >= 500
    # Retry on transport-level failures (timeouts, connection resets, etc.) so the
    # per-attempt read timeout below turns a stuck server into a bounded retry loop.
    if isinstance(exception, (TimeoutError, httpx.TimeoutException, httpx.TransportError)):
        return True
    return False


# Per-attempt read timeout for admin ops, overridable per call. The read
# timeout converts a hang into a TimeoutException that the retry loop can
# handle. Sized for `/pause`, which drains in-flight requests
# (mode="keep") and so can legitimately take a while.
ADMIN_TIMEOUT_S = 300.0
POLICY_PHASE_TIMEOUT_S = 600.0
# `/update_weights` runs a collective NCCL receive across all DP workers, which
# can take longer than the other admin ops.
UPDATE_WEIGHTS_TIMEOUT_S = 720.0
FULL_CHECKPOINT_STAGE_TIMEOUT_S = 600.0
# Per-step LoRA policy phases use one absolute 600-second deadline.
RESUME_ATTEMPT_TIMEOUT_S = 600.0
RESUME_TOTAL_TIMEOUT_S = 600.0
RESUME_CONFIRM_TIMEOUT_S = 600.0
# Full-model bootstrap keeps its historical, separate resume confirmation
# budget. It is never used when a per-step deadline is supplied.
FULL_RESUME_ATTEMPT_TIMEOUT_S = 900.0
FULL_RESUME_TOTAL_TIMEOUT_S = 1800.0
FULL_RESUME_CONFIRM_TIMEOUT_S = 120.0
RESUME_PROBE_TIMEOUT_S = 10.0
RESUME_PROBE_INTERVAL_S = 2.0


async def _admin_post(
    client: AsyncClient,
    path: str,
    *,
    timeout_s: float = ADMIN_TIMEOUT_S,
    total_timeout_s: float | None = None,
    deadline: float | None = None,
    **kwargs,
) -> None:
    """POST an admin op with a bounded retry budget."""
    await _admin_post_response(
        client,
        path,
        timeout_s=timeout_s,
        total_timeout_s=total_timeout_s,
        deadline=deadline,
        **kwargs,
    )


async def _admin_post_response(
    client: AsyncClient,
    path: str,
    *,
    timeout_s: float = ADMIN_TIMEOUT_S,
    total_timeout_s: float | None = None,
    deadline: float | None = None,
    max_attempts: int = 10,
    **kwargs,
) -> httpx.Response:
    """POST an admin operation and return its successful response."""
    loop = asyncio.get_running_loop()
    if deadline is None:
        retry_budget = 2 * timeout_s if total_timeout_s is None else total_timeout_s
        deadline = loop.time() + retry_budget
    elif total_timeout_s is not None:
        deadline = min(deadline, loop.time() + total_timeout_s)
    delay = 1.0
    for attempt in range(max_attempts):
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError(f"admin operation timed out: {path}")
        attempt_timeout = min(timeout_s, remaining)
        try:
            try:
                response = await asyncio.wait_for(
                    client.post(
                        path,
                        timeout=httpx.Timeout(
                            connect=min(10.0, attempt_timeout),
                            read=attempt_timeout,
                            write=min(60.0, attempt_timeout),
                            pool=min(10.0, attempt_timeout),
                        ),
                        **kwargs,
                    ),
                    timeout=attempt_timeout,
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError(f"admin request timed out: {path}") from exc
            response.raise_for_status()
            return response
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            if (
                not _is_retryable_admin_error(exc)
                or attempt + 1 >= max_attempts
                or deadline - loop.time() <= 0
            ):
                raise
            remaining = deadline - loop.time()
            await asyncio.sleep(min(delay, remaining))
            delay = min(delay * 2.0, 10.0)
    raise RuntimeError(f"admin operation did not complete: {path}")


async def _pause_engines(
    admin_clients: list[AsyncClient], *, step: int, deadline: float | None = None
) -> None:
    """Abort registered requests and pause every inference scheduler.

    The dispatcher has preserved every prompt that needs regeneration.
    Engine-native abort is still required because cancelling an HTTP client can
    leave its decode request registered for several engine steps.
    """
    logger = get_logger()
    logger.info(f"Updating policy in-flight to v{step}")
    tasks = [
        asyncio.create_task(
            _admin_post(
                client,
                "/pause",
                params={"mode": "abort", "clear_cache": "false"},
                deadline=deadline,
            )
        )
        for client in admin_clients
    ]
    try:
        if deadline is None:
            await asyncio.gather(*tasks)
        else:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("policy pause phase exceeded its 600-second deadline")
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=remaining)
    except BaseException:
        await _cancel_and_reap(tasks)
        raise
    logger.debug("All inference engines paused")


async def _resume_engines(
    admin_clients: list[AsyncClient], *, deadline: float | None = None
) -> None:
    """Resume all inference engines after weight update.

    Resuming is idempotent. A timeout does not prove that the request failed:
    the engine can finish the request after the HTTP response deadline. Confirm
    the server state before reporting a failed policy update.
    """
    logger = get_logger()

    async def _read_resume_state(
        client: AsyncClient, phase_deadline: float | None
    ) -> tuple[str | None, bool, bool]:
        """Return state, route availability, and explicit route absence."""
        try:
            remaining = (
                phase_deadline - asyncio.get_running_loop().time()
                if phase_deadline is not None
                else RESUME_PROBE_TIMEOUT_S
            )
            if remaining <= 0:
                return None, False, False
            response = await client.get(
                "/resume_status",
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=min(RESUME_PROBE_TIMEOUT_S, remaining),
                    write=min(10.0, remaining),
                    pool=min(10.0, remaining),
                ),
            )
        except (TimeoutError, httpx.TimeoutException, httpx.TransportError):
            return None, False, False
        if response.status_code == 404:
            return None, False, True
        response.raise_for_status()
        state = response.json().get("state")
        if state not in {"pausing", "paused", "resuming", "running", "unknown"}:
            raise RuntimeError(f"invalid resume state from {client.base_url}: {state!r}")
        return state, True, False

    async def _is_live(client: AsyncClient, phase_deadline: float | None) -> bool:
        try:
            remaining = (
                phase_deadline - asyncio.get_running_loop().time()
                if phase_deadline is not None
                else RESUME_PROBE_TIMEOUT_S
            )
            if remaining <= 0:
                return False
            response = await client.get(
                "/liveness",
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=min(RESUME_PROBE_TIMEOUT_S, remaining),
                    write=min(10.0, remaining),
                    pool=min(10.0, remaining),
                ),
            )
        except (TimeoutError, httpx.TimeoutException, httpx.TransportError):
            return False
        return response.status_code == 200

    async def _confirm_resume(
        client: AsyncClient, phase_deadline: float | None
    ) -> tuple[bool, bool]:
        confirm_deadline = (
            phase_deadline
            if phase_deadline is not None
            else asyncio.get_running_loop().time() + FULL_RESUME_CONFIRM_TIMEOUT_S
        )
        state_endpoint_available = False
        while True:
            state, route_available, route_missing = await _read_resume_state(client, confirm_deadline)
            if route_available:
                state_endpoint_available = True
                if state == "running" and await _is_live(client, confirm_deadline):
                    return True, True
            elif route_missing and await _is_live(client, confirm_deadline):
                # Older servers have no state route. Liveness is weaker evidence,
                # so the log must identify that this path was used.
                return True, False

            remaining = confirm_deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return False, state_endpoint_available
            await asyncio.sleep(min(RESUME_PROBE_INTERVAL_S, remaining))

    async def _resume_one(client: AsyncClient, phase_deadline: float | None) -> None:
        try:
            await _admin_post(
                client,
                "/resume",
                timeout_s=(
                    FULL_RESUME_ATTEMPT_TIMEOUT_S
                    if phase_deadline is None
                    else RESUME_ATTEMPT_TIMEOUT_S
                ),
                total_timeout_s=(
                    FULL_RESUME_TOTAL_TIMEOUT_S
                    if phase_deadline is None
                    else None
                ),
                deadline=phase_deadline,
            )
        except (TimeoutError, httpx.TimeoutException, httpx.TransportError) as exc:
            confirmed, state_endpoint_available = await _confirm_resume(client, phase_deadline)
            if confirmed:
                evidence = "resume state and liveness" if state_endpoint_available else "liveness only"
                logger.warning(
                    f"Resume response from {client.base_url} timed out, but {evidence} "
                    "confirmed the engine is running; accepting the resume"
                )
                return
            raise RuntimeError(
                f"Resume timed out on {client.base_url} and was not confirmed by "
                "resume state and liveness probes"
            ) from exc

    tasks = [asyncio.create_task(_resume_one(client, deadline)) for client in admin_clients]
    try:
        if deadline is None:
            await asyncio.gather(*tasks)
        else:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("policy resume phase exceeded its 600-second deadline")
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=remaining)
    except BaseException:
        await _cancel_and_reap(tasks)
        raise
    logger.debug("All inference engines resumed")


async def stage_lora_adapter(
    admin_clients: list[AsyncClient],
    lora_path: Path | str,
    *,
    require_delta: bool = False,
    policy_step: int | None = None,
    delta_deadline: float | None = None,
    origin_base_urls: list[str] | None = None,
) -> dict[str, object]:
    """Stage one adapter copy per inference host.

    A LoRA delta is valid only for the adjacent policy version. The receiver
    checks the manifest and its verified base. This function sends that
    requirement explicitly and returns measured staging time for each host.
    """
    lora_path_posix = str(lora_path)
    if policy_step is not None and (
        type(policy_step) is not int or policy_step <= 0
    ):
        raise RuntimeError("adapter policy step must be a positive integer")
    if require_delta and policy_step is None:
        raise RuntimeError("adapter policy step is required for delta staging")
    if delta_deadline is not None and not require_delta:
        raise RuntimeError("adapter delta deadline requires delta staging")
    if not lora_path_posix.startswith(("http://", "https://")):
        return {
            "wire_stage_seconds": 0.0,
            "transfer_wall_seconds": 0.0,
            "reconstruction_wall_seconds": 0.0,
            "reconstruction_worker_seconds": 0.0,
            "final_verification_wall_seconds": 0.0,
            "total_stage_wall_seconds": 0.0,
            "cache_hit": False,
            "host_count": 0,
            "adapter_identity": None,
        }
    if not admin_clients:
        raise RuntimeError("cannot stage an adapter without an inference endpoint")

    def host_key(client: AsyncClient) -> tuple[str, str, int | None]:
        url = client.base_url
        effective_port = url.port
        if effective_port is None:
            effective_port = {"http": 80, "https": 443}.get(url.scheme)
        return (url.scheme, url.host or str(url), effective_port)

    hosts: dict[tuple[str, str, int | None], AsyncClient] = {}
    for client in admin_clients:
        hosts.setdefault(host_key(client), client)

    def normalized_url(client: AsyncClient) -> str:
        return str(client.base_url).rstrip("/").removesuffix("/v1")

    if origin_base_urls is None:
        ordered_hosts = list(hosts.values())
        origin_count = len(ordered_hosts)
        peer_groups = []
    else:
        configured = {
            str(url).rstrip("/").removesuffix("/v1")
            for url in origin_base_urls
            if isinstance(url, str) and str(url).strip()
        }
        if not configured:
            raise RuntimeError("adapter origin base URLs are not configured")
        unknown_origins = configured - {normalized_url(client) for client in hosts.values()}
        if unknown_origins:
            raise RuntimeError(
                "configured adapter origin is not an admin endpoint: "
                + ", ".join(sorted(unknown_origins))
            )
        origin_hosts = [client for client in hosts.values() if normalized_url(client) in configured]
        peer_hosts = [client for client in hosts.values() if normalized_url(client) not in configured]
        if not origin_hosts:
            raise RuntimeError("configured adapter origin is not an admin endpoint")
        ordered_hosts = origin_hosts
        origin_count = len(origin_hosts)
        peer_groups = [peer_hosts] if peer_hosts else []

    async def _stage_one(
        client: AsyncClient,
        *,
        remaining_delta_seconds: float | None = None,
    ) -> dict[str, object]:
        started = time.perf_counter()
        stage_timeout_s = (
            remaining_delta_seconds
            if remaining_delta_seconds is not None
            else LORA_STAGE_ATTEMPT_TIMEOUT_S
        )
        stage_total_timeout_s = stage_timeout_s if require_delta else LORA_STAGE_TOTAL_TIMEOUT_S
        payload = {
            "lora_path": lora_path_posix,
            "require_delta": require_delta,
        }
        if policy_step is not None:
            payload["policy_step"] = policy_step
        if remaining_delta_seconds is not None:
            payload["stage_timeout_seconds"] = remaining_delta_seconds
        response = await _admin_post_response(
            client,
            "/stage_lora_adapter",
            json=payload,
            timeout_s=stage_timeout_s,
            total_timeout_s=stage_total_timeout_s,
            max_attempts=1,
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"invalid adapter stage response from {client.base_url}") from exc
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise RuntimeError(f"adapter stage failed on {client.base_url}: {payload!r}")
        adapter_identity = _validate_adapter_identity(
            payload.get("adapter_identity"), lora_path_posix
        )
        def metric(name: str, fallback: float | None = None) -> float | None:
            value = payload.get(name, fallback)
            return float(value) if _is_finite_number(value) and float(value) >= 0 else None

        total_stage_seconds = metric("total_stage_wall_seconds", metric("stage_seconds"))
        if total_stage_seconds is None:
            total_stage_seconds = time.perf_counter() - started
        transfer_seconds = metric("transfer_wall_seconds")
        legacy_wire_seconds = metric("wire_stage_seconds", metric("stage_seconds", transfer_seconds))
        reconstruction_seconds = metric("reconstruction_wall_seconds", metric("reconstruct_seconds"))
        worker_seconds = metric("reconstruction_worker_seconds")
        verification_seconds = metric("final_verification_wall_seconds")
        return {
            "wire_stage_seconds": legacy_wire_seconds,
            "transfer_wall_seconds": transfer_seconds,
            "reconstruction_wall_seconds": reconstruction_seconds,
            "reconstruct_seconds": reconstruction_seconds,
            "reconstruction_worker_seconds": worker_seconds,
            "final_verification_wall_seconds": verification_seconds,
            "total_stage_wall_seconds": total_stage_seconds,
            "cache_hit": bool(payload.get("cache_hit", False)),
            "bytes_done": int(payload.get("bytes_done", 0) or 0),
            "bytes_total": int(payload.get("bytes_total", 0) or 0),
            "request_seconds": time.perf_counter() - started,
            "adapter_identity": adapter_identity,
        }

    async def _stage_one_with_retries(
        client: AsyncClient,
        *,
        absolute_deadline: float | None = None,
    ) -> dict[str, object]:
        for attempt in range(1, 4):
            remaining = None
            if absolute_deadline is not None:
                remaining = absolute_deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("adapter delta staging exceeded its time limit")
            try:
                result = await _stage_one(
                    client,
                    remaining_delta_seconds=remaining if require_delta else None,
                )
                if absolute_deadline is not None and time.monotonic() >= absolute_deadline:
                    raise TimeoutError("adapter delta staging exceeded its time limit")
                return result
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                retryable = True
                if isinstance(exc, RuntimeError):
                    message = str(exc).lower()
                    retryable = not any(
                        marker in message
                        for marker in ("identity", "checksum", "policy step", "manifest")
                    )
                if attempt == 3 or not retryable:
                    raise
                delay = min(0.5 * 2 ** (attempt - 1), 10.0)
                if absolute_deadline is not None:
                    remaining = absolute_deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("adapter delta staging exceeded its time limit") from exc
                    delay = min(delay, remaining)
                await asyncio.sleep(delay)
        raise RuntimeError("adapter stage retry loop ended unexpectedly")

    started = time.perf_counter()
    configured_delta_timeout = _delta_stage_timeout_s() if require_delta else None
    if not require_delta and delta_deadline is None:
        delta_deadline = time.monotonic() + LORA_STAGE_TOTAL_TIMEOUT_S
    stage_limit = None
    predictive_transport = False
    if require_delta:
        # The caller has already supplied the policy step. Select its
        # operational limit before probing the manifest so a later step
        # cannot spend the full server maximum in this request.
        if policy_step is None:
            raise RuntimeError("adapter policy step is required for delta staging")
        stage_limit = adapter_stage_limit_seconds(policy_step)
        # A policy adapter stage has one absolute phase deadline. Keep any
        # caller-provided deadline as the stricter bound, and never reset it
        # while probing the manifest or retrying a host.
        selected_deadline = time.monotonic() + POLICY_PHASE_TIMEOUT_S
        selected_deadline = min(selected_deadline, time.monotonic() + min(stage_limit, configured_delta_timeout))
        if delta_deadline is None:
            delta_deadline = selected_deadline
        else:
            delta_deadline = min(delta_deadline, selected_deadline)
        remaining = delta_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("adapter delta staging exceeded its time limit")
        try:
            async with httpx.AsyncClient(follow_redirects=False) as probe:
                response = await asyncio.wait_for(
                    probe.get(
                        f"{lora_path_posix.rstrip('/')}/adapter_manifest.json",
                        timeout=httpx.Timeout(
                            connect=remaining,
                            read=remaining,
                            write=remaining,
                            pool=remaining,
                        ),
                    ),
                    timeout=remaining,
                )
                response.raise_for_status()
                manifest = response.json()
        except asyncio.TimeoutError as exc:
            raise TimeoutError("adapter delta staging exceeded its time limit") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError("cannot determine adapter transport before staging") from exc
        if not isinstance(manifest, dict):
            raise RuntimeError("adapter transport manifest is not an object")
        source_name = Path(urlsplit(lora_path_posix).path.rstrip("/")).name
        match = re.fullmatch(r"step_(\d+)", source_name)
        target_step = manifest.get("target_step")
        if (
            match is None
            or type(target_step) is not int
            or target_step <= 0
            or int(match.group(1)) != target_step
        ):
            raise RuntimeError("cannot determine adapter policy step from validated manifest")
        if manifest.get("version") == 1:
            if manifest.get("base_step") != target_step - 1:
                raise RuntimeError("adapter manifest steps are not consecutive")
        elif manifest.get("version") == 2:
            if (
                manifest.get("base_step") != target_step - 1
                or manifest.get("base2_step") != target_step - 2
            ):
                raise RuntimeError("adapter manifest steps are not consecutive")
        else:
            raise RuntimeError("adapter manifest version is unsupported")
        if policy_step != target_step:
            raise RuntimeError("requested adapter policy step differs from its manifest")
        predictive_transport = (
            manifest.get("version") == 2
            and manifest.get("transport") == "chunked-bf16-linear-zigzag-sparse-zstd"
        )
    results = []
    group_seconds: list[float] = []
    groups = [ordered_hosts, *peer_groups]
    if predictive_transport:
        # v2 origin and peer transfers share one absolute deadline. If the
        # origin fails, cancel and await every peer task before returning.
        remaining = delta_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("adapter delta staging exceeded its time limit")
        tasks = [
            asyncio.create_task(
                _stage_one_with_retries(client, absolute_deadline=delta_deadline)
            )
            for group in groups
            for client in group
        ]
        try:
            staged = await asyncio.wait_for(asyncio.gather(*tasks), timeout=remaining)
        except BaseException:
            await _cancel_and_reap(tasks)
            raise
        if time.monotonic() >= delta_deadline:
            raise TimeoutError("adapter delta staging exceeded its time limit")
        results.extend(staged)
        position = 0
        for group in groups:
            group_results = staged[position : position + len(group)]
            position += len(group)
            group_seconds.append(
                max(
                    (float(item["request_seconds"]) for item in group_results),
                    default=0.0,
                )
            )
    else:
        for group in groups:
            group_started = time.perf_counter()
            stage_tasks = [
                asyncio.create_task(
                    _stage_one_with_retries(
                        client,
                        absolute_deadline=delta_deadline,
                    )
                )
                for client in group
            ]
            if delta_deadline is None:
                try:
                    staged = await asyncio.gather(*stage_tasks)
                except BaseException:
                    await _cancel_and_reap(stage_tasks)
                    raise
            else:
                remaining = delta_deadline - time.monotonic()
                if remaining <= 0:
                    await _cancel_and_reap(stage_tasks)
                    raise TimeoutError("adapter delta staging exceeded its time limit")
                try:
                    staged = await asyncio.wait_for(asyncio.gather(*stage_tasks), timeout=remaining)
                except BaseException as exc:
                    await _cancel_and_reap(stage_tasks)
                    if isinstance(exc, asyncio.TimeoutError):
                        raise TimeoutError("adapter delta staging exceeded its time limit") from exc
                    raise
                if time.monotonic() >= delta_deadline:
                    raise TimeoutError("adapter delta staging exceeded its time limit")
            results.extend(staged)
            group_seconds.append(time.perf_counter() - group_started)
    if delta_deadline is not None and time.monotonic() >= delta_deadline:
        raise TimeoutError("adapter delta staging exceeded its time limit")
    origin_stage_seconds = group_seconds[0] if group_seconds else 0.0
    peer_stage_seconds = sum(group_seconds[1:])
    transfer_values = [
        float(result["transfer_wall_seconds"])
        for result in results
        if result["transfer_wall_seconds"] is not None
    ]
    legacy_wire_values = [
        float(result["wire_stage_seconds"])
        for result in results
        if result["wire_stage_seconds"] is not None
    ]
    reconstruct_values = [
        float(result["reconstruction_wall_seconds"])
        for result in results
        if result["reconstruction_wall_seconds"] is not None
    ]
    worker_values = [
        float(result["reconstruction_worker_seconds"])
        for result in results
        if result["reconstruction_worker_seconds"] is not None
    ]
    verification_values = [
        float(result["final_verification_wall_seconds"])
        for result in results
        if result["final_verification_wall_seconds"] is not None
    ]
    identities = [result["adapter_identity"] for result in results]
    if not identities or any(identity != identities[0] for identity in identities[1:]):
        raise RuntimeError("inference hosts reported different adapter identities")
    adapter_identity = identities[0]
    assert isinstance(adapter_identity, dict)
    return {
        "wire_stage_seconds": max(legacy_wire_values) if legacy_wire_values else None,
        "transfer_wall_seconds": max(transfer_values) if transfer_values else None,
        "origin_stage_seconds": origin_stage_seconds,
        "peer_stage_seconds": peer_stage_seconds,
        "reconstruction_wall_seconds": max(reconstruct_values) if reconstruct_values else None,
        "reconstruct_seconds": max(reconstruct_values) if reconstruct_values else None,
        "reconstruction_worker_seconds": max(worker_values) if worker_values else None,
        "final_verification_wall_seconds": max(verification_values) if verification_values else None,
        "total_stage_wall_seconds": time.perf_counter() - started,
        "cache_hit": bool(results) and all(bool(result["cache_hit"]) for result in results),
        "bytes_done": sum(int(result["bytes_done"]) for result in results),
        "bytes_total": sum(int(result["bytes_total"]) for result in results),
        "host_count": len(hosts),
        "origin_host_count": origin_count,
        "peer_host_count": len(hosts) - origin_count,
        "adapter_identity": adapter_identity,
        "policy_step": policy_step,
        "stage_limit_seconds": stage_limit if require_delta else None,
        "acceptance_target_seconds": 300.0 if require_delta else None,
    }


async def stage_full_checkpoint(admin_clients: list[AsyncClient], checkpoint_url: Path | str) -> None:
    """Download one immutable full checkpoint on every inference host."""
    checkpoint_url_posix = str(checkpoint_url)
    async def _stage(client: AsyncClient) -> None:
        try:
            await _admin_post(
                client,
                "/stage_full_checkpoint",
                json={"checkpoint_url": checkpoint_url_posix},
                timeout_s=FULL_CHECKPOINT_STAGE_TIMEOUT_S,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            await _admin_post(
                client,
                "/collective_rpc",
                json={"method": "stage_full_checkpoint", "args": [checkpoint_url_posix]},
                timeout_s=FULL_CHECKPOINT_STAGE_TIMEOUT_S,
            )

    tasks = [asyncio.create_task(_stage(client)) for client in admin_clients]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        await _cancel_and_reap(tasks)
        raise


async def update_weights(
    admin_clients: list[AsyncClient],
    weight_dir: Path | str | None,
    lora_name: str | None = None,
    step: int = 0,
    require_delta: bool = False,
    prefetched: bool = False,
    expected_adapter_identity: Mapping[str, object] | None = None,
) -> list[AdapterIdentity] | None:
    """Update weights on static inference servers.

    Pauses all engines first to drain in-flight requests, then performs the
    weight update, then resumes. This ensures all DP workers are idle and can
    participate in the collective weight transfer.

    Note: the prefix cache is intentionally not reset on weight update. The orchestrator
    salts the prefix cache per weight version (``cache_salt`` in the sampling request, see
    ``orchestrator/envs.py``), so KV computed under old weights is never reused.
    """
    logger = get_logger()

    weight_dir_posix = str(weight_dir) if weight_dir is not None else None

    if lora_name is not None and weight_dir is not None:
        # Pause admission while the collective LoRA load runs.
        pause_deadline = time.monotonic() + POLICY_PHASE_TIMEOUT_S
        await _pause_engines(admin_clients, step=step, deadline=pause_deadline)
        try:
            load_deadline = time.monotonic() + POLICY_PHASE_TIMEOUT_S
            receipts = await load_lora_adapter(
                admin_clients,
                lora_name,
                weight_dir,
                require_delta=require_delta,
                prefetched=prefetched,
                expected_identity=expected_adapter_identity,
                deadline=load_deadline,
            )
        finally:
            resume_deadline = time.monotonic() + POLICY_PHASE_TIMEOUT_S
            await _resume_engines(admin_clients, deadline=resume_deadline)
        return receipts
    else:
        # Pause engines so all DP workers drain in-flight work and can join the NCCL broadcast
        await _pause_engines(admin_clients, step=step)

        try:
            # Create ready marker before servers enter receive path (used by NCCL broadcast)
            if isinstance(weight_dir, Path):
                nccl_ready_file = weight_dir / NCCL_READY_MARKER
                nccl_ready_file.parent.mkdir(parents=True, exist_ok=True)
                nccl_ready_file.touch()
                logger.debug(f"Created NCCL_READY marker at {nccl_ready_file}")

            async def _update(admin_client: AsyncClient) -> None:
                try:
                    await _admin_post(
                        admin_client,
                        "/update_weights",
                        json={"weight_dir": weight_dir_posix},
                        timeout_s=UPDATE_WEIGHTS_TIMEOUT_S,
                    )
                except httpx.HTTPStatusError as exc:
                    # New vLLM reserves /update_weights for its typed transport
                    # API. Its development RPC route invokes the same mounted
                    # worker extension without changing established engines.
                    if exc.response.status_code != 400:
                        raise
                    await _admin_post(
                        admin_client,
                        "/collective_rpc",
                        json={"method": "update_weights_from_path", "args": [weight_dir_posix]},
                        timeout_s=UPDATE_WEIGHTS_TIMEOUT_S,
                    )

            tasks = [asyncio.create_task(_update(admin_client)) for admin_client in admin_clients]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                await _cancel_and_reap(tasks)
                raise
        finally:
            await _resume_engines(admin_clients)


def _is_retryable_lora_error(exception: BaseException) -> bool:
    """Check if an exception should trigger a retry for LoRA loading."""
    if isinstance(exception, httpx.HTTPStatusError):
        # Retry on 404 (adapter not found) or 500 (server error during loading)
        return exception.response.status_code in (404, 500)
    # Retry on transport-level failures (timeouts, connection resets, etc.) so
    # the per-call read timeout below turns a stuck server into a bounded retry
    # loop instead of propagating as a hard failure on the first hiccup.
    if isinstance(exception, (TimeoutError, httpx.TimeoutException, httpx.TransportError)):
        return True
    return False


# Adapter files cross the control tunnel before vLLM loads them. A rank-32 GLM
# adapter is about 28.5 GiB, so staging needs a longer transport budget than
# the local adapter load.
LORA_LOAD_READ_TIMEOUT_S = 600.0
LORA_LOAD_TOTAL_TIMEOUT_S = 600.0
LORA_STAGE_ATTEMPT_TIMEOUT_S = 600.0
LORA_STAGE_TOTAL_TIMEOUT_S = 600.0
default_delta_stage_timeout_s = 600.0


def adapter_stage_limit_seconds(policy_step: int) -> float:
    """Return the bounded operational staging limit for one adapter step."""
    if type(policy_step) is not int or policy_step <= 0:
        raise RuntimeError("adapter policy step must be a positive integer")
    if policy_step == 1:
        return 600.0
    if policy_step == 2:
        return 600.0
    return 300.0


def _delta_stage_timeout_s() -> float:
    configured = os.environ.get("PRIME_LORA_DELTA_STAGE_TIMEOUT_S")
    if configured is None:
        value = default_delta_stage_timeout_s
    else:
        try:
            value = float(configured)
        except ValueError as exc:
            raise RuntimeError("PRIME_LORA_DELTA_STAGE_TIMEOUT_S must be a number") from exc
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not _is_finite_number(value)
        or value <= 0
    ):
        raise RuntimeError("adapter delta stage timeout must be positive")
    return float(value)


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


async def load_lora_adapter(
    admin_clients: list[AsyncClient],
    lora_name: str,
    lora_path: Path | str,
    *,
    require_delta: bool = False,
    prefetched: bool = False,
    expected_identity: Mapping[str, object] | None = None,
    deadline: float | None = None,
) -> list[AdapterIdentity]:
    """Make a HTTP post request to the vLLM server to load a LoRA adapter.

    Uses our wrapper around vLLM's /v1/load_lora_adapter. The prefix cache is not reset
    here; the orchestrator salts it per weight version (see ``orchestrator/envs.py``) so
    KV computed under old weights is never reused.

    Retries with exponential backoff if the adapter files are not found,
    which can happen due to NFS propagation delays.
    """
    logger = get_logger()
    lora_path_posix = str(lora_path)
    remote_adapter = lora_path_posix.startswith(("http://", "https://"))
    if remote_adapter and not admin_clients:
        raise RuntimeError("cannot load a remote adapter without an inference endpoint")
    effective_expected_identity = expected_identity
    if remote_adapter and effective_expected_identity is None:
        raise RuntimeError("remote LoRA load has no verified staged adapter identity")
    if deadline is None:
        deadline = time.monotonic() + LORA_LOAD_TOTAL_TIMEOUT_S

    async def _read_loaded_receipt(admin_client: AsyncClient) -> AdapterIdentity:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("LoRA load phase exceeded its 600-second deadline")
        response = await admin_client.get(
            "/staging_status",
            timeout=httpx.Timeout(
                connect=min(10.0, remaining),
                read=min(30.0, remaining),
                write=min(30.0, remaining),
                pool=min(30.0, remaining),
            ),
        )
        response.raise_for_status()
        payload = response.json()
        receipts = payload.get("loaded_receipts") if isinstance(payload, dict) else None
        receipt = receipts.get(lora_name) if isinstance(receipts, dict) else None
        return _validate_loaded_receipt(
            receipt,
            source_url=lora_path_posix,
            lora_name=lora_name,
            expected=effective_expected_identity,
        )

    async def _load_lora_adapter(admin_client: AsyncClient) -> AdapterIdentity | None:
        delay = 1.0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("LoRA load phase exceeded its 600-second deadline")
            logger.debug(f"Sending request to load LoRA adapter {lora_name} from {lora_path}")
            try:
                read_timeout = min(LORA_LOAD_READ_TIMEOUT_S, remaining)
                # Some vLLM workers finish the collective LoRA load but never
                # finish the HTTP response. Enforce one absolute deadline in
                # addition to httpx's idle read timeout.
                response = await asyncio.wait_for(
                    admin_client.post(
                        "/load_lora_adapter",
                        json={
                            "lora_name": lora_name,
                            "lora_path": lora_path_posix,
                            "require_delta": require_delta,
                            "prefetched": prefetched,
                        },
                        timeout=httpx.Timeout(
                            connect=min(10.0, read_timeout),
                            read=read_timeout,
                            write=min(60.0, read_timeout),
                            pool=min(10.0, read_timeout),
                        ),
                    ),
                    timeout=read_timeout,
                )
                response.raise_for_status()
                if not remote_adapter:
                    return None
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise RuntimeError(f"invalid LoRA load response from {admin_client.base_url}") from exc
                if not isinstance(payload, dict) or payload.get("status") != "ok":
                    raise RuntimeError(f"LoRA load failed on {admin_client.base_url}: {payload!r}")
                receipt = payload.get("loaded_receipt") if isinstance(payload, dict) else None
                return _validate_loaded_receipt(
                    receipt,
                    source_url=lora_path_posix,
                    lora_name=lora_name,
                    expected=effective_expected_identity,
                )
            except (TimeoutError, httpx.TimeoutException, httpx.TransportError) as exc:
                if remote_adapter:
                    try:
                        receipt = await _read_loaded_receipt(admin_client)
                    except Exception:
                        receipt = None
                    if receipt is not None:
                        logger.warning(
                            f"LoRA load response from {admin_client.base_url} did not finish, "
                            "but its exact loaded adapter receipt is available; accepting the load"
                        )
                        return receipt
                retry_error: BaseException = exc
            except Exception as exc:
                retry_error = exc

            if not _is_retryable_lora_error(retry_error):
                raise retry_error
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("LoRA load phase exceeded its 600-second deadline") from retry_error
            await asyncio.sleep(min(delay, remaining))
            delay = min(delay * 2.0, 10.0)

    tasks = [asyncio.create_task(_load_lora_adapter(admin_client)) for admin_client in admin_clients]
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        await _cancel_and_reap(tasks)
        raise
    return [result for result in results if result is not None]


async def unload_lora_adapter(admin_clients: list[AsyncClient], lora_name: str) -> None:
    """Make a HTTP post request to the vLLM server to unload a LoRA adapter."""
    logger = get_logger()

    async def _unload_lora_adapter(admin_client: AsyncClient) -> None:
        logger.debug(f"Sending request to unload LoRA adapter {lora_name}")
        response = await admin_client.post(
            "/v1/unload_lora_adapter",
            json={"lora_name": lora_name},
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0),
        )
        response.raise_for_status()
        # TODO: The first one can fail, but subsequent ones should succeed.
        # response.raise_for_status()

    tasks = [asyncio.create_task(_unload_lora_adapter(admin_client)) for admin_client in admin_clients]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        await _cancel_and_reap(tasks)
        raise


async def init_nccl_broadcast(
    admin_clients: list[AsyncClient],
    host: str,
    port: int,
    timeout: int,
    inference_world_size: int | None = None,
    quantize_in_weight_transfer: bool = False,
) -> None:
    """Initialize NCCL broadcast on all inference servers.

    Each admin client represents one vLLM server. The function computes
    per-server rank_offset and gpus_per_server so that every inference GPU
    gets a unique rank in the NCCL broadcast group.
    """
    logger = get_logger()

    if inference_world_size is None:
        inference_world_size = len(admin_clients)
        logger.warning(
            f"inference_world_size not provided, defaulting to {inference_world_size} (one GPU per admin client)"
        )

    gpus_per_server = inference_world_size // len(admin_clients)

    logger.info(
        f"Initializing NCCL broadcast: {len(admin_clients)} servers, "
        f"inference_world_size={inference_world_size}, gpus_per_server={gpus_per_server}"
    )

    async def _init_nccl_broadcast(admin_client: AsyncClient, rank_offset: int) -> None:
        try:
            response = await admin_client.post(
                "/init_broadcaster",
                json={
                    "host": host,
                    "port": port,
                    "rank_offset": rank_offset,
                    "inference_world_size": inference_world_size,
                    "timeout": timeout,
                    "quantize_in_weight_transfer": quantize_in_weight_transfer,
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning("The route /init_broadcaster does not exist. Skipping NCCL broadcast initialization.")
                return

    await asyncio.gather(
        *[
            _init_nccl_broadcast(admin_client, client_num * gpus_per_server)
            for client_num, admin_client in enumerate(admin_clients)
        ]
    )


async def init_nixl_broadcast(
    admin_clients: list[AsyncClient],
    host: str,
    port: int,
    timeout: int,
    inference_world_size: int,
    session_id: str,
) -> None:
    """Configure every vLLM worker for NIXL + ModelExpress pulls."""
    workers_per_server = inference_world_size // len(admin_clients)

    async def initialize(admin_client: AsyncClient, rank_offset: int) -> None:
        await _admin_post(
            admin_client,
            "/init_broadcaster",
            timeout_s=max(ADMIN_TIMEOUT_S, timeout),
            json={
                "host": host,
                "port": port,
                "rank_offset": rank_offset,
                "inference_world_size": inference_world_size,
                "timeout": timeout,
                "quantize_in_weight_transfer": False,
                "session_id": session_id,
            },
        )

    await asyncio.gather(
        *[initialize(admin_client, index * workers_per_server) for index, admin_client in enumerate(admin_clients)]
    )


async def prefill_logprobs(openai: AsyncOpenAI, model: str, token_ids: list[int]) -> list[float]:
    """Prefill-score ``token_ids`` under ``model`` via ``/inference/v1/generate``
    + ``prompt_logprobs`` (the prime-rl server-side extension in
    ``inference/vllm/serving_tokens.py``). Returns one logprob per token (0.0 for
    the leading token, which has no preceding context)."""
    from vllm.entrypoints.serve.disagg.protocol import GenerateResponse

    # `/inference/v1/generate` is mounted at server root, not under `/v1`: pass an
    # absolute URL so the SDK skips the base-url merge. vLLM's `GenerateResponse`
    # isn't an `openai.BaseModel`, so the SDK parse layer rejects it as `cast_to`;
    # `cast_to=httpx.Response` lets the SDK still build the request (auth, retries,
    # timeouts) and hand back the raw response for us to validate.
    base = str(openai.base_url).rstrip("/").removesuffix("/v1")
    http_response = await openai.post(
        f"{base}/inference/v1/generate",
        cast_to=httpx.Response,
        body={
            "model": model,
            "token_ids": token_ids,
            "sampling_params": {"max_tokens": 1, "temperature": 1.0, "top_p": 1.0, "prompt_logprobs": 1},
        },
    )
    response = GenerateResponse.model_validate_json(http_response.content)
    # `prompt_logprobs[i]` is a `{token_id: Logprob}` dict, or `None` for the
    # leading token (no preceding context). Flatten to `list[float]`.
    flat: list[float] = []
    for entry in response.prompt_logprobs or []:
        if not entry:
            flat.append(0.0)
            continue
        first = next(iter(entry.values()))
        lp = first.logprob if hasattr(first, "logprob") else first.get("logprob")
        flat.append(float(lp) if lp is not None else 0.0)
    return flat
