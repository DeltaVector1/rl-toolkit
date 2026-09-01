from __future__ import annotations

import asyncio
import math
import os
import time
import tomllib
from collections.abc import Mapping
from itertools import cycle
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx
import verifiers.v1 as vf
from httpx import AsyncClient
from openai import AsyncOpenAI, NotFoundError
from renderers import RendererConfig
from tenacity import AsyncRetrying, retry, retry_if_exception, stop_after_attempt, stop_after_delay, wait_exponential
from verifiers.v1.clients.config import EvalClientConfig, TrainClientConfig

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.logger import get_logger

# Identity tuple used by ``select_train_client`` to key load counts. ``base_url``
# distinguishes servers; ``X-data-parallel-rank`` distinguishes DP shards within a
# server, since the router uses that header to route to specific GPU ranks.
ClientIdentity = tuple[str, str | None]


def client_identity(client: vf.ClientConfig) -> ClientIdentity:
    """Stable identity for load balancing across inference clients."""
    return (client.base_url, client.headers.get("X-data-parallel-rank"))


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
    ) -> None:
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
    ) -> None:
        await update_weights(
            self._admin_clients,
            weight_dir,
            lora_name=lora_name,
            step=step,
            require_delta=require_delta,
            prefetched=prefetched,
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
            timeout=httpx.Timeout(None),
        )

    return [_setup_admin_client(base_url) for base_url in urls]


async def maybe_check_has_model(
    admin_clients: list[AsyncClient], model_name: str, skip_model_check: bool = False
) -> None:
    if skip_model_check:
        return
    logger = get_logger()
    logger.debug(f"Checking if model {model_name} is in the inference pool")
    results = await asyncio.gather(*[admin_client.get("/v1/models") for admin_client in admin_clients])
    for admin_client, result in zip(admin_clients, results):
        models = result.json()["data"]
        if not any(model["id"] == model_name for model in models):
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
        wait_time = 0
        logger.debug("Starting pinging /health to check health")
        while wait_time < timeout:
            try:
                await admin_client.get("/health")
                logger.debug(f"Inference pool is ready after {wait_time} seconds")
                return
            except NotFoundError:
                logger.warning("The route /health does not exist. Skipping health check.")
                return
            except Exception as e:
                if wait_time % log_interval == 0 and wait_time > 0:
                    logger.warning(
                        f"Inference server was not reached after {wait_time} seconds (Error: {e}) on {admin_client.base_url}"
                    )
                await asyncio.sleep(interval)
                wait_time += interval
        msg = f"Inference server is not ready after {wait_time} (>{timeout}) seconds. Aborting..."
        logger.error(msg)
        raise TimeoutError(msg)

    await asyncio.gather(*[_check_health(admin_client) for admin_client in admin_clients])


NCCL_READY_MARKER = "NCCL_READY"


def _is_retryable_admin_error(exception: BaseException) -> bool:
    """Check if an exception should trigger a retry for an admin op (pause/resume/update_weights)."""
    if isinstance(exception, httpx.HTTPStatusError):
        # Retry on transient server errors (5xx, e.g. engine briefly unresponsive);
        # client errors (4xx) won't fix themselves on retry.
        return exception.response.status_code >= 500
    # Retry on transport-level failures (timeouts, connection resets, etc.) so the
    # per-attempt read timeout below turns a stuck server into a bounded retry loop
    # instead of hanging forever on the global timeout=None admin client.
    if isinstance(exception, (httpx.TimeoutException, httpx.TransportError)):
        return True
    return False


# Per-attempt read timeout for admin ops, overridable per call. The admin
# AsyncClient uses `timeout=None`, so without this a stuck server would hang the
# weight update forever: the read timeout converts a hang into a TimeoutException
# that tenacity retries. Sized for `/pause`, which drains in-flight requests
# (mode="keep") and so can legitimately take a while.
ADMIN_TIMEOUT_S = 300.0
# `/update_weights` runs a collective NCCL receive across all DP workers, which
# can take longer than the other admin ops.
UPDATE_WEIGHTS_TIMEOUT_S = 720.0
FULL_CHECKPOINT_STAGE_TIMEOUT_S = 7200.0
# A TP8 LoRA load can leave the engine busy after the load request has completed.
# Keep resume separate from the shorter pause and update budgets.
RESUME_ATTEMPT_TIMEOUT_S = 900.0
RESUME_TOTAL_TIMEOUT_S = 1800.0
RESUME_CONFIRM_TIMEOUT_S = 120.0
RESUME_PROBE_TIMEOUT_S = 10.0
RESUME_PROBE_INTERVAL_S = 2.0


async def _admin_post(
    client: AsyncClient,
    path: str,
    *,
    timeout_s: float = ADMIN_TIMEOUT_S,
    total_timeout_s: float | None = None,
    **kwargs,
) -> None:
    """POST an admin op with a bounded per-attempt timeout, retrying transient errors.

    The total wall-clock budget across all retries is twice the per-attempt
    timeout unless the caller provides an explicit budget.
    """
    retry_budget = 2 * timeout_s if total_timeout_s is None else total_timeout_s
    async for attempt in AsyncRetrying(
        retry=retry_if_exception(_is_retryable_admin_error),
        stop=stop_after_delay(retry_budget) | stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    ):
        with attempt:
            response = await client.post(
                path,
                timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=60.0, pool=10.0),
                **kwargs,
            )
            response.raise_for_status()


async def _admin_post_response(
    client: AsyncClient,
    path: str,
    *,
    timeout_s: float = ADMIN_TIMEOUT_S,
    total_timeout_s: float | None = None,
    **kwargs,
) -> httpx.Response:
    """POST an admin operation and return its successful response."""
    retry_budget = 2 * timeout_s if total_timeout_s is None else total_timeout_s
    async for attempt in AsyncRetrying(
        retry=retry_if_exception(_is_retryable_admin_error),
        stop=stop_after_delay(retry_budget) | stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    ):
        with attempt:
            response = await client.post(
                path,
                timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=60.0, pool=10.0),
                **kwargs,
            )
            response.raise_for_status()
            return response
    raise RuntimeError(f"admin operation did not complete: {path}")


async def _pause_engines(admin_clients: list[AsyncClient], *, step: int) -> None:
    """Abort registered requests and pause every inference scheduler.

    The dispatcher has preserved every prompt that needs regeneration.
    Engine-native abort is still required because cancelling an HTTP client can
    leave its decode request registered for several engine steps.
    """
    logger = get_logger()
    logger.info(f"Updating policy in-flight to v{step}")
    await asyncio.gather(
        *[_admin_post(client, "/pause", params={"mode": "abort", "clear_cache": "false"}) for client in admin_clients]
    )
    logger.debug("All inference engines paused")


async def _resume_engines(admin_clients: list[AsyncClient]) -> None:
    """Resume all inference engines after weight update.

    Resuming is idempotent. A timeout does not prove that the request failed:
    the engine can finish the request after the HTTP response deadline. Confirm
    the server state before reporting a failed policy update.
    """
    logger = get_logger()

    async def _read_resume_state(client: AsyncClient) -> tuple[str | None, bool, bool]:
        """Return state, route availability, and explicit route absence."""
        try:
            response = await client.get(
                "/resume_status",
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=RESUME_PROBE_TIMEOUT_S,
                    write=10.0,
                    pool=10.0,
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

    async def _is_live(client: AsyncClient) -> bool:
        try:
            response = await client.get(
                "/liveness",
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=RESUME_PROBE_TIMEOUT_S,
                    write=10.0,
                    pool=10.0,
                ),
            )
        except (TimeoutError, httpx.TimeoutException, httpx.TransportError):
            return False
        return response.status_code == 200

    async def _confirm_resume(client: AsyncClient) -> tuple[bool, bool]:
        deadline = asyncio.get_running_loop().time() + RESUME_CONFIRM_TIMEOUT_S
        state_endpoint_available = False
        while True:
            state, route_available, route_missing = await _read_resume_state(client)
            if route_available:
                state_endpoint_available = True
                if state == "running" and await _is_live(client):
                    return True, True
            elif route_missing and await _is_live(client):
                # Older servers have no state route. Liveness is weaker evidence,
                # so the log must identify that this path was used.
                return True, False

            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return False, state_endpoint_available
            await asyncio.sleep(min(RESUME_PROBE_INTERVAL_S, remaining))

    async def _resume_one(client: AsyncClient) -> None:
        try:
            await _admin_post(
                client,
                "/resume",
                timeout_s=RESUME_ATTEMPT_TIMEOUT_S,
                total_timeout_s=RESUME_TOTAL_TIMEOUT_S,
            )
        except (TimeoutError, httpx.TimeoutException, httpx.TransportError) as exc:
            confirmed, state_endpoint_available = await _confirm_resume(client)
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

    await asyncio.gather(*[_resume_one(client) for client in admin_clients])
    logger.debug("All inference engines resumed")


async def stage_lora_adapter(
    admin_clients: list[AsyncClient],
    lora_path: Path | str,
    *,
    require_delta: bool = False,
    origin_base_urls: list[str] | None = None,
) -> dict[str, float | int | None]:
    """Stage one adapter copy per inference host.

    A LoRA delta is valid only for the adjacent policy version. The receiver
    checks the manifest and its verified base. This function sends that
    requirement explicitly and returns measured staging time for each host.
    """
    lora_path_posix = str(lora_path)
    if not lora_path_posix.startswith(("http://", "https://")):
        return {"wire_stage_seconds": 0.0, "host_count": 0}
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
    ) -> dict[str, float | None]:
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
        if remaining_delta_seconds is not None:
            payload["stage_timeout_seconds"] = remaining_delta_seconds
        response = await _admin_post_response(
            client,
            "/stage_lora_adapter",
            json=payload,
            timeout_s=stage_timeout_s,
            total_timeout_s=stage_total_timeout_s,
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"invalid adapter stage response from {client.base_url}") from exc
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise RuntimeError(f"adapter stage failed on {client.base_url}: {payload!r}")
        stage_seconds = payload.get("stage_seconds")
        if not isinstance(stage_seconds, (int, float)) or stage_seconds < 0:
            stage_seconds = time.perf_counter() - started
        reconstruct_seconds = payload.get("reconstruct_seconds")
        if not isinstance(reconstruct_seconds, (int, float)) or reconstruct_seconds < 0:
            reconstruct_seconds = None
        return {
            "wire_stage_seconds": float(stage_seconds),
            "reconstruct_seconds": float(reconstruct_seconds)
            if reconstruct_seconds is not None
            else None,
            "request_seconds": time.perf_counter() - started,
        }

    started = time.perf_counter()
    delta_deadline = started + _delta_stage_timeout_s() if require_delta else None
    predictive_transport = False
    if require_delta:
        remaining = _delta_stage_timeout_s() - (time.perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError("adapter delta staging exceeded its time limit")
        try:
            async with httpx.AsyncClient(follow_redirects=False) as probe:
                response = await probe.get(
                    f"{lora_path_posix.rstrip('/')}/adapter_manifest.json",
                    timeout=httpx.Timeout(
                        connect=remaining,
                        read=remaining,
                        write=remaining,
                        pool=remaining,
                    ),
                )
                response.raise_for_status()
                manifest = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError("cannot determine adapter transport before staging") from exc
        if not isinstance(manifest, dict):
            raise RuntimeError("adapter transport manifest is not an object")
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
        remaining = delta_deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("adapter delta staging exceeded its time limit")
        tasks = [
            asyncio.create_task(
                _stage_one(client, remaining_delta_seconds=remaining)
            )
            for group in groups
            for client in group
        ]
        try:
            staged = await asyncio.wait_for(asyncio.gather(*tasks), timeout=remaining)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
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
            if delta_deadline is None:
                results.extend(await asyncio.gather(*(_stage_one(client) for client in group)))
            else:
                remaining = delta_deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError("adapter delta staging exceeded its time limit")
                try:
                    staged = await asyncio.wait_for(
                        asyncio.gather(
                            *(
                                _stage_one(client, remaining_delta_seconds=remaining)
                                for client in group
                            )
                        ),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError as exc:
                    raise TimeoutError("adapter delta staging exceeded its time limit") from exc
                results.extend(staged)
            group_seconds.append(time.perf_counter() - group_started)
    origin_stage_seconds = group_seconds[0] if group_seconds else 0.0
    peer_stage_seconds = sum(group_seconds[1:])
    wire_stage_seconds = time.perf_counter() - started
    reconstruct_values = [
        float(result["reconstruct_seconds"])
        for result in results
        if result["reconstruct_seconds"] is not None
    ]
    return {
        "wire_stage_seconds": wire_stage_seconds,
        "origin_stage_seconds": origin_stage_seconds,
        "peer_stage_seconds": peer_stage_seconds,
        "reconstruct_seconds": max(reconstruct_values) if reconstruct_values else None,
        "host_count": len(hosts),
        "origin_host_count": origin_count,
        "peer_host_count": len(hosts) - origin_count,
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

    await asyncio.gather(*[_stage(client) for client in admin_clients])


async def update_weights(
    admin_clients: list[AsyncClient],
    weight_dir: Path | str | None,
    lora_name: str | None = None,
    step: int = 0,
    require_delta: bool = False,
    prefetched: bool = False,
) -> None:
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
        # vLLM's fused MoE LoRA load is a worker collective. Running it while
        # decode requests remain registered repeatedly stalled h200b in x64p:
        # the token counter stopped and EngineCore logged one unavailable
        # shared-memory broadcast block per minute until the watchdog restart.
        # The dispatcher's version-pending hook drains registered requests
        # before this function runs. Pause here to prevent the engine from
        # admitting work during the collective. Always resume, including after
        # a bounded load failure.
        await _pause_engines(admin_clients, step=step)
        try:
            await load_lora_adapter(
                admin_clients,
                lora_name,
                weight_dir,
                require_delta=require_delta,
                prefetched=prefetched,
            )
        finally:
            await _resume_engines(admin_clients)
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

            await asyncio.gather(*[_update(admin_client) for admin_client in admin_clients])
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
    if isinstance(exception, (httpx.TimeoutException, httpx.TransportError)):
        return True
    return False


# Adapter files cross the control tunnel before vLLM loads them. A rank-32 GLM
# adapter is about 28.5 GiB, so staging needs a longer transport budget than
# the local adapter load.
LORA_LOAD_READ_TIMEOUT_S = 900.0
LORA_LOAD_TOTAL_TIMEOUT_S = 1200.0
LORA_STAGE_ATTEMPT_TIMEOUT_S = 3600.0
LORA_STAGE_TOTAL_TIMEOUT_S = 7200.0


def _delta_stage_timeout_s() -> float:
    configured = os.environ.get("PRIME_LORA_DELTA_STAGE_TIMEOUT_S")
    if configured is not None:
        try:
            value = float(configured)
        except ValueError as exc:
            raise RuntimeError("PRIME_LORA_DELTA_STAGE_TIMEOUT_S must be a number") from exc
    else:
        config_path = Path(
            os.environ.get(
                "PRIME_RL_OPERATIONS_CONFIG",
                "/data/glm52-rl/deployment/configs/operations.toml",
            )
        )
        try:
            with config_path.open("rb") as stream:
                config = tomllib.load(stream)
        except FileNotFoundError as exc:
            raise RuntimeError(f"operations config is missing: {config_path}") from exc
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise RuntimeError(f"cannot read operations config: {config_path}") from exc
        else:
            limits = config.get("limits")
            if not isinstance(limits, dict) or "adapter_delta_stage_timeout_seconds" not in limits:
                raise RuntimeError("operations config has no adapter delta stage timeout")
            value = limits["adapter_delta_stage_timeout_seconds"]
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
) -> None:
    """Make a HTTP post request to the vLLM server to load a LoRA adapter.

    Uses our wrapper around vLLM's /v1/load_lora_adapter. The prefix cache is not reset
    here; the orchestrator salts it per weight version (see ``orchestrator/envs.py``) so
    KV computed under old weights is never reused.

    Retries with exponential backoff if the adapter files are not found,
    which can happen due to NFS propagation delays.
    """
    logger = get_logger()
    lora_path_posix = str(lora_path)

    @retry(
        retry=retry_if_exception(_is_retryable_lora_error),
        # An engine restart takes 5 to 10 minutes. Keep retrying for the
        # complete wall-clock budget instead of stopping after ten fast
        # connection failures.
        stop=stop_after_delay(LORA_LOAD_TOTAL_TIMEOUT_S),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _load_lora_adapter(admin_client: AsyncClient) -> None:
        logger.debug(f"Sending request to load LoRA adapter {lora_name} from {lora_path}")
        try:
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
                        connect=10.0,
                        read=LORA_LOAD_READ_TIMEOUT_S,
                        write=60.0,
                        pool=10.0,
                    ),
                ),
                timeout=LORA_LOAD_READ_TIMEOUT_S,
            )
            response.raise_for_status()
        except (TimeoutError, httpx.TimeoutException, httpx.TransportError):
            if await model_is_visible(admin_client, lora_name):
                logger.warning(
                    f"LoRA load response from {admin_client.base_url} did not finish, "
                    f"but adapter {lora_name} is visible; accepting the completed load"
                )
                return
            raise

    await asyncio.gather(*[_load_lora_adapter(admin_client) for admin_client in admin_clients])


async def unload_lora_adapter(admin_clients: list[AsyncClient], lora_name: str) -> None:
    """Make a HTTP post request to the vLLM server to unload a LoRA adapter."""
    logger = get_logger()

    async def _unload_lora_adapter(admin_client: AsyncClient) -> None:
        logger.debug(f"Sending request to unload LoRA adapter {lora_name}")
        await admin_client.post("/v1/unload_lora_adapter", json={"lora_name": lora_name})
        # TODO: The first one can fail, but subsequent ones should succeed.
        # response.raise_for_status()

    await asyncio.gather(*[_unload_lora_adapter(admin_client) for admin_client in admin_clients])


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
