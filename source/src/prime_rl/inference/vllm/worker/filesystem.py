import fcntl
import hashlib
import os
import shutil
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from torch.nn import Module
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader

from prime_rl.inference.vllm.worker.weight_transfer import load_weights_checkpoint_layerwise


def adapter_cache_path(source_url: str) -> Path:
    return Path(os.environ.get("PRIME_LORA_CACHE_DIR", "/tmp/prime-rl-lora")) / hashlib.sha256(
        source_url.encode()
    ).hexdigest()


def download(source_url: str, destination: Path) -> None:
    for attempt in range(10):
        try:
            with urllib.request.urlopen(urllib.request.Request(source_url, method="HEAD"), timeout=60) as response:
                total = int(response.headers["Content-Length"])
            offset = destination.stat().st_size if destination.exists() else 0
            if offset == total:
                return
            if offset > total:
                destination.unlink()
                offset = 0
            request = urllib.request.Request(source_url, headers={"Range": f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and response.status != 206:
                    destination.unlink(missing_ok=True)
                    raise OSError("adapter server did not honor byte range")
                with destination.open("ab" if offset else "wb") as output:
                    shutil.copyfileobj(response, output, length=16 * 1024 * 1024)
            if destination.stat().st_size != total:
                raise OSError(f"incomplete adapter file: {destination.name}")
            return
        except OSError:
            if attempt == 9:
                raise
            time.sleep(min(2**attempt, 10))


# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object


class FileSystemWeightUpdateWorker(Worker):
    """vLLM worker extension for updating weights in-place using shared filesystem."""

    def init_broadcaster(self) -> None:
        """Initialize the broadcaster."""
        ...

    def liveness_probe(self) -> None:
        """No-op RPC used by the API server liveness endpoint."""
        return None

    def stage_lora_adapter(self, source_url: str) -> None:
        destination = adapter_cache_path(source_url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (destination.parent / f"{destination.name}.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (destination / "READY").exists():
                return
            temporary = destination.with_name(f"{destination.name}.partial")
            temporary.mkdir(exist_ok=True)
            for filename in ("adapter_config.json", "adapter_model.safetensors"):
                download(f"{source_url.rstrip('/')}/{filename}", temporary / filename)
            (temporary / "READY").touch()
            shutil.rmtree(destination, ignore_errors=True)
            temporary.rename(destination)

    def update_weights_from_path(self, weight_path: str) -> None:
        """Update weights from a specified path in shared filesystem containing a HF-compatible checkpoint."""
        # Get vLLM model runner and model
        # When enforce_eager=True, model isn't wrapped by torch.compile so no .runnable attr
        model_runner = self.model_runner
        if hasattr(model_runner.model, "runnable"):
            model = model_runner.model.runnable
        else:
            model = model_runner.model
        assert isinstance(model, Module)

        # Get vLLM model loader
        model_loader = get_model_loader(self.load_config)
        assert isinstance(model_loader, DefaultModelLoader)
        local_source = DefaultModelLoader.Source(
            weight_path,
            revision=None,  # TODO: Check that this is correct or if we should use the default (model_config.revision)
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        weights_iterator = model_loader._get_weights_iterator(local_source)
        load_weights_checkpoint_layerwise(
            model,
            weights_iterator,
            self.model_runner.model_config,
            self.vllm_config,
        )
