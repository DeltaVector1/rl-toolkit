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


if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object


class FileSystemWeightUpdateWorker(Worker):
    def init_broadcaster(self) -> None:
        ...

    def liveness_probe(self) -> None:
        return None

    def stage_lora_adapter(self, source_url: str) -> None:
        destination = adapter_cache_path(source_url)
        if (destination / "READY").exists():
            return
        if self.rank:
            deadline = time.monotonic() + 600
            while not (destination / "READY").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"adapter download timed out: {source_url}")
                time.sleep(0.25)
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (destination.parent / f"{destination.name}.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (destination / "READY").exists():
                return
            temporary = destination.with_name(f"{destination.name}.{os.getpid()}")
            shutil.rmtree(temporary, ignore_errors=True)
            temporary.mkdir()
            for filename in ("adapter_config.json", "adapter_model.safetensors"):
                with urllib.request.urlopen(f"{source_url.rstrip('/')}/{filename}", timeout=600) as response:
                    with (temporary / filename).open("wb") as output:
                        shutil.copyfileobj(response, output, length=16 * 1024 * 1024)
                        if output.tell() != int(response.headers["Content-Length"]):
                            raise OSError(f"incomplete adapter file: {filename}")
            (temporary / "READY").touch()
            shutil.rmtree(destination, ignore_errors=True)
            temporary.rename(destination)

    def update_weights_from_path(self, weight_path: str) -> None:
        model_runner = self.model_runner
        model = model_runner.model.runnable if hasattr(model_runner.model, "runnable") else model_runner.model
        assert isinstance(model, Module)
        model_loader = get_model_loader(self.load_config)
        assert isinstance(model_loader, DefaultModelLoader)
        load_weights_checkpoint_layerwise(
            model,
            model_loader._get_weights_iterator(
                DefaultModelLoader.Source(
                    weight_path,
                    revision=None,
                    prefix="",
                    fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
                    allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
                )
            ),
            self.model_runner.model_config,
            self.vllm_config,
        )
