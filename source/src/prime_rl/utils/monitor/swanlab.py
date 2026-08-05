from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prime_rl.configs.shared import SwanlabConfig
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor.base import Monitor

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout


class SwanlabMonitor(Monitor):
    def __init__(
        self,
        config: SwanlabConfig,
        output_dir: Path | None = None,
        run_config: BaseConfig | None = None,
        keep_full_history: bool = True,
    ):
        self.history: list[dict[str, Any]] = []
        self.keep_full_history = keep_full_history
        self.enabled = int(os.environ.get("RANK", os.environ.get("DP_RANK", "0"))) == 0
        if not self.enabled:
            return

        import swanlab

        self.swanlab = swanlab
        self.run = swanlab.init(
            project=config.project,
            workspace=config.workspace,
            name=config.name,
            group=config.group,
            tags=config.tags,
            description=config.description,
            mode=config.mode,
            log_dir=str(output_dir) if output_dir else None,
            config=run_config.model_dump(mode="json") if run_config else None,
        )
        get_logger().info(f"Logging metrics to SwanLab run {self.run.id}")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self.history = [*self.history, metrics] if self.keep_full_history else [metrics]
        if self.enabled:
            self.swanlab.log(metrics, step=step)

    def log_samples(self, rollouts: list[Rollout], step: int) -> None:
        pass

    def log_eval_samples(self, rollouts: list[Rollout], env_name: str, step: int) -> None:
        pass

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        pass

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        pass

    def close(self) -> None:
        if self.enabled:
            self.swanlab.finish()
