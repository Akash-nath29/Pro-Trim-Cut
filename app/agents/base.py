import time
import traceback
from abc import ABC, abstractmethod

from app.core.workspace import JobWorkspace
from app.core.logging import get_logger


class BaseAgent(ABC):
    """Every agent in the pipeline inherits from this."""

    def __init__(self):
        self.logger = get_logger(f"agent.{self.name}")

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def output_artifact(self) -> str: ...

    @abstractmethod
    def process(self, job_id: str, workspace: JobWorkspace) -> dict: ...

    def execute(self, job_id: str, workspace: JobWorkspace) -> dict:
        self.logger.info(f"[{job_id}] ▶ {self.name} starting")
        start = time.perf_counter()

        try:
            # skip if we already ran this (resume support)
            if workspace.artifact_exists(self.output_artifact):
                self.logger.info(f"[{job_id}] ⏭ {self.name} already done, loading cached result")
                return workspace.load_artifact(self.output_artifact)

            result = self.process(job_id, workspace)
            workspace.save_artifact(self.output_artifact, result)

            elapsed = time.perf_counter() - start
            self.logger.info(f"[{job_id}] ✓ {self.name} done in {elapsed:.2f}s")
            return result

        except Exception as e:
            elapsed = time.perf_counter() - start
            self.logger.error(f"[{job_id}] ✗ {self.name} failed after {elapsed:.2f}s: {e}")
            self.logger.debug(traceback.format_exc())
            raise
