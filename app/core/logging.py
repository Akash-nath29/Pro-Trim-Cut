import logging
import sys
import time
from contextlib import contextmanager


def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("mediapipe").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"autotrim.{name}")


class StageTimer:
    """Tracks how long each pipeline stage takes."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.timings: dict[str, float] = {}
        self.logger = get_logger("timer")

    @contextmanager
    def track(self, stage_name: str):
        self.logger.info(f"[{self.job_id}] ▶ Starting: {stage_name}")
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timings[stage_name] = round(elapsed, 3)
            self.logger.info(f"[{self.job_id}] ✓ Completed: {stage_name} in {elapsed:.2f}s")

    def get_report(self) -> dict:
        total = sum(self.timings.values())
        return {
            "job_id": self.job_id,
            "stages": self.timings,
            "total_seconds": round(total, 3),
        }


def log_device_info() -> None:
    logger = get_logger("system")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            logger.info(f"🖥️  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            logger.info(f"🔧 CUDA: {torch.version.cuda}")
            mode = "GPU"
        else:
            logger.info("🖥️  No GPU detected — running on CPU")
            mode = "CPU"
    except ImportError:
        logger.info("🖥️  PyTorch not available — running on CPU")
        mode = "CPU"

    logger.info(f"⚡ Processing mode: {mode}")
    return mode
