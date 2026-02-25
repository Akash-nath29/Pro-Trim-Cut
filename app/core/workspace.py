import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("workspace")


class JobWorkspace:
    """Each job gets its own folder to store all artifacts and video files."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.root = settings.storage_dir / job_id
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def input_video(self) -> Path:
        return self.root / "input.mp4"

    @property
    def audio_path(self) -> Path:
        return self.root / "audio.wav"

    @property
    def output_video(self) -> Path:
        return self.root / "output.mp4"

    def artifact_path(self, name: str) -> Path:
        return self.root / name

    def save_artifact(self, name: str, data: Any) -> Path:
        path = self.artifact_path(name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.debug(f"[{self.job_id}] Saved artifact: {name}")
        return path

    def load_artifact(self, name: str) -> Any:
        path = self.artifact_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def artifact_exists(self, name: str) -> bool:
        return self.artifact_path(name).exists()

    def save_upload(self, file_data: bytes) -> Path:
        with open(self.input_video, "wb") as f:
            f.write(file_data)
        logger.info(f"[{self.job_id}] Saved input video: {self.input_video}")
        return self.input_video

    def cleanup(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
            logger.info(f"[{self.job_id}] Workspace cleaned up")


def create_job() -> tuple[str, JobWorkspace]:
    job_id = str(uuid.uuid4())[:8]
    workspace = JobWorkspace(job_id)
    logger.info(f"Created job workspace: {job_id}")
    return job_id, workspace


def get_workspace(job_id: str) -> JobWorkspace:
    workspace = JobWorkspace(job_id)
    if not workspace.root.exists():
        raise FileNotFoundError(f"Job workspace not found: {job_id}")
    return workspace
