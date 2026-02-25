from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStage(str, Enum):
    QUEUED = "queued"
    INGESTING = "ingesting"
    ANALYZING_SPEECH = "analyzing_speech"
    ANALYZING_VISION = "analyzing_vision"
    ANALYZING_SEMANTIC = "analyzing_semantic"
    BUILDING_TIMELINE = "building_timeline"
    PLANNING_CUTS = "planning_cuts"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatus(BaseModel):
    job_id: str
    stage: JobStage = JobStage.QUEUED
    progress: float = 0.0
    message: str = ""
    error: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    timings: dict[str, float] = Field(default_factory=dict)
    device: str = "cpu"

    def update(self, stage: JobStage, progress: float, message: str = "") -> "JobStatus":
        self.stage = stage
        self.progress = progress
        self.message = message
        self.updated_at = datetime.now().isoformat()
        return self


class JobResponse(BaseModel):
    job_id: str
    status: str = "queued"
    message: str = "Job created successfully"


class JobStatusResponse(BaseModel):
    job_id: str
    stage: str
    progress: float
    message: str
    error: Optional[str] = None
    created_at: str
    updated_at: str
    timings: dict[str, float] = Field(default_factory=dict)
    device: str = "cpu"
    output_ready: bool = False
