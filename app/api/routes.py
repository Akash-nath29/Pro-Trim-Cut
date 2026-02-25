from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.workspace import create_job, get_workspace, JobWorkspace
from app.core.logging import get_logger
from app.schemas.job import JobResponse, JobStatusResponse, JobStatus, JobStage
from app.graph.runner import run_graph   # ← LangGraph runner replaces old orchestrator

logger = get_logger("api")

router = APIRouter()

# in-memory job tracker (swap for redis in production)
_job_registry: dict[str, JobStatus] = {}


def get_job_status(job_id: str) -> JobStatus:
    if job_id in _job_registry:
        return _job_registry[job_id]

    try:
        ws = get_workspace(job_id)
        if ws.artifact_exists("job_status.json"):
            data = ws.load_artifact("job_status.json")
            status = JobStatus(**data)
            _job_registry[job_id] = status
            return status
    except FileNotFoundError:
        pass

    raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@router.post("/auto-trim", response_model=JobResponse)
async def auto_trim(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Raw MP4 video file"),
):
    if not file.filename or not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(status_code=400, detail="Invalid file type. Supported: .mp4, .mov, .avi, .mkv")

    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)

    if file_size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File too large. Max: {settings.MAX_UPLOAD_SIZE_MB}MB")

    job_id, workspace = create_job()
    workspace.save_upload(contents)

    status = JobStatus(
        job_id=job_id,
        stage=JobStage.QUEUED,
        message=f"Job created. File: {file.filename} ({file_size_mb:.1f}MB)",
        device=settings.resolved_device,
    )
    _job_registry[job_id] = status
    workspace.save_artifact("job_status.json", status.model_dump())

    logger.info(f"Job {job_id} created — {file.filename} ({file_size_mb:.1f}MB)")

    background_tasks.add_task(_run_job, job_id, workspace)

    return JobResponse(job_id=job_id)


async def _run_job(job_id: str, workspace: JobWorkspace):
    status = _job_registry[job_id]
    try:
        await run_graph(job_id, workspace, status)
    except Exception as e:
        logger.error(f"[{job_id}] Pipeline failed: {e}", exc_info=True)
        status.update(JobStage.FAILED, 0, f"Pipeline failed: {str(e)}")
        status.error = str(e)
        workspace.save_artifact("job_status.json", status.model_dump())


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    status = get_job_status(job_id)
    try:
        ws = get_workspace(job_id)
        output_ready = ws.output_video.exists()
    except FileNotFoundError:
        output_ready = False

    return JobStatusResponse(
        job_id=status.job_id,
        stage=status.stage.value,
        progress=status.progress,
        message=status.message,
        error=status.error,
        created_at=status.created_at,
        updated_at=status.updated_at,
        timings=status.timings,
        device=status.device,
        output_ready=output_ready,
    )


@router.get("/jobs/{job_id}/download")
async def download_result(job_id: str):
    status = get_job_status(job_id)

    if status.stage != JobStage.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Job not done yet. Current stage: {status.stage.value}")

    ws = get_workspace(job_id)
    if not ws.output_video.exists():
        raise HTTPException(status_code=404, detail="Output video not found")

    return FileResponse(
        path=str(ws.output_video),
        media_type="video/mp4",
        filename=f"trimmed_{job_id}.mp4",
    )
