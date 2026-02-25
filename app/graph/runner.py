"""
Graph Runner — async wrapper that drives the LangGraph pipeline.

Converts async FastAPI context → LangGraph sync execution in a thread pool,
and streams status updates back to the JobStatus object throughout.
"""

import asyncio
import time
import traceback
from typing import Optional

from app.graph.pipeline import get_compiled_graph
from app.graph.state import GraphState
from app.core.config import settings
from app.core.logging import get_logger
from app.core.workspace import JobWorkspace
from app.schemas.job import JobStatus, JobStage

logger = get_logger("graph.runner")

# Maps internal graph stage names → JobStage + progress + human message
_STAGE_MAP = {
    "ingest": (JobStage.INGESTING, 10, "Extracting audio and reading metadata..."),
    "speech": (JobStage.ANALYZING_SPEECH, 20, "Transcribing speech with AI..."),
    "vision": (JobStage.ANALYZING_VISION, 30, "Analyzing faces and gaze..."),
    "semantic": (JobStage.ANALYZING_SEMANTIC, 50, "Running semantic duplicate detection..."),
    "timeline": (JobStage.BUILDING_TIMELINE, 60, "Building unified edit timeline..."),
    "cut_planner": (JobStage.PLANNING_CUTS, 75, "AI editor planning cuts..."),
    "quality_review": (JobStage.PLANNING_CUTS, 82, "Senior AI editor reviewing the plan..."),
    "render": (JobStage.RENDERING, 90, "Rendering final video..."),
}


async def run_graph(job_id: str, workspace: JobWorkspace, status: JobStatus) -> None:
    """Run the full LangGraph pipeline for a job, updating `status` at each stage.

    This is the single entry-point called by the FastAPI background task.
    It runs the synchronous LangGraph in a thread pool so the event loop stays free.

    Args:
        job_id: Unique job identifier.
        workspace: JobWorkspace with paths to all artifacts.
        status: Mutable JobStatus object — updated live so the API can report progress.
    """
    loop = asyncio.get_event_loop()

    logger.info(f"[{job_id}] ═══ LangGraph pipeline starting ═══")
    logger.info(f"[{job_id}] Device: {settings.resolved_device} | LLM: {settings.LLM_MODEL}")

    initial_state: GraphState = {
        "job_id": job_id,
        "input_video_path": str(workspace.input_video),
        "audio_path": None,
        "transcript_path": None,
        "vision_analysis_path": None,
        "semantic_analysis_path": None,
        "timeline_path": None,
        "edit_plan_path": None,
        "quality_review_path": None,
        "final_video_path": None,
        "processing_mode": settings.resolved_device,
        "jobs_dir": str(settings.storage_dir),
        "current_stage": "starting",
        "progress": 5.0,
        "logs": [f"[{job_id}] Pipeline started"],
        "timings": {},
        "error": None,
    }

    try:
        # Run the compiled graph in a thread pool so asyncio isn't blocked
        final_state = await loop.run_in_executor(
            None,
            _run_sync_with_status,
            initial_state, status, workspace,
        )

        # Collect timing report
        timings = final_state.get("timings", {})
        status.timings = timings
        status.update(JobStage.COMPLETED, 100, "Processing complete!")
        workspace.save_artifact("job_status.json", status.model_dump())

        total = sum(timings.values())
        logger.info(f"[{job_id}] ═══ Pipeline done in {total:.1f}s ═══")
        logger.info(f"[{job_id}] Stage timings: {timings}")

    except Exception as e:
        logger.error(f"[{job_id}] Pipeline failed: {e}")
        logger.debug(traceback.format_exc())
        status.update(JobStage.FAILED, 0, f"Failed: {str(e)}")
        status.error = str(e)
        workspace.save_artifact("job_status.json", status.model_dump())
        raise


def _run_sync_with_status(
    initial_state: GraphState,
    status: JobStatus,
    workspace: JobWorkspace,
) -> GraphState:
    """Synchronous graph execution — called from a thread pool via run_in_executor.

    Streams node events from LangGraph so we can push status updates
    to the JobStatus object as each node completes.
    """
    graph = get_compiled_graph()
    job_id = initial_state["job_id"]
    final_state = initial_state.copy()

    t_pipeline = time.perf_counter()

    # Stream node-level events so we can update the status live
    for event in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "__interrupt__" or node_output is None:
                continue

            # merge the node's partial state into our running state
            for k, v in node_output.items():
                if k == "logs" and isinstance(v, list):
                    final_state.setdefault("logs", [])
                    final_state["logs"] = final_state["logs"] + v
                elif k == "timings" and isinstance(v, dict):
                    final_state.setdefault("timings", {})
                    final_state["timings"] = {**final_state["timings"], **v}
                else:
                    final_state[k] = v

            # push status update to the API layer
            if node_name in _STAGE_MAP:
                stage, progress, message = _STAGE_MAP[node_name]
                status.update(stage, progress, message)
                workspace.save_artifact("job_status.json", status.model_dump())
                logger.info(f"[{job_id}] ✓ Node '{node_name}' completed → {stage.value} ({progress}%)")

            # surface errors from nodes
            if node_output.get("error"):
                raise RuntimeError(node_output["error"])

    return final_state
