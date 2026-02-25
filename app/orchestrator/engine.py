"""
Orchestrator — the main pipeline driver.

Runs all agents in order, handles parallelism, retries, and status updates.
Think of it as the director of the whole editing session.

Flow:
  Ingest → (Speech + Vision in parallel, then Semantic) → Timeline → Cut Planner → Quality Review → Render
"""

import asyncio
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.agents.ingest import IngestAgent
from app.agents.speech import SpeechAgent
from app.agents.vision import VisionAgent
from app.agents.semantic import SemanticAgent
from app.agents.timeline_builder import TimelineBuilderAgent
from app.agents.cut_planner import CutPlannerAgent
from app.agents.quality_review import QualityReviewAgent
from app.agents.render import RenderAgent
from app.core.config import settings
from app.core.logging import get_logger, StageTimer
from app.core.workspace import JobWorkspace
from app.schemas.job import JobStatus, JobStage

logger = get_logger("orchestrator")


class PipelineOrchestrator:

    def __init__(self):
        self.ingest = IngestAgent()
        self.speech = SpeechAgent()
        self.vision = VisionAgent()
        self.semantic = SemanticAgent()
        self.timeline = TimelineBuilderAgent()
        self.cut_planner = CutPlannerAgent()
        self.quality_review = QualityReviewAgent()
        self.render = RenderAgent()

    async def run(self, job_id: str, workspace: JobWorkspace, status: JobStatus) -> None:
        timer = StageTimer(job_id)
        logger.info(f"[{job_id}] ═══ Pipeline starting ═══")
        logger.info(f"[{job_id}] Device: {settings.resolved_device} | LLM: {settings.LLM_MODEL}")

        try:
            # 1. ingest
            status.update(JobStage.INGESTING, 5, "Extracting audio and metadata...")
            workspace.save_artifact("job_status.json", status.model_dump())
            with timer.track("ingest"):
                await self._run_agent(self.ingest, job_id, workspace)

            # 2. parallel analysis
            status.update(JobStage.ANALYZING_SPEECH, 15, "AI agents analyzing speech, vision, semantics...")
            workspace.save_artifact("job_status.json", status.model_dump())
            with timer.track("parallel_analysis"):
                await self._run_parallel_analysis(job_id, workspace, status)

            # 3. timeline
            status.update(JobStage.BUILDING_TIMELINE, 60, "Building unified timeline...")
            workspace.save_artifact("job_status.json", status.model_dump())
            with timer.track("timeline_builder"):
                await self._run_agent(self.timeline, job_id, workspace)

            # 4. cut planning (LLM editor)
            status.update(JobStage.PLANNING_CUTS, 70, "AI editor making cut decisions...")
            workspace.save_artifact("job_status.json", status.model_dump())
            with timer.track("cut_planner"):
                await self._run_agent(self.cut_planner, job_id, workspace)

            # 5. quality review (LLM senior editor)
            status.update(JobStage.PLANNING_CUTS, 80, "Senior AI editor reviewing the plan...")
            workspace.save_artifact("job_status.json", status.model_dump())
            with timer.track("quality_review"):
                await self._run_agent(self.quality_review, job_id, workspace)

            # 6. render
            status.update(JobStage.RENDERING, 90, "Rendering final video...")
            workspace.save_artifact("job_status.json", status.model_dump())
            with timer.track("render"):
                await self._run_agent(self.render, job_id, workspace)

            # done!
            status.timings = timer.timings
            status.update(JobStage.COMPLETED, 100, "Processing complete!")
            workspace.save_artifact("job_status.json", status.model_dump())
            workspace.save_artifact("timing_report.json", timer.get_report())

            logger.info(f"[{job_id}] ═══ Pipeline done ═══")
            logger.info(f"[{job_id}] {timer.get_report()}")

        except Exception as e:
            logger.error(f"[{job_id}] Pipeline failed: {e}")
            logger.debug(traceback.format_exc())
            status.update(JobStage.FAILED, 0, f"Failed: {str(e)}")
            status.error = str(e)
            status.timings = timer.timings
            workspace.save_artifact("job_status.json", status.model_dump())
            raise

    async def _run_parallel_analysis(self, job_id, workspace, status):
        """Speech and Vision run at the same time. Semantic waits for the transcript."""
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=3) as pool:
            speech_future = loop.run_in_executor(pool, self._run_with_retry, self.speech, job_id, workspace, "speech")
            vision_future = loop.run_in_executor(pool, self._run_with_retry, self.vision, job_id, workspace, "vision")

            await speech_future  # semantic needs the transcript

            status.update(JobStage.ANALYZING_VISION, 35, "Speech done. Running semantic analysis...")
            workspace.save_artifact("job_status.json", status.model_dump())

            semantic_future = loop.run_in_executor(pool, self._run_with_retry, self.semantic, job_id, workspace, "semantic")
            await asyncio.gather(vision_future, semantic_future)

            status.update(JobStage.ANALYZING_SEMANTIC, 55, "All analysis complete.")
            workspace.save_artifact("job_status.json", status.model_dump())

    def _run_with_retry(self, agent, job_id, workspace, stage_name) -> dict:
        last_error = None
        for attempt in range(settings.MAX_RETRIES + 1):
            try:
                return agent.execute(job_id, workspace)
            except Exception as e:
                last_error = e
                if attempt < settings.MAX_RETRIES:
                    logger.warning(f"[{job_id}] {stage_name} attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(settings.RETRY_DELAY_SECONDS)
                    path = workspace.artifact_path(agent.output_artifact)
                    if path.exists():
                        path.unlink()

        raise RuntimeError(f"{stage_name} failed after {settings.MAX_RETRIES + 1} attempts: {last_error}")

    async def _run_agent(self, agent, job_id, workspace) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_with_retry, agent, job_id, workspace, agent.name)


_orchestrator: Optional[PipelineOrchestrator] = None


def _get_orchestrator() -> PipelineOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PipelineOrchestrator()
    return _orchestrator


async def run_pipeline(job_id: str, workspace: JobWorkspace, status: JobStatus) -> None:
    orchestrator = _get_orchestrator()
    await orchestrator.run(job_id, workspace, status)
