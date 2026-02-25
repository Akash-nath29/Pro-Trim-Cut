"""
LangGraph shared state definition.

This is the single source of truth that flows through every node in the graph.
All agents read from and write to this state — nothing else is shared between them.
"""

from typing import Annotated, Optional
from typing_extensions import TypedDict
import operator


def _append_log(existing: list, new: list) -> list:
    """Custom reducer that appends new log entries to the existing list."""
    return existing + new


def _merge_timings(existing: dict, new: dict) -> dict:
    """Custom reducer that merges timing dicts (last writer wins per key)."""
    return {**existing, **new}


class GraphState(TypedDict):
    # ─── Job identity ───────────────────────────────────────────────
    job_id: str
    input_video_path: str          # absolute path to uploaded MP4

    # ─── Artifact paths (written by agents, read by downstream) ─────
    audio_path: Optional[str]                # audio.wav
    transcript_path: Optional[str]           # transcript.json
    vision_analysis_path: Optional[str]      # vision.json
    semantic_analysis_path: Optional[str]    # semantic.json
    timeline_path: Optional[str]             # timeline.json
    edit_plan_path: Optional[str]            # edit_plan.json
    quality_review_path: Optional[str]       # quality_review.json
    final_video_path: Optional[str]          # output.mp4

    # ─── Runtime context ────────────────────────────────────────────
    processing_mode: str           # "cpu" or "cuda"
    jobs_dir: str                  # root jobs directory

    # ─── Progress tracking (don't write these from parallel nodes!) ─
    current_stage: str
    progress: float                # 0–100

    # ─── Observability ──────────────────────────────────────────────
    logs: Annotated[list[str], _append_log]       # stage logs accumulate
    timings: Annotated[dict[str, float], _merge_timings]  # stage → seconds
    error: Optional[str]
