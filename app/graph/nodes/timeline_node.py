"""
Timeline Node — LangGraph node that merges all signals into a unified segment timeline.

Depends on: transcript_path, vision_analysis_path, semantic_analysis_path
Tools used: merge_multimodal_signals, build_segments
Writes to state: timeline_path
"""

import json
import time
from pathlib import Path

from app.graph.state import GraphState
from app.tools.timeline_tools import merge_multimodal_signals, build_segments
from app.core.logging import get_logger

logger = get_logger("node.timeline")


def timeline_node(state: GraphState) -> dict:
    """LangGraph node: merge speech + vision + semantic into a unified timeline."""
    job_id = state["job_id"]
    jobs_dir = state["jobs_dir"]

    job_dir = Path(jobs_dir) / job_id
    timeline_out = str(job_dir / "timeline.json")

    if Path(timeline_out).exists():
        logger.info(f"[{job_id}] ⏭ timeline already done, using cached result")
        return {"timeline_path": timeline_out, "logs": [f"[{job_id}] timeline: cached"]}

    logger.info(f"[{job_id}] ▶ timeline_builder starting")
    t0 = time.perf_counter()

    # load artifacts
    with open(state["transcript_path"], encoding="utf-8") as f:
        transcript = json.load(f)
    with open(state["vision_analysis_path"], encoding="utf-8") as f:
        vision_data = json.load(f)
    with open(state["semantic_analysis_path"], encoding="utf-8") as f:
        semantic_data = json.load(f)

    metadata_path = job_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    # ── Tool 1: merge signals ─────────────────────────────────────────────
    merged = merge_multimodal_signals.invoke({
        "transcript": transcript,
        "vision_data": vision_data,
        "semantic_data": semantic_data,
        "metadata": metadata,
    })

    # ── Tool 2: build segments ────────────────────────────────────────────
    timeline = build_segments.invoke({"merged": merged})

    segments = timeline["segments"]
    duration = timeline["total_duration"]

    if segments:
        first_speech = segments[0]["start"]
        if first_speech > 1.0:
            logger.info(f"[{job_id}] Dead time at start: {first_speech:.1f}s")
        last_speech = segments[-1]["end"]
        if duration - last_speech > 1.0:
            logger.info(f"[{job_id}] Dead time at end: {duration - last_speech:.1f}s")

    logger.info(f"[{job_id}] Built timeline with {len(segments)} segments")

    with open(timeline_out, "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2, ensure_ascii=False)

    elapsed = round(time.perf_counter() - t0, 3)
    logger.info(f"[{job_id}] ✓ timeline_builder done in {elapsed:.2f}s")

    return {
        "timeline_path": timeline_out,
        "current_stage": "building_timeline",
        "progress": 60.0,
        "logs": [f"[{job_id}] timeline: {len(segments)} segments in {elapsed}s"],
        "timings": {"timeline": elapsed},
    }
