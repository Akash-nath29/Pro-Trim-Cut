"""
Render Node — LangGraph node that produces the final trimmed MP4.

Depends on: edit_plan_path
Tools used: render_final_video
Writes to state: final_video_path
"""

import json
import time
from pathlib import Path

from app.graph.state import GraphState
from app.tools.render_tools import render_final_video
from app.core.logging import get_logger

logger = get_logger("node.render")


def render_node(state: GraphState) -> dict:
    """LangGraph node: render the final video from the approved edit plan."""
    job_id = state["job_id"]
    jobs_dir = state["jobs_dir"]

    job_dir = Path(jobs_dir) / job_id
    output_path = str(job_dir / "output.mp4")

    if Path(output_path).exists():
        logger.info(f"[{job_id}] ⏭ render already done, using cached result")
        return {
            "final_video_path": output_path,
            "current_stage": "completed",
            "progress": 100.0,
            "logs": [f"[{job_id}] render: cached"],
        }

    logger.info(f"[{job_id}] ▶ render starting")
    t0 = time.perf_counter()

    with open(state["edit_plan_path"], encoding="utf-8") as f:
        edit_plan = json.load(f)

    metadata_path = job_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    keep_ranges = edit_plan["keep_ranges"]
    input_path = state["input_video_path"]

    if keep_ranges:
        logger.info(f"[{job_id}] Rendering {len(keep_ranges)} segments...")
    else:
        logger.info(f"[{job_id}] No cuts needed — copying original video as output")

    # ── Tool: render ──────────────────────────────────────────────────────
    result = render_final_video.invoke({
        "job_id": job_id,
        "input_path": input_path,
        "output_path": output_path,
        "keep_ranges": keep_ranges,
        "jobs_dir": jobs_dir,
    })

    size_mb = result["output_size_bytes"] / (1024 * 1024)
    elapsed = round(time.perf_counter() - t0, 3)

    logger.info(
        f"[{job_id}] ✓ render done in {elapsed:.2f}s — "
        f"{size_mb:.1f}MB, {result['output_duration']:.1f}s "
        f"(was {metadata['duration']:.1f}s)"
    )
    if result.get("note"):
        logger.info(f"[{job_id}] Note: {result['note']}")

    return {
        "final_video_path": output_path,
        "current_stage": "completed",
        "progress": 100.0,
        "logs": [
            f"[{job_id}] render: {size_mb:.1f}MB, {result['output_duration']:.1f}s "
            f"from {metadata['duration']:.1f}s in {elapsed}s"
        ],
        "timings": {"render": elapsed},
    }
