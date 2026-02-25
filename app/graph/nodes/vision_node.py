"""
Vision Node — LangGraph node for face/gaze/motion video analysis.

Tools used: detect_face_presence, motion_energy_analysis
Writes to state: vision_analysis_path
"""

import json
import time
from pathlib import Path

from app.graph.state import GraphState
from app.tools.vision_tools import detect_face_presence, motion_energy_analysis
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("node.vision")


def vision_node(state: GraphState) -> dict:
    """LangGraph node: analyze video frames for face presence, gaze, and motion."""
    job_id = state["job_id"]
    video_path = state["input_video_path"]
    jobs_dir = state["jobs_dir"]

    job_dir = Path(jobs_dir) / job_id
    vision_out = str(job_dir / "vision.json")

    # skip if cached
    if Path(vision_out).exists():
        logger.info(f"[{job_id}] ⏭ vision already done, using cached result")
        return {"vision_analysis_path": vision_out, "logs": [f"[{job_id}] vision: cached"]}

    logger.info(f"[{job_id}] ▶ vision starting")
    t0 = time.perf_counter()

    # load video fps from metadata
    metadata_path = job_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    video_fps = metadata["fps"]
    sample_fps = settings.VISION_SAMPLE_FPS

    logger.info(f"[{job_id}] Sampling at {sample_fps} FPS (every {max(1, int(video_fps / sample_fps))} frames)")

    # ── Tool 1: detect faces + gaze per frame ────────────────────────────
    frames = detect_face_presence.invoke({
        "video_path": video_path,
        "sample_fps": sample_fps,
        "video_fps": video_fps,
    })

    # ── Tool 2: aggregate motion statistics ──────────────────────────────
    motion_stats = motion_energy_analysis.invoke({"frames": frames})

    avg_face = sum(1 for f in frames if f["face_present"]) / max(len(frames), 1)
    avg_gaze = sum(f["looking_at_camera_score"] for f in frames) / max(len(frames), 1)

    logger.info(
        f"[{job_id}] {len(frames)} frames | "
        f"avg face: {avg_face:.2f} | avg gaze: {avg_gaze:.2f} | "
        f"avg motion: {motion_stats['avg_motion']:.2f}"
    )

    result = {
        "frames": frames,
        "avg_face_presence": round(avg_face, 3),
        "avg_gaze_score": round(avg_gaze, 3),
        "motion_stats": motion_stats,
    }

    with open(vision_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    elapsed = round(time.perf_counter() - t0, 3)
    logger.info(f"[{job_id}] ✓ vision done in {elapsed:.2f}s")

    return {
        "vision_analysis_path": vision_out,
        "logs": [f"[{job_id}] vision: {len(frames)} frames, gaze={avg_gaze:.2f} in {elapsed}s"],
        "timings": {"vision": elapsed},
    }
