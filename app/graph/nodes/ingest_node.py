"""
Ingest Node — LangGraph node that extracts audio and reads video metadata.

Tools used: extract_audio, get_video_metadata
Writes to state: audio_path (via workspace artifact), current_stage, progress
"""

import time
from pathlib import Path

from app.graph.state import GraphState
from app.tools.ingest_tools import extract_audio, get_video_metadata
from app.core.logging import get_logger

logger = get_logger("node.ingest")


def ingest_node(state: GraphState) -> dict:
    """LangGraph node: ingest the uploaded video and extract audio."""
    job_id = state["job_id"]
    video_path = state["input_video_path"]
    jobs_dir = state["jobs_dir"]

    job_dir = Path(jobs_dir) / job_id
    audio_out = str(job_dir / "audio.wav")
    metadata_out = str(job_dir / "metadata.json")

    logger.info(f"[{job_id}] ▶ ingest starting")
    t0 = time.perf_counter()

    # ── Tool 1: read metadata ─────────────────────────────────────────────
    metadata = get_video_metadata.invoke({"video_path": video_path})
    logger.info(
        f"[{job_id}] Video: {metadata['width']}x{metadata['height']} "
        f"@ {metadata['fps']:.2f}fps, {metadata['duration']:.1f}s, "
        f"codec: {metadata['video_codec']}"
    )

    # save metadata artifact
    import json
    with open(metadata_out, "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Tool 2: extract audio ─────────────────────────────────────────────
    if not Path(audio_out).exists():
        result = extract_audio.invoke({"video_path": video_path, "output_path": audio_out})
        if not result["success"]:
            raise RuntimeError(f"Audio extraction failed: {result['error']}")
        logger.info(f"[{job_id}] Audio extracted: {audio_out}")
    else:
        logger.info(f"[{job_id}] Audio already exists, skipping extraction")

    elapsed = round(time.perf_counter() - t0, 3)
    logger.info(f"[{job_id}] ✓ ingest done in {elapsed:.2f}s")

    return {
        "audio_path": audio_out,
        "current_stage": "ingested",
        "progress": 10.0,
        "logs": [f"[{job_id}] ingest completed in {elapsed}s — {metadata['duration']:.1f}s video"],
        "timings": {"ingest": elapsed},
        "error": None,
    }
