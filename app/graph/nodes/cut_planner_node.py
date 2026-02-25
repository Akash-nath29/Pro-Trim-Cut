"""
Cut Planner Node — LangGraph node that generates the final edit plan.

Depends on: timeline_path, transcript_path, semantic_analysis_path
Tools used: score_segments, generate_edit_plan (with select_best_takes as fallback)
Writes to state: edit_plan_path
"""

import json
import time
from pathlib import Path

from app.graph.state import GraphState
from app.tools.cut_planning_tools import score_segments, generate_edit_plan
from app.core.logging import get_logger

logger = get_logger("node.cut_planner")


def cut_planner_node(state: GraphState) -> dict:
    """LangGraph node: score segments and generate an LLM-guided edit plan."""
    job_id = state["job_id"]
    jobs_dir = state["jobs_dir"]

    job_dir = Path(jobs_dir) / job_id
    edit_plan_out = str(job_dir / "edit_plan.json")

    if Path(edit_plan_out).exists():
        logger.info(f"[{job_id}] ⏭ cut_planner already done, using cached result")
        return {"edit_plan_path": edit_plan_out, "logs": [f"[{job_id}] cut_planner: cached"]}

    logger.info(f"[{job_id}] ▶ cut_planner starting")
    t0 = time.perf_counter()

    with open(state["timeline_path"], encoding="utf-8") as f:
        timeline = json.load(f)
    with open(state["transcript_path"], encoding="utf-8") as f:
        transcript = json.load(f)
    with open(state["semantic_analysis_path"], encoding="utf-8") as f:
        semantic = json.load(f)

    segments = timeline["segments"]
    total_duration = timeline["total_duration"]

    if not segments:
        logger.info(f"[{job_id}] No segments — returning empty edit plan")
        edit_plan = {
            "keep_ranges": [], "removed_ranges": [],
            "total_input_duration": total_duration, "total_output_duration": 0,
            "cut_count": 0, "mode": "empty",
        }
        with open(edit_plan_out, "w") as f:
            json.dump(edit_plan, f, indent=2)
        return {
            "edit_plan_path": edit_plan_out,
            "logs": [f"[{job_id}] cut_planner: empty (no segments)"],
            "timings": {"cut_planner": 0.0},
        }

    # ── Tool 1: score segments deterministically ──────────────────────────
    scored = score_segments.invoke({"segments": segments})

    # ── Tool 2: generate edit plan (LLM + fallback) ───────────────────────
    speech_analysis = transcript.get("llm_analysis", {})
    semantic_analysis = semantic.get("llm_analysis", {})

    logger.info(f"[{job_id}] LLM editor analyzing {len(scored)} segments...")
    edit_plan = generate_edit_plan.invoke({
        "job_id": job_id,
        "scored_segments": scored,
        "total_duration": total_duration,
        "pause_classifications": speech_analysis.get("pause_classification", []),
        "sentence_restarts": transcript.get("sentence_restarts", []),
        "rehearsal_sentences": semantic_analysis.get("rehearsal_sentences", []),
        "narrative_order": semantic_analysis.get("narrative_order", []),
    })

    if edit_plan.get("llm_explanation"):
        logger.info(f"[{job_id}] LLM: {edit_plan['llm_explanation']}")

    logger.info(
        f"[{job_id}] Keep {len(edit_plan['keep_ranges'])} ranges "
        f"({edit_plan['total_output_duration']:.1f}s from {total_duration:.1f}s)"
    )

    with open(edit_plan_out, "w", encoding="utf-8") as f:
        json.dump(edit_plan, f, indent=2, ensure_ascii=False)

    elapsed = round(time.perf_counter() - t0, 3)
    logger.info(f"[{job_id}] ✓ cut_planner done in {elapsed:.2f}s")

    return {
        "edit_plan_path": edit_plan_out,
        "current_stage": "planning_cuts",
        "progress": 75.0,
        "logs": [
            f"[{job_id}] cut_planner: {len(edit_plan['keep_ranges'])} keep ranges "
            f"({edit_plan['total_output_duration']:.1f}s) in {elapsed}s"
        ],
        "timings": {"cut_planner": elapsed},
    }
