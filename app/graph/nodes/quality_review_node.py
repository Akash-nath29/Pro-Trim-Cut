"""quality review node — senior LLM editor reviews the edit plan before rendering"""

import json
import time
from pathlib import Path

from app.graph.state import GraphState
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("node.quality_review")


def quality_review_node(state: GraphState) -> dict:
    job_id = state["job_id"]
    jobs_dir = state["jobs_dir"]

    job_dir = Path(jobs_dir) / job_id
    qr_out = str(job_dir / "quality_review.json")

    if Path(qr_out).exists():
        logger.info(f"[{job_id}] ⏭ quality_review already done, using cached result")
        return {"quality_review_path": qr_out, "logs": [f"[{job_id}] quality_review: cached"]}

    logger.info(f"[{job_id}] ▶ quality_review starting")
    t0 = time.perf_counter()

    with open(state["edit_plan_path"], encoding="utf-8") as f:
        edit_plan = json.load(f)
    with open(state["timeline_path"], encoding="utf-8") as f:
        timeline = json.load(f)

    keep_ranges = edit_plan["keep_ranges"]
    removed_ranges = edit_plan.get("removed_ranges", [])
    segments = timeline["segments"]
    total_duration = edit_plan["total_input_duration"]
    output_duration = edit_plan["total_output_duration"]

    if not keep_ranges:
        review = {"approved": True, "quality_score": 5, "issues": [], "adjustments": [],
                  "review_notes": "No segments to review"}
        with open(qr_out, "w") as f:
            json.dump(review, f, indent=2)
        return {"quality_review_path": qr_out, "logs": [f"[{job_id}] quality_review: n/a (empty plan)"]}

    review = _llm_review(job_id, keep_ranges, removed_ranges, segments, total_duration, output_duration)

    if review.get("adjustments"):
        adjusted = _apply_adjustments(edit_plan, review["adjustments"], segments, total_duration)
        with open(state["edit_plan_path"], "w", encoding="utf-8") as f:
            json.dump(adjusted, f, indent=2, ensure_ascii=False)
        logger.info(f"[{job_id}] Applied {len(review['adjustments'])} quality adjustments")

    with open(qr_out, "w", encoding="utf-8") as f:
        json.dump(review, f, indent=2, ensure_ascii=False)

    elapsed = round(time.perf_counter() - t0, 3)
    logger.info(f"[{job_id}] ✓ quality_review done in {elapsed:.2f}s")

    return {
        "quality_review_path": qr_out,
        "current_stage": "quality_review",
        "progress": 82.0,
        "logs": [
            f"[{job_id}] quality_review: score={review.get('quality_score', '?')}/10 "
            f"({'approved' if review.get('approved') else 'needs work'}) in {elapsed}s"
        ],
        "timings": {"quality_review": elapsed},
    }


def _llm_review(job_id, keep_ranges, removed_ranges, segments, total_duration, output_duration) -> dict:
    from app.core.llm import get_llm
    llm = get_llm()
    seg_map = {s["id"]: s for s in segments}

    kept = [{
        "start": r["start"], "end": r["end"],
        "duration": round(r["end"] - r["start"], 2),
        "text": seg_map.get(r.get("segment_id", -1), {}).get("text", "")[:200],
        "reason": r.get("reason", ""),
    } for r in keep_ranges]

    removed = [{
        "start": r["start"], "end": r["end"],
        "text": seg_map.get(r.get("segment_id", -1), {}).get("text", "")[:200],
        "reason": r.get("reason", ""),
    } for r in removed_ranges[:20]]

    gaps = []
    for i in range(1, len(keep_ranges)):
        gap = keep_ranges[i]["start"] - keep_ranges[i - 1]["end"]
        if gap > 0.05:
            gaps.append({"between": f"{keep_ranges[i-1].get('segment_id')} → {keep_ranges[i].get('segment_id')}", "gap_s": round(gap, 2)})

    system_prompt = """You are a senior video editor reviewing a final edit plan before rendering.

Check: jump cuts, missing context, over/under-trimming (output should be 25-80% of input for talking-head video), flow.

CRITICAL RULES:
- NEVER re-include segments that were removed as sentence restarts, rehearsal lines, or false starts. Those removals are CORRECT.
- If there's a jump cut, prefer extending an adjacent kept segment rather than re-including bad content.
- The output MUST be shorter than the input. If the kept ranges already cover most of the good content, approve the plan.
- Do NOT add content back just to fill gaps. Gaps between good takes are fine.

Return ONLY valid JSON:
{
  "approved": true/false,
  "quality_score": 0-10,
  "issues": [{"type": "jump_cut", "description": "...", "severity": "low/medium/high"}],
  "adjustments": [
    {"segment_id": 3, "action": "extend_end", "amount": 0.2, "reason": "needs breathing room"}
  ],
  "review_notes": "brief assessment"
}

Adjustment actions: extend_start, extend_end, trim_start, trim_end."""

    user_prompt = f"""Review this edit plan:

Original: {total_duration:.1f}s → Output: {output_duration:.1f}s ({output_duration/max(total_duration,1)*100:.0f}% kept)

KEPT:\n{json.dumps(kept, indent=2)}

REMOVED:\n{json.dumps(removed, indent=2)}

GAPS:\n{json.dumps(gaps, indent=2)}"""

    logger.info(f"[{job_id}] Quality review: {len(keep_ranges)} kept, {len(removed_ranges)} removed...")
    try:
        result = llm.ask_json(system_prompt, user_prompt, temperature=0.1)
        score = result.get("quality_score", 7)
        approved = result.get("approved", True)
        logger.info(f"[{job_id}] {'✓ APPROVED' if approved else '✗ NEEDS WORK'} (score: {score}/10)")
        if result.get("review_notes"):
            logger.info(f"[{job_id}] Review: {result['review_notes']}")
        return result
    except Exception as e:
        logger.warning(f"[{job_id}] Quality review failed: {e}, auto-approving")
        return {"approved": True, "quality_score": 6, "issues": [], "adjustments": [],
                "review_notes": f"Auto-approved (LLM unavailable: {e})"}


def _apply_adjustments(edit_plan, adjustments, segments, total_duration) -> dict:
    keep_ranges = edit_plan["keep_ranges"]
    keep_by_seg = {r.get("segment_id"): i for i, r in enumerate(keep_ranges) if r.get("segment_id") is not None}

    for adj in adjustments:
        seg_id = adj.get("segment_id")
        action = adj.get("action", "")
        amount = adj.get("amount", 0.1)
        if action == "re_include":
            logger.warning(f"Ignoring re_include for segment {seg_id} — removed segments stay removed")
            continue
        if seg_id in keep_by_seg:
            r = keep_ranges[keep_by_seg[seg_id]]
            if action == "extend_start":
                r["start"] = round(max(0, r["start"] - amount), 3)
            elif action == "extend_end":
                r["end"] = round(min(total_duration, r["end"] + amount), 3)
            elif action == "trim_start":
                r["start"] = round(r["start"] + amount, 3)
            elif action == "trim_end":
                r["end"] = round(r["end"] - amount, 3)

    keep_ranges.sort(key=lambda x: x["start"])
    keep_ranges = _merge_overlapping(keep_ranges)
    edit_plan["keep_ranges"] = keep_ranges
    edit_plan["total_output_duration"] = round(sum(r["end"] - r["start"] for r in keep_ranges), 3)
    edit_plan["quality_reviewed"] = True
    return edit_plan


def _merge_overlapping(ranges: list) -> list:
    if not ranges:
        return []
    merged = [ranges[0].copy()]
    for r in ranges[1:]:
        last = merged[-1]
        if r["start"] <= last["end"] + 0.05:
            last["end"] = max(last["end"], r["end"])
            last["score"] = max(last.get("score", 0), r.get("score", 0))
        else:
            merged.append(r.copy())
    return merged
