"""cut planning tools"""

import json

from langchain_core.tools import tool

from app.core.config import settings


@tool
def score_segments(segments: list) -> list:
    """Score each segment using editing heuristics (filler, gaze, pauses, semantics, motion)."""
    scored = []
    for seg in segments:
        filler_score = 1.0 - min(seg["filler_ratio"] * 3, 1.0)
        gaze_score = seg["looking_score"]

        total_pause = seg["pause_before"] + seg["pause_after"]
        pause_score = 0.2 if total_pause > 3.0 else (0.5 if total_pause > 1.5 else 1.0)

        semantic_score = 1.0 if seg["is_best_take"] else 0.3

        motion = seg["avg_motion_energy"]
        motion_score = 0.3 if motion < 0.5 else (0.5 if motion > 20.0 else 1.0)

        score = round(
            filler_score * settings.SCORE_WEIGHT_FILLER
            + gaze_score * settings.SCORE_WEIGHT_GAZE
            + pause_score * settings.SCORE_WEIGHT_PAUSE
            + semantic_score * settings.SCORE_WEIGHT_SEMANTIC
            + motion_score * settings.SCORE_WEIGHT_MOTION,
            3,
        )
        scored.append({**seg, "deterministic_score": score})

    return scored


@tool
def select_best_takes(scored_segments: list, total_duration: float) -> dict:
    """Deterministic fallback — picks best takes when LLM isn't available."""
    padding = settings.CUT_PADDING_MS / 1000.0
    keep_ranges, removed_ranges = [], []
    group_best: dict = {}

    for seg in scored_segments:
        score = seg.get("deterministic_score", 0)
        gid = seg["semantic_group"]

        if gid >= 0 and seg["is_duplicate"]:
            if gid not in group_best or score > group_best[gid][1]:
                if gid in group_best:
                    old_seg, old_score = group_best[gid]
                    removed_ranges.append({
                        "segment_id": old_seg["id"], "action": "remove",
                        "start": old_seg["start"], "end": old_seg["end"],
                        "reason": "duplicate – lower score", "score": round(old_score, 3),
                    })
                group_best[gid] = (seg, score)
            else:
                removed_ranges.append({
                    "segment_id": seg["id"], "action": "remove",
                    "start": seg["start"], "end": seg["end"],
                    "reason": "duplicate – lower score", "score": round(score, 3),
                })
        else:
            if score >= settings.MIN_SEGMENT_SCORE:
                keep_ranges.append({
                    "segment_id": seg["id"], "action": "keep",
                    "start": round(max(0, seg["start"] - padding), 3),
                    "end": round(min(total_duration, seg["end"] + padding), 3),
                    "reason": f"score={score:.2f}", "score": round(score, 3),
                })
            else:
                removed_ranges.append({
                    "segment_id": seg["id"], "action": "remove",
                    "start": seg["start"], "end": seg["end"],
                    "reason": f"low score={score:.2f}", "score": round(score, 3),
                })

    for seg, score in group_best.values():
        keep_ranges.append({
            "segment_id": seg["id"], "action": "keep",
            "start": round(max(0, seg["start"] - padding), 3),
            "end": round(min(total_duration, seg["end"] + padding), 3),
            "reason": f"best take, score={score:.2f}", "score": round(score, 3),
        })

    keep_ranges.sort(key=lambda x: x["start"])
    keep_ranges = _merge_ranges(keep_ranges)
    output_duration = sum(r["end"] - r["start"] for r in keep_ranges)

    return {
        "keep_ranges": keep_ranges,
        "removed_ranges": removed_ranges,
        "total_input_duration": round(total_duration, 3),
        "total_output_duration": round(output_duration, 3),
        "cut_count": len(keep_ranges),
        "mode": "deterministic",
    }


@tool
def generate_edit_plan(
    job_id: str,
    scored_segments: list,
    total_duration: float,
    pause_classifications: list,
    sentence_restarts: list,
    rehearsal_sentences: list,
    narrative_order: list,
) -> dict:
    """LLM-driven edit plan with deterministic fallback."""
    from app.core.llm import get_llm

    llm = get_llm()

    # flag segments that overlap with known sentence restarts
    restart_set = set()
    for seg in scored_segments:
        for sr in sentence_restarts:
            if seg["start"] <= sr.get("end", 0) and seg["end"] >= sr.get("start", 0):
                restart_set.add(seg["id"])
                break

    segment_data = [{
        "id": s["id"], "start": s["start"], "end": s["end"],
        "duration": round(s["end"] - s["start"], 2),
        "text": s["text"][:300],
        "filler_ratio": s["filler_ratio"], "filler_count": s["filler_count"],
        "looking_score": s["looking_score"], "face_present": s["face_present_ratio"],
        "semantic_group": s["semantic_group"],
        "is_duplicate": s["is_duplicate"], "is_best_take": s["is_best_take"],
        "is_fragment": s.get("is_fragment", False),
        "is_restart": s["id"] in restart_set,
        "deterministic_score": s.get("deterministic_score", 0.5),
        "pause_before": s["pause_before"], "pause_after": s["pause_after"],
    } for s in scored_segments]

    system_prompt = """You are a PROFESSIONAL video editor working on a talking-head creator video. Make this raw footage into a clean, tight, and watchable final cut.

YOUR EDITING PHILOSOPHY:
- The final video should feel like the creator edited it themselves — natural, not robotic
- Keep natural breathing room between sentences (don't cut TOO tight)
- Remove obvious mistakes but keep the creator's personality
- If someone says "um" naturally in the middle of a good sentence, usually KEEP it
- Only remove filler words when they're clearly stalling or restarting
- For duplicate takes, keep the one with the best delivery
- Remove dead camera time at start/end
- Remove long pauses but keep short natural pauses
- Cuts MUST occur at natural word/silence boundaries, never mid-word

WHAT TO DEFINITELY REMOVE:
- Fragments (is_fragment=true) — these are short rehearsal bits, false starts, or throwaway words
- Restarts (is_restart=true) — the speaker abandoned this line and started over, ALWAYS remove these
- Rehearsal/practice lines and sentence restarts
- Duplicate inferior takes
- Dead silence > 2 seconds
- Sections where speaker is clearly not delivering content (looking away repeatedly)
- Everything before the speaker's actual content begins (false starts, "all right", "so hey", etc.)

WHAT TO DEFINITELY KEEP:
- The best take of every unique thought
- Natural transitions
- Brief natural pauses (0.3–0.8s)
- Engaging sections where speaker looks at camera

IMPORTANT: The output MUST be SHORTER than the input. You are TRIMMING, not adding.

For each segment, decide: KEEP or REMOVE.

Return ONLY valid JSON:
{
  "decisions": [
    {"segment_id": 0, "action": "keep", "trim_start": 0.0, "trim_end": 0.0, "reason": "Good delivery"},
    {"segment_id": 1, "action": "remove", "reason": "Duplicate, worse delivery than segment 3"}
  ],
  "editing_explanation": "brief overall explanation",
  "natural_pause_between_cuts_ms": 150
}

trim_start/trim_end are offsets in seconds: positive = trim inward, negative = add breathing room."""

    user_prompt = f"""Edit this raw video ({total_duration:.1f}s total).

SEGMENTS:
{json.dumps(segment_data, indent=2)}

CONTEXT:
- Sentence restarts: {json.dumps(sentence_restarts[:10])}
- Rehearsal lines: {json.dumps(rehearsal_sentences[:10])}
- Narrative order: {narrative_order}
- Pause classifications: {json.dumps(pause_classifications[:15])}"""

    try:
        result = llm.ask_json(system_prompt, user_prompt, temperature=0.15)
        return _build_edit_plan_from_llm(result, scored_segments, total_duration, segment_data)
    except Exception:
        return select_best_takes.func(scored_segments, total_duration)


def _merge_ranges(ranges: list) -> list:
    if not ranges:
        return []
    merged = [ranges[0].copy()]
    for r in ranges[1:]:
        last = merged[-1]
        if r["start"] <= last["end"] + 0.1:
            last["end"] = max(last["end"], r["end"])
            last["score"] = max(last["score"], r["score"])
        else:
            merged.append(r.copy())
    return merged


def _build_edit_plan_from_llm(llm_result: dict, segments: list, total_duration: float, segment_data: list = None) -> dict:
    decisions = llm_result.get("decisions", [])
    explanation = llm_result.get("editing_explanation", "")
    padding = settings.CUT_PADDING_MS / 1000.0

    keep_ranges, removed_ranges = [], []
    seg_map = {s["id"]: s for s in segments}

    for d in decisions:
        seg_id = d.get("segment_id")
        if seg_id is None or seg_id not in seg_map:
            continue
        seg = seg_map[seg_id]
        if d.get("action") == "keep":
            start = max(0, seg["start"] + d.get("trim_start", 0) - padding)
            end = min(total_duration, seg["end"] + d.get("trim_end", 0) + padding)
            keep_ranges.append({
                "segment_id": seg_id, "action": "keep",
                "start": round(start, 3), "end": round(end, 3),
                "reason": d.get("reason", ""), "score": seg.get("deterministic_score", 0.5),
            })
        else:
            removed_ranges.append({
                "segment_id": seg_id, "action": "remove",
                "start": seg["start"], "end": seg["end"],
                "reason": d.get("reason", ""), "score": seg.get("deterministic_score", 0),
            })

    # anything the LLM didn't mention — use deterministic rules
    mentioned = {d.get("segment_id") for d in decisions}
    restart_ids = {sd["id"] for sd in (segment_data or []) if sd.get("is_restart")}
    for seg in segments:
        if seg["id"] not in mentioned:
            score = seg.get("deterministic_score", 0)
            is_fragment = seg.get("is_fragment", False)
            is_restart = seg["id"] in restart_ids
            if is_fragment or is_restart or score < settings.MIN_SEGMENT_SCORE:
                removed_ranges.append({
                    "segment_id": seg["id"], "action": "remove",
                    "start": seg["start"], "end": seg["end"],
                    "reason": "fragment/restart/low score, not mentioned by LLM", "score": score,
                })
            else:
                keep_ranges.append({
                    "segment_id": seg["id"], "action": "keep",
                    "start": round(max(0, seg["start"] - padding), 3),
                    "end": round(min(total_duration, seg["end"] + padding), 3),
                    "reason": "kept by score (not mentioned by LLM)", "score": score,
                })

    keep_ranges.sort(key=lambda x: x["start"])
    keep_ranges = _merge_ranges(keep_ranges)
    output_duration = sum(r["end"] - r["start"] for r in keep_ranges)

    return {
        "keep_ranges": keep_ranges,
        "removed_ranges": removed_ranges,
        "total_input_duration": round(total_duration, 3),
        "total_output_duration": round(output_duration, 3),
        "cut_count": len(keep_ranges),
        "mode": "llm",
        "llm_explanation": explanation,
    }


CUT_PLANNING_TOOLS = [score_segments, select_best_takes, generate_edit_plan]
