"""
Cut Planner — the LLM acts as the primary editor.

It sees all the analysis data (fillers, gaze, duplicates, restarts, rehearsals) and
makes human-like decisions about what to keep and what to cut. The deterministic
scorer runs first to give the LLM a baseline, but the LLM has final say.
"""

import json

from app.agents.base import BaseAgent
from app.core.config import settings
from app.core.llm import get_llm
from app.core.workspace import JobWorkspace


class CutPlannerAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "cut_planner"

    @property
    def output_artifact(self) -> str:
        return "edit_plan.json"

    def process(self, job_id: str, workspace: JobWorkspace) -> dict:
        timeline = workspace.load_artifact("timeline.json")
        semantic = workspace.load_artifact("semantic.json")
        transcript = workspace.load_artifact("transcript.json")

        segments = timeline["segments"]
        total_duration = timeline["total_duration"]

        if not segments:
            return {"keep_ranges": [], "removed_ranges": [], "total_input_duration": total_duration,
                    "total_output_duration": 0, "cut_count": 0, "mode": "llm"}

        # score everything deterministically first — gives the LLM a baseline
        scored_segments = []
        for seg in segments:
            seg_copy = {**seg, "deterministic_score": self._score_segment(seg)}
            scored_segments.append(seg_copy)

        # gather all the context from other agents
        speech_analysis = transcript.get("llm_analysis", {})
        semantic_analysis = semantic.get("llm_analysis", {})

        # let the LLM make the final call
        return self._llm_plan_cuts(
            job_id=job_id,
            segments=scored_segments,
            total_duration=total_duration,
            pause_classifications=speech_analysis.get("pause_classification", []),
            sentence_restarts=transcript.get("sentence_restarts", []),
            rehearsal_sentences=semantic_analysis.get("rehearsal_sentences", []),
            narrative_order=semantic_analysis.get("narrative_order", []),
        )

    def _score_segment(self, seg: dict) -> float:
        """Quick deterministic score — mostly for the LLM's reference."""
        filler_score = 1.0 - min(seg["filler_ratio"] * 3, 1.0)
        gaze_score = seg["looking_score"]

        total_pause = seg["pause_before"] + seg["pause_after"]
        pause_score = 0.2 if total_pause > 3.0 else (0.5 if total_pause > 1.5 else 1.0)

        semantic_score = 1.0 if seg["is_best_take"] else 0.3

        motion = seg["avg_motion_energy"]
        motion_score = 0.3 if motion < 0.5 else (0.5 if motion > 20.0 else 1.0)

        return round(
            filler_score * settings.SCORE_WEIGHT_FILLER
            + gaze_score * settings.SCORE_WEIGHT_GAZE
            + pause_score * settings.SCORE_WEIGHT_PAUSE
            + semantic_score * settings.SCORE_WEIGHT_SEMANTIC
            + motion_score * settings.SCORE_WEIGHT_MOTION,
            3,
        )

    def _llm_plan_cuts(self, job_id, segments, total_duration, pause_classifications,
                       sentence_restarts, rehearsal_sentences, narrative_order) -> dict:
        llm = get_llm()

        segment_data = [{
            "id": s["id"], "start": s["start"], "end": s["end"],
            "duration": round(s["end"] - s["start"], 2),
            "text": s["text"][:300],
            "filler_ratio": s["filler_ratio"], "filler_count": s["filler_count"],
            "looking_score": s["looking_score"], "face_present": s["face_present_ratio"],
            "semantic_group": s["semantic_group"],
            "is_duplicate": s["is_duplicate"], "is_best_take": s["is_best_take"],
            "deterministic_score": s["deterministic_score"],
            "pause_before": s["pause_before"], "pause_after": s["pause_after"],
        } for s in segments]

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
- Cuts should happen at natural word boundaries, never mid-word

WHAT TO DEFINITELY REMOVE:
- Rehearsal/practice lines
- Duplicate inferior takes
- Sentence restarts (the abandoned part)
- Dead silence > 2 seconds
- Sections where speaker is clearly not delivering content

WHAT TO DEFINITELY KEEP:
- The best take of every unique thought
- Natural transitions
- Brief natural pauses (0.3-0.8s)
- Engaging sections where speaker looks at camera

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

trim_start/trim_end are offsets: positive = trim inward, negative = add breathing room."""

        user_prompt = f"""Edit this raw video ({total_duration:.1f}s total).

SEGMENTS:
{json.dumps(segment_data, indent=2)}

CONTEXT:
- Sentence restarts: {json.dumps(sentence_restarts[:10])}
- Rehearsal lines: {json.dumps(rehearsal_sentences[:10])}
- Narrative order: {narrative_order}
- Pause classifications: {json.dumps(pause_classifications[:15])}"""

        self.logger.info(f"[{job_id}] LLM editor analyzing {len(segments)} segments...")

        try:
            result = llm.ask_json(system_prompt, user_prompt, temperature=0.15)
        except Exception as e:
            self.logger.warning(f"[{job_id}] LLM failed: {e}, falling back to deterministic")
            return self._plan_deterministic_fallback(job_id, segments, total_duration)

        return self._build_edit_plan(job_id, result, segments, total_duration)

    def _build_edit_plan(self, job_id, llm_result, segments, total_duration) -> dict:
        decisions = llm_result.get("decisions", [])
        explanation = llm_result.get("editing_explanation", "")

        if explanation:
            self.logger.info(f"[{job_id}] LLM: {explanation}")

        keep_ranges = []
        removed_ranges = []
        seg_map = {s["id"]: s for s in segments}

        for d in decisions:
            seg_id = d.get("segment_id")
            if seg_id is None or seg_id not in seg_map:
                continue

            seg = seg_map[seg_id]
            if d.get("action") == "keep":
                padding = settings.CUT_PADDING_MS / 1000.0
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

        # anything the LLM didn't mention — use the deterministic score as tiebreaker
        mentioned = {d.get("segment_id") for d in decisions}
        for seg in segments:
            if seg["id"] not in mentioned:
                padding = settings.CUT_PADDING_MS / 1000.0
                if seg.get("deterministic_score", 0) >= settings.MIN_SEGMENT_SCORE:
                    keep_ranges.append({
                        "segment_id": seg["id"], "action": "keep",
                        "start": round(max(0, seg["start"] - padding), 3),
                        "end": round(min(total_duration, seg["end"] + padding), 3),
                        "reason": "Kept by score (LLM didn't mention)", "score": seg.get("deterministic_score", 0.5),
                    })
                else:
                    removed_ranges.append({
                        "segment_id": seg["id"], "action": "remove",
                        "start": seg["start"], "end": seg["end"],
                        "reason": "Low score, not mentioned by LLM", "score": seg.get("deterministic_score", 0),
                    })

        keep_ranges.sort(key=lambda x: x["start"])
        keep_ranges = self._merge_ranges(keep_ranges)
        output_duration = sum(r["end"] - r["start"] for r in keep_ranges)

        self.logger.info(f"[{job_id}] Keep {len(keep_ranges)} ranges ({output_duration:.1f}s from {total_duration:.1f}s)")

        return {
            "keep_ranges": keep_ranges, "removed_ranges": removed_ranges,
            "total_input_duration": round(total_duration, 3),
            "total_output_duration": round(output_duration, 3),
            "cut_count": len(keep_ranges), "mode": "llm",
            "llm_explanation": explanation,
        }

    def _merge_ranges(self, ranges: list[dict]) -> list[dict]:
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

    def _plan_deterministic_fallback(self, job_id, segments, total_duration) -> dict:
        """If the LLM is down, fall back to pure scoring."""
        keep_ranges, removed_ranges = [], []
        group_best = {}

        for seg in segments:
            score = seg.get("deterministic_score", self._score_segment(seg))
            gid = seg["semantic_group"]

            if gid >= 0 and seg["is_duplicate"]:
                if gid not in group_best or score > group_best[gid][1]:
                    if gid in group_best:
                        old = group_best[gid][0]
                        removed_ranges.append({"segment_id": old["id"], "action": "remove", "start": old["start"], "end": old["end"], "reason": "duplicate", "score": round(group_best[gid][1], 3)})
                    group_best[gid] = (seg, score)
                else:
                    removed_ranges.append({"segment_id": seg["id"], "action": "remove", "start": seg["start"], "end": seg["end"], "reason": "duplicate", "score": round(score, 3)})
            else:
                padding = settings.CUT_PADDING_MS / 1000.0
                if score >= settings.MIN_SEGMENT_SCORE:
                    keep_ranges.append({"segment_id": seg["id"], "action": "keep", "start": round(max(0, seg["start"] - padding), 3), "end": round(min(total_duration, seg["end"] + padding), 3), "reason": f"score {score:.2f}", "score": round(score, 3)})
                else:
                    removed_ranges.append({"segment_id": seg["id"], "action": "remove", "start": seg["start"], "end": seg["end"], "reason": f"low score {score:.2f}", "score": round(score, 3)})

        for seg, score in group_best.values():
            padding = settings.CUT_PADDING_MS / 1000.0
            keep_ranges.append({"segment_id": seg["id"], "action": "keep", "start": round(max(0, seg["start"] - padding), 3), "end": round(min(total_duration, seg["end"] + padding), 3), "reason": f"best take, score {score:.2f}", "score": round(score, 3)})

        keep_ranges.sort(key=lambda x: x["start"])
        keep_ranges = self._merge_ranges(keep_ranges)
        output_duration = sum(r["end"] - r["start"] for r in keep_ranges)

        return {"keep_ranges": keep_ranges, "removed_ranges": removed_ranges, "total_input_duration": round(total_duration, 3), "total_output_duration": round(output_duration, 3), "cut_count": len(keep_ranges), "mode": "deterministic_fallback"}
