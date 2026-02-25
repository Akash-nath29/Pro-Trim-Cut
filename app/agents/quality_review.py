"""
Quality Review Agent — the "second pair of eyes."

After the cut planner makes its decisions, this agent asks the LLM to review
the whole edit plan before we render anything. Catches jump cuts, over-trimming,
missing context, and flow issues.
"""

import json

from app.agents.base import BaseAgent
from app.core.config import settings
from app.core.llm import get_llm
from app.core.workspace import JobWorkspace


class QualityReviewAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "quality_review"

    @property
    def output_artifact(self) -> str:
        return "quality_review.json"

    def process(self, job_id: str, workspace: JobWorkspace) -> dict:
        edit_plan = workspace.load_artifact("edit_plan.json")
        timeline = workspace.load_artifact("timeline.json")

        keep_ranges = edit_plan["keep_ranges"]
        removed_ranges = edit_plan.get("removed_ranges", [])
        segments = timeline["segments"]
        total_duration = edit_plan["total_input_duration"]
        output_duration = edit_plan["total_output_duration"]

        if not keep_ranges:
            return {"approved": False, "reason": "Nothing to keep", "adjustments": []}

        review = self._llm_review(job_id, keep_ranges, removed_ranges, segments, total_duration, output_duration)

        if review.get("adjustments"):
            adjusted = self._apply_adjustments(edit_plan, review["adjustments"], segments, total_duration, job_id)
            workspace.save_artifact("edit_plan.json", adjusted)
            self.logger.info(f"[{job_id}] Applied {len(review['adjustments'])} adjustments")

        return review

    def _llm_review(self, job_id, keep_ranges, removed_ranges, segments, total_duration, output_duration) -> dict:
        llm = get_llm()
        seg_map = {s["id"]: s for s in segments}

        kept = [{"start": r["start"], "end": r["end"], "duration": round(r["end"] - r["start"], 2),
                 "text": seg_map.get(r.get("segment_id", -1), {}).get("text", "")[:200],
                 "reason": r.get("reason", "")} for r in keep_ranges]

        removed = [{"start": r["start"], "end": r["end"],
                     "text": seg_map.get(r.get("segment_id", -1), {}).get("text", "")[:200],
                     "reason": r.get("reason", "")} for r in removed_ranges[:20]]

        # check for gaps between cuts
        gaps = []
        for i in range(1, len(keep_ranges)):
            gap = keep_ranges[i]["start"] - keep_ranges[i - 1]["end"]
            if gap > 0.05:
                gaps.append({"between": f"{keep_ranges[i-1].get('segment_id')} → {keep_ranges[i].get('segment_id')}", "gap_s": round(gap, 2)})

        system_prompt = """You are a senior video editor reviewing an edit plan before rendering.

Check for:
1. Jump cuts — would adjacent kept segments create awkward visual jumps?
2. Missing context — was something important removed?
3. Over-trimming — output should be at least 25% of input for talking-head video
4. Under-trimming — are obvious mistakes still in?
5. Flow — does the sequence feel natural?

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

Adjustment actions: extend_start, extend_end, trim_start, trim_end, re_include (with start/end)."""

        user_prompt = f"""Review this edit plan:

Original: {total_duration:.1f}s → Output: {output_duration:.1f}s ({output_duration/max(total_duration,1)*100:.0f}% kept)

KEPT:\n{json.dumps(kept, indent=2)}

REMOVED:\n{json.dumps(removed, indent=2)}

GAPS:\n{json.dumps(gaps, indent=2)}"""

        self.logger.info(f"[{job_id}] Quality review: {len(keep_ranges)} kept, {len(removed_ranges)} removed...")

        try:
            result = llm.ask_json(system_prompt, user_prompt, temperature=0.1)
            approved = result.get("approved", True)
            score = result.get("quality_score", 7)
            self.logger.info(f"[{job_id}] {'✓ APPROVED' if approved else '✗ NEEDS WORK'} (score: {score}/10)")
            if result.get("review_notes"):
                self.logger.info(f"[{job_id}] Review: {result['review_notes']}")
            return result
        except Exception as e:
            self.logger.warning(f"[{job_id}] Quality review failed: {e}, approving as-is")
            return {"approved": True, "quality_score": 6, "issues": [], "adjustments": [],
                    "review_notes": f"Auto-approved (LLM unavailable: {e})"}

    def _apply_adjustments(self, edit_plan, adjustments, segments, total_duration, job_id) -> dict:
        keep_ranges = edit_plan["keep_ranges"]

        keep_by_seg = {}
        for i, r in enumerate(keep_ranges):
            sid = r.get("segment_id")
            if sid is not None:
                keep_by_seg[sid] = i

        for adj in adjustments:
            seg_id = adj.get("segment_id")
            action = adj.get("action", "")
            amount = adj.get("amount", 0.1)

            if action == "re_include":
                start, end = adj.get("start", 0), adj.get("end", 0)
                if start and end:
                    keep_ranges.append({"segment_id": seg_id or -1, "action": "keep",
                                        "start": round(start, 3), "end": round(end, 3),
                                        "reason": f"re-included: {adj.get('reason', '')}", "score": 0.5})
            elif seg_id in keep_by_seg:
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
        edit_plan["keep_ranges"] = keep_ranges
        edit_plan["total_output_duration"] = round(sum(r["end"] - r["start"] for r in keep_ranges), 3)
        edit_plan["quality_reviewed"] = True
        return edit_plan
