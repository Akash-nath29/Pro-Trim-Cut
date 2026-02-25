"""
Timeline Builder — merges speech, vision, and semantic signals into one unified timeline.
Each segment gets all the data attached so the cut planner can make informed decisions.
"""

from app.agents.base import BaseAgent
from app.core.workspace import JobWorkspace


class TimelineBuilderAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "timeline_builder"

    @property
    def output_artifact(self) -> str:
        return "timeline.json"

    def process(self, job_id: str, workspace: JobWorkspace) -> dict:
        transcript = workspace.load_artifact("transcript.json")
        vision_data = workspace.load_artifact("vision.json")
        semantic_data = workspace.load_artifact("semantic.json")
        metadata = workspace.load_artifact("metadata.json")

        words = transcript["words"]
        pauses = transcript["pauses"]
        vision_frames = vision_data["frames"]
        semantic_groups = semantic_data["groups"]
        sentences = semantic_data["sentences"]
        duration = metadata["duration"]

        # map each sentence to its semantic group
        sentence_group_map = {}
        for group in semantic_groups:
            for sent in group["sentences"]:
                is_best = sent["id"] == group["representative_id"]
                sentence_group_map[sent["id"]] = (group["group_id"], is_best, group["is_duplicate"])

        segments = []
        for i, sent in enumerate(sentences):
            seg_start = sent["start"]
            seg_end = sent["end"]

            # pauses around this segment
            pause_before = 0.0
            pause_after = 0.0
            for p in pauses:
                if abs(p["end"] - seg_start) < 0.1:
                    pause_before = p["duration"]
                if abs(p["start"] - seg_end) < 0.1:
                    pause_after = p["duration"]

            # filler ratio
            seg_words = [w for w in words if w["start"] >= seg_start and w["end"] <= seg_end]
            filler_count = sum(1 for w in seg_words if w.get("is_filler", False))
            filler_ratio = filler_count / max(len(seg_words), 1)

            # vision data for this time range
            seg_vision = [f for f in vision_frames if seg_start <= f["timestamp"] <= seg_end]

            if seg_vision:
                looking_score = sum(f["looking_at_camera_score"] for f in seg_vision) / len(seg_vision)
                face_ratio = sum(1 for f in seg_vision if f["face_present"]) / len(seg_vision)
                avg_motion = sum(f["motion_energy"] for f in seg_vision) / len(seg_vision)
            else:
                looking_score, face_ratio, avg_motion = 0.5, 0.5, 0.0

            # semantic group
            group_id, is_best_take, is_duplicate = sentence_group_map.get(sent["id"], (-1, True, False))

            segments.append({
                "id": i,
                "start": round(seg_start, 3),
                "end": round(seg_end, 3),
                "duration": round(seg_end - seg_start, 3),
                "text": sent["text"],
                "word_count": sent["word_count"],
                "pause_before": round(pause_before, 3),
                "pause_after": round(pause_after, 3),
                "filler_ratio": round(filler_ratio, 3),
                "filler_count": filler_count,
                "looking_score": round(looking_score, 3),
                "face_present_ratio": round(face_ratio, 3),
                "avg_motion_energy": round(avg_motion, 3),
                "semantic_group": group_id,
                "is_duplicate": is_duplicate,
                "is_best_take": is_best_take,
            })

        if segments:
            first_speech = segments[0]["start"]
            if first_speech > 1.0:
                self.logger.info(f"[{job_id}] Dead time at start: {first_speech:.1f}s")
            last_speech = segments[-1]["end"]
            if duration - last_speech > 1.0:
                self.logger.info(f"[{job_id}] Dead time at end: {duration - last_speech:.1f}s")

        self.logger.info(f"[{job_id}] Built timeline with {len(segments)} segments")

        return {
            "segments": segments,
            "total_duration": round(duration, 3),
            "dead_time_start": round(segments[0]["start"], 3) if segments else 0,
            "dead_time_end": round(duration - segments[-1]["end"], 3) if segments else duration,
        }
