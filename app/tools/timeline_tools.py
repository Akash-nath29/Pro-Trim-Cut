"""timeline tools"""

from langchain_core.tools import tool


@tool
def merge_multimodal_signals(
    transcript: dict,
    vision_data: dict,
    semantic_data: dict,
    metadata: dict,
) -> dict:
    """Join speech, vision, and semantic data into one package for segment building."""
    sentence_group_map = {}
    for group in semantic_data.get("groups", []):
        for sent in group["sentences"]:
            is_best = sent["id"] == group["representative_id"]
            sentence_group_map[sent["id"]] = (
                group["group_id"], is_best, group["is_duplicate"]
            )

    return {
        "words": transcript.get("words", []),
        "pauses": transcript.get("pauses", []),
        "fillers": transcript.get("fillers", []),
        "vision_frames": vision_data.get("frames", []),
        "sentences": semantic_data.get("sentences", []),
        "sentence_group_map": sentence_group_map,
        "duration": metadata["duration"],
    }


@tool
def build_segments(merged: dict) -> dict:
    """Build timeline segments from merged multimodal data."""
    words = merged["words"]
    pauses = merged["pauses"]
    vision_frames = merged["vision_frames"]
    sentences = merged["sentences"]
    sentence_group_map = merged["sentence_group_map"]
    duration = merged["duration"]

    # fix whisperx bugs where one word spans way too long
    for w in words:
        if w["end"] - w["start"] > 2.0:
            w["end"] = round(w["start"] + 0.5, 3)

    segments = []
    for i, sent in enumerate(sentences):
        seg_start = sent["start"]
        seg_end = sent["end"]

        pause_before = 0.0
        pause_after = 0.0
        for p in pauses:
            if abs(p["end"] - seg_start) < 0.1:
                pause_before = p["duration"]
            if abs(p["start"] - seg_end) < 0.1:
                pause_after = p["duration"]

        seg_words = [w for w in words if w["start"] >= seg_start and w["end"] <= seg_end]
        filler_count = sum(1 for w in seg_words if w.get("is_filler", False))
        filler_ratio = filler_count / max(len(seg_words), 1)

        seg_vision = [f for f in vision_frames if seg_start <= f["timestamp"] <= seg_end]
        if seg_vision:
            looking_score = sum(f["looking_at_camera_score"] for f in seg_vision) / len(seg_vision)
            face_ratio = sum(1 for f in seg_vision if f["face_present"]) / len(seg_vision)
            avg_motion = sum(f["motion_energy"] for f in seg_vision) / len(seg_vision)
        else:
            looking_score, face_ratio, avg_motion = 0.5, 0.5, 0.0

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
            "is_fragment": sent.get("is_fragment", False),
        })

    return {
        "segments": segments,
        "total_duration": round(duration, 3),
        "dead_time_start": round(segments[0]["start"], 3) if segments else 0,
        "dead_time_end": round(duration - segments[-1]["end"], 3) if segments else duration,
    }


TIMELINE_TOOLS = [merge_multimodal_signals, build_segments]
