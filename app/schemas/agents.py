from typing import Optional
from pydantic import BaseModel, Field


# ingest

class IngestResult(BaseModel):
    fps: float
    duration: float
    width: int
    height: int
    audio_path: str
    video_codec: str = ""
    audio_codec: str = ""
    file_size_bytes: int = 0


# speech

class TranscriptWord(BaseModel):
    word: str
    start: float
    end: float
    confidence: float = 1.0
    is_filler: bool = False


class PauseRegion(BaseModel):
    start: float
    end: float
    duration: float
    type: str = "silence"


class FillerRegion(BaseModel):
    start: float
    end: float
    word: str
    confidence: float = 1.0


class SpeechResult(BaseModel):
    words: list[TranscriptWord]
    pauses: list[PauseRegion]
    fillers: list[FillerRegion]
    full_text: str = ""
    total_speech_duration: float = 0.0
    total_pause_duration: float = 0.0


# vision

class VisionFrame(BaseModel):
    timestamp: float
    face_present: bool = False
    face_count: int = 0
    looking_at_camera_score: float = 0.0
    motion_energy: float = 0.0


class VisionResult(BaseModel):
    frames: list[VisionFrame]
    avg_face_presence: float = 0.0
    avg_gaze_score: float = 0.0


# semantic

class SentenceSegment(BaseModel):
    id: int
    text: str
    start: float
    end: float
    word_count: int = 0
    embedding_index: int = -1


class SemanticGroup(BaseModel):
    group_id: int
    sentences: list[SentenceSegment]
    representative_id: int
    is_duplicate: bool = False
    similarity_score: float = 0.0


class SemanticResult(BaseModel):
    sentences: list[SentenceSegment]
    groups: list[SemanticGroup]
    duplicate_count: int = 0


# timeline

class TimelineSegment(BaseModel):
    id: int
    start: float
    end: float
    duration: float = 0.0
    text: str = ""
    word_count: int = 0
    pause_before: float = 0.0
    pause_after: float = 0.0
    filler_ratio: float = 0.0
    filler_count: int = 0
    looking_score: float = 0.0
    face_present_ratio: float = 0.0
    avg_motion_energy: float = 0.0
    semantic_group: int = -1
    is_duplicate: bool = False
    is_best_take: bool = True


# cut planner

class EditDecision(BaseModel):
    segment_id: int
    action: str
    start: float
    end: float
    reason: str = ""
    score: float = 0.0


class EditPlan(BaseModel):
    keep_ranges: list[EditDecision]
    removed_ranges: list[EditDecision]
    total_input_duration: float = 0.0
    total_output_duration: float = 0.0
    cut_count: int = 0
    mode: str = "deterministic"
