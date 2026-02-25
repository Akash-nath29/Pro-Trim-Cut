import os
from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):

    # server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # storage
    STORAGE_PATH: str = "./jobs"
    MAX_UPLOAD_SIZE_MB: int = 500

    # device
    DEVICE: Literal["auto", "cpu", "cuda"] = "auto"
    COMPUTE_TYPE: str = "int8"

    # whisper
    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_LANGUAGE: str = "en"
    WHISPER_BATCH_SIZE: int = 16

    # vad
    VAD_THRESHOLD: float = 0.5

    # pause detection — anything longer than this gets flagged
    PAUSE_MIN_DURATION: float = 0.8
    PAUSE_KEEP_DURATION: float = 0.3

    # obvious fillers that are never intentional
    FILLER_WORDS: list[str] = [
        "um", "umm", "uh", "uhh", "uh huh",
        "ah", "ahh", "aa", "aah",
        "er", "err", "hmm", "hm",
    ]

    # vision
    VISION_SAMPLE_FPS: float = 2.0
    GAZE_THRESHOLD: float = 0.6
    MOTION_ENERGY_THRESHOLD: float = 500.0

    # semantic
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    SIMILARITY_THRESHOLD: float = 0.82
    MIN_SENTENCE_LENGTH: int = 5

    # cut planner scoring weights
    SCORE_WEIGHT_FILLER: float = 0.25
    SCORE_WEIGHT_GAZE: float = 0.20
    SCORE_WEIGHT_PAUSE: float = 0.20
    SCORE_WEIGHT_SEMANTIC: float = 0.20
    SCORE_WEIGHT_MOTION: float = 0.15
    MIN_SEGMENT_SCORE: float = 0.4
    CUT_PADDING_MS: int = 50

    # llm (azure ai inference / github models)
    GITHUB_TOKEN: Optional[str] = None
    LLM_ENDPOINT: str = "https://models.github.ai/inference"
    LLM_MODEL: str = "openai/gpt-4o-mini"

    # render
    OUTPUT_VIDEO_CODEC: str = "libx264"
    OUTPUT_AUDIO_CODEC: str = "aac"
    OUTPUT_CRF: int = 23
    OUTPUT_PRESET: str = "medium"

    # orchestrator
    MAX_RETRIES: int = 2
    RETRY_DELAY_SECONDS: float = 2.0

    @property
    def resolved_device(self) -> str:
        if self.DEVICE == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.DEVICE

    @property
    def storage_dir(self) -> Path:
        return Path(self.STORAGE_PATH).resolve()

    model_config = {"env_prefix": "", "env_file": ".env", "extra": "ignore"}


settings = Settings()
