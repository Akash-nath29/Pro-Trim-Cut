"""speech tools"""

from langchain_core.tools import tool

from app.core.config import settings


@tool
def transcribe_with_timestamps(
    audio_path: str,
    device: str = "cpu",
    model_size: str = "base",
    language: str = "en",
    batch_size: int = 16,
) -> dict:
    """Transcribe audio to words with precise timestamps using WhisperX."""
    import whisperx

    compute_type = "int8" if device == "cpu" else settings.COMPUTE_TYPE

    model = whisperx.load_model(model_size, device=device, compute_type=compute_type, language=language)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size, language=language)

    align_model, align_metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(
        result["segments"], align_model, align_metadata, audio,
        device=device, return_char_alignments=False,
    )

    words = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            if "start" in w and "end" in w:
                words.append({
                    "word": w["word"].strip(),
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                    "confidence": round(w.get("score", 1.0), 3),
                    "is_filler": False,
                })

    import torch
    del model, align_model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {"words": words, "raw_segments": result.get("segments", [])}


@tool
def detect_pauses(words: list, min_duration: float = 0.8) -> list:
    """Find gaps between words that are long enough to be considered pauses."""
    pauses = []
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap >= min_duration:
            pauses.append({
                "start": round(words[i - 1]["end"], 3),
                "end": round(words[i]["start"], 3),
                "duration": round(gap, 3),
                "type": "long_pause" if gap > 2.0 else "silence",
            })
    return pauses


@tool
def detect_filler_words(words: list, filler_set: list = None) -> list:
    """Catch obvious filler words like um, uh, ah."""
    if filler_set is None:
        filler_set = settings.FILLER_WORDS

    fillers = []
    filler_lower = set(f.lower() for f in filler_set)
    for w in words:
        word_lower = w["word"].lower().strip(".,!?;:")
        if word_lower in filler_lower:
            fillers.append({
                "start": w["start"],
                "end": w["end"],
                "word": w["word"],
                "confidence": 0.95,
                "source": "pattern",
            })
    return fillers


SPEECH_TOOLS = [transcribe_with_timestamps, detect_pauses, detect_filler_words]
