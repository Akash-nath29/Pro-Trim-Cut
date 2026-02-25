"""
Speech Node — LangGraph node for speech transcription and analysis.

Tools used: transcribe_with_timestamps, detect_pauses, detect_filler_words
Writes to state: transcript_path
"""

import json
import time
from pathlib import Path

from app.graph.state import GraphState
from app.tools.speech_tools import transcribe_with_timestamps, detect_pauses, detect_filler_words
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("node.speech")


def speech_node(state: GraphState) -> dict:
    """LangGraph node: transcribe audio and detect speech patterns."""
    job_id = state["job_id"]
    audio_path = state["audio_path"]
    jobs_dir = state["jobs_dir"]
    device = state["processing_mode"]

    job_dir = Path(jobs_dir) / job_id
    transcript_out = str(job_dir / "transcript.json")

    # skip if cached
    if Path(transcript_out).exists():
        logger.info(f"[{job_id}] ⏭ speech already done, using cached result")
        return {"transcript_path": transcript_out, "logs": [f"[{job_id}] speech: cached"]}

    logger.info(f"[{job_id}] ▶ speech starting | device: {device}")
    t0 = time.perf_counter()

    # ── Tool 1: transcribe ────────────────────────────────────────────────
    logger.info(f"[{job_id}] Loading WhisperX ({settings.WHISPER_MODEL_SIZE}) and transcribing...")
    transcription = transcribe_with_timestamps.invoke({
        "audio_path": audio_path,
        "device": device,
        "model_size": settings.WHISPER_MODEL_SIZE,
        "language": settings.WHISPER_LANGUAGE,
        "batch_size": settings.WHISPER_BATCH_SIZE,
    })
    words = transcription["words"]
    logger.info(f"[{job_id}] Got {len(words)} words")

    # ── Tool 2: detect pauses ─────────────────────────────────────────────
    pauses = detect_pauses.invoke({"words": words, "min_duration": settings.PAUSE_MIN_DURATION})

    # ── Tool 3: detect filler words ───────────────────────────────────────
    basic_fillers = detect_filler_words.invoke({"words": words, "filler_set": settings.FILLER_WORDS})

    # LLM speech analysis (only if there's something to analyze)
    llm_analysis = {"context_dependent_fillers": [], "sentence_restarts": [], "pause_classification": []}
    if words:
        llm_analysis = _llm_classify_speech(job_id, words, pauses)

    # merge filler results
    fillers = _merge_fillers(basic_fillers, llm_analysis)
    filler_set = {(f["start"], f["end"]) for f in fillers}
    for w in words:
        if (w["start"], w["end"]) in filler_set:
            w["is_filler"] = True

    # mark false starts
    restart_ranges = llm_analysis.get("sentence_restarts", [])
    for w in words:
        for restart in restart_ranges:
            if w["start"] >= restart.get("start", 9999) and w["end"] <= restart.get("end", -1):
                w["is_filler"] = True

    full_text = " ".join(w["word"] for w in words)
    total_speech = sum(w["end"] - w["start"] for w in words) if words else 0
    total_pause = sum(p["duration"] for p in pauses)

    result = {
        "words": words,
        "pauses": pauses,
        "fillers": fillers,
        "sentence_restarts": restart_ranges,
        "full_text": full_text,
        "total_speech_duration": round(total_speech, 3),
        "total_pause_duration": round(total_pause, 3),
        "llm_analysis": llm_analysis,
    }

    with open(transcript_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    elapsed = round(time.perf_counter() - t0, 3)
    logger.info(
        f"[{job_id}] ✓ speech done in {elapsed:.2f}s — "
        f"{len(words)} words, {len(pauses)} pauses, {len(fillers)} fillers"
    )

    return {
        "transcript_path": transcript_out,
        "logs": [f"[{job_id}] speech: {len(words)} words, {len(pauses)} pauses in {elapsed}s"],
        "timings": {"speech": elapsed},
    }


def _llm_classify_speech(job_id: str, words: list, pauses: list) -> dict:
    """Ask the LLM to classify context-dependent fillers, restarts, and pause types."""
    from app.core.llm import get_llm

    llm = get_llm()
    lines = []
    for w in words:
        lines.append(f"[{w['start']:.2f}-{w['end']:.2f}] {w['word']}")
    for p in pauses:
        lines.append(f"  ** PAUSE {p['start']:.2f}-{p['end']:.2f} ({p['duration']:.1f}s) **")
    lines.sort(
        key=lambda l: float(l.split("[")[1].split("-")[0]) if "[" in l
        else float(l.split("PAUSE ")[1].split("-")[0])
    )
    transcript_text = "\n".join(lines)

    system_prompt = """You are a professional video editor analyzing a transcript to identify what to cut.

Return ONLY valid JSON:
{
  "context_dependent_fillers": [
    {"start": 1.23, "end": 1.45, "word": "like", "reason": "verbal filler, not meaningful"}
  ],
  "sentence_restarts": [
    {"start": 5.0, "end": 7.2, "reason": "speaker started then restarted"}
  ],
  "pause_classification": [
    {"start": 3.0, "end": 3.8, "type": "thinking_pause", "reason": "mid-thought pause"}
  ]
}"""

    logger.info(f"[{job_id}] Asking LLM to classify speech patterns...")
    try:
        return llm.ask_json(system_prompt, f"Analyze:\n\n{transcript_text}", temperature=0.1)
    except Exception as e:
        logger.warning(f"[{job_id}] LLM speech analysis failed: {e}, using basic detection only")
        return {"context_dependent_fillers": [], "sentence_restarts": [], "pause_classification": []}


def _merge_fillers(basic: list, llm_analysis: dict) -> list:
    merged = list(basic)
    seen = {(f["start"], f["end"]) for f in merged}
    for lf in llm_analysis.get("context_dependent_fillers", []):
        key = (lf["start"], lf["end"])
        if key not in seen:
            merged.append({
                "start": lf["start"], "end": lf["end"],
                "word": lf.get("word", ""), "confidence": 0.85,
                "source": "llm", "reason": lf.get("reason", ""),
            })
            seen.add(key)
    return merged
