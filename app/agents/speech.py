"""
Speech Intelligence Agent

Transcribes audio with WhisperX, then hands the transcript to the LLM so it can
figure out which words are real fillers vs just how the person naturally talks.
Also catches sentence restarts and classifies pauses.
"""

import re
from pathlib import Path

from app.agents.base import BaseAgent
from app.core.config import settings
from app.core.llm import get_llm
from app.core.workspace import JobWorkspace


class SpeechAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "speech"

    @property
    def output_artifact(self) -> str:
        return "transcript.json"

    def process(self, job_id: str, workspace: JobWorkspace) -> dict:
        audio_path = workspace.audio_path
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        device = settings.resolved_device
        self.logger.info(f"[{job_id}] Device: {device}")

        words = self._transcribe(audio_path, device, job_id)
        pauses = self._detect_pauses(words, job_id)
        basic_fillers = self._detect_obvious_fillers(words, job_id)

        # let the LLM analyze the transcript for nuanced stuff
        if words:
            llm_analysis = self._llm_classify_speech(words, pauses, job_id)
        else:
            self.logger.info(f"[{job_id}] No words found, skipping LLM analysis")
            llm_analysis = {"context_dependent_fillers": [], "sentence_restarts": [], "pause_classification": []}

        fillers = self._merge_filler_results(basic_fillers, llm_analysis, words)

        # mark fillers in the word list
        filler_ranges = {(f["start"], f["end"]) for f in fillers}
        for w in words:
            if (w["start"], w["end"]) in filler_ranges:
                w["is_filler"] = True

        # mark false starts the LLM found
        restart_ranges = llm_analysis.get("sentence_restarts", [])
        for w in words:
            for restart in restart_ranges:
                if w["start"] >= restart["start"] and w["end"] <= restart["end"]:
                    w["is_filler"] = True

        full_text = " ".join(w["word"] for w in words)
        total_speech = sum(w["end"] - w["start"] for w in words) if words else 0
        total_pause = sum(p["duration"] for p in pauses)

        self.logger.info(
            f"[{job_id}] {len(words)} words, {len(pauses)} pauses ({total_pause:.1f}s), "
            f"{len(fillers)} fillers, {len(restart_ranges)} restarts"
        )

        return {
            "words": words,
            "pauses": pauses,
            "fillers": fillers,
            "sentence_restarts": restart_ranges,
            "full_text": full_text,
            "total_speech_duration": round(total_speech, 3),
            "total_pause_duration": round(total_pause, 3),
            "llm_analysis": llm_analysis,
        }

    def _transcribe(self, audio_path: Path, device: str, job_id: str) -> list[dict]:
        import whisperx
        import torch

        compute_type = settings.COMPUTE_TYPE
        if device == "cpu":
            compute_type = "int8"

        self.logger.info(f"[{job_id}] Loading WhisperX ({settings.WHISPER_MODEL_SIZE})...")
        model = whisperx.load_model(
            settings.WHISPER_MODEL_SIZE, device=device,
            compute_type=compute_type, language=settings.WHISPER_LANGUAGE,
        )

        self.logger.info(f"[{job_id}] Transcribing...")
        audio = whisperx.load_audio(str(audio_path))
        result = model.transcribe(audio, batch_size=settings.WHISPER_BATCH_SIZE, language=settings.WHISPER_LANGUAGE)

        self.logger.info(f"[{job_id}] Aligning word timestamps...")
        align_model, align_metadata = whisperx.load_align_model(language_code=settings.WHISPER_LANGUAGE, device=device)
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

        del model, align_model
        if device == "cuda":
            torch.cuda.empty_cache()

        self.logger.info(f"[{job_id}] Got {len(words)} words")
        return words

    def _detect_pauses(self, words: list[dict], job_id: str) -> list[dict]:
        pauses = []
        min_pause = settings.PAUSE_MIN_DURATION

        for i in range(1, len(words)):
            gap = words[i]["start"] - words[i - 1]["end"]
            if gap >= min_pause:
                pauses.append({
                    "start": round(words[i - 1]["end"], 3),
                    "end": round(words[i]["start"], 3),
                    "duration": round(gap, 3),
                    "type": "long_pause" if gap > 2.0 else "silence",
                })

        return pauses

    def _detect_obvious_fillers(self, words: list[dict], job_id: str) -> list[dict]:
        """Catches the fillers that are never intentional — um, uh, ah, etc."""
        fillers = []
        for w in words:
            word_lower = w["word"].lower().strip(".,!?;:")
            if word_lower in settings.FILLER_WORDS:
                fillers.append({
                    "start": w["start"], "end": w["end"],
                    "word": w["word"], "confidence": 0.95, "source": "pattern",
                })
        return fillers

    def _llm_classify_speech(self, words: list[dict], pauses: list[dict], job_id: str) -> dict:
        """
        The interesting part — asks the LLM to read the transcript and figure out:
        - which "like" and "so" are filler vs actually part of the sentence
        - where the speaker restarted a sentence mid-way
        - which pauses are thinking vs dead air
        """
        llm = get_llm()

        transcript_lines = []
        for w in words:
            transcript_lines.append(f"[{w['start']:.2f}-{w['end']:.2f}] {w['word']}")
        for p in pauses:
            transcript_lines.append(f"  ** PAUSE at {p['start']:.2f}-{p['end']:.2f} ({p['duration']:.1f}s) **")

        transcript_lines.sort(
            key=lambda l: float(l.split("[")[1].split("-")[0]) if "[" in l else float(l.split("at ")[1].split("-")[0])
        )
        transcript_text = "\n".join(transcript_lines)

        system_prompt = """You are a professional video editor analyzing a transcript of a talking-head video recording.

Your job is to identify speech patterns that should be edited out to make the video feel professionally trimmed.

Analyze the transcript and identify:

1. **context_dependent_fillers**: Words like "like", "so", "well", "right", "basically", "actually", "you know", "I mean" that are being used as FILLERS (not intentionally). Only flag the ones used as verbal crutches, NOT when used meaningfully in a sentence.

2. **sentence_restarts**: Places where the speaker starts saying something, stops, and restarts. These are false starts. Include the timestamps of the aborted attempt that should be cut.

3. **pause_classification**: For each pause, classify it as:
   - "natural_breath" — keep it
   - "thinking_pause" — trim it down
   - "dead_air" — remove entirely

Return ONLY valid JSON:
{
  "context_dependent_fillers": [
    {"start": 1.23, "end": 1.45, "word": "like", "reason": "used as verbal filler before restarting thought"}
  ],
  "sentence_restarts": [
    {"start": 5.0, "end": 7.2, "reason": "speaker starts 'I think we should' then restarts with 'So what we need to do is'"}
  ],
  "pause_classification": [
    {"start": 3.0, "end": 3.8, "type": "thinking_pause", "reason": "speaker pauses mid-thought"}
  ]
}"""

        self.logger.info(f"[{job_id}] Asking LLM to analyze speech patterns...")

        try:
            result = llm.ask_json(system_prompt, f"Analyze this transcript:\n\n{transcript_text}", temperature=0.1)
            self.logger.info(
                f"[{job_id}] LLM found {len(result.get('context_dependent_fillers', []))} context fillers, "
                f"{len(result.get('sentence_restarts', []))} restarts"
            )
            return result
        except Exception as e:
            self.logger.warning(f"[{job_id}] LLM speech analysis failed: {e}, using basic detection only")
            return {"context_dependent_fillers": [], "sentence_restarts": [], "pause_classification": []}

    def _merge_filler_results(self, basic_fillers: list[dict], llm_analysis: dict, words: list[dict]) -> list[dict]:
        merged = list(basic_fillers)
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
