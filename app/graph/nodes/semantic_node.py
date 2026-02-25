"""
Semantic Node — LangGraph node for semantic duplicate detection and take analysis.

Depends on: transcript_path (written by speech_node)
Tools used: sentence_segmentation, embedding_generation, cluster_similar_takes, detect_rehearsal_phrases
Writes to state: semantic_analysis_path
"""

import json
import time
from pathlib import Path

from app.graph.state import GraphState
from app.tools.semantic_tools import (
    sentence_segmentation, embedding_generation,
    cluster_similar_takes, detect_rehearsal_phrases,
)
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("node.semantic")


def semantic_node(state: GraphState) -> dict:
    """LangGraph node: segment sentences, embed them, cluster duplicates, detect rehearsals."""
    job_id = state["job_id"]
    transcript_path = state["transcript_path"]
    jobs_dir = state["jobs_dir"]

    job_dir = Path(jobs_dir) / job_id
    semantic_out = str(job_dir / "semantic.json")

    # skip if cached
    if Path(semantic_out).exists():
        logger.info(f"[{job_id}] ⏭ semantic already done, using cached result")
        return {"semantic_analysis_path": semantic_out, "logs": [f"[{job_id}] semantic: cached"]}

    logger.info(f"[{job_id}] ▶ semantic starting")
    t0 = time.perf_counter()

    with open(transcript_path, encoding="utf-8") as f:
        transcript = json.load(f)

    words = transcript.get("words", [])

    if not words:
        logger.warning(f"[{job_id}] Empty transcript — skipping semantic analysis")
        result = {"sentences": [], "groups": [], "duplicate_count": 0, "llm_analysis": {}}
        with open(semantic_out, "w") as f:
            json.dump(result, f, indent=2)
        return {
            "semantic_analysis_path": semantic_out,
            "logs": [f"[{job_id}] semantic: skipped (no words)"],
            "timings": {"semantic": 0.0},
        }

    # ── Tool 1: segment into sentences ───────────────────────────────────
    sentences = sentence_segmentation.invoke({
        "words": words,
        "min_sentence_length": settings.MIN_SENTENCE_LENGTH,
    })
    logger.info(f"[{job_id}] Segmented into {len(sentences)} sentences")

    if not sentences:
        result = {"sentences": [], "groups": [], "duplicate_count": 0, "llm_analysis": {}}
        with open(semantic_out, "w") as f:
            json.dump(result, f, indent=2)
        return {"semantic_analysis_path": semantic_out, "logs": [f"[{job_id}] semantic: no valid sentences"]}

    # ── Tool 2: generate embeddings ───────────────────────────────────────
    logger.info(f"[{job_id}] Generating embeddings with {settings.EMBEDDING_MODEL}...")
    embeddings = embedding_generation.invoke({
        "sentences": sentences,
        "model_name": settings.EMBEDDING_MODEL,
    })

    # ── Tool 3: cluster duplicates ────────────────────────────────────────
    groups = cluster_similar_takes.invoke({
        "sentences": sentences,
        "embeddings": embeddings,
        "threshold": settings.SIMILARITY_THRESHOLD,
    })
    dup_count = sum(1 for g in groups if g["is_duplicate"])
    logger.info(f"[{job_id}] {len(groups)} groups, {dup_count} with duplicates")

    # ── LLM: pick best takes and identify rehearsals ──────────────────────
    llm_json_str = _ask_llm_for_takes(job_id, sentences, groups)

    # ── Tool 4: apply LLM decisions ───────────────────────────────────────
    applied = detect_rehearsal_phrases.invoke({
        "sentences": sentences,
        "groups": groups,
        "llm_json_str": llm_json_str,
    })
    final_groups = applied["groups"]
    llm_analysis = applied["llm_analysis"]

    result = {
        "sentences": sentences,
        "groups": final_groups,
        "duplicate_count": dup_count,
        "llm_analysis": llm_analysis,
    }

    with open(semantic_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    elapsed = round(time.perf_counter() - t0, 3)
    logger.info(f"[{job_id}] ✓ semantic done in {elapsed:.2f}s")

    return {
        "semantic_analysis_path": semantic_out,
        "current_stage": "analyzed_semantic",
        "progress": 55.0,
        "logs": [f"[{job_id}] semantic: {len(sentences)} sentences, {dup_count} duplicates in {elapsed}s"],
        "timings": {"semantic": elapsed},
    }


def _ask_llm_for_takes(job_id: str, sentences: list, groups: list) -> str:
    """Ask LLM to pick best takes from duplicate groups and flag rehearsal lines."""
    import json as _json
    from app.core.llm import get_llm

    llm = get_llm()
    sentences_data = [
        {"id": s["id"], "text": s["text"], "start": s["start"], "end": s["end"], "word_count": s["word_count"]}
        for s in sentences
    ]
    dup_groups = [
        {"group_id": g["group_id"], "takes": [
            {"id": s["id"], "text": s["text"], "start": s["start"]}
            for s in g["sentences"]
        ]}
        for g in groups if g["is_duplicate"]
    ]

    system_prompt = """You are a professional video editor choosing best takes.

1. Choose the best take for each duplicate group (most natural, confident, complete).
2. Identify rehearsal/practice lines.
3. Suggest narrative sentence order.

Return ONLY valid JSON:
{
  "best_takes": {
    "<group_id>": {"best_sentence_id": <id>, "reason": "why this take wins"}
  },
  "rehearsal_sentences": [
    {"id": <sentence_id>, "reason": "why this is practice"}
  ],
  "narrative_order": [<sentence IDs in best order>],
  "editing_notes": "brief notes"
}"""

    logger.info(f"[{job_id}] Asking LLM to pick best takes from {len(dup_groups)} duplicate groups...")
    try:
        result = llm.ask_json(
            system_prompt,
            f"ALL SENTENCES:\n{_json.dumps(sentences_data, indent=2)}\n\n"
            f"DUPLICATE GROUPS:\n{_json.dumps(dup_groups, indent=2)}",
            temperature=0.1,
        )
        if result.get("editing_notes"):
            logger.info(f"[{job_id}] LLM: {result['editing_notes']}")
        return _json.dumps(result)
    except Exception as e:
        logger.warning(f"[{job_id}] LLM semantic analysis failed: {e}")
        return _json.dumps({
            "best_takes": {}, "rehearsal_sentences": [],
            "narrative_order": [s["id"] for s in sentences],
            "editing_notes": f"LLM unavailable: {e}",
        })
