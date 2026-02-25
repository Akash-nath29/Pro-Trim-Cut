"""semantic tools"""

import json
import re

import numpy as np
from langchain_core.tools import tool


@tool
def sentence_segmentation(words: list, min_sentence_length: int = 5) -> list:
    """Break words into logical sentences based on punctuation, pauses, or length."""
    sentences = []
    current_words = []
    sentence_id = 0
    end_pattern = re.compile(r'[.!?]$')

    # fix whisperx bugs where one word spans like 8 seconds
    for w in words:
        if w["end"] - w["start"] > 2.0:
            w["end"] = round(w["start"] + 0.5, 3)

    for i, w in enumerate(words):
        current_words.append(w)

        is_end = bool(end_pattern.search(w["word"]))
        if i < len(words) - 1 and words[i + 1]["start"] - w["end"] > 0.5:
            is_end = True
        if len(current_words) >= 30:
            is_end = True

        if is_end:
            if len(current_words) >= 2:
                text = " ".join(cw["word"] for cw in current_words).strip()
                entry = {
                    "id": sentence_id,
                    "text": text,
                    "start": round(current_words[0]["start"], 3),
                    "end": round(current_words[-1]["end"], 3),
                    "word_count": len(current_words),
                    "embedding_index": sentence_id,
                }
                if len(text.split()) < min_sentence_length:
                    entry["is_fragment"] = True
                sentences.append(entry)
                sentence_id += 1
            # always flush on break, even single-word junk
            current_words = []

    if current_words and len(current_words) >= 2:
        text = " ".join(cw["word"] for cw in current_words).strip()
        entry = {
            "id": sentence_id,
            "text": text,
            "start": round(current_words[0]["start"], 3),
            "end": round(current_words[-1]["end"], 3),
            "word_count": len(current_words),
            "embedding_index": sentence_id,
        }
        if len(text.split()) < min_sentence_length:
            entry["is_fragment"] = True
        sentences.append(entry)

    return sentences


@tool
def embedding_generation(sentences: list, model_name: str = "all-MiniLM-L6-v2") -> list:
    """Generate semantic embeddings for sentences."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [s["text"] for s in sentences]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return [e.tolist() for e in embeddings]


@tool
def cluster_similar_takes(
    sentences: list,
    embeddings: list,
    threshold: float = 0.82,
) -> list:
    """Group sentences that are semantically similar (duplicate takes)."""
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

    if not sentences:
        return []

    emb_array = np.array(embeddings)
    sim_matrix = sk_cosine(emb_array)
    n = len(sentences)

    assigned = set()
    groups = []
    group_id = 0

    for i in range(n):
        if i in assigned:
            continue
        group_members = [i]
        for j in range(i + 1, n):
            if j not in assigned and sim_matrix[i][j] >= threshold:
                group_members.append(j)
        for m in group_members:
            assigned.add(m)

        best_idx = max(group_members, key=lambda idx: sentences[idx]["word_count"])
        is_duplicate = len(group_members) > 1

        groups.append({
            "group_id": group_id,
            "sentences": [sentences[m] for m in group_members],
            "representative_id": sentences[best_idx]["id"],
            "is_duplicate": is_duplicate,
            "similarity_score": round(
                float(max(sim_matrix[group_members[0]][m] for m in group_members[1:]))
                if is_duplicate else 0.0, 3,
            ),
        })
        group_id += 1

    return groups


@tool
def detect_rehearsal_phrases(
    sentences: list,
    groups: list,
    llm_json_str: str,
) -> dict:
    """Apply LLM's best-take and rehearsal decisions to the group data."""
    try:
        llm_result = json.loads(llm_json_str)
    except (json.JSONDecodeError, TypeError):
        return {
            "groups": groups,
            "rehearsal_ids": [],
            "llm_analysis": {"best_takes": {}, "rehearsal_sentences": [], "narrative_order": []},
        }

    best_takes = llm_result.get("best_takes", {})
    rehearsal_sentences = llm_result.get("rehearsal_sentences", [])

    for group in groups:
        gid_str = str(group["group_id"])
        if gid_str in best_takes:
            choice = best_takes[gid_str]
            best_id = choice.get("best_sentence_id")
            if best_id is not None:
                group_ids = {s["id"] for s in group["sentences"]}
                if best_id in group_ids:
                    group["representative_id"] = best_id
                    group["llm_reason"] = choice.get("reason", "")

    rehearsal_ids = {r["id"] for r in rehearsal_sentences if isinstance(r, dict)}
    for group in groups:
        for sent in group["sentences"]:
            if sent["id"] in rehearsal_ids:
                sent["is_rehearsal"] = True

    return {
        "groups": groups,
        "rehearsal_ids": list(rehearsal_ids),
        "llm_analysis": llm_result,
    }


SEMANTIC_TOOLS = [sentence_segmentation, embedding_generation, cluster_similar_takes, detect_rehearsal_phrases]
