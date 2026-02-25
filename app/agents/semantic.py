"""
Semantic Intelligence Agent

Groups sentences by meaning using embeddings, finds duplicate takes,
then asks the LLM to pick the best take from each group and spot rehearsal lines.
"""

import re
import json

import numpy as np

from app.agents.base import BaseAgent
from app.core.config import settings
from app.core.llm import get_llm
from app.core.workspace import JobWorkspace


class SemanticAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "semantic"

    @property
    def output_artifact(self) -> str:
        return "semantic.json"

    def process(self, job_id: str, workspace: JobWorkspace) -> dict:
        transcript = workspace.load_artifact("transcript.json")
        words = transcript["words"]

        if not words:
            self.logger.warning(f"[{job_id}] Empty transcript, nothing to analyze")
            return {"sentences": [], "groups": [], "duplicate_count": 0, "llm_analysis": {}}

        sentences = self._segment_sentences(words, job_id)
        if not sentences:
            return {"sentences": [], "groups": [], "duplicate_count": 0, "llm_analysis": {}}

        embeddings = self._generate_embeddings(sentences, job_id)
        groups = self._cluster_duplicates(sentences, embeddings, job_id)

        # let the LLM pick the best takes and find rehearsals
        llm_analysis = self._llm_analyze_takes(sentences, groups, job_id)
        self._apply_llm_decisions(groups, llm_analysis, job_id)

        duplicate_count = sum(1 for g in groups if g["is_duplicate"])
        self.logger.info(f"[{job_id}] {len(sentences)} sentences, {len(groups)} groups, {duplicate_count} duplicates")

        return {
            "sentences": sentences,
            "groups": groups,
            "duplicate_count": duplicate_count,
            "llm_analysis": llm_analysis,
        }

    def _segment_sentences(self, words: list[dict], job_id: str) -> list[dict]:
        sentences = []
        current_words = []
        sentence_id = 0
        end_pattern = re.compile(r'[.!?]$')

        for i, w in enumerate(words):
            current_words.append(w)

            is_end = False
            if end_pattern.search(w["word"]):
                is_end = True
            if i < len(words) - 1 and words[i + 1]["start"] - w["end"] > 0.5:
                is_end = True
            if len(current_words) >= 30:
                is_end = True

            if is_end and len(current_words) >= 2:
                text = " ".join(cw["word"] for cw in current_words).strip()
                if len(text.split()) >= settings.MIN_SENTENCE_LENGTH:
                    sentences.append({
                        "id": sentence_id,
                        "text": text,
                        "start": round(current_words[0]["start"], 3),
                        "end": round(current_words[-1]["end"], 3),
                        "word_count": len(current_words),
                        "embedding_index": sentence_id,
                    })
                    sentence_id += 1
                current_words = []

        # leftover words
        if current_words and len(current_words) >= 2:
            text = " ".join(cw["word"] for cw in current_words).strip()
            if len(text.split()) >= settings.MIN_SENTENCE_LENGTH:
                sentences.append({
                    "id": sentence_id, "text": text,
                    "start": round(current_words[0]["start"], 3),
                    "end": round(current_words[-1]["end"], 3),
                    "word_count": len(current_words),
                    "embedding_index": sentence_id,
                })

        self.logger.info(f"[{job_id}] Segmented into {len(sentences)} sentences")
        return sentences

    def _generate_embeddings(self, sentences: list[dict], job_id: str) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        self.logger.info(f"[{job_id}] Loading {settings.EMBEDDING_MODEL}...")
        model = SentenceTransformer(settings.EMBEDDING_MODEL)
        texts = [s["text"] for s in sentences]
        self.logger.info(f"[{job_id}] Encoding {len(texts)} sentences...")
        return np.array(model.encode(texts, show_progress_bar=False, normalize_embeddings=True))

    def _cluster_duplicates(self, sentences: list[dict], embeddings: np.ndarray, job_id: str) -> list[dict]:
        from sklearn.metrics.pairwise import cosine_similarity

        n = len(sentences)
        if n == 0:
            return []

        sim_matrix = cosine_similarity(embeddings)
        threshold = settings.SIMILARITY_THRESHOLD

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

            # default: pick longest (LLM will override if it has a better pick)
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

        self.logger.info(f"[{job_id}] {len(groups)} groups, {sum(1 for g in groups if g['is_duplicate'])} with duplicates")
        return groups

    def _llm_analyze_takes(self, sentences: list[dict], groups: list[dict], job_id: str) -> dict:
        """
        Asks the LLM: "you're a pro editor — which take sounds best,
        and which lines are just the creator practicing?"
        """
        llm = get_llm()

        sentences_data = [{"id": s["id"], "text": s["text"], "start": s["start"], "end": s["end"], "word_count": s["word_count"]} for s in sentences]
        duplicate_groups = [{"group_id": g["group_id"], "takes": [{"id": s["id"], "text": s["text"], "start": s["start"]} for s in g["sentences"]]} for g in groups if g["is_duplicate"]]

        system_prompt = """You are a professional video editor choosing the best takes from a raw talking-head video.

1. **Choose the best take** for each duplicate group — the one that sounds most natural, confident, and complete.

2. **Identify rehearsal/practice lines** — things like "Let me try that again", mumbled first attempts, the speaker talking to themselves about how to say something, mic checks, etc.

3. **Suggest the narrative order** of sentences that would make the best final video.

Return ONLY valid JSON:
{
  "best_takes": {
    "<group_id>": {"best_sentence_id": <id>, "reason": "why this take wins"}
  },
  "rehearsal_sentences": [
    {"id": <sentence_id>, "reason": "why this is practice, not real content"}
  ],
  "narrative_order": [<sentence IDs in the best order>],
  "editing_notes": "brief notes on the overall approach"
}"""

        self.logger.info(f"[{job_id}] Asking LLM to pick best takes from {len(duplicate_groups)} groups...")

        try:
            result = llm.ask_json(
                system_prompt,
                f"ALL SENTENCES:\n{json.dumps(sentences_data, indent=2)}\n\nDUPLICATE GROUPS:\n{json.dumps(duplicate_groups, indent=2)}",
                temperature=0.1,
            )
            self.logger.info(f"[{job_id}] LLM: {len(result.get('best_takes', {}))} take decisions, {len(result.get('rehearsal_sentences', []))} rehearsals")
            if result.get("editing_notes"):
                self.logger.info(f"[{job_id}] LLM says: {result['editing_notes']}")
            return result
        except Exception as e:
            self.logger.warning(f"[{job_id}] LLM semantic analysis failed: {e}")
            return {"best_takes": {}, "rehearsal_sentences": [], "narrative_order": [s["id"] for s in sentences], "editing_notes": "LLM unavailable"}

    def _apply_llm_decisions(self, groups: list[dict], llm_analysis: dict, job_id: str) -> None:
        best_takes = llm_analysis.get("best_takes", {})

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

        rehearsal_ids = {r["id"] for r in llm_analysis.get("rehearsal_sentences", []) if isinstance(r, dict)}
        for group in groups:
            for sent in group["sentences"]:
                if sent["id"] in rehearsal_ids:
                    sent["is_rehearsal"] = True
