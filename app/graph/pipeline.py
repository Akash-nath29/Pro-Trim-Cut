"""
LangGraph Pipeline Definition — Pro Auto-Trim

Graph flow:
  START → ingest
        → [speech ‖ vision] (parallel)
        → speech → semantic → timeline → cut_planner → quality_review → render → END
        → vision → END (writes to shared state, read by timeline)

Parallel execution of Speech and Vision is the key performance optimization.
Semantic runs after speech because it needs the transcript.
Vision writes its analysis to shared state; timeline reads it after semantic triggers.
"""

from typing import Any
from langgraph.graph import StateGraph, END, START

from app.graph.state import GraphState
from app.graph.nodes.ingest_node import ingest_node
from app.graph.nodes.speech_node import speech_node
from app.graph.nodes.vision_node import vision_node
from app.graph.nodes.semantic_node import semantic_node
from app.graph.nodes.timeline_node import timeline_node
from app.graph.nodes.cut_planner_node import cut_planner_node
from app.graph.nodes.quality_review_node import quality_review_node
from app.graph.nodes.render_node import render_node


def build_graph() -> StateGraph:
    """Build the Pro Auto-Trim LangGraph StateGraph."""
    graph = StateGraph(GraphState)

    # ── Register all nodes ────────────────────────────────────────────────
    graph.add_node("ingest", ingest_node)
    graph.add_node("speech", speech_node)
    graph.add_node("vision", vision_node)
    graph.add_node("semantic", semantic_node)
    graph.add_node("timeline", timeline_node)
    graph.add_node("cut_planner", cut_planner_node)
    graph.add_node("quality_review", quality_review_node)
    graph.add_node("render", render_node)

    # ── Sequential start ──────────────────────────────────────────────────
    graph.add_edge(START, "ingest")

    # ── Parallel fan-out: Speech + Vision run simultaneously after ingest ──
    graph.add_edge("ingest", "speech")
    graph.add_edge("ingest", "vision")

    # ── Semantic runs after speech (needs transcript) ─────────────────────
    graph.add_edge("speech", "semantic")

    # ── Vision branch terminates after writing to shared state ────────────
    graph.add_edge("vision", END)

    # ── Timeline runs after semantic (vision state is already available) ──
    graph.add_edge("semantic", "timeline")

    # ── Sequential tail ───────────────────────────────────────────────────
    graph.add_edge("timeline", "cut_planner")
    graph.add_edge("cut_planner", "quality_review")
    graph.add_edge("quality_review", "render")
    graph.add_edge("render", END)

    return graph


# module-level compiled graph (singleton — avoids recompiling per request)
_compiled: Any = None


def get_compiled_graph() -> Any:
    """Return the compiled graph, building it once and caching it."""
    global _compiled
    if _compiled is None:
        _compiled = build_graph().compile()
    return _compiled
