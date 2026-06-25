"""
main.py

LangGraph graph definition and session runner.

Defines StoryState, builds the agent graph with all nodes and edges,
compiles with SQLite checkpointer for discussion prompt interrupt,
and exposes run_story_session() as the public API for app.py.

Graph structure:
    story_architect → writer → validator → [writer | narrator]
    narrator → discussion_prompt (interrupt)
    discussion_prompt → puzzle → voice_input
    voice_input → answer_validator → [voice_input | memory_save]
    memory_save → END

Post-session (outside graph):
    evaluate_trajectory(final_state)

Public API:
    run_story_session(profile, topic, story_length, on_story_chunk)
    resume_story_session(session_id)
"""

from dotenv import load_dotenv
load_dotenv()   # must run before any LangChain/LangGraph imports

import json
import os
import uuid
import time
import threading
from pathlib import Path
from typing import TypedDict, Optional, Callable, Literal

from langgraph.graph import StateGraph, END
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import NodeInterrupt

from src.tools.sandbox_manager import call_scorer
from src.agents.architect import architect_node
from src.agents.writer import writer_node
from src.agents.validator import validator_node
from src.agents.narrator import narrator_node
from src.agents.discussion_prompt import discussion_prompt_node
from src.agents.puzzle import puzzle_node
from src.agents.answer import answer_validator_node
from src.evaluation.trajectory import evaluate_trajectory
from src.memory.sqlite import (
    initialise_db, get_lessons_covered, save_session
)
from src.tools.stt import listen, delete_recording
from src.error_handler import safe_run
from src.logger import get_logger

log = get_logger("main")

# ── Callback registry — keeps non-serialisable callables out of LangGraph state ─
_story_chunk_callbacks: dict = {}

# ── Persistent SQLite checkpointer — survives app restarts ───────────────────
# check_same_thread=False required for Streamlit background threads
_checkpoint_conn = sqlite3.connect("./data/checkpoints.db", check_same_thread=False)
_checkpointer = SqliteSaver(_checkpoint_conn)

# ── Settings ──────────────────────────────────────────────────────────────────
_settings = json.loads(
    (Path(__file__).parent / "SETTINGS.json").read_text()
)
MAX_REWRITE_ATTEMPTS = _settings.get("MAX_REWRITE_ATTEMPTS", 1)
MAX_HINT_COUNT       = _settings.get("MAX_HINT_COUNT", 3)
SANDBOX_IMAGE        = _settings.get("SANDBOX_IMAGE", "storynest-sandbox")
SANDBOX_TIMEOUT      = _settings.get("SANDBOX_TIMEOUT_S", 10)
DATA_DIR             = Path(_settings.get("DATA_DIR", "./data"))
SPECS_DIR            = Path(_settings.get("SPECS_DIR", "./specs"))

# ── Initialise database on import ─────────────────────────────────────────────
initialise_db()


# ── StoryState ────────────────────────────────────────────────────────────────

class StoryState(TypedDict):
    session_id            : str
    session_started_at    : str
    child_name            : str
    child_age             : int
    interests             : list
    avoid                 : list
    lessons_covered       : list
    story_length          : str
    topic                 : str
    story_arc             : str
    moral_lesson          : str
    protagonist_name      : str
    fetched_facts         : str
    story_text            : str
    rewrite_attempts      : int
    rewrite_instructions  : str
    validation_score      : dict
    validation_passed     : bool
    validation_history    : list
    narration_failed      : bool
    awaiting_discussion   : bool
    discussion_complete   : bool
    discussion_prompt_text: str
    puzzle_question       : str
    correct_answer        : str
    answer_keywords       : list
    child_answer          : str
    pronunciation_score   : Optional[float]
    answer_result         : Literal["correct", "wrong", "unclear"]
    hint_count            : int
    current_hint          : str
    hints_given           : list
    total_tokens          : int


# ── Docker scorer ─────────────────────────────────────────────────────────────

def _call_scorer(audio_path: Path, session_id: str) -> Optional[float]:
    """
    Scores pronunciation via pre-warmed scorer container.
    Sends POST /score with filename — audio read from /data mount inside container.
    Returns score float or None on failure.
    """
    result = call_scorer(audio_path, session_id)
    if result.get("success"):
        score = float(result["score"])
        log.info("pronunciation_scored", session_id=session_id, score=score)
        return score
    log.warning("pronunciation_scorer_returned_failure",
                session_id=session_id, error=result.get("error"))
    return None


# ── Inline nodes ──────────────────────────────────────────────────────────────

def voice_input_node(state: StoryState) -> StoryState:
    """Records voice, transcribes, scores pronunciation, deletes audio."""
    session_id = state["session_id"]
    log.info("voice_input_started", session_id=session_id)

    text, audio_path = listen(session_id=session_id)

    pron_score = None
    if audio_path and audio_path.exists():
        pron_score = safe_run(
            _call_scorer, audio_path, session_id,
            default=None, session_id=session_id,
        )
        delete_recording(audio_path, session_id)

    log.info("voice_input_complete", session_id=session_id,
             transcribed=bool(text), pronunciation_score=pron_score)

    return {
        **state,
        "child_answer":        text or "",
        "pronunciation_score": pron_score,
    }


def memory_save_node(state: StoryState) -> StoryState:
    """Saves completed session to SQLite."""
    session_id = state["session_id"]
    safe_run(save_session, dict(state), session_id=session_id)
    log.info("session_saved", session_id=session_id,
             moral=state.get("moral_lesson"),
             answer_result=state.get("answer_result"),
             total_tokens=state.get("total_tokens"))
    return state


# ── Conditional edges ─────────────────────────────────────────────────────────

def route_validation(state: StoryState) -> str:
    """Routes after Validator — rewrite loop or narrator."""
    if state.get("validation_passed"):
        return "narrator"
    if state.get("rewrite_attempts", 0) >= MAX_REWRITE_ATTEMPTS:
        log.warning("max_rewrites_reached_using_current",
                    session_id=state["session_id"],
                    attempts=state.get("rewrite_attempts"))
        return "narrator"
    return "writer"


def route_answer(state: StoryState) -> str:
    """Routes after Answer Validator — hint loop or memory save."""
    result     = state.get("answer_result", "unclear")
    hint_count = state.get("hint_count", 0)
    if result == "correct":
        return "memory_save"
    if hint_count >= MAX_HINT_COUNT:
        return "memory_save"
    return "voice_input"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """Builds and compiles the LangGraph story agent graph."""
    g = StateGraph(StoryState)

    g.add_node("story_architect",   architect_node)
    g.add_node("writer",            writer_node)
    g.add_node("validator",         validator_node)
    g.add_node("narrator",          narrator_node)
    g.add_node("discussion_prompt", discussion_prompt_node)
    g.add_node("puzzle",            puzzle_node)
    g.add_node("voice_input",       voice_input_node)
    g.add_node("answer_validator",  answer_validator_node)
    g.add_node("memory_save",       memory_save_node)

    g.set_entry_point("story_architect")

    g.add_edge("story_architect",   "writer")
    g.add_edge("writer",            "validator")
    g.add_edge("narrator",          "discussion_prompt")
    g.add_edge("discussion_prompt", "puzzle")
    g.add_edge("puzzle",            "voice_input")
    g.add_edge("voice_input",       "answer_validator")
    g.add_edge("memory_save",       END)

    g.add_conditional_edges("validator", route_validation, {
        "writer":   "writer",
        "narrator": "narrator",
    })

    g.add_conditional_edges("answer_validator", route_answer, {
        "voice_input": "voice_input",
        "memory_save": "memory_save",
    })

    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=["discussion_prompt"],
    )


# ── Session runner ────────────────────────────────────────────────────────────

def run_story_session(
    profile       : dict,
    topic         : str,
    story_length  : str,
    on_story_chunk: Optional[Callable] = None,
    session_id    : Optional[str] = None,
) -> dict:
    """
    Runs a story session from start to discussion_prompt pause.
    Called by app.py in a background thread.

    session_id may be pre-generated by app.py so the UI can locate
    specs/{session_id}/story_final.md before the session completes.

    Returns partial state with awaiting_discussion=True when
    discussion_prompt interrupt fires. app.py shows Ready button.
    Parent taps Ready → app.py calls resume_story_session().
    """
    session_id = session_id or str(uuid.uuid4())[:8]
    SPECS_DIR.joinpath(session_id).mkdir(parents=True, exist_ok=True)

    if on_story_chunk:
        _story_chunk_callbacks[session_id] = on_story_chunk


    story_graph = build_graph(checkpointer=_checkpointer)
    config      = {"configurable": {"thread_id": session_id}}

    log.info("session_started", session_id=session_id,
             child_name=profile.get("child_name"),
             topic=topic or "(gemini picks)",
             story_length=story_length)

    lessons_covered = safe_run(
        get_lessons_covered, default=[], session_id=session_id,
    ) or []

    initial_state: StoryState = {
        "session_id":           session_id,
        "session_started_at":   "",
        "child_name":           profile.get("child_name", ""),
        "child_age":            profile.get("child_age", 5),
        "interests":            profile.get("interests", []),
        "avoid":                profile.get("avoid", []),
        "lessons_covered":      lessons_covered,
        "story_length":         story_length,
        "topic":                topic.strip() if topic else "",
        "story_arc":            "",
        "moral_lesson":         "",
        "protagonist_name":     "",
        "fetched_facts":        "",
        "story_text":           "",
        "rewrite_attempts":     0,
        "rewrite_instructions": "",
        "validation_score":     {},
        "validation_passed":    False,
        "validation_history":   [],
        "narration_failed":     False,
        "awaiting_discussion":  False,
        "discussion_complete":  False,
        "discussion_prompt_text": "",
        "puzzle_question":      "",
        "correct_answer":       "",
        "answer_keywords":      [],
        "child_answer":         "",
        "pronunciation_score":  None,
        "answer_result":        "unclear",
        "hint_count":           0,
        "current_hint":         "",
        "hints_given":          [],
        "total_tokens":         0,
    }

    try:
        result = story_graph.invoke(initial_state, config=config)

        # LangGraph 1.x: interrupt_before no longer raises NodeInterrupt —
        # invoke() returns the state snapshot silently. Check if the graph
        # is paused at an interrupt point by inspecting pending next nodes.
        graph_state = story_graph.get_state(config)
        if graph_state.next:
            log.info("session_paused_at_interrupt",
                     session_id=session_id,
                     next=list(graph_state.next),
                     story_text_len=len((graph_state.values or {}).get("story_text", "")))
            _story_chunk_callbacks.pop(session_id, None)
            # Use graph_state.values (the checkpoint) not invoke() return —
            # invoke() may return {} when hitting an interrupt in LangGraph 1.x
            return {
                **(graph_state.values or {}),
                "awaiting_discussion": True,
                "session_id":          session_id,
            }

        if result.get("narration_failed"):
            log.warning("session_stopped_narration_failed",
                        session_id=session_id)
            _story_chunk_callbacks.pop(session_id, None)
            return {**result, "error": "narration_failed"}

        _story_chunk_callbacks.pop(session_id, None)
        return result

    except NodeInterrupt:
        # Fallback for older LangGraph behaviour
        log.info("session_paused_node_interrupt", session_id=session_id)
        _story_chunk_callbacks.pop(session_id, None)
        return {
            **initial_state,
            "awaiting_discussion": True,
            "session_id":          session_id,
        }
    except Exception as exc:
        log.error("session_graph_failed", session_id=session_id,
                  error=str(exc), exc_info=True)
        _story_chunk_callbacks.pop(session_id, None)
        return {"error": str(exc), "session_id": session_id}


def resume_story_session(session_id: str) -> dict:
    """
    Resumes a paused session after parent taps Ready.
    Called by app.py when parent confirms discussion is done.
    Runs puzzle, voice input, answer loop, memory save, trajectory.
    """
    log.info("session_resuming", session_id=session_id)

    story_graph = build_graph(checkpointer=_checkpointer)
    config      = {"configurable": {"thread_id": session_id}}

    try:
        # Update the checkpoint to mark discussion as complete before resuming.
        # discussion_prompt_node checks awaiting_discussion to decide Pass 1 vs
        # Pass 2. The checkpoint was saved before the node ran (interrupt_before),
        # so awaiting_discussion=False there. Without this update the node raises
        # NodeInterrupt again on resume and the puzzle is never reached.
        story_graph.update_state(config, {"awaiting_discussion": True})

        final_state = story_graph.invoke(None, config=config)

        safe_run(evaluate_trajectory, final_state, session_id=session_id)
        _story_chunk_callbacks.pop(session_id, None)

        log.info("session_complete_after_resume", session_id=session_id,
                 moral=final_state.get("moral_lesson"),
                 answer_result=final_state.get("answer_result"),
                 total_tokens=final_state.get("total_tokens"))

        return final_state

    except Exception as exc:
        log.error("session_resume_failed", session_id=session_id,
                  error=str(exc), exc_info=True)
        _story_chunk_callbacks.pop(session_id, None)
        return {"error": str(exc), "session_id": session_id}
