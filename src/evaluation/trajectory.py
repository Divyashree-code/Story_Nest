"""
src/evaluation/trajectory.py

Post-session trajectory evaluator.

Scores five agent decisions from the completed session (1-3 each).
Runs AFTER LangGraph completes — cannot affect the current session.
Called from main.py once per session completion.

NOT a LangGraph node — plain Python function called post-session.

Five decisions scored:
    topic_selection    — age-appropriate, aligned with interests?
    moral_selection    — not recently covered, suitable?
    rewrite_quality    — did rewrites improve meaningfully?
    puzzle_difficulty  — genuinely answerable for this age?
    hint_effectiveness — hints guided without revealing?

Scores: 1=poor, 2=acceptable, 3=optimal
Overall: mean of five scores / 3 → 0-1 float

Outputs:
    1. SQLite trajectory_scores table (via sqlite.py)
    2. specs/{session_id}/trajectory.md (via spec_writer.py)
    3. app.log structured JSON entry with timestamp + full input

Temperature: 0.0 — consistent scoring across sessions.

Public API:
    evaluate_trajectory(state) -> dict
        Never raises. Returns neutral scores on any failure.
"""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
import os

from src.logger import get_logger
from src.memory.sqlite import save_trajectory
from src.tools.spec_writer import write_spec
from src.error_handler import safe_run

load_dotenv()
log = get_logger("trajectory")

# ── Gemini setup — temperature 0 for consistent scoring ──────────────────────
genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))

_settings = json.loads(
    (Path(__file__).parent.parent.parent / "SETTINGS.json").read_text()
)
GEMINI_MODEL = _settings.get("GEMINI_MODEL", "gemini-2.5-flash")

gemini_trajectory = genai.GenerativeModel(
    GEMINI_MODEL,
    generation_config=genai.types.GenerationConfig(temperature=0.0),
)

# ── Neutral fallback result ───────────────────────────────────────────────────
NEUTRAL_RESULT = {
    "step_scores": {
        "topic_selection":    2,
        "moral_selection":    2,
        "rewrite_quality":    2,
        "puzzle_difficulty":  2,
        "hint_effectiveness": 2,
    },
    "trajectory_score": 0.67,   # 2/3
    "weakest_step":     "unknown",
    "recommendation":   "",
}


# ── Session context builder ───────────────────────────────────────────────────

def _build_session_context(state: dict) -> dict:
    """
    Extracts relevant session data for trajectory evaluation.
    Stored in SQLite as JSON so Tab 3 can show input given.
    """
    return {
        "child_age":          state.get("child_age"),
        "interests":          state.get("interests", []),
        "topic":              state.get("topic"),
        "moral_lesson":       state.get("moral_lesson"),
        "lessons_covered":    state.get("lessons_covered", []),
        "rewrite_attempts":   state.get("rewrite_attempts", 0),
        "validation_history": state.get("validation_history", []),
        "puzzle_question":    state.get("puzzle_question"),
        "correct_answer":     state.get("correct_answer"),
        "child_answer":       state.get("child_answer"),
        "hint_count":         state.get("hint_count", 0),
        "answer_result":      state.get("answer_result"),
        "hints_given":        state.get("hints_given", []),
        "pronunciation_score": state.get("pronunciation_score"),
        "total_tokens":       state.get("total_tokens", 0),
    }


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_trajectory_prompt(context: dict) -> str:
    """Builds the trajectory evaluation prompt from session context."""

    validation_summary = ""
    history = context.get("validation_history", [])
    if history:
        scores_list = [
            f"Attempt {h['attempt']}: total={h['total']}, passed={h['passed']}"
            for h in history
        ]
        validation_summary = "\n".join(scores_list)
    else:
        validation_summary = "No validation history (first attempt passed)"

    hints_summary = ""
    hints = context.get("hints_given", [])
    if hints:
        hints_summary = "\n".join(
            f"Hint {h['hint_number']}: '{h['hint_text'][:60]}' "
            f"(child said: '{h['child_answer'][:40]}')"
            for h in hints
        )
    else:
        hints_summary = "No hints needed" if context.get("answer_result") == "correct" else "No hints given"

    return (
        f"You evaluate AI agent decisions for a children's storytelling system.\n\n"
        f"Score each of the 5 agent decisions 1-3:\n"
        f"  3 = optimal\n"
        f"  2 = acceptable, minor issues\n"
        f"  1 = poor, should be different next session\n\n"
        f"SESSION DATA:\n"
        f"  Child age: {context.get('child_age')}\n"
        f"  Interests: {context.get('interests')}\n"
        f"  Topic chosen: '{context.get('topic')}'\n"
        f"  Moral chosen: '{context.get('moral_lesson')}'\n"
        f"  Morals already covered: {context.get('lessons_covered')}\n"
        f"  Rewrite attempts: {context.get('rewrite_attempts')}\n"
        f"  Validation scores per attempt:\n{validation_summary}\n"
        f"  Puzzle question: '{context.get('puzzle_question')}'\n"
        f"  Correct answer: '{context.get('correct_answer')}'\n"
        f"  Child's answer: '{context.get('child_answer')}'\n"
        f"  Answer result: '{context.get('answer_result')}'\n"
        f"  Hint count: {context.get('hint_count')}\n"
        f"  Hints given:\n{hints_summary}\n\n"
        f"EVALUATION CRITERIA:\n"
        f"  topic_selection:    Was topic age-appropriate and interest-aligned?\n"
        f"  moral_selection:    Was moral not recently covered and suitable?\n"
        f"  rewrite_quality:    If rewrites happened, did scores improve each time?\n"
        f"                      Score 3 if no rewrites needed (passed first time).\n"
        f"  puzzle_difficulty:  Was puzzle genuinely answerable for age {context.get('child_age')}?\n"
        f"  hint_effectiveness: Did hints guide without revealing? "
        f"Did child answer correctly after hint?\n\n"
        f"Also provide:\n"
        f"  weakest_step:   the dimension with the lowest score\n"
        f"  recommendation: ONE specific sentence for the Story Architect to improve next session\n\n"
        f"Respond ONLY with valid JSON, no markdown:\n"
        f"{{\n"
        f'  "step_scores": {{\n'
        f'    "topic_selection": int,\n'
        f'    "moral_selection": int,\n'
        f'    "rewrite_quality": int,\n'
        f'    "puzzle_difficulty": int,\n'
        f'    "hint_effectiveness": int\n'
        f'  }},\n'
        f'  "weakest_step": string,\n'
        f'  "recommendation": string\n'
        f"}}"
    )


# ── Parse and validate scores ─────────────────────────────────────────────────

def _parse_trajectory_result(raw: str, session_id: str) -> dict:
    """
    Parses Gemini JSON response into trajectory result dict.
    Returns neutral result on parse failure.
    """
    try:
        clean  = re.sub(r"```json|```", "", raw).strip()
        parsed = json.loads(clean)

        step_scores = parsed.get("step_scores", {})
        required    = {
            "topic_selection", "moral_selection", "rewrite_quality",
            "puzzle_difficulty", "hint_effectiveness"
        }

        if not required.issubset(step_scores.keys()):
            raise ValueError(f"Missing step scores: {step_scores.keys()}")

        # Clamp scores to 1-3
        for key in required:
            step_scores[key] = max(1, min(3, int(step_scores[key])))

        # Calculate overall trajectory score (mean / 3 → 0-1)
        mean_score       = sum(step_scores.values()) / len(step_scores)
        trajectory_score = round(mean_score / 3, 4)

        # Find weakest step
        weakest = min(step_scores, key=step_scores.get)
        if parsed.get("weakest_step") and parsed["weakest_step"] in required:
            weakest = parsed["weakest_step"]

        return {
            "step_scores":      step_scores,
            "trajectory_score": trajectory_score,
            "weakest_step":     weakest,
            "recommendation":   parsed.get("recommendation", ""),
        }

    except Exception as exc:
        log.warning(
            "trajectory_parse_failed_using_neutral",
            session_id=session_id,
            error=str(exc),
            raw_preview=raw[:200],
        )
        return NEUTRAL_RESULT.copy()


# ── Spec file writer ──────────────────────────────────────────────────────────

def _write_trajectory_spec(
    session_id: str,
    result: dict,
    context: dict,
    evaluated_at: str,
) -> None:
    """Writes specs/{session_id}/trajectory.md atomically."""
    scores    = result["step_scores"]
    scores_md = "\n".join(
        f"- **{k.replace('_', ' ').title()}**: {v}/3"
        for k, v in scores.items()
    )

    context_md = json.dumps(context, indent=2, default=str)

    content = (
        f"# Trajectory Evaluation\n\n"
        f"**Session:** {session_id}\n"
        f"**Evaluated at:** {evaluated_at}\n"
        f"**Overall trajectory score:** {result['trajectory_score']:.2f}/1.0\n"
        f"**Weakest step:** {result['weakest_step']}\n\n"
        f"## Step Scores\n\n{scores_md}\n\n"
        f"## Recommendation for Next Session\n\n"
        f"{result['recommendation'] or '_No specific recommendation_'}\n\n"
        f"## Session Context (input given to evaluator)\n\n"
        f"```json\n{context_md}\n```\n"
    )

    safe_run(
        write_spec,
        session_id, "trajectory.md", content,
        session_id=session_id,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate_trajectory(state: dict) -> dict:
    """
    Post-session trajectory evaluation.
    Called from main.py after story_graph.invoke() completes.

    Never raises — returns neutral scores on any failure.
    Session is already complete so failure has no user-facing impact.

    Args:
        state: final LangGraph StoryState dict

    Returns:
        Trajectory result dict with step_scores, trajectory_score,
        weakest_step, recommendation, session_context, evaluated_at
    """
    session_id   = state["session_id"]
    evaluated_at = datetime.now(timezone.utc).isoformat()
    start        = time.perf_counter()

    log.info(
        "trajectory_evaluation_started",
        session_id=session_id,
        topic=state.get("topic"),
        moral=state.get("moral_lesson"),
        answer_result=state.get("answer_result"),
    )

    # ── Build session context ─────────────────────────────────────────────────
    context = _build_session_context(state)

    # ── Call Gemini ───────────────────────────────────────────────────────────
    try:
        prompt   = _build_trajectory_prompt(context)
        response = gemini_trajectory.generate_content(prompt)
        raw      = response.text.strip()
        result   = _parse_trajectory_result(raw, session_id)

    except Exception as exc:
        log.warning(
            "trajectory_gemini_failed_using_neutral",
            session_id=session_id,
            error=str(exc),
        )
        result = NEUTRAL_RESULT.copy()

    # Attach session context and timestamp to result
    result["session_context"] = context
    result["evaluated_at"]    = evaluated_at

    duration_ms = round((time.perf_counter() - start) * 1000)

    # ── Save to SQLite ────────────────────────────────────────────────────────
    safe_run(
        save_trajectory,
        result, session_id,
        session_id=session_id,
    )

    # ── Write spec file ───────────────────────────────────────────────────────
    _write_trajectory_spec(session_id, result, context, evaluated_at)

    # ── Log structured entry with timestamp + full input ──────────────────────
    log.info(
        "trajectory_evaluation_complete",
        session_id=session_id,
        trajectory_score=result["trajectory_score"],
        step_scores=result["step_scores"],
        weakest_step=result["weakest_step"],
        recommendation=result["recommendation"],
        session_context=context,   # full input logged here
        evaluated_at=evaluated_at,
        duration_ms=duration_ms,
    )

    return result
