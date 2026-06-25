"""
src/agents/puzzle.py

Puzzle Generator — sixth LangGraph node in the pipeline.

Generates one puzzle question and correct answer from the approved
story text and moral lesson. Speaks the question via TTS immediately.
No rewrite loop — one attempt only.

If Gemini returns malformed JSON, falls back to a simple default
question based on the moral lesson. Session never stops for puzzle
generation failure.

Reads from state:
    story_text, moral_lesson, protagonist_name,
    child_age, child_name, session_id, total_tokens

Writes to state:
    puzzle_question, correct_answer, answer_keywords,
    hint_count, current_hint, total_tokens

Writes to specs:
    specs/{session_id}/puzzle.md
"""

import json
import re
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
import os

from src.errors import StoryAgentError
from src.error_handler import timed, safe_run
from src.logger import get_logger
from src.tools.tts import speak
from src.tools.spec_writer import write_spec

load_dotenv()
log = get_logger("puzzle")

# ── Gemini setup ──────────────────────────────────────────────────────────────
genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))

_settings = json.loads(
    (Path(__file__).parent.parent.parent / "SETTINGS.json").read_text()
)
GEMINI_MODEL = _settings.get("GEMINI_MODEL", "gemini-2.5-flash")

# Puzzle uses temperature=0.3 — some variation for question phrasing
# but mostly consistent structure
gemini_puzzle = genai.GenerativeModel(
    GEMINI_MODEL,
    generation_config=genai.types.GenerationConfig(temperature=0.3),
)

# ── Fallback questions by moral ───────────────────────────────────────────────
FALLBACK_QUESTIONS = {
    "sharing":         "What did {protagonist} do to help their friend?",
    "honesty":         "Why was it important for {protagonist} to tell the truth?",
    "courage":         "What brave thing did {protagonist} do in the story?",
    "kindness":        "How did {protagonist} show kindness to someone?",
    "patience":        "How did waiting help {protagonist} in the end?",
    "empathy":         "How did {protagonist} show they understood how someone felt?",
    "gratitude":       "What was {protagonist} thankful for in the story?",
    "perseverance":    "What did {protagonist} keep trying even when it was hard?",
    "responsibility":  "What important thing did {protagonist} take care of?",
    "forgiveness":     "How did {protagonist} show forgiveness in the story?",
    "generosity":      "What did {protagonist} give to make someone happy?",
    "respect":         "How did {protagonist} show respect for others?",
    "teamwork":        "How did working together help {protagonist} and their friends?",
    "confidence":      "What did {protagonist} believe they could do?",
    "compassion":      "How did {protagonist} help someone who was struggling?",
}

DEFAULT_FALLBACK = "What did {protagonist} learn in today's story?"


def _get_fallback_question(moral: str, protagonist: str) -> dict:
    """Returns a simple fallback question when Gemini fails."""
    template = FALLBACK_QUESTIONS.get(moral, DEFAULT_FALLBACK)
    question = template.format(protagonist=protagonist)
    return {
        "question":        question,
        "correct_answer":  f"Something about {moral}",
        "answer_keywords": [moral],
    }


# ── Puzzle generation ─────────────────────────────────────────────────────────

def _generate_puzzle(state: dict) -> tuple[dict, int]:
    """
    Calls Gemini to generate puzzle question, correct answer,
    and answer keywords.

    Returns (puzzle_dict, tokens_used).
    Falls back to default question on any failure.
    """
    session_id  = state["session_id"]
    age         = state.get("child_age", 5)
    moral       = state.get("moral_lesson", "kindness")
    protagonist = state.get("protagonist_name") or "the hero"
    story_text  = state.get("story_text", "")

    prompt = (
        f"You create puzzle questions for {age}-year-old children.\n\n"
        f"Story moral: '{moral}'\n"
        f"Protagonist: '{protagonist}'\n\n"
        f"Create ONE question that:\n"
        f"- Tests understanding of the moral, not plot details\n"
        f"- References a specific story moment to help the child remember\n"
        f"- Is answerable in one sentence by a {age}-year-old\n"
        f"- Has clear vocabulary appropriate for age {age}\n"
        f"- Has one clear correct answer but accepts paraphrasing\n\n"
        f"Do NOT ask 'what is the moral?' — that is too abstract.\n"
        f"Do NOT ask about colours, numbers, or trivial plot details.\n\n"
        f"Respond ONLY with valid JSON, no markdown:\n"
        f"{{\n"
        f'  "question": string,\n'
        f'  "correct_answer": string,\n'
        f'  "answer_keywords": [list of 3-5 key words/phrases that indicate correct answer]\n'
        f"}}\n\n"
        f"Story:\n{story_text[:1500]}"  # cap story to avoid huge prompt
    )

    try:
        response = gemini_puzzle.generate_content(prompt)
        raw      = response.text.strip()
        tokens   = getattr(
            response.usage_metadata, "total_token_count", 0
        ) or 0

        # Parse JSON
        clean  = re.sub(r"```json|```", "", raw).strip()
        puzzle = json.loads(clean)

        # Validate required keys
        if not all(k in puzzle for k in ("question", "correct_answer", "answer_keywords")):
            raise ValueError(f"Missing keys in puzzle JSON: {puzzle.keys()}")

        log.info(
            "puzzle_generated",
            session_id=session_id,
            question=puzzle["question"],
            moral=moral,
            tokens=tokens,
        )
        return puzzle, tokens

    except Exception as exc:
        log.warning(
            "puzzle_generation_failed_using_fallback",
            session_id=session_id,
            error=str(exc),
            moral=moral,
        )
        return _get_fallback_question(moral, protagonist), 0


# ── Spec file ─────────────────────────────────────────────────────────────────

def _write_puzzle_spec(state: dict, puzzle: dict) -> None:
    """Writes specs/{session_id}/puzzle.md atomically."""
    session_id = state["session_id"]
    keywords   = ", ".join(puzzle.get("answer_keywords", []))

    content = (
        f"# Puzzle\n\n"
        f"**Session:** {session_id}\n"
        f"**Moral tested:** {state.get('moral_lesson')}\n"
        f"**Protagonist:** {state.get('protagonist_name')}\n\n"
        f"## Question\n\n{puzzle['question']}\n\n"
        f"## Correct Answer\n\n{puzzle['correct_answer']}\n\n"
        f"## Answer Keywords\n\n{keywords}\n"
    )

    safe_run(
        write_spec,
        session_id, "puzzle.md", content,
        session_id=session_id,
    )


# ── Main node function ────────────────────────────────────────────────────────

@timed("puzzle", "puzzle_node")
def puzzle_node(state: dict) -> dict:
    """
    LangGraph node — Puzzle Generator.

    Generates question and correct answer.
    Speaks question via TTS.
    Writes puzzle.md spec.

    Args:
        state: LangGraph StoryState dict

    Returns:
        Updated state with puzzle_question, correct_answer,
        answer_keywords, hint_count, current_hint, total_tokens
    """
    session_id = state["session_id"]

    log.info(
        "puzzle_started",
        session_id=session_id,
        moral=state.get("moral_lesson"),
        child_age=state.get("child_age"),
    )

    # ── Generate puzzle ───────────────────────────────────────────────────────
    puzzle, tokens   = _generate_puzzle(state)
    total_tokens     = state.get("total_tokens", 0) + tokens

    # ── Write spec file ───────────────────────────────────────────────────────
    _write_puzzle_spec(state, puzzle)

    # ── Speak question via TTS ────────────────────────────────────────────────
    question_text  = puzzle["question"]
    tts_success    = safe_run(
        speak,
        question_text,
        session_id,
        default=False,
        session_id=session_id,
    )

    if not tts_success:
        log.warning(
            "puzzle_tts_failed_question_shown_in_ui",
            session_id=session_id,
            question=question_text,
        )

    log.info(
        "puzzle_node_complete",
        session_id=session_id,
        question=question_text,
        tts_success=tts_success,
        total_tokens=total_tokens,
    )

    return {
        **state,
        "puzzle_question":  question_text,
        "correct_answer":   puzzle["correct_answer"],
        "answer_keywords":  puzzle.get("answer_keywords", []),
        "hint_count":       0,
        "current_hint":     "",
        "total_tokens":     total_tokens,
    }
