"""
src/agents/answer.py

Answer Validator — seventh LangGraph agent node.

Judges the child's transcribed spoken answer against the correct answer.
Generates contextual hints if wrong. Speaks hint + question via TTS.
Routes to memory save on correct, loops back on wrong/unclear.

Three outcomes:
    correct — essence of moral captured, generous judgment
    wrong   — missed the point, hint generated and spoken
    unclear — answer too short/quiet, re-ask without penalty

Hint progression (controlled by hint_count in state):
    hint_count=0 → gentle hint referencing story moment
    hint_count=1 → more direct, almost gives it away
    hint_count=2 → very direct, kind reveal

After MAX_HINT_COUNT reached, route_answer() in main.py
routes to memory save. This node speaks a kind farewell message.

Reads from state:
    child_answer, puzzle_question, correct_answer, answer_keywords,
    moral_lesson, story_text, protagonist_name,
    hint_count, current_hint, hints_given,
    pronunciation_score, session_id, total_tokens,
    child_age, child_name

Writes to state:
    answer_result, current_hint, hint_count,
    hints_given, total_tokens
"""

import json
import re
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
import os

from src.errors import AnswerValidatorError
from src.error_handler import timed, safe_run
from src.logger import get_logger
from src.tools.tts import speak, speak_hint

load_dotenv()
log = get_logger("answer_validator")

# ── Gemini setup ──────────────────────────────────────────────────────────────
genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))

_settings = json.loads(
    (Path(__file__).parent.parent.parent / "SETTINGS.json").read_text()
)
GEMINI_MODEL   = _settings.get("GEMINI_MODEL", "gemini-2.5-flash")
MAX_HINT_COUNT = _settings.get("MAX_HINT_COUNT", 3)

# Answer validator uses temperature=0.3
# Low enough for consistent judgment, slight variation for hint phrasing
gemini_answer = genai.GenerativeModel(
    GEMINI_MODEL,
    generation_config=genai.types.GenerationConfig(temperature=0.3),
)

# ── Kind farewell when max hints reached ──────────────────────────────────────
MAX_HINTS_MESSAGE = (
    "Great effort today! You listened so well to the story. "
    "The answer was {correct_answer}. "
    "Well done for trying so hard!"
)


def _build_judge_prompt(state: dict) -> str:
    """
    Builds the LLM Judge prompt for answer evaluation.
    Includes pronunciation score as additional context.
    """
    age          = state.get("child_age", 5)
    hint_count   = state.get("hint_count", 0)
    pron_score   = state.get("pronunciation_score")
    child_answer = state.get("child_answer", "")

    # Pronunciation context for Gemini
    pron_note = ""
    if pron_score is not None:
        if pron_score < 0.4:
            pron_note = (
                f"\nPronunciation clarity score: {pron_score:.2f}/1.0 (low). "
                f"Child may not have spoken clearly — be more lenient with 'unclear' judgment."
            )
        elif pron_score >= 0.7:
            pron_note = (
                f"\nPronunciation clarity score: {pron_score:.2f}/1.0 (high). "
                f"Child spoke clearly — their words are likely well captured."
            )

    # Hint directness based on count
    if hint_count == 0:
        hint_instruction = (
            "If wrong: give a GENTLE hint referencing a specific story moment. "
            "Do not give away the answer. Just nudge them toward it."
        )
    elif hint_count == 1:
        hint_instruction = (
            "If wrong: give a MORE DIRECT hint. The child has tried twice. "
            "You can almost give away the answer but leave the last step to them."
        )
    else:
        hint_instruction = (
            "If wrong: give a VERY DIRECT hint. The child has tried many times. "
            "Make the answer very obvious — just one small step from revealing it."
        )

    return (
        f"You are judging a {age}-year-old child's verbal answer to a story puzzle.\n\n"
        f"Be GENEROUS — accept paraphrasing, related ideas, partial answers.\n"
        f"A {age}-year-old uses simple language — do not penalise grammar or short answers.\n"
        f"If the child captured the ESSENCE of the moral, mark as correct.\n\n"
        f"Story moral: '{state.get('moral_lesson')}'\n"
        f"Correct answer: '{state.get('correct_answer')}'\n"
        f"Answer keywords: {state.get('answer_keywords', [])}\n"
        f"Child said: '{child_answer}'\n"
        f"Hints given so far: {hint_count}{pron_note}\n\n"
        f"If child said nothing or only 1-2 meaningless words → 'unclear'\n"
        f"If child captured the moral essence → 'correct'\n"
        f"If child tried but missed the point → 'wrong'\n\n"
        f"{hint_instruction}\n\n"
        f"The hint must reference the story protagonist: "
        f"'{state.get('protagonist_name', 'the hero')}'\n\n"
        f"Respond ONLY with valid JSON, no markdown:\n"
        f"{{\n"
        f'  "result": "correct" | "wrong" | "unclear",\n'
        f'  "hint": string  (empty string if correct or unclear)\n'
        f"}}"
    )


def _judge_answer(state: dict) -> tuple[str, str, int]:
    """
    Calls Gemini to judge the child's answer.
    Returns (result, hint_text, tokens_used).
    Falls back to 'unclear' on any failure.
    """
    session_id = state["session_id"]

    try:
        prompt   = _build_judge_prompt(state)
        response = gemini_answer.generate_content(prompt)
        raw      = response.text.strip()
        tokens   = getattr(
            response.usage_metadata, "total_token_count", 0
        ) or 0

        clean  = re.sub(r"```json|```", "", raw).strip()
        parsed = json.loads(clean)

        result = parsed.get("result", "unclear")
        hint   = parsed.get("hint", "")

        # Validate result is one of the three valid values
        if result not in ("correct", "wrong", "unclear"):
            log.warning(
                "answer_validator_invalid_result",
                session_id=session_id,
                result=result,
            )
            result = "unclear"

        log.info(
            "answer_judged",
            session_id=session_id,
            result=result,
            hint_count=state.get("hint_count", 0),
            child_answer=state.get("child_answer", "")[:50],
            has_hint=bool(hint),
        )
        return result, hint, tokens

    except Exception as exc:
        log.warning(
            "answer_validator_failed_defaulting_unclear",
            session_id=session_id,
            error=str(exc),
        )
        return "unclear", "", 0


# ── Main node function ────────────────────────────────────────────────────────

@timed("answer_validator", "answer_validator_node")
def answer_validator_node(state: dict) -> dict:
    """
    LangGraph node — Answer Validator.

    Judges child's answer, generates hint if wrong,
    speaks hint + question via TTS, updates hint history.

    Args:
        state: LangGraph StoryState dict

    Returns:
        Updated state with answer_result, current_hint,
        hint_count, hints_given, total_tokens
    """
    session_id   = state["session_id"]
    hint_count   = state.get("hint_count", 0)
    child_answer = state.get("child_answer", "")

    log.info(
        "answer_validator_started",
        session_id=session_id,
        hint_count=hint_count,
        child_answer=child_answer[:50],
        pronunciation_score=state.get("pronunciation_score"),
    )

    # ── Judge the answer ──────────────────────────────────────────────────────
    result, hint_text, tokens = _judge_answer(state)
    total_tokens = state.get("total_tokens", 0) + tokens

    # ── Update hints history ──────────────────────────────────────────────────
    hints_given = list(state.get("hints_given", []))

    if result == "wrong" and hint_text:
        new_hint_count = hint_count + 1
        hints_given.append({
            "hint_number":  new_hint_count,
            "hint_text":    hint_text,
            "child_answer": child_answer,
        })

        # Speak hint + question via TTS
        safe_run(
            speak_hint,
            hint_text,
            state.get("puzzle_question", ""),
            session_id,
            default=False,
            session_id=session_id,
        )

        log.info(
            "hint_spoken",
            session_id=session_id,
            hint_number=new_hint_count,
            hint_text=hint_text[:60],
        )

    elif result == "correct":
        new_hint_count = hint_count
        # Speak congratulations
        congrats = (
            f"Well done, {state.get('child_name', 'superstar')}! "
            f"That is exactly right!"
        )
        safe_run(speak, congrats, session_id,
                 default=False, session_id=session_id)

    elif result == "unclear":
        new_hint_count = hint_count   # unclear does not increment hint count
        # Speak gentle re-ask
        rephrase = (
            f"I did not quite catch that. "
            f"Could you try again? {state.get('puzzle_question', '')}"
        )
        safe_run(speak, rephrase, session_id,
                 default=False, session_id=session_id)

    else:
        new_hint_count = hint_count

    # ── Check if max hints reached — speak farewell ───────────────────────────
    if result == "wrong" and new_hint_count >= MAX_HINT_COUNT:
        farewell = MAX_HINTS_MESSAGE.format(
            correct_answer=state.get("correct_answer", "something wonderful")
        )
        safe_run(speak, farewell, session_id,
                 default=False, session_id=session_id)
        log.info(
            "max_hints_reached_ending_puzzle",
            session_id=session_id,
            hint_count=new_hint_count,
        )

    log.info(
        "answer_validator_complete",
        session_id=session_id,
        result=result,
        hint_count=new_hint_count,
        total_tokens=total_tokens,
    )

    return {
        **state,
        "answer_result": result,
        "current_hint":  hint_text,
        "hint_count":    new_hint_count,
        "hints_given":   hints_given,
        "total_tokens":  total_tokens,
    }
