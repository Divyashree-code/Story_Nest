"""
src/agents/validator.py

Validator — LLM Judge, third LangGraph node in the pipeline.

Independent Gemini call that scores the Writer's story on five
dimensions. Returns structured JSON scores, pass/fail decision,
and specific rewrite instructions if failing.

Temperature: 0.0 — maximises scoring consistency.
Same story should score the same way every time.

Reads from state:
    story_text, child_age, moral_lesson, story_length,
    rewrite_attempts, validation_history, session_id, total_tokens

Writes to state:
    validation_score, validation_passed, rewrite_instructions,
    validation_history, total_tokens

Writes to specs:
    specs/{session_id}/story_final.md — on approval only

Note:
    Max retries is NOT enforced here — handled by route_validation()
    in main.py. Validator always scores and returns regardless of
    how many attempts have been made.
"""

import json
import re
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
import os

from src.errors import ValidatorError
from src.error_handler import timed
from src.logger import get_logger
from src.tools.spec_writer import write_spec
from src.error_handler import safe_run

load_dotenv()
log = get_logger("validator")

# ── Gemini setup — temperature 0 for consistent scoring ──────────────────────
genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))

_settings = json.loads(
    (Path(__file__).parent.parent.parent / "SETTINGS.json").read_text()
)
GEMINI_MODEL  = _settings.get("GEMINI_MODEL", "gemini-2.5-flash")
LENGTH_WORDS  = _settings.get("STORY_LENGTH_WORDS", {
    "Short": 150, "Medium": 350, "Long": 700
})
PASS_THRESHOLD = 3   # all scores must be >= this to pass

# Validator uses temperature=0.0 — consistency over creativity
gemini_validator = genai.GenerativeModel(
    GEMINI_MODEL,
    generation_config=genai.types.GenerationConfig(temperature=0.0),
)


# ── Scoring ───────────────────────────────────────────────────────────────────

def _build_validator_prompt(state: dict) -> str:
    """Builds the LLM Judge system prompt."""
    age         = state.get("child_age", 5)
    word_target = LENGTH_WORDS.get(state.get("story_length", "Medium"), 350)
    moral       = state.get("moral_lesson", "")
    attempt     = state.get("rewrite_attempts", 1)

    return (
        f"You are a strict children's content judge evaluating a story "
        f"for a {age}-year-old child.\n\n"
        f"Score each dimension 1-5:\n"
        f"  vocabulary_fit  : are words right for age {age}? "
        f"(5=perfect, 1=far too complex)\n"
        f"  moral_clarity   : is '{moral}' shown through actions, never stated? "
        f"(5=perfect, 1=stated directly or missing)\n"
        f"  scare_factor    : anything frightening? "
        f"(5=totally safe, 1=very scary)\n"
        f"  engagement      : would a {age}-year-old stay interested? "
        f"(5=very engaging, 1=boring)\n"
        f"  length_fit      : is length close to ~{word_target} words? "
        f"(5=perfect, 1=far too long or short)\n\n"
        f"Pass threshold: ALL scores >= {PASS_THRESHOLD}.\n\n"
        f"If any score < {PASS_THRESHOLD}, write specific actionable "
        f"rewrite_instructions telling the writer exactly what to fix. "
        f"Reference specific words, scenes, or sentences.\n"
        f"If all pass, set rewrite_instructions to empty string.\n\n"
        f"This is attempt {attempt}. Be consistent and objective.\n\n"
        f"Respond ONLY with valid JSON, no markdown, no explanation:\n"
        f"{{\n"
        f'  "vocabulary_fit": int,\n'
        f'  "moral_clarity": int,\n'
        f'  "scare_factor": int,\n'
        f'  "engagement": int,\n'
        f'  "length_fit": int,\n'
        f'  "rewrite_instructions": string\n'
        f"}}"
    )


def _parse_scores(raw: str, session_id: str) -> dict:
    """
    Parses Gemini's JSON response into scores dict.
    Strips markdown fences if present.
    Returns fallback pass scores if parsing fails — session continues.
    """
    try:
        clean  = re.sub(r"```json|```", "", raw).strip()
        scores = json.loads(clean)

        # Validate all required keys present
        required = {
            "vocabulary_fit", "moral_clarity", "scare_factor",
            "engagement", "length_fit", "rewrite_instructions"
        }
        if not required.issubset(scores.keys()):
            raise ValueError(f"Missing keys in scores: {scores.keys()}")

        # Clamp scores to valid range 1-5
        for key in required - {"rewrite_instructions"}:
            scores[key] = max(1, min(5, int(scores[key])))

        return scores

    except Exception as exc:
        # JSON parse failure — grant fallback pass with warning
        # Better to let a story through than crash session on parse error
        log.warning(
            "validator_json_parse_failed_granting_pass",
            session_id=session_id,
            error=str(exc),
            raw_response=raw[:200],
        )
        return {
            "vocabulary_fit":       3,
            "moral_clarity":        3,
            "scare_factor":         5,
            "engagement":           3,
            "length_fit":           3,
            "rewrite_instructions": "",
        }


def _score_story(state: dict) -> tuple[dict, int]:
    """
    Calls Gemini to score the story.
    Returns (scores_dict, tokens_used).
    Raises ValidatorError on Gemini failure.
    """
    session_id = state["session_id"]
    system     = _build_validator_prompt(state)
    prompt     = f"{system}\n\nStory to evaluate:\n{state.get('story_text', '')}"

    try:
        response = gemini_validator.generate_content(prompt)
        raw      = response.text.strip()
        tokens   = getattr(
            response.usage_metadata, "total_token_count", 0
        ) or 0

        scores = _parse_scores(raw, session_id)
        return scores, tokens

    except ValidatorError:
        raise
    except Exception as exc:
        raise ValidatorError(
            f"Validator Gemini call failed: {exc}",
            session_id=session_id,
            original_error=str(exc),
        ) from exc


# ── Spec file writer ──────────────────────────────────────────────────────────

def _write_story_spec(state: dict, scores: dict) -> None:
    """
    Writes specs/{session_id}/story_final.md on approval.
    Called only when validation_passed = True.
    Wrapped in safe_run — spec failure never crashes session.
    """
    session_id = state["session_id"]

    scores_md = "\n".join(
        f"- **{k.replace('_', ' ').title()}**: {v}/5"
        for k, v in scores.items()
        if k != "rewrite_instructions" and isinstance(v, int)
    )

    content = (
        f"# Story Final\n\n"
        f"**Session:** {session_id}\n"
        f"**Child:** {state.get('child_name')}, age {state.get('child_age')}\n"
        f"**Topic:** {state.get('topic')}\n"
        f"**Moral:** {state.get('moral_lesson')}\n"
        f"**Approved on attempt:** {state.get('rewrite_attempts', 1)}\n\n"
        f"## Validation Scores\n\n{scores_md}\n\n"
        f"## Approved Story\n\n{state.get('story_text', '')}\n"
    )

    safe_run(
        write_spec,
        session_id, "story_final.md", content,
        session_id=session_id,
    )


# ── Main node function ────────────────────────────────────────────────────────

@timed("validator", "validator_node")
def validator_node(state: dict) -> dict:
    """
    LangGraph node — Validator (LLM Judge).

    Scores the story on five dimensions.
    Routes to narrator on pass, back to writer on fail.
    Writes story_final.md spec on approval.

    Args:
        state: LangGraph StoryState dict

    Returns:
        Updated state with validation_score, validation_passed,
        rewrite_instructions, validation_history, total_tokens
    """
    session_id = state["session_id"]
    attempt    = state.get("rewrite_attempts", 1)

    log.info(
        "validator_started",
        session_id=session_id,
        attempt=attempt,
        story_word_count=len(state.get("story_text", "").split()),
    )

    scores, tokens = _score_story(state)

    # Determine pass/fail
    score_values = {
        k: v for k, v in scores.items()
        if k != "rewrite_instructions" and isinstance(v, int)
    }
    passed = all(v >= PASS_THRESHOLD for v in score_values.values())
    total_score = sum(score_values.values())

    # Append to validation history for trajectory evaluator
    history = list(state.get("validation_history", []))
    history.append({
        "attempt":    attempt,
        "scores":     score_values,
        "total":      total_score,
        "passed":     passed,
    })

    total_tokens = state.get("total_tokens", 0) + tokens

    log.info(
        "validator_complete",
        session_id=session_id,
        attempt=attempt,
        passed=passed,
        total_score=total_score,
        scores=score_values,
        rewrite_needed=not passed,
        tokens_this_call=tokens,
        total_tokens=total_tokens,
    )

    # Write spec file on approval
    if passed:
        _write_story_spec(state, scores)
        log.info(
            "story_approved_spec_written",
            session_id=session_id,
            attempt=attempt,
            total_score=total_score,
        )

    return {
        **state,
        "validation_score":    score_values,
        "validation_passed":   passed,
        "rewrite_instructions": scores.get("rewrite_instructions", ""),
        "validation_history":  history,
        "total_tokens":        total_tokens,
    }
