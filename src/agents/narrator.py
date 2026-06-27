"""
src/agents/narrator.py

Narrator — fourth LangGraph node in the pipeline.

Plays the approved story text aloud using Microsoft Edge TTS (edge-tts).
No Gemini call — pure audio output.

After story plays, speaks a personalised discussion prompt
inviting parent and child to talk before the puzzle begins.

Reads from state:
    story_text, protagonist_name, child_name,
    session_id, total_tokens

Writes to state:
    narration_failed (bool)

Retry:
    speak() in tts.py handles one internal retry with 2s delay.
    If speak() returns False after retry:
        narration_failed = True in state
        Tab 1 in Streamlit shows clean message to parent
        Session continues — puzzle and interaction still happen
        Parent can read story text already visible in Tab 1

Discussion prompt:
    Spoken after story completes.
    Invites parent and child to discuss before puzzle.
    Skipped silently if TTS fails — less critical than story.
    LangGraph waits for parent to tap Ready before continuing.
"""

import json
from pathlib import Path

from src.error_handler import timed
from src.logger import get_logger
from src.tools.tts import speak

_settings = json.loads(
    (Path(__file__).parent.parent.parent / "SETTINGS.json").read_text()
)
SKIP_NARRATION = _settings.get("SKIP_NARRATION", False)

log = get_logger("narrator")

# ── Discussion prompt template ─────────────────────────────────────────────────
DISCUSSION_PROMPT = (
    "What a wonderful story! "
    "Talk with your parent about what {protagonist} learned today. "
    "Take your time — press the button when you are ready for a question."
)


def _build_discussion_prompt(state: dict) -> str:
    """Builds personalised discussion prompt using protagonist name."""
    protagonist = state.get("protagonist_name") or "our hero"
    return DISCUSSION_PROMPT.format(protagonist=protagonist)


# ── Main node function ────────────────────────────────────────────────────────

@timed("narrator", "narrator_node")
def narrator_node(state: dict) -> dict:
    """
    LangGraph node — Narrator.

    Plays story text via Microsoft Edge TTS.
    Speaks discussion prompt after story completes.
    Sets narration_failed flag if TTS fails after retry.

    Args:
        state: LangGraph StoryState dict

    Returns:
        Updated state with narration_failed bool
    """
    session_id  = state["session_id"]
    story_text  = state.get("story_text", "")
    word_count  = len(story_text.split())

    log.info(
        "narrator_started",
        session_id=session_id,
        word_count=word_count,
        protagonist=state.get("protagonist_name"),
    )

    # ── Play story ────────────────────────────────────────────────────────────
    if SKIP_NARRATION:
        log.info("narration_skipped", session_id=session_id,
                 reason="SKIP_NARRATION=true in SETTINGS.json")
        return {**state, "narration_failed": False, "discussion_complete": False}

    story_success = speak(story_text, session_id)

    if not story_success:
        log.error(
            "narration_failed",
            session_id=session_id,
            word_count=word_count,
            note="story text visible in Tab 1 — session continues",
        )
        return {
            **state,
            "narration_failed": True,
        }

    log.info(
        "narration_complete",
        session_id=session_id,
        word_count=word_count,
    )

    # ── Speak discussion prompt ───────────────────────────────────────────────
    # Non-critical — skipped silently if TTS fails
    discussion_prompt = _build_discussion_prompt(state)
    discussion_success = speak(discussion_prompt, session_id)

    if not discussion_success:
        log.warning(
            "discussion_prompt_tts_failed_continuing",
            session_id=session_id,
            note="parent sees story text — can prompt discussion manually",
        )

    log.info(
        "narrator_node_complete",
        session_id=session_id,
        discussion_prompt_spoken=discussion_success,
    )

    return {
        **state,
        "narration_failed":        False,
        "discussion_complete":     False,  # set True when parent taps Ready
    }
