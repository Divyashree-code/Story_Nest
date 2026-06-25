"""
src/agents/architect.py

Story Architect — first LangGraph node in the pipeline.

Responsibilities:
    1. Pick topic if parent left it empty (Gemini call)
    2. Select moral lesson not yet covered (SQLite + deterministic)
    3. Fetch external facts via enrichment skill (Docker + Model Armor)
    4. Design 3-act story arc (Gemini call)
    5. Write specs/{session_id}/arc.md

Reads from state:
    topic, child_name, child_age, interests, avoid,
    story_length, session_id

Writes to state:
    topic, moral_lesson, protagonist_name,
    story_arc, fetched_facts, total_tokens,
    session_started_at

Reads from SQLite:
    get_lessons_covered()              — avoid repeating morals
    get_last_trajectory_recommendation() — adjust from last session

Writes to specs:
    specs/{session_id}/arc.md
"""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
import os

from src.errors import StoryArchitectError
from src.error_handler import timed, safe_run
from src.logger import get_logger
from src.memory.sqlite import get_lessons_covered, get_last_trajectory_recommendation
from src.skills.enrichment import pick_enrichment_skill
from src.tools.spec_writer import write_spec
from src.tools.gemini_limiter import gemini_limiter

load_dotenv()
log = get_logger("architect")

# ── Gemini setup ──────────────────────────────────────────────────────────────
genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))

# Read settings
_settings = json.loads(
    (Path(__file__).parent.parent.parent / "SETTINGS.json").read_text()
)
GEMINI_MODEL   = _settings.get("GEMINI_MODEL", "gemini-2.5-flash")
LENGTH_WORDS   = _settings.get("STORY_LENGTH_WORDS", {"Short": 150, "Medium": 350, "Long": 700})

gemini = genai.GenerativeModel(GEMINI_MODEL)

# ── Available moral lessons ───────────────────────────────────────────────────
ALL_MORALS = [
    "sharing", "honesty", "courage", "kindness", "patience",
    "empathy", "gratitude", "perseverance", "responsibility", "forgiveness",
    "generosity", "respect", "teamwork", "confidence", "compassion",
]

# ── Baseline avoid list — always applied regardless of parent preferences ─────
BASELINE_AVOID = [
    "death", "violence", "weapons", "darkness", "scary monsters",
    "adult relationships", "drugs", "alcohol", "horror", "war",
    "political content", "religious conflict",
]

# ── Default fallback topics by age group ─────────────────────────────────────
DEFAULT_TOPICS = {
    range(3, 5):  ["friendly animals", "colorful butterflies", "little stars"],
    range(5, 7):  ["dinosaurs", "ocean creatures", "space exploration"],
    range(7, 9):  ["ancient Egypt", "rainforest animals", "volcanoes"],
    range(9, 11): ["deep sea mysteries", "famous inventors", "the solar system"],
}


def _get_default_topic(age: int) -> str:
    """Returns an age-appropriate fallback topic."""
    import random
    for age_range, topics in DEFAULT_TOPICS.items():
        if age in age_range:
            return random.choice(topics)
    return "friendly animals"


# ── Gemini helper ─────────────────────────────────────────────────────────────

def _call_gemini(system: str, user: str, session_id: str) -> tuple[str, int]:
    """
    Calls Gemini and returns (response_text, token_count).
    Proactively rate-limited; retries once on 429 using the suggested delay.
    Raises StoryArchitectError on failure.
    """
    prompt = f"{system}\n\n{user}"
    for attempt in range(1, 3):
        gemini_limiter.wait(session_id)
        try:
            response = gemini.generate_content(prompt)
            text     = response.text.strip()
            tokens   = getattr(response.usage_metadata, "total_token_count", 0) or 0
            return text, tokens
        except Exception as exc:
            if attempt == 1 and gemini_limiter.backoff(exc, session_id):
                continue
            raise StoryArchitectError(
                f"Gemini call failed in architect: {exc}",
                session_id=session_id,
                original_error=str(exc),
            ) from exc


# ── Step 1: Topic picking ─────────────────────────────────────────────────────

def _pick_topic(state: dict) -> tuple[str, int]:
    """
    If topic is empty, asks Gemini to pick an age-appropriate topic
    based on the child's age and interests.

    Returns (topic, tokens_used).
    """
    if state.get("topic", "").strip():
        return state["topic"].strip(), 0

    interests_str = (
        ", ".join(state.get("interests", [])) or "general nature and animals"
    )

    try:
        topic, tokens = _call_gemini(
            system=(
                f"You pick engaging story topics for children aged {state['child_age']}. "
                f"Return ONLY the topic — 2-4 words, no explanation, no punctuation."
            ),
            user=(
                f"Child interests: {interests_str}. "
                f"Pick one concrete, age-appropriate story topic."
            ),
            session_id=state["session_id"],
        )
        log.info(
            "topic_picked_by_gemini",
            session_id=state["session_id"],
            topic=topic,
            child_age=state["child_age"],
        )
        return topic, tokens

    except Exception:
        # Topic pick is optional — skip Gemini call entirely and use fallback
        # rather than waiting on rate limits for a non-critical step.
        fallback = _get_default_topic(state["child_age"])
        log.warning(
            "topic_pick_failed_using_fallback",
            session_id=state["session_id"],
            fallback=fallback,
        )
        return fallback, 0


# ── Step 2: Moral selection ───────────────────────────────────────────────────

def _pick_moral(session_id: str) -> str:
    """
    Picks the next moral lesson not yet covered.
    Cycles back to the start if all morals have been taught.
    """
    covered = set(get_lessons_covered())
    for moral in ALL_MORALS:
        if moral not in covered:
            log.info(
                "moral_selected",
                session_id=session_id,
                moral=moral,
                covered_count=len(covered),
            )
            return moral

    # All morals covered — start again
    log.info(
        "all_morals_covered_restarting",
        session_id=session_id,
        total_covered=len(covered),
    )
    return ALL_MORALS[0]


# ── Step 3: Arc design ────────────────────────────────────────────────────────

def _design_arc(state: dict, topic: str, moral: str, facts: str) -> tuple[str, str, int]:
    """
    Designs a 3-act story arc with a named protagonist.
    Returns (arc_text, protagonist_name, tokens_used).
    """
    avoid_list  = BASELINE_AVOID + state.get("avoid", [])
    avoid_str   = ", ".join(avoid_list)
    word_target = LENGTH_WORDS.get(state.get("story_length", "Medium"), 350)
    interests   = ", ".join(state.get("interests", [])) or "general topics"

    # Read last trajectory recommendation if available
    last_rec = get_last_trajectory_recommendation()
    rec_note = f"\n\nLast session feedback: {last_rec}" if last_rec else ""

    facts_note = (
        f"\n\nReal facts to weave in naturally: {facts}"
        if facts else ""
    )

    arc, tokens = _call_gemini(
        system=(
            f"You are a children's story architect designing for "
            f"{state['child_name']}, age {state['child_age']}.\n"
            f"Child interests: {interests}.\n"
            f"NEVER include: {avoid_str}.\n"
            f"Target story length: ~{word_target} words.\n"
            f"Moral to weave in (never state it directly): {moral}.\n"
            f"Design a warm, engaging 3-act arc with a named protagonist.\n"
            f"Return:\n"
            f"PROTAGONIST: [name]\n"
            f"ACT 1: [setup]\n"
            f"ACT 2: [conflict]\n"
            f"ACT 3: [resolution]"
            f"{rec_note}"
        ),
        user=f"Topic: {topic}{facts_note}",
        session_id=state["session_id"],
    )

    # Extract protagonist name from arc
    protagonist = "the little hero"
    match = re.search(r"PROTAGONIST:\s*(.+)", arc, re.IGNORECASE)
    if match:
        protagonist = match.group(1).strip().split("\n")[0].strip()

    log.info(
        "arc_designed",
        session_id=state["session_id"],
        topic=topic,
        moral=moral,
        protagonist=protagonist,
        word_target=word_target,
    )
    return arc, protagonist, tokens


# ── Main node function ────────────────────────────────────────────────────────

@timed("architect", "architect_node")
def architect_node(state: dict) -> dict:
    """
    LangGraph node — Story Architect.

    Args:
        state: LangGraph StoryState dict

    Returns:
        Updated state with topic, moral_lesson, protagonist_name,
        story_arc, fetched_facts, total_tokens, session_started_at
    """
    session_id    = state["session_id"]
    total_tokens  = state.get("total_tokens", 0)

    # Record session start time
    session_started_at = datetime.now(timezone.utc).isoformat()

    log.info(
        "architect_started",
        session_id=session_id,
        child_name=state.get("child_name"),
        child_age=state.get("child_age"),
        topic_provided=bool(state.get("topic", "").strip()),
    )

    # ── Step 1: Pick topic ────────────────────────────────────────────────────
    topic, tok = _pick_topic(state)
    total_tokens += tok

    # ── Step 2: Pick moral ────────────────────────────────────────────────────
    moral = _pick_moral(session_id)

    # ── Step 3: Fetch external facts via enrichment skill ─────────────────────
    skill = pick_enrichment_skill(topic)
    log.info(
        "enrichment_skill_selected",
        session_id=session_id,
        skill=skill.name,
        topic=topic,
    )
    facts = safe_run(
        skill.run,
        topic,
        session_id,
        default="",
        session_id=session_id,
    )

    # ── Step 4: Design story arc ──────────────────────────────────────────────
    arc, protagonist, tok = _design_arc(state, topic, moral, facts)
    total_tokens += tok

    # ── Step 5: Write spec file ───────────────────────────────────────────────
    arc_md = (
        f"# Story Arc\n\n"
        f"**Session:** {session_id}\n"
        f"**Child:** {state.get('child_name')}, age {state.get('child_age')}\n"
        f"**Topic:** {topic}\n"
        f"**Moral:** {moral}\n"
        f"**Protagonist:** {protagonist}\n"
        f"**Story length:** {state.get('story_length', 'Medium')}\n"
        f"**Enrichment skill:** {skill.name}\n\n"
        f"## External Facts\n\n{facts or '_No external facts — abstract topic_'}\n\n"
        f"## Story Arc\n\n{arc}\n"
    )
    safe_run(
        write_spec,
        session_id, "arc.md", arc_md,
        session_id=session_id,
    )

    log.info(
        "architect_complete",
        session_id=session_id,
        topic=topic,
        moral=moral,
        protagonist=protagonist,
        facts_fetched=bool(facts),
        total_tokens=total_tokens,
    )

    return {
        **state,
        "topic":               topic,
        "moral_lesson":        moral,
        "protagonist_name":    protagonist,
        "story_arc":           arc,
        "fetched_facts":       facts,
        "total_tokens":        total_tokens,
        "session_started_at":  session_started_at,
    }
