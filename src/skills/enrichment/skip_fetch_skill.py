"""
src/skills/enrichment/skip_fetch_skill.py

Returns empty string immediately for abstract topics.
No Docker call, no network, no Model Armor.

Chosen by pick_enrichment_skill() when the topic is abstract —
kindness, sharing, courage, honesty, feelings, friendship.

Wikipedia has no useful factual summary for these topics.
Fetching would return philosophical or sociological content
that adds nothing to a children's story.

The Story Architect uses Gemini's own knowledge for these topics
which handles abstract moral concepts naturally.

Why this exists as a class:
    Keeps Story Architect code clean — always calls
    chosen_skill.run(topic, session_id) with no if/else branching.
    The null object pattern — does nothing, returns empty, safely.
"""

from src.skills.base_skill import BaseSkill
from src.logger import get_logger

log = get_logger("skip_fetch_skill")


class SkipFetchSkill(BaseSkill):
    """
    No-op skill for abstract topics.
    Returns empty string immediately — no external calls made.
    """

    @property
    def name(self) -> str:
        return "skip_fetch"

    @property
    def description(self) -> str:
        return (
            "Use for abstract or emotional topics where Wikipedia "
            "has no useful factual content — kindness, sharing, "
            "courage, honesty, feelings, friendship, patience. "
            "Gemini's own knowledge handles these topics better "
            "than external facts."
        )

    def run(self, topic: str, session_id: str) -> str:
        """
        Returns empty string immediately.
        Story Architect uses Gemini's own knowledge for this topic.

        Args:
            topic:      story topic (abstract — no facts needed)
            session_id: for logging

        Returns:
            Empty string always.
        """
        log.debug(
            "skip_fetch_selected",
            session_id=session_id,
            topic=topic,
            reason="abstract topic — no Wikipedia facts needed",
        )
        return ""
