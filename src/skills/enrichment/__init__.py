"""
src/skills/enrichment/__init__.py

Deterministic skill selector for story topic enrichment.

pick_enrichment_skill(topic) returns either:
    WebFetchSkill()  — concrete topic, Wikipedia facts useful
    SkipFetchSkill() — abstract topic, use Gemini's own knowledge

Selection is rule-based, not LLM-driven:
    - Deterministic — same topic always returns same skill
    - No latency — no extra Gemini call for selection
    - No hallucination risk — LLM cannot pick wrong skill
    - Independently testable — pure function with clear inputs/outputs

Abstract topic detection:
    Checks if any word in the topic matches the ABSTRACT_CONCEPTS list.
    "a story about kindness and dinosaurs" → matches "kindness" → skip fetch
    The concrete element (dinosaurs) is still used in the story arc —
    we just skip the Wikipedia enrichment step since "kindness" dominates.

Usage:
    from src.skills.enrichment import pick_enrichment_skill

    skill = pick_enrichment_skill("dinosaurs")   → WebFetchSkill
    skill = pick_enrichment_skill("sharing")     → SkipFetchSkill
    facts = skill.run("dinosaurs", session_id)
"""

from src.skills.enrichment.web_fetch_skill import WebFetchSkill
from src.skills.enrichment.skip_fetch_skill import SkipFetchSkill
from src.skills.base_skill import BaseSkill

# ── Abstract concept list ─────────────────────────────────────────────────────
# Topics where Wikipedia has no useful factual content for a children's story.
# Gemini handles these better from its own training knowledge.
ABSTRACT_CONCEPTS = {
    # Moral values
    "kindness", "sharing", "courage", "honesty", "patience",
    "empathy", "respect", "gratitude", "forgiveness", "loyalty",
    "bravery", "generosity", "compassion", "friendship", "trust",
    "fairness", "helpfulness", "responsibility", "perseverance",
    # Emotions
    "feelings", "emotions", "happiness", "sadness", "fear",
    "anger", "love", "jealousy", "loneliness", "joy", "hope",
    # Abstract concepts
    "family", "teamwork", "cooperation", "community", "belonging",
    "identity", "diversity", "inclusion", "mindfulness", "confidence",
}


def pick_enrichment_skill(topic: str) -> BaseSkill:
    """
    Deterministically selects the enrichment skill for a given topic.

    Checks if any word in the topic (lowercased) matches the
    ABSTRACT_CONCEPTS set. If yes, returns SkipFetchSkill.
    If no match, returns WebFetchSkill.

    Args:
        topic: story topic string e.g. "dinosaurs" or "sharing"

    Returns:
        WebFetchSkill  — for concrete topics with Wikipedia facts
        SkipFetchSkill — for abstract topics without useful facts

    Examples:
        pick_enrichment_skill("dinosaurs")       → WebFetchSkill
        pick_enrichment_skill("kindness")        → SkipFetchSkill
        pick_enrichment_skill("space")           → WebFetchSkill
        pick_enrichment_skill("sharing berries") → SkipFetchSkill
        pick_enrichment_skill("Dubai")           → WebFetchSkill
        pick_enrichment_skill("")                → SkipFetchSkill
    """
    if not topic or not topic.strip():
        return SkipFetchSkill()

    # Tokenise topic into words and check against abstract concepts
    topic_words = set(topic.lower().split())

    if topic_words & ABSTRACT_CONCEPTS:
        return SkipFetchSkill()

    return WebFetchSkill()


__all__ = [
    "pick_enrichment_skill",
    "WebFetchSkill",
    "SkipFetchSkill",
    "ABSTRACT_CONCEPTS",
]
