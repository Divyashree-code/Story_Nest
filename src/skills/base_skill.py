"""
src/skills/base_skill.py

Abstract base class for all skills in the Kids Storytelling Agent.

A skill is a self-contained capability with a clear input/output
contract. Skills are independently testable and reusable across
agents without rewriting logic.

Current skills:
    src/skills/enrichment/web_fetch_skill.py  — fetch Wikipedia facts
    src/skills/enrichment/skip_fetch_skill.py — skip fetch for abstract topics

Selection is deterministic — pick_enrichment_skill() in
src/skills/enrichment/__init__.py uses rule-based logic, not LLM,
to choose the right skill based on topic type.

Usage:
    from src.skills.enrichment import pick_enrichment_skill

    skill = pick_enrichment_skill(topic="dinosaurs")
    facts = skill.run(topic="dinosaurs", session_id="abc123")
"""

from abc import ABC, abstractmethod


class BaseSkill(ABC):
    """
    Abstract base for all skills.

    Subclasses must define:
        name        (str)  — short identifier e.g. "web_fetch"
        description (str)  — what this skill does and when to use it
        run()              — executes the skill, returns string result

    Python raises TypeError at import time if any of these
    are missing from a subclass — catches errors early.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this skill e.g. 'web_fetch'."""

    @property
    @abstractmethod
    def description(self) -> str:
        """
        What this skill does and when it should be chosen.
        Written clearly enough that a developer reading it
        immediately understands the selection criteria.
        """

    @abstractmethod
    def run(self, topic: str, session_id: str) -> str:
        """
        Executes the skill for the given topic.

        Args:
            topic:      the story topic e.g. "dinosaurs"
            session_id: for logging and file scoping

        Returns:
            String result — facts, empty string, or fallback content.
            Never raises — all errors handled internally and logged.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
