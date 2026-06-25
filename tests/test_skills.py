"""
tests/test_skills.py

Tests for src/skills/enrichment/ — skill selection and behaviour.

Tests:
    - pick_enrichment_skill returns correct class for concrete topics
    - pick_enrichment_skill returns correct class for abstract topics
    - pick_enrichment_skill handles edge cases (empty, mixed, case)
    - SkipFetchSkill.run() always returns empty string
    - WebFetchSkill.run() returns empty string on Docker failure
    - BaseSkill enforces abstract interface
    - ABSTRACT_CONCEPTS coverage
"""

import pytest
import unittest.mock as mock
import sys

# ── Mock heavy dependencies before any imports ────────────────────────────────
@pytest.fixture(autouse=True)
def mock_dependencies():
    mocks = {
        'docker':                    mock.MagicMock(),
        'docker.errors':             mock.MagicMock(),
        'google.cloud':              mock.MagicMock(),
        'google.cloud.modelarmor_v1': mock.MagicMock(),
        'google.api_core':           mock.MagicMock(),
        'google.api_core.client_options': mock.MagicMock(),
        'google.generativeai':       mock.MagicMock(),
        'google.generativeai.types': mock.MagicMock(),
    }
    with mock.patch.dict('sys.modules', mocks):
        yield


# ── Imports ───────────────────────────────────────────────────────────────────
from src.skills.base_skill import BaseSkill
from src.skills.enrichment import pick_enrichment_skill, ABSTRACT_CONCEPTS
from src.skills.enrichment.web_fetch_skill import WebFetchSkill
from src.skills.enrichment.skip_fetch_skill import SkipFetchSkill


# ── BaseSkill tests ───────────────────────────────────────────────────────────

class TestBaseSkill:
    def test_abstract_class_cannot_be_instantiated(self):
        """BaseSkill raises TypeError if instantiated directly."""
        with pytest.raises(TypeError):
            BaseSkill()

    def test_subclass_without_name_raises(self):
        """Subclass missing name property raises TypeError."""
        class Incomplete(BaseSkill):
            @property
            def description(self): return "test"
            def run(self, topic, session_id): return ""

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_without_run_raises(self):
        """Subclass missing run method raises TypeError."""
        class Incomplete(BaseSkill):
            @property
            def name(self): return "test"
            @property
            def description(self): return "test"

        with pytest.raises(TypeError):
            Incomplete()

    def test_complete_subclass_instantiates(self):
        """Complete subclass instantiates without error."""
        class Complete(BaseSkill):
            @property
            def name(self): return "complete"
            @property
            def description(self): return "A complete skill"
            def run(self, topic, session_id): return "result"

        skill = Complete()
        assert skill.name == "complete"
        assert skill.run("test", "session") == "result"

    def test_repr_contains_class_name(self):
        """__repr__ contains the class name."""
        skill = WebFetchSkill()
        assert "WebFetchSkill" in repr(skill)

        skill = SkipFetchSkill()
        assert "SkipFetchSkill" in repr(skill)


# ── pick_enrichment_skill tests ───────────────────────────────────────────────

class TestPickEnrichmentSkill:
    def test_concrete_topics_return_web_fetch(self):
        """Concrete topics return WebFetchSkill."""
        concrete = [
            "dinosaurs", "space", "elephants", "Dubai",
            "volcanoes", "ocean", "butterflies", "Egypt",
        ]
        for topic in concrete:
            skill = pick_enrichment_skill(topic)
            assert isinstance(skill, WebFetchSkill), \
                f"Expected WebFetchSkill for '{topic}', got {type(skill).__name__}"

    def test_abstract_topics_return_skip_fetch(self):
        """Abstract moral/emotional topics return SkipFetchSkill."""
        abstract = [
            "kindness", "sharing", "courage", "honesty", "patience",
            "empathy", "gratitude", "friendship", "feelings", "love",
            "teamwork", "respect", "compassion", "forgiveness", "trust",
        ]
        for topic in abstract:
            skill = pick_enrichment_skill(topic)
            assert isinstance(skill, SkipFetchSkill), \
                f"Expected SkipFetchSkill for '{topic}', got {type(skill).__name__}"

    def test_mixed_topic_with_abstract_word_returns_skip(self):
        """If topic contains any abstract word, SkipFetchSkill is returned."""
        mixed_topics = [
            "sharing berries in the forest",
            "a story about kindness and dinosaurs",
            "courage under the sea",
        ]
        for topic in mixed_topics:
            skill = pick_enrichment_skill(topic)
            assert isinstance(skill, SkipFetchSkill), \
                f"Expected SkipFetchSkill for mixed '{topic}'"

    def test_empty_topic_returns_skip_fetch(self):
        """Empty topic returns SkipFetchSkill."""
        assert isinstance(pick_enrichment_skill(""), SkipFetchSkill)
        assert isinstance(pick_enrichment_skill("   "), SkipFetchSkill)

    def test_case_insensitive_matching(self):
        """Abstract concept matching is case-insensitive."""
        assert isinstance(pick_enrichment_skill("Kindness"), SkipFetchSkill)
        assert isinstance(pick_enrichment_skill("SHARING"), SkipFetchSkill)
        assert isinstance(pick_enrichment_skill("Dinosaurs"), WebFetchSkill)

    def test_skill_names_correct(self):
        """Skill name properties return correct values."""
        assert WebFetchSkill().name == "web_fetch"
        assert SkipFetchSkill().name == "skip_fetch"

    def test_skill_descriptions_non_empty(self):
        """Skill descriptions are non-empty strings."""
        assert len(WebFetchSkill().description) > 20
        assert len(SkipFetchSkill().description) > 20


# ── ABSTRACT_CONCEPTS tests ───────────────────────────────────────────────────

class TestAbstractConcepts:
    def test_contains_core_morals(self):
        """ABSTRACT_CONCEPTS contains core moral lessons."""
        core = {"kindness", "sharing", "courage", "honesty", "patience"}
        assert core.issubset(ABSTRACT_CONCEPTS)

    def test_does_not_contain_concrete_topics(self):
        """ABSTRACT_CONCEPTS does not contain concrete/physical topics."""
        concrete = {"dinosaurs", "space", "ocean", "volcano", "elephant"}
        assert not concrete.intersection(ABSTRACT_CONCEPTS)

    def test_minimum_coverage(self):
        """ABSTRACT_CONCEPTS has at least 30 entries."""
        assert len(ABSTRACT_CONCEPTS) >= 30


# ── SkipFetchSkill tests ──────────────────────────────────────────────────────

class TestSkipFetchSkill:
    def test_run_always_returns_empty_string(self):
        """SkipFetchSkill.run() always returns empty string."""
        skill = SkipFetchSkill()
        assert skill.run("kindness", "session123") == ""
        assert skill.run("", "session123") == ""
        assert skill.run("sharing berries", "session123") == ""

    def test_run_never_raises(self):
        """SkipFetchSkill.run() never raises any exception."""
        skill = SkipFetchSkill()
        try:
            result = skill.run("anything", "session")
            assert result == ""
        except Exception as e:
            pytest.fail(f"SkipFetchSkill.run() raised: {e}")


# ── WebFetchSkill tests ───────────────────────────────────────────────────────

class TestWebFetchSkill:
    def test_run_returns_empty_on_docker_not_found(self):
        """WebFetchSkill returns empty string when Docker image not found."""
        import docker.errors as docker_errors
        docker_errors.ImageNotFound = Exception

        skill = WebFetchSkill()
        with mock.patch('src.skills.enrichment.web_fetch_skill.docker') as mock_docker:
            mock_docker.from_env.return_value.containers.run.side_effect = \
                Exception("Image not found")
            result = skill.run("dinosaurs", "session123")

        assert result == ""

    def test_run_returns_empty_on_docker_timeout(self):
        """WebFetchSkill returns empty string when Docker times out."""
        skill = WebFetchSkill()
        with mock.patch('src.skills.enrichment.web_fetch_skill.docker') as mock_docker:
            mock_docker.from_env.return_value.containers.run.side_effect = \
                Exception("Timeout")
            result = skill.run("dinosaurs", "session123")

        assert result == ""

    def test_run_never_raises(self):
        """WebFetchSkill.run() never raises — safe_run catches everything."""
        skill = WebFetchSkill()
        with mock.patch('src.skills.enrichment.web_fetch_skill.docker') as mock_docker:
            mock_docker.from_env.side_effect = Exception("Docker not running")
            try:
                result = skill.run("dinosaurs", "session123")
                assert isinstance(result, str)
            except Exception as e:
                pytest.fail(f"WebFetchSkill.run() raised: {e}")
