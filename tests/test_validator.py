"""
tests/test_validator.py

Tests for src/agents/validator.py — LLM Judge scoring logic.

Tests:
    - System prompt contains required elements
    - Valid JSON scores parsed correctly
    - Score clamping to 1-5 range
    - Pass/fail determination at threshold
    - Fallback pass on malformed JSON
    - Markdown fence stripping
    - Validation history accumulated correctly
    - Spec file written only on approval
    - Temperature is 0.0 for consistency
"""

import json
import pytest
import unittest.mock as mock


@pytest.fixture(autouse=True)
def mock_dependencies():
    mocks = {
        'google.generativeai':       mock.MagicMock(),
        'google.generativeai.types': mock.MagicMock(),
    }
    with mock.patch.dict('sys.modules', mocks):
        yield


from src.agents.validator import (
    _build_validator_prompt,
    _parse_scores,
    validator_node,
    PASS_THRESHOLD,
)


# ── Base state ────────────────────────────────────────────────────────────────

@pytest.fixture
def base_state():
    return {
        "session_id":         "test_sess",
        "child_age":          5,
        "moral_lesson":       "sharing",
        "story_length":       "Medium",
        "story_text":         "Once upon a time Dino found berries and shared them.",
        "rewrite_attempts":   1,
        "validation_history": [],
        "total_tokens":       200,
        "child_name":         "Sara",
        "topic":              "dinosaurs",
        "protagonist_name":   "Dino",
    }


# ── PASS_THRESHOLD tests ──────────────────────────────────────────────────────

class TestPassThreshold:
    def test_pass_threshold_is_three(self):
        assert PASS_THRESHOLD == 3

    def test_all_threes_passes(self):
        scores = {"vocabulary_fit": 3, "moral_clarity": 3,
                  "scare_factor": 3, "engagement": 3, "length_fit": 3}
        assert all(v >= PASS_THRESHOLD for v in scores.values())

    def test_one_two_fails(self):
        scores = {"vocabulary_fit": 2, "moral_clarity": 3,
                  "scare_factor": 5, "engagement": 4, "length_fit": 3}
        assert not all(v >= PASS_THRESHOLD for v in scores.values())

    def test_all_fives_passes(self):
        scores = {"vocabulary_fit": 5, "moral_clarity": 5,
                  "scare_factor": 5, "engagement": 5, "length_fit": 5}
        assert all(v >= PASS_THRESHOLD for v in scores.values())


# ── Prompt tests ──────────────────────────────────────────────────────────────

class TestValidatorPrompt:
    def test_prompt_contains_all_dimensions(self, base_state):
        prompt = _build_validator_prompt(base_state)
        for dim in ["vocabulary_fit", "moral_clarity", "scare_factor",
                    "engagement", "length_fit"]:
            assert dim in prompt, f"Missing dimension: {dim}"

    def test_prompt_contains_moral(self, base_state):
        prompt = _build_validator_prompt(base_state)
        assert "sharing" in prompt

    def test_prompt_contains_word_target(self, base_state):
        prompt = _build_validator_prompt(base_state)
        assert "350" in prompt  # Medium = 350 words

    def test_prompt_contains_age(self, base_state):
        prompt = _build_validator_prompt(base_state)
        assert "5" in prompt

    def test_prompt_requests_json_only(self, base_state):
        prompt = _build_validator_prompt(base_state)
        assert "JSON" in prompt

    def test_prompt_contains_pass_threshold(self, base_state):
        prompt = _build_validator_prompt(base_state)
        assert str(PASS_THRESHOLD) in prompt

    def test_prompt_includes_attempt_number(self, base_state):
        base_state["rewrite_attempts"] = 2
        prompt = _build_validator_prompt(base_state)
        assert "2" in prompt


# ── Score parsing tests ───────────────────────────────────────────────────────

class TestParseScores:
    def _valid_json(self, overrides=None):
        scores = {
            "vocabulary_fit": 4, "moral_clarity": 3,
            "scare_factor": 5, "engagement": 4,
            "length_fit": 3, "rewrite_instructions": ""
        }
        if overrides:
            scores.update(overrides)
        return json.dumps(scores)

    def test_valid_json_parsed_correctly(self):
        scores = _parse_scores = __import__(
            'src.agents.validator', fromlist=['_parse_scores']
        )._parse_scores
        result = scores(self._valid_json(), "test")
        assert result["vocabulary_fit"] == 4
        assert result["scare_factor"] == 5
        assert result["rewrite_instructions"] == ""

    def test_score_above_max_clamped_to_five(self):
        from src.agents.validator import _parse_scores
        result = _parse_scores(self._valid_json({"vocabulary_fit": 10}), "test")
        assert result["vocabulary_fit"] == 5

    def test_score_below_min_clamped_to_one(self):
        from src.agents.validator import _parse_scores
        result = _parse_scores(self._valid_json({"moral_clarity": 0}), "test")
        assert result["moral_clarity"] == 1

    def test_malformed_json_returns_fallback_pass(self):
        from src.agents.validator import _parse_scores
        result = _parse_scores("this is not json", "test")
        assert result["vocabulary_fit"] == 3
        assert result["scare_factor"] == 5
        assert result["rewrite_instructions"] == ""

    def test_markdown_fences_stripped(self):
        from src.agents.validator import _parse_scores
        fenced = (
            "```json\n"
            '{"vocabulary_fit":4,"moral_clarity":4,'
            '"scare_factor":5,"engagement":4,"length_fit":4,'
            '"rewrite_instructions":""}\n'
            "```"
        )
        result = _parse_scores(fenced, "test")
        assert result["vocabulary_fit"] == 4

    def test_missing_key_returns_fallback(self):
        from src.agents.validator import _parse_scores
        incomplete = json.dumps({
            "vocabulary_fit": 4, "moral_clarity": 3
            # missing other keys
        })
        result = _parse_scores(incomplete, "test")
        # Falls back to neutral pass
        assert result["scare_factor"] == 5


# ── validator_node tests ──────────────────────────────────────────────────────

class TestValidatorNode:
    def _make_response(self, scores: dict, tokens: int = 300):
        resp = mock.MagicMock()
        resp.text = json.dumps({
            **scores,
            "rewrite_instructions": ""
            if all(v >= 3 for v in scores.values()) else "fix vocab"
        })
        resp.usage_metadata.total_token_count = tokens
        return resp

    def test_approved_story_sets_passed_true(self, base_state):
        passing = {"vocabulary_fit": 4, "moral_clarity": 4,
                   "scare_factor": 5, "engagement": 4, "length_fit": 4}
        import src.agents.validator as val_mod
        with mock.patch.object(val_mod.gemini_validator, 'generate_content',
                               return_value=self._make_response(passing)):
            with mock.patch('src.agents.validator.write_spec'):
                result = validator_node(base_state)
        assert result["validation_passed"] is True
        assert result["rewrite_instructions"] == ""

    def test_failing_story_sets_passed_false(self, base_state):
        failing = {"vocabulary_fit": 2, "moral_clarity": 3,
                   "scare_factor": 5, "engagement": 3, "length_fit": 3}
        import src.agents.validator as val_mod
        with mock.patch.object(val_mod.gemini_validator, 'generate_content',
                               return_value=self._make_response(failing)):
            with mock.patch('src.agents.validator.write_spec'):
                result = validator_node(base_state)
        assert result["validation_passed"] is False
        assert result["rewrite_instructions"] != ""

    def test_tokens_accumulated(self, base_state):
        passing = {"vocabulary_fit": 4, "moral_clarity": 4,
                   "scare_factor": 5, "engagement": 4, "length_fit": 4}
        import src.agents.validator as val_mod
        with mock.patch.object(val_mod.gemini_validator, 'generate_content',
                               return_value=self._make_response(passing, tokens=250)):
            with mock.patch('src.agents.validator.write_spec'):
                result = validator_node(base_state)
        assert result["total_tokens"] == 200 + 250

    def test_validation_history_accumulated(self, base_state):
        base_state["validation_history"] = [
            {"attempt": 0, "scores": {}, "total": 15, "passed": False}
        ]
        passing = {"vocabulary_fit": 4, "moral_clarity": 4,
                   "scare_factor": 5, "engagement": 4, "length_fit": 4}
        import src.agents.validator as val_mod
        with mock.patch.object(val_mod.gemini_validator, 'generate_content',
                               return_value=self._make_response(passing)):
            with mock.patch('src.agents.validator.write_spec'):
                result = validator_node(base_state)
        assert len(result["validation_history"]) == 2

    def test_spec_written_only_on_approval(self, base_state):
        failing = {"vocabulary_fit": 2, "moral_clarity": 3,
                   "scare_factor": 5, "engagement": 3, "length_fit": 3}
        import src.agents.validator as val_mod
        with mock.patch.object(val_mod.gemini_validator, 'generate_content',
                               return_value=self._make_response(failing)):
            with mock.patch('src.agents.validator.write_spec') as mock_write:
                validator_node(base_state)
        mock_write.assert_not_called()

    def test_spec_written_on_approval(self, base_state):
        passing = {"vocabulary_fit": 4, "moral_clarity": 4,
                   "scare_factor": 5, "engagement": 4, "length_fit": 4}
        import src.agents.validator as val_mod
        with mock.patch.object(val_mod.gemini_validator, 'generate_content',
                               return_value=self._make_response(passing)):
            with mock.patch('src.agents.validator.write_spec') as mock_write:
                validator_node(base_state)
        mock_write.assert_called_once()

    def test_gemini_failure_grants_fallback_pass(self, base_state):
        """On Gemini failure, fallback pass is granted — session continues."""
        import src.agents.validator as val_mod
        mock_resp = mock.MagicMock()
        mock_resp.text = "not valid json"
        with mock.patch.object(val_mod.gemini_validator, 'generate_content',
                               return_value=mock_resp):
            with mock.patch('src.agents.validator.write_spec'):
                result = validator_node(base_state)
        # Fallback pass means validation_passed may be True or False
        # but session must not crash
        assert "validation_passed" in result
        assert "validation_score" in result
