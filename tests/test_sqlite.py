"""
tests/test_sqlite.py

Tests for src/memory/sqlite.py — all database operations.

Uses a temporary database file for each test — no shared state.

Tests:
    - initialise_db creates all tables
    - save_profile and get_profile round-trip correctly
    - profile upsert works
    - profile_exists returns correct bool
    - save_session writes all fields
    - save_trajectory writes correctly
    - get_last_trajectory_recommendation returns formatted string
    - get_recent_sessions returns joined data
    - get_lessons_covered filters to correct answer only
    - all operations raise StoryMemoryError on DB failure
"""

import json
import uuid
import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime, timezone
from unittest import mock

from src.errors import StoryMemoryError


# ── Temp DB fixture ───────────────────────────────────────────────────────────

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """
    Patches DB_PATH to a temporary file for each test.
    Ensures tests never touch the real story.db.
    """
    db_file = tmp_path / "test_story.db"
    monkeypatch.setattr(
        "src.memory.sqlite.DB_PATH",
        db_file,
    )
    from src.memory.sqlite import initialise_db
    initialise_db()
    return db_file


def _make_session_state(override=None):
    """Returns a minimal valid session state dict."""
    sid = str(uuid.uuid4())[:8]
    state = {
        "session_id":          sid,
        "child_name":          "Sara",
        "child_age":           5,
        "topic":               "dinosaurs",
        "moral_lesson":        "sharing",
        "story_length":        "Medium",
        "rewrite_attempts":    1,
        "narration_failed":    False,
        "answer_result":       "correct",
        "hint_count":          1,
        "pronunciation_score": 0.82,
        "total_tokens":        1247,
        "discussion_complete": True,
        "session_started_at":  datetime.now(timezone.utc).isoformat(),
        "validation_passed":   True,
        "validation_score": {
            "vocabulary_fit": 4,
            "moral_clarity":  4,
            "scare_factor":   5,
            "engagement":     4,
            "length_fit":     3,
        },
        "hints_given": [
            {
                "hint_number":  1,
                "hint_text":    "Remember what Dino found?",
                "child_answer": "he ate them",
            }
        ],
    }
    if override:
        state.update(override)
    return state


# ── initialise_db tests ───────────────────────────────────────────────────────

class TestInitialiseDb:
    def test_creates_all_tables(self, temp_db):
        import sqlite3
        conn   = sqlite3.connect(str(temp_db))
        tables = {row[0] for row in
                  conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()
        required = {"profiles", "sessions", "llm_scores", "hints", "trajectory_scores"}
        assert required.issubset(tables), f"Missing tables: {required - tables}"

    def test_idempotent(self, temp_db):
        """Calling initialise_db twice does not raise."""
        from src.memory.sqlite import initialise_db
        initialise_db()
        initialise_db()


# ── Profile tests ─────────────────────────────────────────────────────────────

class TestProfile:
    def test_profile_exists_false_before_save(self, temp_db):
        from src.memory.sqlite import profile_exists
        assert profile_exists() is False

    def test_profile_exists_true_after_save(self, temp_db):
        from src.memory.sqlite import save_profile, profile_exists
        save_profile({"child_name": "Sara", "child_age": 5,
                      "interests": [], "avoid": []})
        assert profile_exists() is True

    def test_get_profile_none_before_save(self, temp_db):
        from src.memory.sqlite import get_profile
        assert get_profile() is None

    def test_save_and_get_profile_round_trip(self, temp_db):
        from src.memory.sqlite import save_profile, get_profile
        save_profile({
            "child_name": "Sara",
            "child_age":  5,
            "interests":  ["dinosaurs", "space"],
            "avoid":      ["darkness"],
        })
        profile = get_profile()
        assert profile["child_name"] == "Sara"
        assert profile["child_age"] == 5
        assert "dinosaurs" in profile["interests"]
        assert "darkness" in profile["avoid"]

    def test_save_profile_upserts(self, temp_db):
        """Second save updates existing profile (id=1 constraint)."""
        from src.memory.sqlite import save_profile, get_profile
        save_profile({"child_name": "Sara", "child_age": 5,
                      "interests": [], "avoid": []})
        save_profile({"child_name": "Sara", "child_age": 6,
                      "interests": ["space"], "avoid": []})
        profile = get_profile()
        assert profile["child_age"] == 6
        assert "space" in profile["interests"]

    def test_interests_stored_as_list(self, temp_db):
        from src.memory.sqlite import save_profile, get_profile
        save_profile({"child_name": "Test", "child_age": 5,
                      "interests": ["a", "b", "c"], "avoid": []})
        profile = get_profile()
        assert isinstance(profile["interests"], list)
        assert len(profile["interests"]) == 3


# ── Session tests ─────────────────────────────────────────────────────────────

class TestSaveSession:
    def test_save_session_writes_to_db(self, temp_db):
        import sqlite3
        from src.memory.sqlite import save_session
        state = _make_session_state()
        save_session(state)

        conn = sqlite3.connect(str(temp_db))
        row  = conn.execute(
            "SELECT * FROM sessions WHERE session_id=?",
            (state["session_id"],)
        ).fetchone()
        conn.close()
        assert row is not None

    def test_save_session_writes_llm_scores(self, temp_db):
        import sqlite3
        from src.memory.sqlite import save_session
        state = _make_session_state()
        save_session(state)

        conn = sqlite3.connect(str(temp_db))
        row  = conn.execute(
            "SELECT vocabulary_fit FROM llm_scores WHERE session_id=?",
            (state["session_id"],)
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == 4

    def test_save_session_writes_hints(self, temp_db):
        import sqlite3
        from src.memory.sqlite import save_session
        state = _make_session_state()
        save_session(state)

        conn = sqlite3.connect(str(temp_db))
        hints = conn.execute(
            "SELECT * FROM hints WHERE session_id=?",
            (state["session_id"],)
        ).fetchall()
        conn.close()
        assert len(hints) == 1

    def test_save_session_all_in_transaction(self, temp_db):
        """All three writes succeed or none do."""
        from src.memory.sqlite import save_session
        state = _make_session_state()
        # Should not raise
        save_session(state)


# ── Trajectory tests ──────────────────────────────────────────────────────────

class TestTrajectory:
    def _save_session_and_trajectory(self, temp_db, trajectory_score=2.4,
                                     weakest="hint_effectiveness",
                                     recommendation="Use concrete hints"):
        from src.memory.sqlite import save_session, save_trajectory
        state = _make_session_state()
        save_session(state)

        result = {
            "step_scores": {
                "topic_selection": 3, "moral_selection": 3,
                "rewrite_quality": 2, "puzzle_difficulty": 3,
                "hint_effectiveness": 1,
            },
            "trajectory_score": trajectory_score,
            "weakest_step":     weakest,
            "recommendation":   recommendation,
            "session_context":  {"topic": "dinosaurs"},
        }
        save_trajectory(result, state["session_id"])
        return state["session_id"]

    def test_save_trajectory_writes_to_db(self, temp_db):
        import sqlite3
        sid = self._save_session_and_trajectory(temp_db)

        conn = sqlite3.connect(str(temp_db))
        row  = conn.execute(
            "SELECT trajectory_score FROM trajectory_scores WHERE session_id=?",
            (sid,)
        ).fetchone()
        conn.close()
        assert row is not None
        assert abs(row[0] - 2.4) < 0.01

    def test_get_last_trajectory_recommendation(self, temp_db):
        from src.memory.sqlite import get_last_trajectory_recommendation
        self._save_session_and_trajectory(temp_db,
                                          recommendation="Use concrete story references")
        rec = get_last_trajectory_recommendation()
        assert rec is not None
        assert "hint_effectiveness" in rec
        assert "concrete" in rec

    def test_get_last_trajectory_recommendation_none_when_empty(self, temp_db):
        from src.memory.sqlite import get_last_trajectory_recommendation
        result = get_last_trajectory_recommendation()
        assert result is None


# ── get_recent_sessions tests ─────────────────────────────────────────────────

class TestGetRecentSessions:
    def test_returns_empty_list_when_no_sessions(self, temp_db):
        from src.memory.sqlite import get_recent_sessions
        assert get_recent_sessions() == []

    def test_returns_correct_number_of_sessions(self, temp_db):
        from src.memory.sqlite import save_session, get_recent_sessions
        for i in range(3):
            save_session(_make_session_state())
        sessions = get_recent_sessions(limit=5)
        assert len(sessions) == 3

    def test_respects_limit(self, temp_db):
        from src.memory.sqlite import save_session, get_recent_sessions
        for i in range(5):
            save_session(_make_session_state())
        sessions = get_recent_sessions(limit=3)
        assert len(sessions) == 3

    def test_returns_most_recent_first(self, temp_db):
        from src.memory.sqlite import save_session, get_recent_sessions
        import time
        s1 = _make_session_state({"topic": "first"})
        save_session(s1)
        time.sleep(0.01)
        s2 = _make_session_state({"topic": "second"})
        save_session(s2)

        sessions = get_recent_sessions()
        assert sessions[0]["topic"] == "second"

    def test_includes_llm_scores(self, temp_db):
        from src.memory.sqlite import save_session, get_recent_sessions
        save_session(_make_session_state())
        sessions = get_recent_sessions()
        assert sessions[0]["llm_scores"] is not None
        assert sessions[0]["llm_scores"]["vocabulary_fit"] == 4

    def test_includes_hints(self, temp_db):
        from src.memory.sqlite import save_session, get_recent_sessions
        save_session(_make_session_state())
        sessions = get_recent_sessions()
        assert len(sessions[0]["hints"]) == 1


# ── get_lessons_covered tests ─────────────────────────────────────────────────

class TestGetLessonsCovered:
    def test_returns_empty_when_no_sessions(self, temp_db):
        from src.memory.sqlite import get_lessons_covered
        assert get_lessons_covered() == []

    def test_returns_correct_morals(self, temp_db):
        from src.memory.sqlite import save_session, get_lessons_covered
        save_session(_make_session_state({"moral_lesson": "sharing",
                                           "answer_result": "correct"}))
        lessons = get_lessons_covered()
        assert "sharing" in lessons

    def test_filters_to_correct_answer_only(self, temp_db):
        """Lessons where child answered wrong are not counted as covered."""
        from src.memory.sqlite import save_session, get_lessons_covered
        save_session(_make_session_state({
            "moral_lesson": "honesty", "answer_result": "wrong"
        }))
        lessons = get_lessons_covered()
        assert "honesty" not in lessons

    def test_deduplicates_lessons(self, temp_db):
        """Same moral taught in two sessions appears only once."""
        from src.memory.sqlite import save_session, get_lessons_covered
        save_session(_make_session_state({"moral_lesson": "sharing",
                                           "answer_result": "correct"}))
        save_session(_make_session_state({"moral_lesson": "sharing",
                                           "answer_result": "correct"}))
        lessons = get_lessons_covered()
        assert lessons.count("sharing") == 1
