"""
src/memory/sqlite.py

All database operations for the Kids Storytelling Agent.
Manages data/story.db with five tables.

Tables:
    profiles         — child profile (one row, upserted)
    sessions         — one row per completed story session
    llm_scores       — five LLM Judge dimension scores per session
    hints            — each hint given during the hint loop per session
    trajectory_scores — post-session trajectory evaluation results

Public API:
    save_profile(profile)                    — upsert child profile
    get_profile() -> dict | None             — read profile for session start
    profile_exists() -> bool                 — check if profile is saved
    save_session(state)                      — write session + scores + hints
    save_trajectory(result, session_id)      — write trajectory evaluation
    get_last_trajectory_recommendation()     — read for Story Architect
    get_recent_sessions(limit) -> list       — read for Tab 3 History

All writes use transactions. Either everything saves or nothing does.
All errors raise StoryMemoryError from src/errors.py.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.errors import StoryMemoryError
from src.logger import get_logger

log = get_logger("sqlite")

# ── Database path ─────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent.parent.parent / "data" / "story.db"


# ── Connection context manager ────────────────────────────────────────────────

@contextmanager
def _get_conn():
    """
    Yields a SQLite connection with WAL mode enabled for better
    concurrent read performance and row_factory set to sqlite3.Row
    so results are accessible by column name.

    Commits on success, rolls back on any exception.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception as exc:
        conn.rollback()
        raise StoryMemoryError(
            f"Database operation failed: {exc}",
            original_error=str(exc)
        ) from exc
    finally:
        conn.close()


# ── Schema creation ───────────────────────────────────────────────────────────

def initialise_db() -> None:
    """
    Creates all tables if they do not exist.
    Called once at application startup from main.py.
    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.
    """
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS profiles (
                id          INTEGER PRIMARY KEY CHECK (id = 1),
                child_name  TEXT    NOT NULL,
                child_age   INTEGER NOT NULL,
                interests   TEXT    NOT NULL DEFAULT '[]',
                avoid       TEXT    NOT NULL DEFAULT '[]',
                created_at  TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id          TEXT    PRIMARY KEY,
                child_name          TEXT    NOT NULL,
                child_age           INTEGER NOT NULL,
                topic               TEXT    NOT NULL,
                moral_lesson        TEXT    NOT NULL,
                story_length        TEXT    NOT NULL,
                rewrite_attempts    INTEGER NOT NULL DEFAULT 0,
                narration_failed    INTEGER NOT NULL DEFAULT 0,
                answer_result       TEXT,
                hint_count          INTEGER NOT NULL DEFAULT 0,
                pronunciation_score REAL,
                total_tokens        INTEGER NOT NULL DEFAULT 0,
                discussion_complete INTEGER NOT NULL DEFAULT 0,
                started_at          TEXT    NOT NULL,
                completed_at        TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS llm_scores (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id       TEXT    NOT NULL REFERENCES sessions(session_id),
                vocabulary_fit   INTEGER NOT NULL,
                moral_clarity    INTEGER NOT NULL,
                scare_factor     INTEGER NOT NULL,
                engagement       INTEGER NOT NULL,
                length_fit       INTEGER NOT NULL,
                passed           INTEGER NOT NULL DEFAULT 0,
                created_at       TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS hints (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT    NOT NULL REFERENCES sessions(session_id),
                hint_number  INTEGER NOT NULL,
                hint_text    TEXT    NOT NULL,
                child_answer TEXT    NOT NULL,
                created_at   TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trajectory_scores (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id          TEXT    NOT NULL REFERENCES sessions(session_id),
                topic_selection     INTEGER NOT NULL,
                moral_selection     INTEGER NOT NULL,
                rewrite_quality     INTEGER NOT NULL,
                puzzle_difficulty   INTEGER NOT NULL,
                hint_effectiveness  INTEGER NOT NULL,
                trajectory_score    REAL    NOT NULL,
                weakest_step        TEXT    NOT NULL,
                recommendation      TEXT    NOT NULL,
                session_context     TEXT    NOT NULL,
                evaluated_at        TEXT    NOT NULL
            );

        """)

    log.info("database_initialised", db_path=str(DB_PATH))


# ── Profile operations ────────────────────────────────────────────────────────

def save_profile(profile: dict) -> None:
    """
    Upserts the child profile. Only one profile row exists (id=1).
    Called from app.py when parent saves or edits the registration form.

    Args:
        profile: dict with keys:
            child_name  (str)
            child_age   (int)
            interests   (list[str])
            avoid       (list[str])
    """
    now = datetime.now(timezone.utc).isoformat()

    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO profiles
                (id, child_name, child_age, interests, avoid, created_at, updated_at)
            VALUES (1, :name, :age, :interests, :avoid, :now, :now)
            ON CONFLICT(id) DO UPDATE SET
                child_name = excluded.child_name,
                child_age  = excluded.child_age,
                interests  = excluded.interests,
                avoid      = excluded.avoid,
                updated_at = excluded.updated_at
        """, {
            "name":      profile["child_name"],
            "age":       profile["child_age"],
            "interests": json.dumps(profile.get("interests", [])),
            "avoid":     json.dumps(profile.get("avoid", [])),
            "now":       now,
        })

    log.info(
        "profile_saved",
        child_name=profile["child_name"],
        child_age=profile["child_age"],
    )


def get_profile() -> Optional[dict]:
    """
    Returns the child profile as a dict, or None if no profile exists.
    Called by Story Architect at the start of every session.

    Returns:
        dict with child_name, child_age, interests (list), avoid (list)
        or None if no profile has been saved yet.
    """
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM profiles WHERE id = 1"
        ).fetchone()

    if row is None:
        return None

    return {
        "child_name": row["child_name"],
        "child_age":  row["child_age"],
        "interests":  json.loads(row["interests"]),
        "avoid":      json.loads(row["avoid"]),
    }


def profile_exists() -> bool:
    """
    Returns True if a child profile has been saved, False otherwise.
    Called by app.py to decide which page to show on startup.
    """
    with _get_conn() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM profiles WHERE id = 1"
        ).fetchone()[0]
    return count > 0


# ── Session operations ────────────────────────────────────────────────────────

def save_session(state: dict) -> None:
    """
    Writes the completed session, LLM Judge scores, and hints
    in a single transaction. Either all save or none do.

    Called by Memory Save node at the end of every LangGraph session.

    Args:
        state: the final LangGraph StoryState dict
    """
    now = datetime.now(timezone.utc).isoformat()
    session_id = state["session_id"]

    with _get_conn() as conn:
        # ── sessions table ────────────────────────────────────────────────
        conn.execute("""
            INSERT OR REPLACE INTO sessions (
                session_id, child_name, child_age, topic, moral_lesson,
                story_length, rewrite_attempts, narration_failed,
                answer_result, hint_count, pronunciation_score,
                total_tokens, discussion_complete, started_at, completed_at
            ) VALUES (
                :session_id, :child_name, :child_age, :topic, :moral_lesson,
                :story_length, :rewrite_attempts, :narration_failed,
                :answer_result, :hint_count, :pronunciation_score,
                :total_tokens, :discussion_complete, :started_at, :completed_at
            )
        """, {
            "session_id":          session_id,
            "child_name":          state.get("child_name", ""),
            "child_age":           state.get("child_age", 0),
            "topic":               state.get("topic", ""),
            "moral_lesson":        state.get("moral_lesson", ""),
            "story_length":        state.get("story_length", "Medium"),
            "rewrite_attempts":    state.get("rewrite_attempts", 0),
            "narration_failed":    int(state.get("narration_failed", False)),
            "answer_result":       state.get("answer_result"),
            "hint_count":          state.get("hint_count", 0),
            "pronunciation_score": state.get("pronunciation_score"),
            "total_tokens":        state.get("total_tokens", 0),
            "discussion_complete": int(state.get("discussion_complete", False)),
            "started_at":          state.get("session_started_at", now),
            "completed_at":        now,
        })

        # ── llm_scores table ──────────────────────────────────────────────
        scores = state.get("validation_score", {})
        if scores:
            conn.execute("""
                INSERT INTO llm_scores (
                    session_id, vocabulary_fit, moral_clarity,
                    scare_factor, engagement, length_fit, passed, created_at
                ) VALUES (
                    :session_id, :vocabulary_fit, :moral_clarity,
                    :scare_factor, :engagement, :length_fit, :passed, :now
                )
            """, {
                "session_id":     session_id,
                "vocabulary_fit": scores.get("vocabulary_fit", 0),
                "moral_clarity":  scores.get("moral_clarity", 0),
                "scare_factor":   scores.get("scare_factor", 0),
                "engagement":     scores.get("engagement", 0),
                "length_fit":     scores.get("length_fit", 0),
                "passed":         int(state.get("validation_passed", False)),
                "now":            now,
            })

        # ── hints table ───────────────────────────────────────────────────
        hints = state.get("hints_given", [])
        for hint in hints:
            conn.execute("""
                INSERT INTO hints (
                    session_id, hint_number, hint_text, child_answer, created_at
                ) VALUES (
                    :session_id, :hint_number, :hint_text, :child_answer, :now
                )
            """, {
                "session_id":  session_id,
                "hint_number": hint["hint_number"],
                "hint_text":   hint["hint_text"],
                "child_answer": hint["child_answer"],
                "now":         now,
            })

    log.info(
        "session_saved",
        session_id=session_id,
        topic=state.get("topic"),
        moral=state.get("moral_lesson"),
        answer_result=state.get("answer_result"),
        total_tokens=state.get("total_tokens", 0),
    )


# ── Trajectory operations ─────────────────────────────────────────────────────

def save_trajectory(result: dict, session_id: str) -> None:
    """
    Writes the post-session trajectory evaluation result.
    Called by src/evaluation/trajectory.py after Gemini scores the session.

    Args:
        result:     dict returned by trajectory evaluator containing
                    step_scores, trajectory_score, weakest_step,
                    recommendation, session_context
        session_id: the session this evaluation belongs to
    """
    now = datetime.now(timezone.utc).isoformat()
    step_scores = result.get("step_scores", {})

    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO trajectory_scores (
                session_id, topic_selection, moral_selection,
                rewrite_quality, puzzle_difficulty, hint_effectiveness,
                trajectory_score, weakest_step, recommendation,
                session_context, evaluated_at
            ) VALUES (
                :session_id, :topic_selection, :moral_selection,
                :rewrite_quality, :puzzle_difficulty, :hint_effectiveness,
                :trajectory_score, :weakest_step, :recommendation,
                :session_context, :evaluated_at
            )
        """, {
            "session_id":         session_id,
            "topic_selection":    step_scores.get("topic_selection", 0),
            "moral_selection":    step_scores.get("moral_selection", 0),
            "rewrite_quality":    step_scores.get("rewrite_quality", 0),
            "puzzle_difficulty":  step_scores.get("puzzle_difficulty", 0),
            "hint_effectiveness": step_scores.get("hint_effectiveness", 0),
            "trajectory_score":   result.get("trajectory_score", 0.0),
            "weakest_step":       result.get("weakest_step", ""),
            "recommendation":     result.get("recommendation", ""),
            "session_context":    json.dumps(result.get("session_context", {})),
            "evaluated_at":       now,
        })

    log.info(
        "trajectory_saved",
        session_id=session_id,
        trajectory_score=result.get("trajectory_score"),
        weakest_step=result.get("weakest_step"),
        evaluated_at=now,
    )


# ── Read operations ───────────────────────────────────────────────────────────

def get_last_trajectory_recommendation() -> Optional[str]:
    """
    Returns the recommendation text from the most recent trajectory
    evaluation. Used by Story Architect at session start to adjust
    its behaviour based on last session's weakest step.

    Returns:
        Recommendation string, or None if no trajectory exists yet.
    """
    with _get_conn() as conn:
        row = conn.execute("""
            SELECT recommendation, weakest_step, trajectory_score
            FROM trajectory_scores
            ORDER BY evaluated_at DESC
            LIMIT 1
        """).fetchone()

    if row is None:
        return None

    return (
        f"Last session weakest step: {row['weakest_step']} "
        f"(score {row['trajectory_score']:.1f}/3). "
        f"Recommendation: {row['recommendation']}"
    )


def get_recent_sessions(limit: int = 5) -> list:
    """
    Returns the most recent sessions with their LLM scores and
    trajectory scores joined. Used by Tab 3 History in app.py.

    Args:
        limit: number of sessions to return, default 5

    Returns:
        List of dicts, most recent first. Each dict contains:
            session_id, child_name, topic, moral_lesson, story_length,
            answer_result, hint_count, pronunciation_score, total_tokens,
            narration_failed, completed_at,
            llm_scores (dict of 5 dimensions),
            trajectory (dict with score, weakest_step, recommendation)
    """
    with _get_conn() as conn:
        sessions = conn.execute("""
            SELECT * FROM sessions
            ORDER BY completed_at DESC
            LIMIT ?
        """, (limit,)).fetchall()

        results = []
        for session in sessions:
            sid = session["session_id"]

            # LLM scores for this session
            score_row = conn.execute("""
                SELECT * FROM llm_scores WHERE session_id = ?
            """, (sid,)).fetchone()

            # Trajectory for this session
            traj_row = conn.execute("""
                SELECT * FROM trajectory_scores WHERE session_id = ?
                ORDER BY evaluated_at DESC LIMIT 1
            """, (sid,)).fetchone()

            # Hints for this session
            hint_rows = conn.execute("""
                SELECT * FROM hints WHERE session_id = ?
                ORDER BY hint_number ASC
            """, (sid,)).fetchall()

            results.append({
                "session_id":          sid,
                "child_name":          session["child_name"],
                "topic":               session["topic"],
                "moral_lesson":        session["moral_lesson"],
                "story_length":        session["story_length"],
                "answer_result":       session["answer_result"],
                "hint_count":          session["hint_count"],
                "pronunciation_score": session["pronunciation_score"],
                "total_tokens":        session["total_tokens"],
                "narration_failed":    bool(session["narration_failed"]),
                "discussion_complete": bool(session["discussion_complete"]),
                "rewrite_attempts":    session["rewrite_attempts"],
                "completed_at":        session["completed_at"],
                "llm_scores": {
                    "vocabulary_fit": score_row["vocabulary_fit"] if score_row else None,
                    "moral_clarity":  score_row["moral_clarity"]  if score_row else None,
                    "scare_factor":   score_row["scare_factor"]   if score_row else None,
                    "engagement":     score_row["engagement"]      if score_row else None,
                    "length_fit":     score_row["length_fit"]      if score_row else None,
                    "passed":         bool(score_row["passed"])    if score_row else None,
                } if score_row else None,
                "trajectory": {
                    "score":          traj_row["trajectory_score"]  if traj_row else None,
                    "weakest_step":   traj_row["weakest_step"]      if traj_row else None,
                    "recommendation": traj_row["recommendation"]    if traj_row else None,
                    "step_scores": {
                        "topic_selection":    traj_row["topic_selection"]    if traj_row else None,
                        "moral_selection":    traj_row["moral_selection"]    if traj_row else None,
                        "rewrite_quality":    traj_row["rewrite_quality"]    if traj_row else None,
                        "puzzle_difficulty":  traj_row["puzzle_difficulty"]  if traj_row else None,
                        "hint_effectiveness": traj_row["hint_effectiveness"] if traj_row else None,
                    },
                    "evaluated_at": traj_row["evaluated_at"] if traj_row else None,
                } if traj_row else None,
                "hints": [
                    {
                        "hint_number":  h["hint_number"],
                        "hint_text":    h["hint_text"],
                        "child_answer": h["child_answer"],
                    }
                    for h in hint_rows
                ],
            })

    return results


def get_session(session_id: str) -> Optional[dict]:
    """
    Returns a single session with LLM scores and trajectory joined.
    Used by Tab 2 in app.py to display results after resume completes.
    Returns None if session not found.
    """
    with _get_conn() as conn:
        session = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()

        if session is None:
            return None

        score_row = conn.execute(
            "SELECT * FROM llm_scores WHERE session_id = ?", (session_id,)
        ).fetchone()

        traj_row = conn.execute(
            "SELECT * FROM trajectory_scores WHERE session_id = ? ORDER BY evaluated_at DESC LIMIT 1",
            (session_id,)
        ).fetchone()

    return {
        "session_id":          session_id,
        "topic":               session["topic"],
        "moral_lesson":        session["moral_lesson"],
        "story_length":        session["story_length"],
        "answer_result":       session["answer_result"],
        "hint_count":          session["hint_count"],
        "pronunciation_score": session["pronunciation_score"],
        "total_tokens":        session["total_tokens"],
        "rewrite_attempts":    session["rewrite_attempts"],
        "narration_failed":    bool(session["narration_failed"]),
        "llm_scores": {
            "vocabulary_fit": score_row["vocabulary_fit"] if score_row else None,
            "moral_clarity":  score_row["moral_clarity"]  if score_row else None,
            "scare_factor":   score_row["scare_factor"]   if score_row else None,
            "engagement":     score_row["engagement"]      if score_row else None,
            "length_fit":     score_row["length_fit"]      if score_row else None,
            "passed":         bool(score_row["passed"])    if score_row else None,
        } if score_row else None,
        "trajectory": {
            "score":          traj_row["trajectory_score"]  if traj_row else None,
            "weakest_step":   traj_row["weakest_step"]      if traj_row else None,
            "recommendation": traj_row["recommendation"]    if traj_row else None,
        } if traj_row else None,
    }


def get_lessons_covered() -> list:
    """
    Returns list of moral lessons taught across all sessions.
    Used by Story Architect to avoid repeating lessons.

    Returns:
        List of moral lesson strings, most recent first.
    """
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT DISTINCT moral_lesson FROM sessions
            WHERE answer_result = 'correct'
            ORDER BY completed_at DESC
        """).fetchall()

    return [row["moral_lesson"] for row in rows]
