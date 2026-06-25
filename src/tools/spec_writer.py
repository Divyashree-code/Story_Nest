"""
src/tools/spec_writer.py

Atomic markdown spec file writer and reader.

Writes approved agent outputs as human-readable markdown files
under specs/{session_id}/ for audit trail and Day 5 spec-driven
development demonstration.

Files written per session:
    arc.md           — story arc + moral (architect.py)
    story_final.md   — approved story text (writer.py, on approval only)
    puzzle.md        — question + correct answer (puzzle.py)
    trajectory.md    — trajectory evaluation scores (trajectory.py)

Public API:
    write_spec(session_id, filename, content)  — atomic write
    read_spec(session_id, filename) -> str | None  — safe read

Write failures raise ToolError but are wrapped in safe_run() at
call sites — a spec write failure never crashes the session.
Spec files are an audit trail, not a critical path.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from src.errors import ToolError
from src.logger import get_logger

log = get_logger("spec_writer")

# ── Specs root directory ──────────────────────────────────────────────────────
SPECS_ROOT = Path(__file__).parent.parent.parent / "specs"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _session_dir(session_id: str) -> Path:
    """Returns the directory path for a session's spec files."""
    return SPECS_ROOT / session_id


def _spec_path(session_id: str, filename: str) -> Path:
    """Returns the full path to a spec file."""
    return _session_dir(session_id) / filename


# ── Public API ────────────────────────────────────────────────────────────────

def write_spec(session_id: str, filename: str, content: str) -> None:
    """
    Atomically writes content to specs/{session_id}/{filename}.

    Atomic write pattern:
        1. Create specs/{session_id}/ directory if needed
        2. Write content to a temp file in the same directory
        3. Rename temp file to final path (atomic on Linux/macOS)
        4. Final file either fully exists or does not — no partial writes

    Args:
        session_id: UUID-scoped session identifier from LangGraph state
        filename:   Target filename e.g. "arc.md", "story_final.md"
        content:    Markdown content to write

    Raises:
        ToolError: if directory creation or file write fails
    """
    session_dir = _session_dir(session_id)
    final_path  = _spec_path(session_id, filename)

    try:
        # Ensure session directory exists
        session_dir.mkdir(parents=True, exist_ok=True)

        # Write to temp file in same directory first
        # Using same directory ensures rename stays on same filesystem
        fd, tmp_path = tempfile.mkstemp(
            prefix=".tmp_",
            suffix=f"_{filename}",
            dir=session_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())   # ensure bytes hit disk before rename

            # Atomic rename — final file appears complete or not at all
            os.replace(tmp_path, final_path)

        except Exception:
            # Clean up temp file if rename or write failed
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    except Exception as exc:
        raise ToolError(
            f"Failed to write spec file {filename} for session {session_id}: {exc}",
            session_id=session_id,
            spec_filename=filename,
            original_error=str(exc),
        ) from exc

    log.info(
        "spec_written",
        session_id=session_id,
        spec_filename=filename,
        bytes=len(content.encode("utf-8")),
        path=str(final_path),
    )


def read_spec(session_id: str, filename: str) -> Optional[str]:
    """
    Reads a spec file and returns its content as a string.

    Returns None if the file does not exist — callers decide
    how to handle a missing spec. Never raises on missing file.

    Args:
        session_id: UUID-scoped session identifier
        filename:   Target filename e.g. "story_final.md"

    Returns:
        File content as string, or None if file does not exist.

    Raises:
        ToolError: only if the file exists but cannot be read
                   (permissions, encoding error, etc.)
    """
    path = _spec_path(session_id, filename)

    if not path.exists():
        log.debug(
            "spec_not_found",
            session_id=session_id,
            spec_filename=filename,
        )
        return None

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        raise ToolError(
            f"Failed to read spec file {filename} for session {session_id}: {exc}",
            session_id=session_id,
            spec_filename=filename,
            original_error=str(exc),
        ) from exc

    log.debug(
        "spec_read",
        session_id=session_id,
        spec_filename=filename,
        bytes=len(content.encode("utf-8")),
    )
    return content


def list_session_specs(session_id: str) -> list:
    """
    Returns list of spec filenames written for a session.
    Useful for debugging — shows which stages completed.

    Args:
        session_id: UUID-scoped session identifier

    Returns:
        List of filenames e.g. ["arc.md", "story_final.md", "puzzle.md"]
        Empty list if session directory does not exist.
    """
    session_dir = _session_dir(session_id)
    if not session_dir.exists():
        return []

    return sorted([
        f.name for f in session_dir.iterdir()
        if f.is_file() and not f.name.startswith(".tmp_")
    ])
