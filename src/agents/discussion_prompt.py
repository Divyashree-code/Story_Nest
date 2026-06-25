"""
src/agents/discussion_prompt.py

Human-in-the-loop interrupt node — NOT an agent node.
No LLM call, no tool use, no decision making.

Sits between Narrator and Puzzle Generator in the LangGraph pipeline.
Pauses graph execution using NodeInterrupt and waits for the parent
to tap "We're ready" in the Streamlit UI before the puzzle begins.

This creates an unmediated discussion space between parent and child
with zero AI involvement. The app steps back completely.

How the pause works:
    1. Node sets awaiting_discussion=True in state
    2. Raises NodeInterrupt — graph pauses here
    3. Streamlit UI detects awaiting_discussion flag
    4. Shows "We're ready for the puzzle" button in Tab 1
    5. Parent taps button → Streamlit resumes graph via invoke()
    6. Node runs again, sets discussion_complete=True
    7. Graph continues to Puzzle Generator

LangSmith visibility:
    NodeInterrupt nodes can be invisible in LangSmith if not traced.
    This node uses langsmith.trace() context manager to explicitly
    create a span so it appears in the LangSmith timeline as
    'discussion_prompt' with duration and state logged.

Reads from state:
    protagonist_name, child_name, session_id, awaiting_discussion

Writes to state:
    awaiting_discussion (bool)
    discussion_complete (bool)
    discussion_prompt_text (str)  — shown in Tab 1
"""

from src.logger import get_logger
from src.error_handler import timed

log = get_logger("discussion_prompt")

# ── LangSmith tracing ─────────────────────────────────────────────────────────
try:
    from langsmith import trace as ls_trace
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    ls_trace = None

# ── NodeInterrupt import ──────────────────────────────────────────────────────
try:
    from langgraph.errors import NodeInterrupt
except ImportError:
    # Fallback for older LangGraph versions
    class NodeInterrupt(Exception):
        pass

# ── Prompt text ───────────────────────────────────────────────────────────────
DISCUSSION_TEXT = (
    "Talk with your child about what {protagonist} learned today. "
    "Take your time. There is no rush."
)


def _build_prompt_text(state: dict) -> str:
    """Builds personalised discussion text for Tab 1."""
    protagonist = state.get("protagonist_name") or "our hero"
    return DISCUSSION_TEXT.format(protagonist=protagonist)


# ── Main node function ────────────────────────────────────────────────────────

def discussion_prompt_node(state: dict) -> dict:
    """
    LangGraph human-in-the-loop interrupt node.

    First pass  — sets awaiting_discussion=True, raises NodeInterrupt.
    Second pass — sets discussion_complete=True, continues to puzzle.

    LangGraph calls this node twice:
        Pass 1: graph pauses here (NodeInterrupt raised)
        Pass 2: graph resumes after parent taps Ready

    LangSmith span created explicitly so this node is visible
    in the LangSmith timeline despite not being an LLM call.

    Args:
        state: LangGraph StoryState dict

    Returns:
        Updated state with discussion_complete=True on resume
        (Pass 1 never returns — NodeInterrupt is raised instead)
    """
    session_id      = state["session_id"]
    prompt_text     = _build_prompt_text(state)
    already_waiting = state.get("awaiting_discussion", False)

    def _run():
        nonlocal already_waiting

        # ── Pass 2: Parent tapped Ready — resume ──────────────────────────
        if already_waiting:
            log.info(
                "discussion_complete",
                session_id=session_id,
                protagonist=state.get("protagonist_name"),
            )
            return {
                **state,
                "awaiting_discussion":   False,
                "discussion_complete":   True,
                "discussion_prompt_text": prompt_text,
            }

        # ── Pass 1: First time through — pause and wait ───────────────────
        log.info(
            "discussion_prompt_pausing",
            session_id=session_id,
            protagonist=state.get("protagonist_name"),
            prompt_text=prompt_text,
        )

        # Update state before interrupting so Streamlit can read it
        updated_state = {
            **state,
            "awaiting_discussion":    True,
            "discussion_complete":    False,
            "discussion_prompt_text": prompt_text,
        }

        # NodeInterrupt pauses the graph here
        # LangGraph checkpointer saves state before raising
        # Graph resumes when Streamlit calls invoke() again
        raise NodeInterrupt(
            f"Waiting for parent to confirm discussion complete "
            f"for session {session_id}"
        )

    # ── LangSmith explicit span ───────────────────────────────────────────────
    # Creates a visible span in LangSmith timeline for this non-LLM node
    if LANGSMITH_AVAILABLE and ls_trace:
        with ls_trace(
            name="discussion_prompt",
            run_type="chain",
            inputs={
                "session_id":    session_id,
                "protagonist":   state.get("protagonist_name"),
                "already_waiting": already_waiting,
            },
            tags=["human-in-the-loop", "interrupt"],
        ) as run:
            result = _run()
            if run and result:
                run.end(outputs={
                    "discussion_complete": result.get("discussion_complete"),
                    "awaiting_discussion": result.get("awaiting_discussion"),
                })
            return result
    else:
        return _run()
