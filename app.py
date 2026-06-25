"""
app.py

Streamlit parent dashboard for StoryNest.

Three pages:
    Page 1 — Registration (first visit, no profile in SQLite)
    Page 2 — Session Start (profile exists, choose length + topic)
    Page 3 — Active Session (three tabs: Story, Results, History)

Calls run_story_session() and resume_story_session() from main.py
in background threads. Results stored in st.session_state.
"""

import threading
import time
from datetime import datetime
from io import BytesIO

import streamlit as st

from src.memory.sqlite import (
    save_profile, get_profile, profile_exists, get_recent_sessions,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StoryNest",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* Story text box */
    .story-box {
        background: #fdf8f0;
        border-left: 4px solid #e8c87a;
        border-radius: 8px;
        padding: 1.4rem 1.6rem;
        font-size: 1.05rem;
        line-height: 2.0;
        color: #3d2b1f;
        min-height: 160px;
        white-space: pre-wrap;
        font-family: Georgia, serif;
    }
    /* Session history card */
    .history-card {
        background: #fafafa;
        border: 1px solid #e8e8e8;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    /* Score badge */
    .score-pass { color: #2d6a2d; font-weight: 600; }
    .score-warn { color: #8a6000; font-weight: 600; }
    .score-fail { color: #8a0000; font-weight: 600; }
    /* Page headers */
    .page-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c2c2c;
        margin-bottom: 0.2rem;
    }
    .page-sub {
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 1.6rem;
    }
    /* Ready button */
    div[data-testid='stButton'] button[kind='primary'] {
        background-color: #4a7c59;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
_defaults = {
    "page":               "register",   # register | start | session
    "story_text":         "",
    "session_running":    False,
    "session_result":     None,
    "session_id":         None,
    "awaiting_discussion": False,
    "resume_running":     False,
    "resume_result":      None,
    # One-shot flags: guarantee a rerun immediately after background threads finish
    "_session_just_finished": False,
    "_resume_just_finished":  False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── PDF export ────────────────────────────────────────────────────────────────

def _make_pdf(result: dict, profile: dict) -> bytes:
    """Generates session summary PDF using reportlab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, HRFlowable
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors

        buf    = BytesIO()
        doc    = SimpleDocTemplate(buf, pagesize=A4,
                                   leftMargin=2*cm, rightMargin=2*cm,
                                   topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        title_style = ParagraphStyle("title", parent=styles["Title"],
                                     fontSize=18, spaceAfter=6)
        body_style  = ParagraphStyle("body", parent=styles["Normal"],
                                     fontSize=11, leading=18)
        head_style  = ParagraphStyle("head", parent=styles["Heading2"],
                                     fontSize=13, spaceBefore=12)

        story.append(Paragraph("📖 StoryNest — Session Summary", title_style))
        story.append(Paragraph(
            f"Child: {profile.get('child_name')} · Age: {profile.get('child_age')} · "
            f"Date: {datetime.now().strftime('%d %b %Y')}",
            body_style
        ))
        story.append(HRFlowable(width="100%", spaceAfter=12))

        story.append(Paragraph("Story", head_style))
        story_text = result.get("story_text", "").replace("\n", "<br/>")
        story.append(Paragraph(story_text, body_style))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Session Details", head_style))
        story.append(Paragraph(
            f"Topic: {result.get('topic', '—')} · "
            f"Moral: {result.get('moral_lesson', '—').title()} · "
            f"Length: {result.get('story_length', '—')}",
            body_style
        ))
        story.append(Paragraph(
            f"Answer result: {result.get('answer_result', '—').title()} · "
            f"Hints needed: {result.get('hint_count', 0)} · "
            f"Rewrites: {result.get('rewrite_attempts', 0)}",
            body_style
        ))

        scores = result.get("validation_score", {})
        if scores:
            story.append(Paragraph("Story Quality Scores", head_style))
            for k, v in scores.items():
                if isinstance(v, int):
                    label = k.replace("_", " ").title()
                    story.append(Paragraph(f"{label}: {v}/5", body_style))

        traj = result.get("_trajectory")
        if traj:
            story.append(Paragraph("Trajectory Evaluation", head_style))
            story.append(Paragraph(
                f"Overall score: {traj.get('trajectory_score', 0):.2f}/1.0 · "
                f"Weakest step: {traj.get('weakest_step', '—')}",
                body_style
            ))
            if traj.get("recommendation"):
                story.append(Paragraph(
                    f"Recommendation: {traj['recommendation']}", body_style
                ))

        doc.build(story)
        buf.seek(0)
        return buf.read()
    except ImportError:
        return b""


# ── Thread-safe result store ──────────────────────────────────────────────────
# Streamlit session_state is not thread-safe — background thread writes are not
# guaranteed to be visible to the main thread on the next rerun.
# We use st.cache_resource to create a singleton dict that survives reruns.
# Plain dict assignment is atomic under the GIL, so background threads can
# write and the main thread can read safely without locks.
# Keys: session_id → result dict.  Cleared by the main thread after pickup.

@st.cache_resource
def _get_thread_store() -> dict:
    """Singleton store persisted across Streamlit reruns."""
    return {"results": {}, "resumes": {}}

_STORE = _get_thread_store()


# ── Background thread helpers ─────────────────────────────────────────────────

def _on_story_chunk(chunk: str):
    # Story is displayed from specs/{session_id}/story_final.md after validation.
    # Callback kept so writer uses streaming mode (lower Gemini latency).
    pass


def _run_session(profile, topic, story_length, session_id):
    try:
        from main import run_story_session
        result = run_story_session(
            profile=profile,
            topic=topic,
            story_length=story_length,
            on_story_chunk=_on_story_chunk,
            session_id=session_id,
        )
        _STORE["results"][session_id] = result
    except Exception as e:
        _STORE["results"][session_id] = {"error": str(e)}
    finally:
        st.session_state.session_running        = False
        st.session_state._session_just_finished = True


def _run_resume(session_id):
    try:
        from main import resume_story_session
        result = resume_story_session(session_id)
        _STORE["resumes"][session_id] = result
    except Exception as e:
        import traceback, logging
        logging.getLogger("app").error(
            "resume_thread_failed session=%s error=%s\n%s",
            session_id, e, traceback.format_exc()
        )
        _STORE["resumes"][session_id] = {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — REGISTRATION
# ══════════════════════════════════════════════════════════════════════════════

def show_registration():
    st.markdown('<div class="page-title">👋 Welcome to StoryNest</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Tell us a little about your child '
                'to personalise every story.</div>', unsafe_allow_html=True)

    # Pre-fill if editing existing profile
    existing = get_profile() or {}

    with st.form("registration_form"):
        child_name = st.text_input(
            "Child's name *",
            value=existing.get("child_name", ""),
            placeholder="Sara",
        )
        child_age = st.slider(
            "Age *", min_value=3, max_value=10,
            value=existing.get("child_age", 5),
        )
        interests_raw = st.text_input(
            "Interests (optional, comma separated)",
            value=", ".join(existing.get("interests", [])),
            placeholder="dinosaurs, space, unicorns",
        )
        avoid_raw = st.text_input(
            "Things to avoid (optional, comma separated)",
            value=", ".join(existing.get("avoid", [])),
            placeholder="darkness, spiders, loud noises",
        )

        submitted = st.form_submit_button(
            "Save Profile →",
            use_container_width=True,
            type="primary",
        )

        if submitted:
            if not child_name.strip():
                st.error("Please enter your child's name.")
            else:
                profile = {
                    "child_name": child_name.strip(),
                    "child_age":  child_age,
                    "interests":  [i.strip() for i in interests_raw.split(",") if i.strip()],
                    "avoid":      [a.strip() for a in avoid_raw.split(",") if a.strip()],
                }
                save_profile(profile)
                st.session_state.editing_profile = False
                st.session_state.page = "start"
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SESSION START
# ══════════════════════════════════════════════════════════════════════════════

def show_session_start():
    profile = get_profile()
    if not profile:
        st.session_state.page = "register"
        st.rerun()
        return

    name = profile["child_name"]

    st.markdown(f'<div class="page-title">✨ Ready for {name}\'s story</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Choose a length and an optional topic, '
                'then press start.</div>', unsafe_allow_html=True)

    story_length = st.radio(
        "Story length",
        options=["Short", "Medium", "Long"],
        index=1,
        horizontal=True,
        help="Short ≈ 2 min · Medium ≈ 5 min · Long ≈ 10 min",
    )

    topic = st.text_input(
        "Topic (optional)",
        placeholder="Leave empty for a surprise…",
        max_chars=60,
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        start = st.button(
            "📖 Tell me a story",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.session_running,
        )
    with col2:
        if st.button("✏️ Edit profile", use_container_width=True):
            st.session_state.editing_profile = True
            st.session_state.page = "register"
            st.rerun()

    if start:
        import uuid as _uuid
        session_id = str(_uuid.uuid4())[:8]   # generate HERE so UI sees it immediately
        st.session_state.story_text           = ""
        st.session_state.session_running      = True
        st.session_state.session_result       = None
        st.session_state.resume_result        = None
        st.session_state.awaiting_discussion  = False
        st.session_state.session_id           = session_id
        st.session_state.page                 = "session"

        t = threading.Thread(
            target=_run_session,
            args=(profile, topic.strip(), story_length, session_id),
            daemon=True,
        )
        t.start()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ACTIVE SESSION
# ══════════════════════════════════════════════════════════════════════════════

def show_session():
    profile = get_profile() or {}

    # ── Pick up background thread results (thread-safe via module-level dicts) ──
    # All session_state flag updates (running=False) happen HERE in the main
    # thread, not in the background thread, to avoid Streamlit thread-safety issues.
    _sid_now = st.session_state.get("session_id", "")
    if _sid_now and _sid_now in _STORE["results"]:
        _r = _STORE["results"].pop(_sid_now)
        st.session_state.session_result      = _r
        st.session_state.session_running     = False   # set by main thread
        st.session_state.story_text          = _r.get("story_text", "")
        st.session_state.awaiting_discussion = _r.get("awaiting_discussion", False)
    if _sid_now and _sid_now in _STORE["resumes"]:
        _rr = _STORE["resumes"].pop(_sid_now)
        st.session_state.resume_result       = _rr
        st.session_state.resume_running      = False   # set by main thread
        st.session_state.awaiting_discussion = False

    result  = st.session_state.session_result or {}
    resume  = st.session_state.resume_result  or {}

    # Combine results — resume result takes priority after Ready is tapped
    final_result = {**result, **resume} if resume else result

    tab1, tab2, tab3 = st.tabs(["📖 Story", "📊 Session Results", "📜 History"])

    # ── TAB 1: STORY ──────────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Story")

        # Story text: read specs/{session_id}/story_final.md — written by the
        # validator on approval, before the narrator runs. Available during
        # narration so the child can read along while listening.
        _live_text = ""
        _sid = st.session_state.get("session_id", "")
        if _sid:
            from pathlib import Path as _Path
            _spec = _Path(f"./specs/{_sid}/story_final.md")
            if _spec.exists():
                _raw = _spec.read_text(encoding="utf-8")
                # story_final.md has story under "## Approved Story" heading
                if "## Approved Story" in _raw:
                    _live_text = _raw.split("## Approved Story", 1)[-1].strip()
                else:
                    _live_text = _raw.strip()
        story_display = (
            _live_text
            or st.session_state.story_text
            or final_result.get("story_text", "")
        )

        # DEBUG — remove once story display is confirmed working
        st.caption(
            f"DEBUG | running={st.session_state.session_running} | "
            f"sid={_sid} | spec_chars={len(_live_text)} | "
            f"awaiting={st.session_state.awaiting_discussion}"
        )

        import html as _html
        story_html = (
            _html.escape(story_display).replace("\n", "<br>")
            if story_display
            else "<span style='color:#bbb'>Your story will appear here…</span>"
        )
        st.markdown(
            f'<div class="story-box">{story_html}</div>',
            unsafe_allow_html=True,
        )

        # Narration failed message
        if final_result.get("narration_failed"):
            st.warning(
                "Audio narration failed. The story is shown above — "
                "you can read it aloud to your child."
            )

        # Spinner while generating / narrating.
        # Poll while session_running=True AND no result has arrived yet.
        # _STORE["results"] is backed by @st.cache_resource and survives reruns,
        # so it is the reliable signal — unlike session_running which is written
        # by the background thread and may not be visible immediately.
        _still_running = (
            st.session_state.session_running
            and st.session_state.session_result is None
            and _sid not in _STORE["results"]
        )
        if _still_running:
            if _live_text:
                st.info("🔊 Narrating your story… please wait.")
            else:
                st.info("✨ Creating your story… please wait.")
            time.sleep(1)
            st.rerun()

        # Discussion prompt — parent and child talk, then tap Ready
        if st.session_state.awaiting_discussion and not st.session_state.resume_running:
            prompt_text = result.get(
                "discussion_prompt_text",
                f"Talk with your child about today's story. Take your time."
            )
            st.info(f"💬 {prompt_text}")

            if st.button(
                "✅ We're ready for the puzzle!",
                type="primary",
                use_container_width=True,
            ):
                sid = st.session_state.session_id
                st.session_state.resume_running = True
                t = threading.Thread(
                    target=_run_resume,
                    args=(sid,),
                    daemon=True,
                )
                t.start()
                st.rerun()

        # Spinner while puzzle/answer interaction running
        _resume_still_running = (
            st.session_state.resume_running
            and st.session_state.resume_result is None
            and _sid not in _STORE["resumes"]
        )
        if _resume_still_running:
            st.info("🎯 Puzzle time… please wait.")
            time.sleep(1)
            st.rerun()
        elif st.session_state.get("_resume_just_finished"):
            st.session_state._resume_just_finished = False
            time.sleep(0.3)
            st.rerun()

        # Session complete — back to start button
        if final_result and not st.session_state.session_running \
                and not st.session_state.resume_running \
                and not st.session_state.awaiting_discussion:
            st.divider()
            if st.button("📖 Tell another story", use_container_width=True):
                for k, v in _defaults.items():
                    st.session_state[k] = v
                st.session_state.page = "start"
                st.rerun()

    # ── TAB 2: SESSION RESULTS ────────────────────────────────────────────────
    with tab2:
        if final_result.get("error") and not final_result.get("story_text"):
            st.error(f"Session error: {final_result['error']}")
        elif final_result.get("moral_lesson"):
            st.markdown("#### Session Results")

            # Key metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Moral taught",    final_result.get("moral_lesson", "—").title())
            c2.metric("Answer",          final_result.get("answer_result", "—").title())
            c3.metric("Hints needed",    final_result.get("hint_count", 0))
            c4.metric("Total tokens",    final_result.get("total_tokens", 0))

            col1, col2 = st.columns(2)
            with col1:
                col1.metric("Rewrites",  final_result.get("rewrite_attempts", 0))
            with col2:
                pron = final_result.get("pronunciation_score")
                col2.metric(
                    "Pronunciation clarity",
                    f"{pron:.2f}" if pron is not None else "N/A"
                )

            # LLM Judge scores
            scores = final_result.get("validation_score", {})
            if scores:
                st.divider()
                st.markdown("**Story quality scores (LLM Judge)**")
                for k, v in scores.items():
                    if isinstance(v, int):
                        label  = k.replace("_", " ").title()
                        colour = "normal" if v >= 4 else ("off" if v == 3 else "inverse")
                        st.progress(v / 5, text=f"{label}: {v}/5")

            # Trajectory scores (if available from resume result)
            traj = final_result.get("_trajectory")
            if traj:
                st.divider()
                st.markdown("**Agent behaviour score (Trajectory)**")
                st.metric(
                    "Overall trajectory score",
                    f"{traj.get('trajectory_score', 0):.2f}/1.0"
                )
                if traj.get("weakest_step"):
                    st.caption(
                        f"⚠️ Weakest step: **{traj['weakest_step'].replace('_', ' ').title()}**"
                    )
                if traj.get("recommendation"):
                    st.caption(f"💡 {traj['recommendation']}")

            # Export buttons
            st.divider()
            st.markdown("**Export**")
            col1, col2 = st.columns(2)

            with col1:
                txt = (
                    f"StoryNest — {profile.get('child_name', '')} — "
                    f"{datetime.now().strftime('%d %b %Y')}\n\n"
                    f"Topic: {final_result.get('topic', '')}\n"
                    f"Moral: {final_result.get('moral_lesson', '')}\n\n"
                    f"{final_result.get('story_text', '')}"
                )
                st.download_button(
                    "📄 Download TXT",
                    data=txt,
                    file_name=f"story_{profile.get('child_name', 'session')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            with col2:
                pdf_bytes = _make_pdf(final_result, profile)
                if pdf_bytes:
                    st.download_button(
                        "📕 Download PDF",
                        data=pdf_bytes,
                        file_name=f"story_{profile.get('child_name', 'session')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
        else:
            st.caption("Session results will appear here after the story completes.")

    # ── TAB 3: HISTORY ────────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### Last 5 Sessions")
        sessions = get_recent_sessions(limit=5)

        if not sessions:
            st.caption("No sessions yet. Tell your first story!")
        else:
            for s in sessions:
                completed = s.get("completed_at", "")
                try:
                    dt  = datetime.fromisoformat(completed)
                    date_str = dt.strftime("%d %b %Y %H:%M")
                except Exception:
                    date_str = completed[:10] if completed else "—"

                with st.expander(
                    f"📖 {s.get('topic', '—').title()} · "
                    f"{s.get('moral_lesson', '—').title()} · {date_str}",
                    expanded=False,
                ):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Answer",       s.get("answer_result", "—").title())
                    c2.metric("Hints",        s.get("hint_count", 0))
                    c3.metric("Tokens",       s.get("total_tokens", 0))

                    c4, c5 = st.columns(2)
                    c4.metric("Story length", s.get("story_length", "—"))
                    pron = s.get("pronunciation_score")
                    c5.metric("Pronunciation",
                              f"{pron:.2f}" if pron is not None else "N/A")

                    # LLM scores
                    llm = s.get("llm_scores")
                    if llm:
                        st.markdown("**Story quality:**")
                        score_parts = [
                            f"{k.replace('_', ' ').title()}: {v}/5"
                            for k, v in llm.items()
                            if isinstance(v, int)
                        ]
                        st.caption(" · ".join(score_parts))

                    # Trajectory
                    traj = s.get("trajectory")
                    if traj and traj.get("score") is not None:
                        st.markdown("**Agent behaviour (trajectory):**")
                        st.caption(
                            f"Score: {traj['score']:.2f}/1.0 · "
                            f"Weakest: {traj.get('weakest_step', '—').replace('_', ' ')}"
                        )
                        if traj.get("recommendation"):
                            st.caption(f"💡 {traj['recommendation']}")


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Determine starting page
    if st.session_state.page == "register":
        if profile_exists() and not st.session_state.get("editing_profile", False):
            # Auto-advance to start if profile already saved
            st.session_state.page = "start"

    page = st.session_state.page

    if page == "register":
        show_registration()
    elif page == "start":
        show_session_start()
    elif page == "session":
        show_session()
    else:
        st.session_state.page = "start"
        st.rerun()


if __name__ == "__main__":
    main()
