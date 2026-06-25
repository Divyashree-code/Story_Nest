"""
src/tools/stt.py

Whisper speech-to-text wrapper for capturing and transcribing
the child's spoken answers during the puzzle interaction.

Two separate concerns kept as separate functions:
    record_audio(duration, session_id) -> Path
        Captures child's voice from microphone, saves to WAV file.
        Returns path to the saved WAV file.

    transcribe(audio_path, session_id) -> str | None
        Runs Whisper on WAV file, returns transcribed text.
        Returns None if transcription is empty or fails.

Why separate:
    The WAV file is shared between this module and the Docker
    pronunciation scorer — both need to read the same recording.
    Keeping record and transcribe separate lets the LangGraph
    voice input node control the file lifecycle explicitly —
    record first, pass path to scorer, transcribe, then delete.

Whisper model:
    Loads whisper base (~74MB) once at module level as singleton.
    Fast and accurate for clear child speech in quiet environment.
    Change WHISPER_MODEL constant to upgrade to small or medium.

Retry:
    Empty transcription returns None — caller re-asks child once.
    STTError on hardware failure — caller re-asks child once.
    Maximum one re-ask — after that answer_result = unclear.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from src.errors import STTError
from src.logger import get_logger

log = get_logger("stt")

# ── Constants ─────────────────────────────────────────────────────────────────
WHISPER_MODEL   = "base"     # base = 74MB, good for clear child speech
SAMPLE_RATE     = 16000      # Whisper native sample rate
DEFAULT_DURATION = 7         # seconds to record — enough for a short answer
AUDIO_DIR       = Path(__file__).parent.parent.parent / "data"

# ── Whisper singleton ─────────────────────────────────────────────────────────
_whisper_model = None


def _get_whisper():
    """
    Returns the Whisper model, loading it once on first call.
    Subsequent calls return the cached instance immediately.

    Raises:
        STTError: if Whisper fails to load
    """
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    try:
        import whisper
        log.info("whisper_loading", model=WHISPER_MODEL)
        _whisper_model = whisper.load_model(WHISPER_MODEL)
        log.info("whisper_loaded", model=WHISPER_MODEL)
        return _whisper_model
    except Exception as exc:
        raise STTError(
            f"Whisper failed to load: {exc}",
            original_error=str(exc),
        ) from exc


# ── Recording ─────────────────────────────────────────────────────────────────

def record_audio(
    session_id: str,
    duration: int = DEFAULT_DURATION,
) -> Path:
    """
    Records audio from the default microphone for the given duration.
    Saves to data/recording_{session_id}.wav and returns the path.

    The saved WAV file is shared with the Docker pronunciation scorer
    which reads it to compute the clarity score. Do not delete this
    file until after scoring is complete — the voice input node in
    LangGraph manages deletion.

    Args:
        session_id: used to scope the filename
        duration:   seconds to record, default 7

    Returns:
        Path to the saved WAV file

    Raises:
        STTError: if microphone recording fails
    """
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = AUDIO_DIR / f"recording_{session_id}.wav"

    try:
        log.info(
            "recording_started",
            session_id=session_id,
            duration_s=duration,
        )

        # Record from default microphone
        # dtype float32 is what Whisper expects natively
        audio = sd.rec(
            frames=int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()   # block until recording complete

        # Save to WAV
        sf.write(str(audio_path), audio, SAMPLE_RATE)

        log.info(
            "recording_saved",
            session_id=session_id,
            path=str(audio_path),
            duration_s=duration,
        )
        return audio_path

    except Exception as exc:
        raise STTError(
            f"Microphone recording failed: {exc}",
            session_id=session_id,
            original_error=str(exc),
        ) from exc


# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe(
    audio_path: Path,
    session_id: str,
) -> Optional[str]:
    """
    Transcribes a WAV file using Whisper and returns the text.

    Returns None if:
    - Transcription result is empty or whitespace only
    - Audio is too short or too quiet for Whisper to detect speech

    The caller (voice input node) handles None by speaking a
    re-ask prompt and recording again. Maximum one re-ask.

    Args:
        audio_path: path to WAV file to transcribe
        session_id: for logging

    Returns:
        Transcribed text string, or None if empty/failed

    Raises:
        STTError: if Whisper fails to process the audio file
    """
    if not audio_path.exists():
        raise STTError(
            f"Audio file not found: {audio_path}",
            session_id=session_id,
            path=str(audio_path),
        )

    start = time.perf_counter()

    try:
        model  = _get_whisper()
        result = model.transcribe(
            str(audio_path),
            language="en",           # English only
            fp16=False,              # fp16 not supported on CPU
            verbose=False,
        )
        duration_ms = round((time.perf_counter() - start) * 1000)
        text = result.get("text", "").strip()

        if not text:
            log.warning(
                "transcription_empty",
                session_id=session_id,
                duration_ms=duration_ms,
            )
            return None

        log.info(
            "transcription_complete",
            session_id=session_id,
            duration_ms=duration_ms,
            text_length=len(text),
            text_preview=text[:50],
        )
        return text

    except Exception as exc:
        raise STTError(
            f"Whisper transcription failed: {exc}",
            session_id=session_id,
            original_error=str(exc),
        ) from exc


# ── Public convenience function ───────────────────────────────────────────────

def listen(
    session_id: str,
    duration: int = DEFAULT_DURATION,
) -> tuple[Optional[str], Path]:
    """
    Records audio and transcribes it in one call.
    Returns both the transcribed text and the audio file path.

    The audio path is returned so the caller can pass it to the
    Docker pronunciation scorer before deleting it.

    Args:
        session_id: for file scoping and logging
        duration:   seconds to record, default 7

    Returns:
        Tuple of (transcribed_text or None, audio_file_path)

    Raises:
        STTError: if recording fails (transcription None is not an error)
    """
    audio_path = record_audio(session_id=session_id, duration=duration)
    text       = transcribe(audio_path=audio_path, session_id=session_id)
    return text, audio_path


def delete_recording(audio_path: Path, session_id: str) -> None:
    """
    Deletes the recording WAV file after both transcription and
    pronunciation scoring are complete.

    Called by the voice input node in LangGraph after the Docker
    scorer has finished reading the file.

    Args:
        audio_path: path to WAV file to delete
        session_id: for logging
    """
    try:
        if audio_path.exists():
            audio_path.unlink()
            log.debug(
                "recording_deleted",
                session_id=session_id,
                path=str(audio_path),
            )
    except OSError as exc:
        # Non-critical — log and continue, do not raise
        log.warning(
            "recording_delete_failed",
            session_id=session_id,
            path=str(audio_path),
            error=str(exc),
        )
