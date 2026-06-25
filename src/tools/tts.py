"""
src/tools/tts.py

Microsoft Edge neural TTS wrapper (edge-tts).

Uses Microsoft's cloud neural TTS service via the edge-tts package —
no API key required. Audio is streamed as MP3, decoded in-memory
via miniaudio (no ffmpeg dependency), and played via sounddevice.

Voice: en-US-JennyNeural — warm, natural, child-friendly.
       Change VOICE constant to switch voices.

Public API:
    speak(text, session_id) -> bool
        Speaks text aloud. Returns True on success, False on failure
        after one retry. Never raises — callers check the return value.

    speak_hint(hint, question, session_id) -> bool
        Speaks hint + question combined.

Audio flow:
    edge-tts streams MP3 bytes from Microsoft servers
        → decoded to float32 PCM via miniaudio
        → played via sounddevice
        → timeout guard prevents indefinite hangs

Retry:
    First failure → wait 2s → retry once.
    Second failure → log error → return False.
    Caller sets narration_failed=True in state.
"""

import asyncio
import time
import threading

import numpy as np
import sounddevice as sd
import edge_tts
import miniaudio

from src.errors import TTSError
from src.logger import get_logger

log = get_logger("tts")

# ── Constants ─────────────────────────────────────────────────────────────────
VOICE       = "en-US-JennyNeural"   # warm, natural, child-friendly voice
RATE        = "-30%"                 # slower pace — clearer for young children
SAMPLE_RATE = 24000                  # target sample rate for playback


# ── Audio generation ──────────────────────────────────────────────────────────

async def _synthesize(text: str) -> bytes:
    """Streams MP3 bytes from Microsoft Edge TTS service."""
    communicate = edge_tts.Communicate(text, VOICE, rate=RATE)
    mp3_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_bytes += chunk["data"]
    return mp3_bytes


def _generate_audio(text: str) -> np.ndarray:
    """
    Synthesises text to a float32 numpy audio array.

    Calls edge-tts (internet required), decodes MP3 via miniaudio.

    Args:
        text: text to synthesise

    Returns:
        float32 numpy array at SAMPLE_RATE

    Raises:
        TTSError: if synthesis or decoding fails
    """
    try:
        log.info("tts_synthesizing", chars=len(text))
        mp3_bytes = asyncio.run(_synthesize(text))

        if not mp3_bytes:
            raise TTSError("Edge TTS returned empty audio", text_length=len(text))

        log.info("tts_synthesized", bytes=len(mp3_bytes))

        # Decode MP3 → PCM using miniaudio (pure C, no ffmpeg needed)
        decoded = miniaudio.decode(
            mp3_bytes,
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=SAMPLE_RATE,
        )
        audio = np.array(decoded.samples, dtype=np.int16).astype(np.float32) / 32768.0

        log.info(
            "tts_decoded",
            samples=len(audio),
            duration_s=round(len(audio) / SAMPLE_RATE, 1),
        )
        return audio

    except TTSError:
        raise
    except Exception as exc:
        raise TTSError(
            f"Edge TTS audio generation failed: {exc}",
            original_error=str(exc),
            text_length=len(text),
        ) from exc


# ── Playback ──────────────────────────────────────────────────────────────────

def _play_audio(audio: np.ndarray, session_id: str) -> None:
    """
    Plays a float32 audio array via sounddevice.
    Blocks until complete, with a hard timeout to prevent hangs.

    Args:
        audio:      float32 numpy array of PCM samples
        session_id: used for logging

    Raises:
        TTSError: if playback fails or times out
    """
    try:
        duration_s = len(audio) / SAMPLE_RATE
        timeout_s  = duration_s + 10.0

        sd.play(audio, samplerate=SAMPLE_RATE)

        done = threading.Event()
        threading.Thread(
            target=lambda: (sd.wait(), done.set()),
            daemon=True,
        ).start()

        if not done.wait(timeout=timeout_s):
            sd.stop()
            raise TTSError(
                f"Audio playback timed out after {timeout_s:.0f}s",
                session_id=session_id,
            )

    except TTSError:
        raise
    except Exception as exc:
        raise TTSError(
            f"Audio playback failed: {exc}",
            session_id=session_id,
            original_error=str(exc),
        ) from exc


# ── Public API ────────────────────────────────────────────────────────────────

def speak(text: str, session_id: str) -> bool:
    """
    Speaks text aloud using Microsoft Edge neural TTS.

    Retries once on failure with a 2 second delay.

    Args:
        text:       text to speak aloud
        session_id: for logging

    Returns:
        True  — audio played successfully
        False — failed after one retry
    """
    if not text or not text.strip():
        log.warning("speak_called_with_empty_text", session_id=session_id)
        return False

    for attempt in range(1, 3):
        start = time.perf_counter()
        try:
            audio = _generate_audio(text)
            _play_audio(audio, session_id)
            duration_ms = round((time.perf_counter() - start) * 1000)

            log.info(
                "speech_complete",
                session_id=session_id,
                attempt=attempt,
                duration_ms=duration_ms,
                text_chars=len(text),
            )
            return True

        except TTSError as exc:
            duration_ms = round((time.perf_counter() - start) * 1000)

            if attempt == 1:
                log.warning(
                    "tts_attempt_failed_retrying",
                    session_id=session_id,
                    attempt=attempt,
                    duration_ms=duration_ms,
                    error=str(exc),
                )
                time.sleep(2.0)
            else:
                log.error(
                    "tts_failed_after_retry",
                    session_id=session_id,
                    duration_ms=duration_ms,
                    error=str(exc),
                    exc_info=True,
                )
                return False

    return False


def speak_hint(hint: str, question: str, session_id: str) -> bool:
    """
    Speaks a hint followed by the question again.
    Used by the hint loop when the child answers incorrectly.

    Args:
        hint:       hint text from Answer Validator
        question:   original puzzle question to repeat
        session_id: for logging

    Returns:
        True on success, False on failure after retry
    """
    combined = f"{hint} ... {question}"
    return speak(combined, session_id)
