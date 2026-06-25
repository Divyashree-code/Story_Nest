"""
sandbox/scorer.py

Scores pronunciation clarity of a child's spoken answer using wav2vec2-base.
Runs inside Docker — binary audio processing is isolated from main process.

Usage:
    python scorer.py --audio /path/to/recording.wav

Output (single JSON line to stdout):
    {"success": true,  "score": 0.82}
    {"success": false, "error": "Audio file too short"}

Scoring method:
    wav2vec2 encodes audio into hidden state representations.
    The mean of the last hidden state is taken as the feature vector.
    The L2 norm of this vector, normalised to 0-1 range, is the
    clarity score. Higher = more confident, well-structured phonemes.
    Score range: 0.0 (unclear/silent) to 1.0 (very clear speech).

Why in Docker:
    wav2vec2 processes binary audio via PyTorch C++ extensions.
    A corrupted or malformed WAV file can cause a C-level segfault.
    Docker contains the crash — main app continues with score = None.

Exit codes:
    0 — always, even on failure. Caller reads JSON success field.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME        = "/opt/models/wav2vec2-base"   # local path baked into Docker image
TARGET_SAMPLE_RATE = 16000    # wav2vec2 native sample rate
MIN_DURATION_S    = 0.5       # reject recordings shorter than 0.5 seconds
MAX_DURATION_S    = 30.0      # reject recordings longer than 30 seconds

# ── Model singleton ───────────────────────────────────────────────────────────
# Loaded once — weights are baked into Docker image at build time
_processor = None
_model     = None


def _load_model():
    global _processor, _model
    if _processor is not None:
        return _processor, _model

    _processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    _model     = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    _model.eval()   # inference mode — no gradient tracking
    return _processor, _model


def _load_audio(audio_path: str) -> tuple:
    """
    Loads WAV file and resamples to 16000Hz if needed.

    Returns (audio_array, sample_rate) tuple.
    Raises ValueError for invalid or too-short audio.
    """
    audio, sample_rate = sf.read(audio_path, dtype="float32")

    # Convert stereo to mono if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration_s = len(audio) / sample_rate

    if duration_s < MIN_DURATION_S:
        raise ValueError(
            f"Audio too short: {duration_s:.2f}s "
            f"(minimum {MIN_DURATION_S}s)"
        )

    if duration_s > MAX_DURATION_S:
        raise ValueError(
            f"Audio too long: {duration_s:.2f}s "
            f"(maximum {MAX_DURATION_S}s)"
        )

    # Resample to 16000Hz if needed
    if sample_rate != TARGET_SAMPLE_RATE:
        import torchaudio
        audio_tensor  = torch.FloatTensor(audio).unsqueeze(0)
        resampler     = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=TARGET_SAMPLE_RATE,
        )
        audio = resampler(audio_tensor).squeeze(0).numpy()

    return audio, TARGET_SAMPLE_RATE


def score_pronunciation(audio_path: str) -> dict:
    """
    Scores pronunciation clarity from a WAV file.

    Returns dict with success/score or success/error.
    Score range: 0.0 to 1.0 (higher = clearer speech).
    """
    path = Path(audio_path)

    if not path.exists():
        return {
            "success": False,
            "error": f"Audio file not found: {audio_path}",
        }

    try:
        # Load and validate audio
        audio, sample_rate = _load_audio(str(path))

        # Load model (cached after first call)
        processor, model = _load_model()

        # Prepare input for wav2vec2
        inputs = processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )

        # Run inference — no gradient needed
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract clarity score from last hidden state
        # Last hidden state shape: (batch, sequence, hidden_size)
        last_hidden = outputs.last_hidden_state

        # Mean pool across sequence dimension → (batch, hidden_size)
        mean_hidden = last_hidden.mean(dim=1)

        # L2 norm → scalar representing feature magnitude
        l2_norm = torch.norm(mean_hidden, p=2, dim=1).item()

        # Normalise to 0-1 range
        # wav2vec2-base typical range is roughly 5-25 for spoken audio
        # Clip and normalise to get a 0-1 score
        NORM_MIN = 5.0
        NORM_MAX = 25.0
        score = (l2_norm - NORM_MIN) / (NORM_MAX - NORM_MIN)
        score = float(np.clip(score, 0.0, 1.0))

        return {
            "success": True,
            "score":   round(score, 4),
        }

    except ValueError as exc:
        # Known validation errors (too short, too long)
        return {
            "success": False,
            "error": str(exc),
        }
    except Exception as exc:
        # Any other error — C-level crash caught here inside Docker
        return {
            "success": False,
            "error": f"Scoring failed: {exc}",
        }


def main():
    parser = argparse.ArgumentParser(
        description="wav2vec2 pronunciation clarity scorer"
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to WAV file to score",
    )
    args = parser.parse_args()

    result = score_pronunciation(args.audio.strip())

    # Print single JSON line to stdout — caller reads and parses this
    print(json.dumps(result))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
