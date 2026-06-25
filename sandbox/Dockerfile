# sandbox/Dockerfile
#
# Single image for two sandbox scripts:
#   web_fetcher.py  — isolated Wikipedia HTTP call
#   scorer.py       — wav2vec2 pronunciation scoring
#
# Called with different commands:
#   docker run storynest-sandbox python web_fetcher.py --topic dinosaurs
#   docker run --network none storynest-sandbox python scorer.py --audio /audio/recording.wav
#
# Security:
#   - Runs as non-root user
#   - Network blocked at runtime via --network none (scorer only)
#   - wav2vec2 weights baked in at build time so no internet needed at runtime
#   - mem_limit set by Python docker SDK caller (512MB)

FROM python:3.12-slim

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user ──────────────────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash sandbox
WORKDIR /home/sandbox

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements_sandbox.txt .
RUN pip install --no-cache-dir -r requirements_sandbox.txt

# ── Download wav2vec2 weights to fixed path ───────────────────────────────────
# Saved directly to /opt/models/wav2vec2-base at build time (root has internet).
# scorer.py loads from this path at runtime — no network call ever needed.
RUN python -c "\
from transformers import Wav2Vec2Processor, Wav2Vec2Model; \
Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base', cache_dir=None).save_pretrained('/opt/models/wav2vec2-base'); \
Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base', cache_dir=None).save_pretrained('/opt/models/wav2vec2-base'); \
print('wav2vec2 weights saved to /opt/models/wav2vec2-base')" \
    && chmod -R 755 /opt/models

# ── Copy sandbox scripts ──────────────────────────────────────────────────────
COPY web_fetcher.py .
COPY scorer.py .

# ── Switch to non-root user ───────────────────────────────────────────────────
USER sandbox

# ── Block any accidental network calls from transformers at runtime ───────────
ENV TRANSFORMERS_OFFLINE=1

# ── Expose scorer Flask port ──────────────────────────────────────────────────
EXPOSE 8080

# ── Default command — scorer Flask server ─────────────────────────────────────
# web_fetcher.py is invoked via exec_run with --topic argument
CMD ["python", "scorer.py", "--server"]
