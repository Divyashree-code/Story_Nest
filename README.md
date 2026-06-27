# StoryNest

An AI-powered storytelling agent that generates personalised, fact-enriched bedtime stories for children aged 3-10. Designed as a co-experience for parent and child — the AI enriches the story with real-world facts and age-appropriate moral lessons, while the parent and child engage together through discussion and a puzzle interaction.

---

## Problem Statement

Parents want to give their children quality story time but often draw from the same stories and themes. This agent generates stories enriched with real facts the parent might not know, teaches moral lessons systematically across sessions, and creates a structured discussion moment between parent and child — expanding the child's thinking beyond the parent's natural repertoire.

---

## Architecture Overview

Seven LangGraph agents orchestrated in a state machine with two conditional loops:

```
Story Architect → Writer → LLM Judge (rewrite loop) → Narrator
→ Puzzle Generator → TTS Question → Whisper STT
→ Pronunciation Scorer (Docker) → Answer Validator (hint loop)
→ Memory Save → Trajectory Evaluator (post-session)
```

**Key technical decisions:**
- LangGraph for stateful agent orchestration with conditional loops
- Gemini 2.5 Flash for all LLM calls (free tier)
- Microsoft Edge TTS (edge-tts) for neural TTS narration — no API key needed
- Whisper base for local STT transcription
- wav2vec2-base in Docker for pronunciation scoring
- web_fetcher in Docker for isolated Wikipedia calls
- Model Armor for web content sanitization
- SQLite for long-term memory across sessions
- Spec-driven MD files per session as audit trail
- LangSmith for automatic LLM-level observability
- Structured JSON logging for business-level events

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| CPU | Any modern 64-bit | 4+ cores |
| RAM | 8GB | 16GB |
| GPU | Not required | Not required |
| Disk | 5GB free | 10GB free |
| Microphone | Required | Required |
| Internet | Required | Required |

Tested on: Windows 11, Python 3.12, 16GB RAM

---

## Software Requirements

| Software | Version | Purpose |
|---|---|---|
| Python | 3.12.x | Runtime |
| Docker Desktop | Latest | Sandbox for web fetcher and pronunciation scorer — must be running before app start |
| FFmpeg | Latest | Required by openai-whisper for audio decoding — must be on system PATH |
| Git | Latest | Version control |

---

## Installation

### Step 1 — Install system dependencies

**Python 3.12**
Download from `python.org/downloads/release/python-3129`
Tick: Add Python to PATH + Install pip

**Docker Desktop**
Download from `docker.com/products/docker-desktop`
Start Docker Desktop and leave it running

**FFmpeg**
Download from `ffmpeg.org/download.html` → Windows builds → gyan.dev
Extract and add `bin/` folder to Windows PATH
Verify: `ffmpeg -version`

### Step 2 — Clone and set up project

```bash
git clone https://github.com/yourusername/storynest.git
cd storynest
```

### Step 3 — Create virtual environment

```bash
py -3.12 -m venv venv312
venv312\Scripts\activate
```

### Step 4 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Configure API keys

```bash
cp .env.example .env
```

Open `.env` and fill in:

| Key | Where to get it |
|---|---|
| `GOOGLE_API_KEY` | `aistudio.google.com` — free |
| `MODEL_ARMOR_PROJECT_ID` | `console.cloud.google.com` — free tier |
| `MODEL_ARMOR_TEMPLATE_ID` | Create template in Model Armor console |
| `GOOGLE_APPLICATION_CREDENTIALS` | Service account JSON from GCP |
| `LANGCHAIN_API_KEY` | `smith.langchain.com` — free |

### Step 6 — Build Docker sandbox

```bash
docker build -t storynest-sandbox ./sandbox
```

This builds a single image used for two containers:
- **scorer** — runs wav2vec2-base as a Flask server for pronunciation scoring
- **web_fetcher** — isolated Wikipedia fetching with Model Armor sanitisation

Both containers start automatically when the app launches and are kept warm throughout the session. Docker Desktop must be running before you start the app.

---

## Running the Application

```bash
# Activate venv if not already active
venv312\Scripts\activate

# Start the app
streamlit run app.py
```

Opens at `http://localhost:8501`

**First run:**
- Page 1 — register child profile (name, age, interests, avoid list)
- Page 2 — choose story length and optional topic
- Story plays by voice, child answers puzzle, parent sees results

---

## Project Structure

See `directory_structure.txt` for full tree.

Key files:
- `app.py` — Streamlit parent dashboard
- `main.py` — LangGraph graph and session runner
- `SETTINGS.json` — all configurable paths and parameters
- `src/agents/` — one file per LangGraph node
- `src/tools/` — TTS, STT, Model Armor, spec writer
- `src/memory/sqlite.py` — all database operations
- `src/evaluation/trajectory.py` — post-session trajectory scoring
- `sandbox/` — Docker image for web fetch and pronunciation scoring

---

## Observability

**LangSmith** — automatic LLM-level tracing
View at: `smith.langchain.com/projects/storynest`
Captures: exact prompts, responses, token counts, latency per node

**logs/app.log** — structured JSON business events
Captures: fallbacks, retries, Model Armor decisions, trajectory scores

**specs/{session_id}/** — human-readable markdown audit trail
Written per session: arc.md, story_final.md, puzzle.md, trajectory.md (4 files per session)

---

## API Keys and Costs

| Service | Cost |
|---|---|
| Gemini 2.5 Flash | check current limits at aistudio.google.com |
| Model Armor | Free tier |
| LangSmith | Free — 5000 traces/month |
| Edge TTS (edge-tts) | Free — Microsoft cloud neural TTS |
| Whisper base | Free — runs locally |
| wav2vec2-base | Free — runs locally in Docker |

**Total cost: $0**

---

## Session Execution Times (approximate)

| Stage | Time |
|---|---|
| Story generation (Medium, first attempt) | 8-12 seconds |
| Story narration (Medium, ~350 words) | 25-35 seconds |
| Puzzle generation | 2-3 seconds |
| Pronunciation scoring | 3-5 seconds |
| Trajectory evaluation | 3-4 seconds |
| Full session end to end | 3-5 minutes |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Future Extensions

- **Multilingual support** — Gemini natively supports Arabic and other languages. The architecture supports this without changes — only the TTS voice and profile language field need updating
- **Multiple child profiles** — UUID-scoped session directories already support concurrent sessions. SQLite schema extension needed for multiple profiles
- **Google ADK** — for production deployment on Vertex AI, Google ADK would replace LangGraph providing native Gemini integration and built-in evaluation
- **Offline TTS** — Edge TTS requires internet (Microsoft cloud). For fully offline deployment, replace edge-tts with a local model such as Coqui TTS or Piper
- **OpenTelemetry** — for distributed tracing when scaling to a multi-service deployment on Google Cloud

---

## References

- LangGraph: `langchain-ai.github.io/langgraph`
- Whisper: Radford et al., 2022 — `arxiv.org/abs/2212.04356`
- wav2vec2: Baevski et al., 2020 — `arxiv.org/abs/2006.11477`
- Edge TTS: `github.com/rany2/edge-tts`
- Google Model Armor: `cloud.google.com/security/products/model-armor`
- LangSmith: `docs.smith.langchain.com`
- Gemini 2.5 Flash: `ai.google.dev/gemini-api/docs`
