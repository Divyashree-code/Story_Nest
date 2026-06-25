# Entry Points

All commands run from the project root directory with the virtual environment activated.

---

## 1. Environment Setup (one time only)

### Create virtual environment
```bash
py -3.12 -m venv venv312
source venv312/Scripts/activate     # Windows Git Bash
# or
venv312\Scripts\activate            # Windows Command Prompt
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Configure environment variables
```bash
cp .env.example .env
# Open .env and fill in:
# GOOGLE_API_KEY        — from aistudio.google.com
# MODEL_ARMOR_PROJECT_ID — from console.cloud.google.com
# LANGCHAIN_API_KEY      — from smith.langchain.com
```

---

## 2. Build Docker Sandbox (one time only)

The sandbox runs the web fetcher and pronunciation scorer in isolation.
Docker Desktop must be running before this step.

```bash
docker build -t storynest-sandbox ./sandbox
```

Verify the image was built:
```bash
docker images | grep storynest-sandbox
```

---

## 3. Run the Application

```bash
streamlit run app.py
```

Opens automatically in your browser at `http://localhost:8501`

---

## 4. Run Tests

```bash
pytest tests/ -v
```

Run a specific test file:
```bash
pytest tests/test_sqlite.py -v
pytest tests/test_skills.py -v
pytest tests/test_error_handler.py -v
pytest tests/test_validator.py -v
```

---

## 5. View Logs

Structured JSON logs written to `logs/app.log`.
View in real time:
```bash
# Windows Git Bash
tail -f logs/app.log

# Pretty print last 10 entries
tail -10 logs/app.log | python -m json.tool
```

---

## 6. View Spec Files

Human-readable markdown outputs written per session:
```bash
ls specs/
ls specs/{session_id}/
cat specs/{session_id}/story_final.md
cat specs/{session_id}/trajectory.md
```

---

## 7. LangSmith Traces

With `LANGCHAIN_TRACING_V2=true` set in `.env`, all agent runs
are automatically traced. View at:
```
https://smith.langchain.com/projects/storynest
```

---

## 8. Reset Database (if needed)

```bash
rm data/story.db
# Database recreated automatically on next app start
```

---

## Key Assumptions

- Docker Desktop must be running before starting the app
- Default microphone must be connected and set as system default
- Internet connection required for Gemini API, Model Armor, and LangSmith
- Python 3.12 required — Kokoro is incompatible with Python 3.13+
- Virtual environment must be activated before running any command
- FFmpeg must be installed and on PATH for Whisper to work
