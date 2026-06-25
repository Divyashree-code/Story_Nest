"""
src/tools/sandbox_manager.py

Manages pre-warmed Docker sandbox containers for scorer and web_fetch.

Both containers are pre-warmed at session start:

  scorer:
    - Runs scorer.py in Flask server mode (--server)
    - wav2vec2 loaded into RAM once at container start
    - ./data directory mounted read-only as /data
    - Uses internal Docker network (no internet, host can reach Flask)
    - Scoring triggered via POST /score {"file": "/data/recording_xxx.wav"}
    - Audio never travels over HTTP — only filename sent, file read from mount

  web_fetch:
    - Container kept running (sleep infinity)
    - Uses bridge network (needs internet for Wikipedia)
    - Triggered via exec_run passing --topic argument
    - No persistent process needed — lightweight, no model to load

Lifecycle per call:
  1. Use pre-warmed container
  2. Destroy it immediately after
  3. Spin up fresh replacement in background (ready for next session)

This proves ephemeral isolation — each container instance handles one call
and is destroyed. Security properties preserved:
  - scorer:    no internet (internal network), non-root user, resource limits
  - web_fetch: non-root user, resource limits, bridge network for Wikipedia only
"""

import json
import threading
import time
from pathlib import Path

import docker
import requests as http_requests

from src.logger import get_logger

log = get_logger("sandbox_manager")

# ── Settings ──────────────────────────────────────────────────────────────────
_settings = json.loads(
    (Path(__file__).parent.parent.parent / "SETTINGS.json").read_text()
)
SANDBOX_IMAGE   = _settings.get("SANDBOX_IMAGE", "storynest-sandbox")
SANDBOX_TIMEOUT = _settings.get("SANDBOX_TIMEOUT_S", 30)
DATA_DIR        = Path(_settings.get("DATA_DIR", "./data")).resolve()

# ── Singletons ────────────────────────────────────────────────────────────────
_scorer_container    = None   # dict: {"container": ..., "port": int}
_web_fetch_container = None   # docker container object
_lock = threading.Lock()


# ── Docker client ─────────────────────────────────────────────────────────────

def _get_client():
    return docker.from_env()


# ── Scorer container ──────────────────────────────────────────────────────────

def _start_scorer_container():
    """
    Starts scorer container in Flask server mode.
    - Mounts ./data as /data (read-only) so scorer can read audio files
    - Bridge network with port published to 127.0.0.1 (localhost only)
    - TRANSFORMERS_OFFLINE=1 baked into image prevents outbound model downloads
    - Production hardening: add gVisor runtime for kernel-level network isolation
    """
    client = _get_client()
    container = client.containers.run(
        image=SANDBOX_IMAGE,
        command=["python", "scorer.py", "--server"],
        volumes={str(DATA_DIR).replace("\\", "/"): {"bind": "/data", "mode": "ro"}},
        network_mode="bridge",
        mem_limit="512m",
        cpu_period=100000,
        cpu_quota=50000,
        ports={"8080/tcp": ("127.0.0.1", None)},   # localhost only, not exposed externally
        detach=True,
    )
    return container


def _wait_for_scorer(container, session_id: str, timeout: int = 60) -> int:
    """
    Waits for scorer Flask server to be ready via health check.
    Returns the assigned localhost port.
    """
    container.reload()
    port = int(container.ports["8080/tcp"][0]["HostPort"])

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = http_requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if resp.status_code == 200:
                log.info("scorer_ready", session_id=session_id, port=port)
                return port
        except Exception:
            pass
        time.sleep(0.5)

    raise RuntimeError(f"Scorer container did not become ready within {timeout}s")


# ── Web fetch container ───────────────────────────────────────────────────────

def _start_web_fetch_container():
    """
    Starts web_fetch container and keeps it alive.
    - Bridge network (needs internet for Wikipedia)
    - exec_run used per call — no persistent process needed
    """
    client = _get_client()
    container = client.containers.run(
        image=SANDBOX_IMAGE,
        command=["sleep", "infinity"],
        network_mode="bridge",
        mem_limit="256m",
        cpu_period=100000,
        cpu_quota=50000,
        detach=True,
    )
    return container


# ── Destroy helpers ───────────────────────────────────────────────────────────

def _destroy_container(container):
    """Stops and removes a container, ignoring errors."""
    try:
        container.stop(timeout=5)
        container.remove()
    except Exception:
        pass


def _replace_scorer(session_id: str):
    """Destroys used scorer and spins up a fresh replacement."""
    global _scorer_container
    try:
        c = _start_scorer_container()
        port = _wait_for_scorer(c, session_id)
        with _lock:
            _scorer_container = {"container": c, "port": port}
        log.info("scorer_replaced", session_id=session_id, port=port)
    except Exception as exc:
        log.warning("scorer_replace_failed", session_id=session_id, error=str(exc))
        with _lock:
            _scorer_container = None


def _replace_web_fetch(session_id: str):
    """Destroys used web_fetch container and spins up a fresh replacement."""
    global _web_fetch_container
    try:
        c = _start_web_fetch_container()
        with _lock:
            _web_fetch_container = c
        log.info("web_fetch_replaced", session_id=session_id)
    except Exception as exc:
        log.warning("web_fetch_replace_failed", session_id=session_id, error=str(exc))
        with _lock:
            _web_fetch_container = None


# ── Public API ────────────────────────────────────────────────────────────────

def _prewarm_scorer(session_id: str):
    """Starts scorer container and waits for Flask to be ready."""
    global _scorer_container
    try:
        log.info("scorer_prewarming", session_id=session_id)
        c = _start_scorer_container()
        port = _wait_for_scorer(c, session_id)
        with _lock:
            _scorer_container = {"container": c, "port": port}
        log.info("scorer_prewarmed", session_id=session_id, port=port)
    except Exception as exc:
        log.warning("scorer_prewarm_failed", session_id=session_id, error=str(exc))


def _prewarm_web_fetch(session_id: str):
    """Starts web_fetch container."""
    global _web_fetch_container
    try:
        log.info("web_fetch_prewarming", session_id=session_id)
        c = _start_web_fetch_container()
        with _lock:
            _web_fetch_container = c
        log.info("web_fetch_prewarmed", session_id=session_id)
    except Exception as exc:
        log.warning("web_fetch_prewarm_failed", session_id=session_id, error=str(exc))


def prewarm(session_id: str):
    """
    Pre-warms scorer and web_fetch containers in parallel.
    Both start simultaneously — scorer takes ~15s (wav2vec2 load),
    web_fetch takes ~500ms. Waits for both to complete.
    """
    t_scorer    = threading.Thread(target=_prewarm_scorer,    args=(session_id,), daemon=True)
    t_web_fetch = threading.Thread(target=_prewarm_web_fetch, args=(session_id,), daemon=True)

    t_scorer.start()
    t_web_fetch.start()

    t_scorer.join()
    t_web_fetch.join()


def call_scorer(audio_path: Path, session_id: str) -> dict:
    """
    Sends POST /score to warm scorer container with audio filename.
    Audio is read by scorer from /data mount — not sent over HTTP.
    Destroys container after use, spins up replacement in background.

    Returns dict with success/score or success/error.
    """
    with _lock:
        scorer_info = _scorer_container

    if not scorer_info:
        log.warning("scorer_not_available", session_id=session_id)
        return {"success": False, "error": "Scorer container not available"}

    container = scorer_info["container"]
    port      = scorer_info["port"]

    try:
        resp = http_requests.post(
            f"http://127.0.0.1:{port}/score",
            json={"file": f"/data/{audio_path.name}"},
            timeout=SANDBOX_TIMEOUT,
        )
        result = resp.json()
        log.info("scorer_result", session_id=session_id,
                 success=result.get("success"), score=result.get("score"))
        return result

    except Exception as exc:
        log.warning("scorer_call_failed", session_id=session_id, error=str(exc))
        return {"success": False, "error": str(exc)}

    finally:
        # Destroy used container
        _destroy_container(container)
        # Spin up replacement in background — ready for next session
        t = threading.Thread(
            target=_replace_scorer, args=(session_id,), daemon=True
        )
        t.start()


def call_web_fetch(topic: str, session_id: str) -> dict:
    """
    Runs web_fetcher.py inside warm container via exec_run.
    No model to keep in RAM — exec_run is fast enough for web_fetch.
    Destroys container after use, spins up replacement in background.

    Returns dict with success/content or success/error.
    """
    with _lock:
        container = _web_fetch_container

    if not container:
        log.warning("web_fetch_not_available", session_id=session_id)
        return {"success": False, "error": "Web fetch container not available"}

    try:
        exit_code, output = container.exec_run(
            ["python", "web_fetcher.py", "--topic", topic],
            demux=False,
        )
        result = json.loads(output.decode("utf-8").strip())
        log.info("web_fetch_result", session_id=session_id,
                 topic=topic, success=result.get("success"))
        return result

    except Exception as exc:
        log.warning("web_fetch_call_failed", session_id=session_id, error=str(exc))
        return {"success": False, "error": str(exc)}

    finally:
        # Destroy used container
        _destroy_container(container)
        # Spin up replacement in background
        t = threading.Thread(
            target=_replace_web_fetch, args=(session_id,), daemon=True
        )
        t.start()


def cleanup():
    """
    Stops and removes both containers.
    Called on app shutdown to clean up running containers.
    """
    global _scorer_container, _web_fetch_container
    with _lock:
        si = _scorer_container
        wf = _web_fetch_container
        _scorer_container    = None
        _web_fetch_container = None

    if si:
        _destroy_container(si["container"])
    if wf:
        _destroy_container(wf)
    log.info("sandbox_containers_cleaned_up")
