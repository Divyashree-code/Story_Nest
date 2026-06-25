"""
src/skills/enrichment/web_fetch_skill.py

Fetches real-world facts from Wikipedia via the Docker sandbox,
then sanitizes the content through Model Armor before returning.

Chosen by pick_enrichment_skill() for concrete topics that have
Wikipedia articles — dinosaurs, space, elephants, Dubai, volcanoes.

Two defence layers:
    Docker sandbox  — contains HTTP execution risk (crash, huge response)
    Model Armor     — sanitizes returned content (prompt injection)

Returns empty string on any failure — Story Architect continues
using Gemini's own knowledge. Never raises.
"""

import json
import os
import tempfile
from pathlib import Path

import docker

from src.skills.base_skill import BaseSkill
from src.error_handler import safe_run
from src.errors import ToolError
from src.logger import get_logger
from src.tools.model_armor import sanitize as armor_sanitize

log = get_logger("web_fetch_skill")

# Read sandbox image name from SETTINGS.json
import json as _json
_settings = _json.loads(
    (Path(__file__).parent.parent.parent.parent / "SETTINGS.json").read_text()
)
SANDBOX_IMAGE   = _settings.get("SANDBOX_IMAGE", "storynest-sandbox")
SANDBOX_TIMEOUT = _settings.get("SANDBOX_TIMEOUT_S", 10)


class WebFetchSkill(BaseSkill):
    """
    Fetches Wikipedia facts for concrete topics via Docker sandbox.
    Safe to call — never raises, returns empty string on any failure.
    """

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetches real-world facts from Wikipedia for concrete topics "
            "such as animals, places, science, history. "
            "Use for topics like 'dinosaurs', 'space', 'elephants', 'Dubai'. "
            "Do NOT use for abstract topics like 'kindness' or 'sharing'."
        )

    def run(self, topic: str, session_id: str) -> str:
        """
        Calls Docker sandbox web_fetcher.py, then sanitizes via Model Armor.

        Args:
            topic:      story topic to fetch facts for
            session_id: for logging

        Returns:
            Clean facts string, or empty string on any failure.
        """
        return safe_run(
            self._fetch_and_sanitize,
            topic,
            session_id,
            default="",
            session_id=session_id,
        )

    def _fetch_and_sanitize(self, topic: str, session_id: str) -> str:
        """
        Internal implementation — called via safe_run for error containment.

        1. Call Docker sandbox web_fetcher.py
        2. Parse JSON output
        3. Pass raw content through Model Armor
        4. Return sanitized facts
        """
        raw_content = self._call_docker(topic, session_id)
        if not raw_content:
            return ""

        # Model Armor sanitizes raw Wikipedia content
        sanitized = armor_sanitize(raw_content, session_id)

        log.info(
            "web_fetch_complete",
            session_id=session_id,
            topic=topic,
            raw_chars=len(raw_content),
            sanitized_chars=len(sanitized),
            blocked=sanitized == "",
        )
        return sanitized

    def _call_docker(self, topic: str, session_id: str) -> str:
        """
        Runs web_fetcher.py inside Docker sandbox and returns raw content.

        Returns empty string if Docker fails or returns success=False.
        """
        try:
            client = docker.from_env()

            output = client.containers.run(
                image=SANDBOX_IMAGE,
                command=["python", "web_fetcher.py", "--topic", topic],
                remove=True,               # auto-delete container after run
                network_mode="bridge",     # web_fetcher needs internet
                mem_limit="256m",          # memory cap
                cpu_period=100000,
                cpu_quota=50000,           # 50% CPU max
            )

            # Parse JSON from stdout
            stdout = output.decode("utf-8").strip()
            result = json.loads(stdout)

            if not result.get("success"):
                log.warning(
                    "web_fetch_sandbox_failed",
                    session_id=session_id,
                    topic=topic,
                    error=result.get("error", "unknown"),
                )
                return ""

            return result.get("content", "")

        except docker.errors.ContainerError as exc:
            log.warning(
                "web_fetch_container_error",
                session_id=session_id,
                topic=topic,
                error=str(exc),
            )
            return ""
        except docker.errors.ImageNotFound:
            log.error(
                "web_fetch_image_not_found",
                session_id=session_id,
                image=SANDBOX_IMAGE,
                hint="Run: docker build -t storynest-sandbox ./sandbox",
            )
            return ""
        except Exception as exc:
            log.warning(
                "web_fetch_docker_error",
                session_id=session_id,
                topic=topic,
                error=str(exc),
            )
            return ""
