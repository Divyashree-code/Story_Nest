"""
src/tools/model_armor.py

Google Cloud Model Armor wrapper.

Sanitizes raw text returned from the Docker web fetcher before
it reaches the Story Architect. The only point in the pipeline
where external untrusted content enters — Model Armor is the
content-level defence layer after Docker's execution-level isolation.

Two distinct failure modes:
    ModelArmorAPIError  — API unavailable/timeout → retry once → pass raw through
    ModelArmorMatchError — injection detected     → block content → return ""

Public API:
    sanitize(content, session_id) -> str
        Returns sanitized content, empty string if blocked,
        or raw content if API unavailable. Never raises.

Authentication:
    Uses GOOGLE_APPLICATION_CREDENTIALS env var pointing to
    service account JSON. google-cloud-modelarmor picks this up
    automatically — no manual auth code needed.

Configuration (from .env):
    MODEL_ARMOR_PROJECT_ID   — GCP project ID
    MODEL_ARMOR_LOCATION     — e.g. us-central1
    MODEL_ARMOR_TEMPLATE_ID  — template name from GCP console
"""

import os
import time
from typing import Optional

from dotenv import load_dotenv

from src.errors import ModelArmorAPIError, ModelArmorMatchError
from src.logger import get_logger

load_dotenv()
log = get_logger("model_armor")

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ID  = os.getenv("MODEL_ARMOR_PROJECT_ID", "")
LOCATION    = os.getenv("MODEL_ARMOR_LOCATION", "us-central1")
TEMPLATE_ID = os.getenv("MODEL_ARMOR_TEMPLATE_ID", "storynest-filter")

# ── Client singleton ──────────────────────────────────────────────────────────
_client        = None
_template_name = None


def _get_client():
    """
    Returns the Model Armor client, initialising it once on first call.
    Uses GOOGLE_APPLICATION_CREDENTIALS for authentication automatically.

    Raises:
        ModelArmorAPIError: if client fails to initialise
    """
    global _client, _template_name

    if _client is not None:
        return _client, _template_name

    if not PROJECT_ID:
        raise ModelArmorAPIError(
            "MODEL_ARMOR_PROJECT_ID not set in .env",
            config_missing=True,
        )

    try:
        from google.api_core.client_options import ClientOptions
        from google.cloud import modelarmor_v1

        _client = modelarmor_v1.ModelArmorClient(
            transport="rest",
            client_options=ClientOptions(
                api_endpoint=f"modelarmor.{LOCATION}.rep.googleapis.com"
            ),
        )
        _template_name = (
            f"projects/{PROJECT_ID}"
            f"/locations/{LOCATION}"
            f"/templates/{TEMPLATE_ID}"
        )

        log.info(
            "model_armor_client_initialised",
            project_id=PROJECT_ID,
            location=LOCATION,
            template_id=TEMPLATE_ID,
        )
        return _client, _template_name

    except Exception as exc:
        raise ModelArmorAPIError(
            f"Model Armor client failed to initialise: {exc}",
            original_error=str(exc),
        ) from exc


# ── Internal sanitize call ────────────────────────────────────────────────────

def _call_model_armor(content: str) -> str:
    """
    Makes one Model Armor API call.

    Returns sanitized content string if clean.

    Raises:
        ModelArmorMatchError: if injection or policy violation detected
        ModelArmorAPIError:   if API call fails
    """
    from google.cloud import modelarmor_v1

    client, template_name = _get_client()

    request = modelarmor_v1.SanitizeUserPromptRequest(
        name=template_name,
        user_prompt_data=modelarmor_v1.DataItem(text=content),
    )

    response = client.sanitize_user_prompt(request=request)
    result   = response.sanitization_result

    # MATCH_FOUND — injection or policy violation detected
    if result.filter_match_state == modelarmor_v1.FilterMatchState.MATCH_FOUND:
        filter_details = str(result.filter_results)
        raise ModelArmorMatchError(
            "Model Armor detected prompt injection or policy violation",
            filter_results=filter_details,
        )

    # NO_MATCH_FOUND — content is clean
    return content


# ── Public API ────────────────────────────────────────────────────────────────

def sanitize(content: str, session_id: str) -> str:
    """
    Sanitizes raw web content through Model Armor before it reaches
    the Story Architect. Called after Docker web_fetcher returns text.

    Three possible outcomes:
        Clean content  → returns content unchanged
        Match detected → returns "" (Story Architect uses own knowledge)
        API failure    → returns raw content after one retry (low risk
                         since content is from Wikipedia)

    Args:
        content:    raw text from Docker web fetcher
        session_id: for logging

    Returns:
        Sanitized content string, "" if blocked, raw if API unavailable.
        Never raises.
    """
    if not content or not content.strip():
        return ""

    start = time.perf_counter()

    for attempt in range(1, 3):   # attempts 1 and 2
        try:
            result      = _call_model_armor(content)
            duration_ms = round((time.perf_counter() - start) * 1000)

            log.info(
                "model_armor_clean",
                session_id=session_id,
                attempt=attempt,
                duration_ms=duration_ms,
                content_chars=len(content),
            )
            return result

        except ModelArmorMatchError as exc:
            # Injection detected — do not retry, block content immediately
            duration_ms = round((time.perf_counter() - start) * 1000)
            log.warning(
                "model_armor_blocked",
                session_id=session_id,
                duration_ms=duration_ms,
                content_chars=len(content),
                filter_results=str(exc.context.get("filter_results", "")),
            )
            return ""   # empty string — architect uses Gemini's own knowledge

        except ModelArmorAPIError as exc:
            duration_ms = round((time.perf_counter() - start) * 1000)

            if attempt == 1:
                log.warning(
                    "model_armor_api_failed_retrying",
                    session_id=session_id,
                    attempt=attempt,
                    duration_ms=duration_ms,
                    error=str(exc),
                )
                time.sleep(2.0)
            else:
                # Both attempts failed — pass raw content through
                # Wikipedia is low risk so this is acceptable
                log.warning(
                    "model_armor_unavailable_passing_raw",
                    session_id=session_id,
                    duration_ms=duration_ms,
                    content_chars=len(content),
                    error=str(exc),
                )
                return content   # raw content — low risk from Wikipedia

        except Exception as exc:
            # Unexpected error — treat as API failure, pass raw through
            duration_ms = round((time.perf_counter() - start) * 1000)
            log.error(
                "model_armor_unexpected_error",
                session_id=session_id,
                duration_ms=duration_ms,
                error=str(exc),
                exc_info=True,
            )
            return content

    return content   # unreachable but satisfies type checker
