"""
sandbox/web_fetcher.py

Fetches a Wikipedia article summary for a given topic.
Runs inside Docker — HTTP call is isolated from the main process.

Usage:
    python web_fetcher.py --topic dinosaurs
    python web_fetcher.py --topic "sharing kindness"

Output (single JSON line to stdout):
    {"success": true,  "content": "Dinosaurs were reptiles..."}
    {"success": false, "error": "Wikipedia article not found"}

Why in Docker:
    External HTTP with unknown response size and encoding.
    A malformed response could crash the process.
    Docker contains the crash — main app is unaffected.

Exit codes:
    0 — always, even on failure. Caller reads JSON success field.
"""

import argparse
import json
import sys
import urllib.parse

import requests

# ── Config ────────────────────────────────────────────────────────────────────
WIKIPEDIA_API   = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
TIMEOUT_SECONDS = 5
MAX_CONTENT_CHARS = 2000   # cap response to avoid huge payloads


def fetch_wikipedia(topic: str) -> dict:
    """
    Fetches Wikipedia article summary for the given topic.

    Returns dict with success/content or success/error.
    """
    # URL-encode topic for safe HTTP request
    encoded = urllib.parse.quote(topic.replace(" ", "_"))
    url     = WIKIPEDIA_API.format(encoded)

    try:
        response = requests.get(
            url,
            timeout=TIMEOUT_SECONDS,
            headers={"User-Agent": "storynest-agent/1.0"},
        )

        if response.status_code == 404:
            return {
                "success": False,
                "error": f"Wikipedia article not found for topic: {topic}",
            }

        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Wikipedia returned status {response.status_code}",
            }

        data    = response.json()
        extract = data.get("extract", "").strip()

        if not extract:
            return {
                "success": False,
                "error": "Wikipedia article exists but has no summary text",
            }

        # Cap content length to avoid sending huge payloads back
        content = extract[:MAX_CONTENT_CHARS]

        return {
            "success": True,
            "content": content,
            "title":   data.get("title", topic),
        }

    except requests.Timeout:
        return {
            "success": False,
            "error": f"Wikipedia request timed out after {TIMEOUT_SECONDS}s",
        }
    except requests.ConnectionError as exc:
        return {
            "success": False,
            "error": f"Wikipedia connection failed: {exc}",
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Unexpected error fetching Wikipedia: {exc}",
        }


def main():
    parser = argparse.ArgumentParser(description="Wikipedia web fetcher sandbox")
    parser.add_argument(
        "--topic",
        required=True,
        help="Topic to fetch from Wikipedia",
    )
    args = parser.parse_args()

    result = fetch_wikipedia(args.topic.strip())

    # Print single JSON line to stdout — caller reads and parses this
    print(json.dumps(result))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
