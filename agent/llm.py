"""Groq free-tier LLM client (OpenAI-compatible)."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv

DEFAULT_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def make_client() -> Any:
    """Create an OpenAI client pointed at Groq."""
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://console.groq.com"
        )
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def chat(
    *,
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    client: Any = None,
    max_retries: int = 4,
) -> str:
    """Chat completion with exponential backoff on rate limits."""
    cli = client or make_client()
    delay = 1.0
    last_err: Exception | None = None

    for attempt in range(max_retries):
        try:
            resp = cli.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            content = resp.choices[0].message.content or ""
            return content
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            msg = str(exc).lower()
            if "429" in msg or "rate" in msg or "timeout" in msg:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise
    raise RuntimeError(f"LLM request failed after retries: {last_err}")


def chat_json(
    *,
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    client: Any = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """
    Request JSON and parse it. On parse failure, one repair retry.
    """
    json_system = (
        system
        + "\n\nYou MUST respond with a single valid JSON object only. "
        "No markdown fences, no commentary."
    )
    raw = chat(
        system=json_system,
        user=user,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        client=client,
    )
    try:
        return _parse_json(raw)
    except json.JSONDecodeError:
        repair_user = (
            "Your previous response was not valid JSON. "
            "Fix it and return ONLY valid JSON.\n\nPrevious response:\n"
            + raw[:4000]
        )
        raw2 = chat(
            system=json_system,
            user=repair_user,
            model=model,
            temperature=0,
            max_tokens=max_tokens,
            client=client,
        )
        return _parse_json(raw2)


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    # strip markdown fences if present
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        # allow top-level list by wrapping
        if isinstance(data, list):
            return {"items": data}
        raise json.JSONDecodeError("Expected JSON object", text, 0)
    return data
