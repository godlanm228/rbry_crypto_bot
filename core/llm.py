"""One Groq chat helper for the explanation layer, news translation and tone.

The LLM never decides anything here: it translates, summarises supplied text
or explains facts computed elsewhere. Every caller must handle ``None``
(no key, rate limit, malformed output) by showing less, never by inventing.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import httpx

import config

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def _payload(system: str, user: str, max_tokens: int) -> dict:
    return {
        "model": config.GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        # gpt-oss models reason before answering; keep it short so the JSON
        # answer is not cut off by max_tokens.
        "reasoning_effort": "low",
        "response_format": {"type": "json_object"},
    }


def _parse(response: httpx.Response) -> Optional[dict]:
    if response.status_code != 200:
        logger.warning("Groq %s: %s", response.status_code, response.text[:200])
        return None
    try:
        content = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, ValueError) as error:
        logger.warning("Groq returned unusable output: %s", error)
        return None
    return parsed if isinstance(parsed, dict) else None


def _headers() -> dict:
    return {"Authorization": f"Bearer {config.GROQ_API_KEY}", "Content-Type": "application/json"}


def groq_json(system: str, user: str, *, max_tokens: int = 800, timeout: float = 25.0) -> Optional[dict]:
    """Blocking JSON chat call; None when unavailable or unusable."""
    if not config.GROQ_API_KEY:
        return None
    try:
        response = httpx.post(GROQ_API_URL, headers=_headers(), json=_payload(system, user, max_tokens), timeout=timeout)
    except httpx.HTTPError as error:
        logger.warning("Groq request failed: %s", error)
        return None
    return _parse(response)


async def groq_json_async(system: str, user: str, *, max_tokens: int = 800, timeout: float = 25.0) -> Optional[dict]:
    """Async variant of :func:`groq_json`."""
    if not config.GROQ_API_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(GROQ_API_URL, headers=_headers(), json=_payload(system, user, max_tokens))
    except httpx.HTTPError as error:
        logger.warning("Groq request failed: %s", error)
        return None
    return _parse(response)
