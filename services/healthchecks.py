import asyncio
import logging
import os
from dataclasses import dataclass, asdict
from typing import Callable, Awaitable, List, Optional

import httpx

import config
from core.finbert_client import get_finbert_client
from locales.translator import get_translator
from core.portfolio_db import get_portfolio_manager

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    name: str
    ok: bool
    detail: str


async def _http_check(name: str, url: str, timeout: float = 5.0) -> HealthCheckResult:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url)
            return HealthCheckResult(name=name, ok=r.status_code < 500, detail=f"{r.status_code} {url}")
    except Exception as e:
        return HealthCheckResult(name=name, ok=False, detail=str(e))


async def _hf_inference_check() -> HealthCheckResult:
    if not config.HUGGINGFACE_API_KEY:
        return HealthCheckResult("huggingface", False, "HUGGINGFACE_API_KEY missing")
    model_id = getattr(config, "FINBERT_MODEL_ID", "ProsusAI/finbert")
    url = os.getenv("HF_INFERENCE_URL", f"https://api-inference.huggingface.co/models/{model_id}")
    headers = {"Authorization": f"Bearer {config.HUGGINGFACE_API_KEY}"}
    payload = {"inputs": "Market sentiment is neutral"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code == 200:
                return HealthCheckResult("huggingface", True, f"200 {model_id}")
            return HealthCheckResult("huggingface", False, f"{r.status_code} {r.text[:120]}")
    except Exception as e:
        return HealthCheckResult("huggingface", False, str(e))


async def _groq_check() -> HealthCheckResult:
    if not config.GROQ_API_KEY:
        return HealthCheckResult("groq", False, "GROQ_API_KEY missing")
    url = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
    model = getattr(config, "GROQ_MODEL", "llama-3.1-8b-instant")
    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return JSON {label, confidence}"},
            {"role": "user", "content": "neutral"},
        ],
        "temperature": 0,
        "max_tokens": 16,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code == 200:
                return HealthCheckResult("groq", True, f"200 {model}")
            return HealthCheckResult("groq", False, f"{r.status_code} {r.text[:120]}")
    except Exception as e:
        return HealthCheckResult("groq", False, str(e))


async def _finbert_status_check() -> HealthCheckResult:
    try:
        fb = get_finbert_client().status
        return HealthCheckResult("finbert", fb.get("available", False), f"{fb}")
    except Exception as e:
        return HealthCheckResult("finbert", False, str(e))


async def _locales_check() -> HealthCheckResult:
    try:
        tr = get_translator()
        # sanity: ensure each language has root keys loaded
        missing = []
        for lang in tr.supported_langs:
            val = tr.get("menu.portfolio", lang=lang)
            if val.startswith("["):
                missing.append(lang)
        if missing:
            return HealthCheckResult("locales", False, f"missing key in: {', '.join(missing)}")
        return HealthCheckResult("locales", True, f"langs={','.join(tr.supported_langs)}")
    except Exception as e:
        return HealthCheckResult("locales", False, str(e))


async def _portfolio_db_check() -> HealthCheckResult:
    try:
        pm = get_portfolio_manager()
        # lightweight call
        users = pm.get_all_users()
        return HealthCheckResult("portfolio_db", True, f"users={len(users)}")
    except Exception as e:
        return HealthCheckResult("portfolio_db", False, str(e))


async def run_healthchecks(context=None):
    if not getattr(config, "HEALTHCHECK_ENABLED", True):
        return

    checks: List[Callable[[], Awaitable[HealthCheckResult]]] = [
        _finbert_status_check,
        _locales_check,
        _portfolio_db_check,
        lambda: _http_check("coingecko", "https://api.coingecko.com/api/v3/ping"),
        lambda: _http_check("telegram", "https://api.telegram.org"),
    ]

    # Optional remote model checks (can be expensive / rate-limited)
    if getattr(config, "HEALTHCHECK_REMOTE_MODELS", True):
        checks.append(_hf_inference_check)
        checks.append(_groq_check)

    results: List[HealthCheckResult] = []
    for chk in checks:
        try:
            results.append(await chk())
        except Exception as e:
            results.append(HealthCheckResult(name=getattr(chk, "__name__", "check"), ok=False, detail=str(e)))

    ok_count = sum(1 for r in results if r.ok)
    total = len(results)

    lines = [f"HEALTHCHECK {ok_count}/{total}"]
    for r in results:
        status = "OK" if r.ok else "FAIL"
        lines.append(f"- {r.name}: {status} ({r.detail})")

    logger.info("\n".join(lines))
