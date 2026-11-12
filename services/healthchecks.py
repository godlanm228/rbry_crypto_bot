import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Awaitable, List

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
    skipped: bool = False


_notification_state = {}


async def _http_check(name: str, url: str, timeout: float = 5.0) -> HealthCheckResult:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url)
            return HealthCheckResult(
                name=name,
                ok=200 <= r.status_code < 400,
                detail=f"{r.status_code} {url}",
            )
    except Exception as e:
        return HealthCheckResult(name=name, ok=False, detail=str(e))


async def _hf_inference_check() -> HealthCheckResult:
    if not config.HUGGINGFACE_API_KEY:
        return HealthCheckResult("huggingface", True, "not configured", skipped=True)
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
        return HealthCheckResult("groq", True, "not configured", skipped=True)
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
        available = fb.get("available", False)
        required = getattr(config, "REQUIRE_FINBERT", False)
        if not available and not required:
            return HealthCheckResult("finbert", True, f"optional: {fb}", skipped=True)
        return HealthCheckResult("finbert", available, f"{fb}")
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


def _snapshot_check(context, name: str, key: str, max_age: float) -> HealthCheckResult:
    application = getattr(context, "application", None)
    service = application.bot_data.get(key) if application else None
    if service is None:
        return HealthCheckResult(name, False, "service missing")
    latest = service.get_latest()
    timestamp = getattr(latest, "ts", None)
    started_at = float(getattr(service, "started_at", 0))
    now = time.time()
    failures = int(getattr(service, "consecutive_failures", 0))
    coverage = float(
        getattr(service, "coverage_ratio", 1.0 if timestamp is not None else 0.0)
    )
    required_coverage = float(getattr(service, "min_coverage", 0.8))
    expected = int(getattr(service, "expected_count", 0))
    available = int(getattr(service, "available_count", 0))
    warmup_grace = (
        config.HISTORY_WARMUP_GRACE_SECONDS
        if key == "history_snapshot"
        else config.SNAPSHOT_WARMUP_GRACE_SECONDS
    )
    if (
        timestamp is None
        and started_at
        and now - started_at <= warmup_grace
        and failures < 3
    ):
        detail = (
            f"warming up coverage={available}/{expected} failures={failures}"
        )
        return HealthCheckResult(name, True, detail, skipped=True)
    age = now - timestamp if timestamp else None
    last_error = getattr(service, "last_error", None)
    stale = age is None or age > max_age
    ok = not stale and failures < 3 and coverage >= required_coverage
    age_text = f"{age:.0f}s" if age is not None else "none"
    detail = (
        f"age={age_text} coverage={coverage:.0%} "
        f"rows={available}/{expected} failures={failures}"
    )
    if last_error:
        detail += f" last_error={last_error[:240]}"
    return HealthCheckResult(name, ok, detail)


def snapshot_health_result(context, name: str, key: str, max_age: float) -> HealthCheckResult:
    """Public read-only snapshot diagnostic used by owner commands."""
    return _snapshot_check(context, name, key, max_age)


async def _notify_owner(context, scope: str, results: List[HealthCheckResult]) -> None:
    failures = [result for result in results if not result.ok and not result.skipped]
    signature = tuple(sorted(result.name for result in failures))
    previous = _notification_state.get(scope, {})
    now = time.time()
    bot = getattr(context, "bot", None) if context else None
    if bot is None and context is not None:
        bot = getattr(getattr(context, "application", None), "bot", None)
    owner_id = getattr(config, "OWNER_ID", None)

    should_send_failure = bool(failures) and (
        signature != previous.get("signature")
        or now - float(previous.get("notified_at", 0)) >= 1800
    )
    should_send_recovery = not failures and bool(previous.get("signature"))

    if bot and owner_id and (should_send_failure or should_send_recovery):
        if failures:
            lines = [f"⚠️ {scope} degraded"] + [
                f"- {result.name}: {result.detail}" for result in failures
            ]
        else:
            lines = [f"✅ {scope} recovered"]
        try:
            await bot.send_message(chat_id=owner_id, text="\n".join(lines)[:3500])
        except Exception:
            logger.exception("Failed to notify owner about %s health", scope)

    _notification_state[scope] = {
        "signature": signature,
        "notified_at": now if should_send_failure else previous.get("notified_at", 0),
    }


async def run_snapshot_watchdog(context):
    results = [
        _snapshot_check(
            context,
            "market_snapshot",
            "market_snapshot",
            max(120, config.SNAPSHOT_REFRESH_SECONDS * 3),
        ),
        _snapshot_check(
            context,
            "history_snapshot",
            "history_snapshot",
            max(600, config.HISTORY_REFRESH_SECONDS * 3),
        ),
    ]
    logger.info(
        "SNAPSHOT_HEALTH\n%s",
        "\n".join(
            f"- {result.name}: "
            f"{'SKIP' if result.skipped else 'OK' if result.ok else 'FAIL'} "
            f"({result.detail})"
            for result in results
        ),
    )
    await _notify_owner(context, "snapshots", results)
    return results


async def run_healthchecks(context=None):
    if not getattr(config, "HEALTHCHECK_ENABLED", True):
        return []

    checks: List[Callable[[], Awaitable[HealthCheckResult]]] = [
        _finbert_status_check,
        _locales_check,
        _portfolio_db_check,
        lambda: _http_check("telegram", "https://api.telegram.org"),
    ]

    # A running application already has richer snapshot diagnostics below.
    # An extra /ping wastes the same IP-wide public budget and can itself cause
    # a 429 while saying nothing about whether cached user data is usable.
    if context is None:
        checks.append(
            lambda: _http_check("coingecko", "https://api.coingecko.com/api/v3/ping")
        )

    # Optional remote model checks (can be expensive / rate-limited)
    if getattr(config, "HEALTHCHECK_REMOTE_MODELS", True):
        if config.HUGGINGFACE_API_KEY:
            checks.append(_hf_inference_check)
        if config.GROQ_API_KEY:
            checks.append(_groq_check)

    results: List[HealthCheckResult] = []
    for chk in checks:
        try:
            results.append(await chk())
        except Exception as e:
            results.append(HealthCheckResult(name=getattr(chk, "__name__", "check"), ok=False, detail=str(e)))

    if context is not None:
        results.extend([
            _snapshot_check(
                context,
                "market_snapshot",
                "market_snapshot",
                max(120, config.SNAPSHOT_REFRESH_SECONDS * 3),
            ),
            _snapshot_check(
                context,
                "history_snapshot",
                "history_snapshot",
                max(600, config.HISTORY_REFRESH_SECONDS * 3),
            ),
        ])

    active = [result for result in results if not result.skipped]
    ok_count = sum(1 for result in active if result.ok)
    total = len(active)

    lines = [f"HEALTHCHECK {ok_count}/{total}"]
    for r in results:
        status = "SKIP" if r.skipped else "OK" if r.ok else "FAIL"
        lines.append(f"- {r.name}: {status} ({r.detail})")

    logger.info("\n".join(lines))
    if context is not None:
        await _notify_owner(
            context,
            "healthchecks",
            [
                result
                for result in results
                if result.name not in {"market_snapshot", "history_snapshot"}
            ],
        )
    return results
