import json
import logging
from typing import Dict, Any

import httpx

import config

logger = logging.getLogger(__name__)

LANG_NAME = {
    "ru": "Russian",
    "uk": "Ukrainian",
    "en": "English",
    "de": "German",
}


def _build_prompt(sections: Dict[str, Dict[str, Any]], lang_code: str) -> Dict[str, str]:
    lang_name = LANG_NAME.get(lang_code, "English")
    keys = ", ".join(sections.keys())
    facts_lines = []
    for key, payload in sections.items():
        facts = payload.get("facts", "")
        facts_lines.append(f"{key.upper()}: {facts}")

    system = (
        "You are a crypto analyst. Return ONLY a valid JSON object. "
        f"Language: {lang_name}. "
        "Use only the provided facts; do not invent data. "
        "Explain in simple words for a beginner, 1-2 sentences per section. "
        "Do not mention missing data. Do not mention sources or time ranges. "
        "Do not use markdown or bullet lists."
    )

    user = (
        "Sections and facts:\n"
        + "\n".join(facts_lines)
        + "\n\n"
        f"Return JSON with keys: {keys}."
    )
    return {"system": system, "user": user}


async def generate_explanations(sections: Dict[str, Dict[str, Any]], lang_code: str) -> Dict[str, str]:
    """
    Use Groq to generate short explanations per section.
    Sections format:
      {
        "tech": {"facts": "..."},
        "patterns": {"facts": "..."},
        ...
      }
    Returns dict {section_key: explanation_text}
    """
    if not sections:
        return {}

    api_key = getattr(config, "GROQ_API_KEY", "") or ""
    if not api_key:
        logger.warning("AI explainer skipped: GROQ_API_KEY missing")
        return {}

    api_url = getattr(config, "GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
    configured_model = getattr(config, "GROQ_MODEL", "") or ""
    model_candidates = [
        configured_model,
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b",
    ]
    # De-dupe while preserving order
    seen = set()
    model_candidates = [m for m in model_candidates if m and not (m in seen or seen.add(m))]

    prompt = _build_prompt(sections, lang_code)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = None
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            for model in model_candidates:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 500,
                }
                resp = await client.post(api_url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    break
                logger.warning(f"AI explainer error: {resp.status_code} model={model} body={resp.text[:200]}")
            if data is None:
                return {}
    except Exception as e:
        logger.warning(f"AI explainer request failed: {e}")
        return {}

    try:
        content = data["choices"][0]["message"]["content"]
        if "```" in content:
            content = content.strip()
            content = content.strip("`")
            content = content.replace("json", "", 1).strip()
        parsed = json.loads(content)
        out: Dict[str, str] = {}
        for key in sections.keys():
            val = parsed.get(key)
            if isinstance(val, str) and val.strip():
                out[key] = val.strip()
        return out
    except Exception as e:
        logger.warning(f"AI explainer parse failed: {e}")
        return {}
