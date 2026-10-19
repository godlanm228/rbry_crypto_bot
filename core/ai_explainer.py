import json
import logging
from typing import Dict, Any

from core.llm import groq_json_async

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

    if not config.GROQ_API_KEY:
        logger.warning("AI explainer skipped: GROQ_API_KEY missing")
        return {}

    prompt = _build_prompt(sections, lang_code)
    parsed = await groq_json_async(prompt["system"], prompt["user"], max_tokens=900)
    if not parsed:
        return {}
    out: Dict[str, str] = {}
    for key in sections.keys():
        val = parsed.get(key)
        if isinstance(val, str) and val.strip():
            out[key] = val.strip()
    return out
