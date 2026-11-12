import logging
import os
from typing import Optional, Tuple

import httpx

from core.llm import groq_json, groq_json_async

import config

logger = logging.getLogger(__name__)

FINBERT_MODEL_ID = getattr(config, "FINBERT_MODEL_ID", "ProsusAI/finbert")
HF_INFERENCE_URL = os.getenv(
    "HF_INFERENCE_URL",
    f"https://api-inference.huggingface.co/models/{FINBERT_MODEL_ID}",
)

# Groq defaults can change; allow explicit override.


class FinBERTClient:
    """News sentiment through a cloud backend: Hugging Face's FinBERT, else Groq. Without a key there is no
    sentiment at all (never a made-up "neutral"). The on-device model (torch, 2.5 GB) was never installed
    anywhere and was removed on 12.11.2026."""

    def __init__(self):
        self.enabled = bool(getattr(config, "ENABLE_FINBERT", False))
        self.require = bool(getattr(config, "REQUIRE_FINBERT", False))
        self.hf_api_key = getattr(config, "HUGGINGFACE_API_KEY", "") or ""
        self.groq_api_key = getattr(config, "GROQ_API_KEY", "") or ""
        self._status = {
            "enabled": self.enabled,
            "available": False,
            "backend": "none",
            "reason": "disabled" if not self.enabled else "not_initialized",
        }

        if not self.enabled:
            return

        if self.hf_api_key:
            self._status.update({"available": True, "backend": "huggingface", "reason": "remote_hf"})
        elif self.groq_api_key:
            self._status.update({"available": True, "backend": "groq", "reason": "remote_groq"})
        else:
            self._status.update({"available": False, "backend": "none", "reason": "no_backend"})

        if self.require and not self._status["available"]:
            raise RuntimeError("FinBERT is required but no backend is available")

    @property
    def status(self):
        return dict(self._status)

    def _normalize_label(self, label: str) -> str:
        if not label:
            return "neutral"
        l = label.strip().lower()
        if l.startswith("pos") or l == "positive":
            return "positive"
        if l.startswith("neg") or l == "negative":
            return "negative"
        if l.startswith("neu") or l == "neutral":
            return "neutral"
        # FinBERT sometimes returns LABEL_0..2; assume neutral if unknown
        return "neutral"

    def _parse_hf_response(self, data) -> Tuple[str, float]:
        # HF Inference returns list of dicts for text-classification
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(str(data["error"]))
        if isinstance(data, list) and data:
            top = data[0]
            label = self._normalize_label(top.get("label"))
            score = float(top.get("score", 0.0))
            return label, score
        return "neutral", 0.5

    def _hf_infer_sync(self, text: str) -> Tuple[str, float]:
        headers = {"Authorization": f"Bearer {self.hf_api_key}"}
        payload = {"inputs": text}
        with httpx.Client(timeout=15) as client:
            r = client.post(HF_INFERENCE_URL, headers=headers, json=payload)
            r.raise_for_status()
            return self._parse_hf_response(r.json())

    async def _hf_infer_async(self, text: str) -> Tuple[str, float]:
        headers = {"Authorization": f"Bearer {self.hf_api_key}"}
        payload = {"inputs": text}
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(HF_INFERENCE_URL, headers=headers, json=payload)
            r.raise_for_status()
            return self._parse_hf_response(r.json())

    def _groq_infer_sync(self, text: str) -> Tuple[str, float]:
        system = (
            "You are a sentiment classifier for crypto news."
            " Answer with a JSON object with keys 'label' and 'confidence'."
            " label is one of: positive, negative, neutral; confidence is 0..1."
        )
        parsed = groq_json(system, f"Classify sentiment:\n{text[:1000]}", max_tokens=300)
        return self._groq_result(parsed)

    async def _groq_infer_async(self, text: str) -> Tuple[str, float]:
        system = (
            "You are a sentiment classifier for crypto news."
            " Answer with a JSON object with keys 'label' and 'confidence'."
            " label is one of: positive, negative, neutral; confidence is 0..1."
        )
        parsed = await groq_json_async(system, f"Classify sentiment:\n{text[:1000]}", max_tokens=300)
        return self._groq_result(parsed)

    def _groq_result(self, parsed) -> Tuple[str, float]:
        # Unusable output is an error, never a silent "neutral".
        if not parsed or str(parsed.get("label", "")).lower() not in ("positive", "negative", "neutral"):
            raise ValueError("Groq sentiment unavailable")
        return self._normalize_label(parsed["label"]), float(parsed.get("confidence", 0.5))

    def analyze_sync(self, text: str) -> Tuple[str, float, str]:
        if not self.enabled:
            return "neutral", 0.5, "disabled"

        if self.hf_api_key:
            try:
                label, score = self._hf_infer_sync(text[:512])
                return label, score, "huggingface"
            except Exception as e:
                logger.warning("HF inference error: %r", e)

        if self.groq_api_key:
            try:
                label, score = self._groq_infer_sync(text[:512])
                return label, score, "groq"
            except Exception as e:
                logger.warning("Groq inference error: %r", e)

        # No backend answered: report it instead of a fabricated "neutral".
        raise RuntimeError("no sentiment backend answered")

    async def analyze_async(self, text: str) -> Tuple[str, float, str]:
        if not self.enabled:
            return "neutral", 0.5, "disabled"

        if self.hf_api_key:
            try:
                label, score = await self._hf_infer_async(text[:512])
                return label, score, "huggingface"
            except Exception as e:
                logger.warning("HF inference async error: %r", e)

        if self.groq_api_key:
            try:
                label, score = await self._groq_infer_async(text[:512])
                return label, score, "groq"
            except Exception as e:
                logger.warning("Groq inference async error: %r", e)

        # No backend answered: report it instead of a fabricated "neutral".
        raise RuntimeError("no sentiment backend answered")


_finbert_client: Optional[FinBERTClient] = None


def get_finbert_client() -> FinBERTClient:
    global _finbert_client
    if _finbert_client is None:
        _finbert_client = FinBERTClient()
    return _finbert_client
