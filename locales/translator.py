"""
Translation Engine with In-Memory Caching

Supports parameter substitution and fallback to Russian for missing keys.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class Translator:
    """
    Translation engine with caching and parameter support

    Features:
    - JSON-based translations
    - In-memory caching for performance
    - Nested key access (e.g., "menu.portfolio")
    - Parameter substitution (e.g., {coin}, {amount})
    - Fallback to Russian for missing keys
    """

    def __init__(self, locales_dir: Path = None):
        if locales_dir is None:
            locales_dir = Path(__file__).parent / "strings"

        self.locales_dir = locales_dir
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.supported_langs = ["ru", "uk", "en", "de"]
        self.default_lang = "ru"

        # Preload all translations on init
        self._preload_translations()

    def _preload_translations(self):
        """Preload all translation files into memory"""
        for lang in self.supported_langs:
            try:
                self._load_language(lang)
                logger.info(f"Loaded translations for {lang}")
            except Exception as e:
                logger.error(f"Failed to load {lang} translations: {e}")

    def _load_language(self, lang: str) -> Dict[str, Any]:
        """Load translation file for a language"""
        if lang in self._cache:
            return self._cache[lang]

        lang_file = self.locales_dir / f"{lang}.json"

        if not lang_file.exists():
            logger.warning(f"Translation file not found: {lang_file}")
            if lang != self.default_lang:
                return self._load_language(self.default_lang)
            return {}

        try:
            # utf-8-sig handles BOM if present, and works for plain utf-8 too
            with open(lang_file, 'r', encoding='utf-8-sig') as f:
                translations = json.load(f)
                self._cache[lang] = translations
                return translations
        except Exception as e:
            logger.error(f"Error loading {lang_file}: {e}")
            if lang != self.default_lang:
                return self._load_language(self.default_lang)
            return {}

    def _get_nested_value(self, data: Dict, key_path: str) -> Optional[Any]:
        """
        Get value from nested dict using dot notation
        Example: "menu.portfolio" -> data["menu"]["portfolio"]
        """
        keys = key_path.split('.')
        value = data

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return None
            else:
                return None

        return value

    def get(self, key: str, lang: str = "ru", **params) -> str:
        """
        Get translated string with parameter substitution

        Args:
            key: Translation key (supports dot notation: "menu.portfolio")
            lang: Language code (ru, uk, en, de)
            **params: Parameters to substitute in the string

        Returns:
            Translated string with parameters substituted

        Example:
            t.get("portfolio.coin_entry", lang="en", coin="BTC", amount=0.5, price=50000)
            -> "💰 BTC: 0.500000 @ $50000.00"
        """
        # Validate language
        if lang not in self.supported_langs:
            lang = self.default_lang

        # Load translations
        translations = self._load_language(lang)

        # Get value
        value = self._get_nested_value(translations, key)

        # Fallback to Russian if key not found
        if value is None and lang != self.default_lang:
            translations = self._load_language(self.default_lang)
            value = self._get_nested_value(translations, key)

        # Still not found - return key itself as fallback
        if value is None:
            logger.warning(f"Translation key not found: {key} (lang: {lang})")
            return f"[{key}]"

        # Handle list values (join with newlines)
        if isinstance(value, list):
            value = "\n".join(str(v) for v in value)

        # Convert to string
        value = str(value)

        # Substitute parameters
        if params:
            try:
                value = value.format(**params)
            except (KeyError, ValueError, IndexError) as e:
                logger.warning(f"Parameter substitution failed for key {key}: {e}")
                # Graceful degradation: substitute the params we DO have,
                # leave unknown placeholders as-is instead of returning raw text
                value = self._partial_format(value, params)

        return value

    @staticmethod
    def _partial_format(template: str, params: Dict[str, Any]) -> str:
        """Replace only the provided {name} / {name:spec} placeholders."""
        def _repl(match):
            name = match.group(1)
            spec = match.group(2)
            if name not in params:
                return match.group(0)
            try:
                return format(params[name], spec) if spec else str(params[name])
            except (ValueError, TypeError):
                return str(params[name])

        return re.sub(r"\{(\w+)(?::([^{}]*))?\}", _repl, template)

    def reload(self):
        """Clear cache and reload all translations"""
        self._cache.clear()
        self._preload_translations()


# Global singleton
_translator: Optional[Translator] = None


def get_translator() -> Translator:
    """Get global translator instance"""
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator


def t(key: str, user_id: int = None, lang: str = None, **params) -> str:
    """
    Convenience function for translations

    Args:
        key: Translation key
        user_id: User ID (to auto-detect language from preferences)
        lang: Language code (overrides user preference)
        **params: Parameters for substitution

    Returns:
        Translated string

    Example:
        t("menu.portfolio", user_id=12345)  # Auto-detects user's language
        t("portfolio.coin_entry", lang="en", coin="ETH", amount=2.5, price=3000)
    """
    translator = get_translator()

    # Auto-detect language from user preferences
    if lang is None and user_id is not None:
        try:
            from modules.user_prefs import get_language
            lang = get_language(user_id)
        except Exception as e:
            logger.warning(f"Failed to get user language: {e}")
            lang = None

    if lang is None:
        lang = "ru"

    if user_id is not None and "user_id" not in params:
        params["user_id"] = user_id

    return translator.get(key, lang=lang, **params)
