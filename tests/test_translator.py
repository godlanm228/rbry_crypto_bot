"""
Unit tests for the translation engine
"""

import pytest
from locales.translator import Translator, get_translator, t
from core.formatting import format_price


class TestTranslator:
    """Test suite for Translator class"""

    def setup_method(self):
        """Setup test translator instance"""
        self.translator = Translator()

    def test_translator_initialization(self):
        """Test translator initializes correctly"""
        assert self.translator is not None
        assert len(self.translator.supported_langs) == 4
        assert "ru" in self.translator.supported_langs
        assert "uk" in self.translator.supported_langs
        assert "en" in self.translator.supported_langs
        assert "de" in self.translator.supported_langs
        assert self.translator.default_lang == "ru"

    def test_basic_translation_russian(self):
        """Test simple key retrieval in Russian"""
        result = self.translator.get("menu.portfolio", lang="ru")
        assert result == "💼 Портфель"

        result = self.translator.get("menu.stats", lang="ru")
        assert result == "📊 Статистика"

    def test_nested_key_access(self):
        """Test nested key access with dot notation"""
        result = self.translator.get("welcome.title", lang="ru")
        assert "RBRY CRYPTOBOT" in result

        result = self.translator.get("settings.button_timezone", lang="ru")
        assert "Часовой пояс" in result

    def test_parameter_substitution_simple(self):
        """Test basic parameter substitution"""
        result = self.translator.get(
            "tier.max_coins",
            lang="ru",
            count=5
        )
        assert "5" in result
        assert "монет" in result

    def test_parameter_substitution_complex(self):
        """Test complex parameter substitution with formatting"""
        result = self.translator.get(
            "portfolio.coin_entry",
            lang="ru",
            coin="BTC",
            amount=0.5,
            price=format_price(50000)
        )
        assert "BTC" in result
        assert "0.500000" in result
        assert "$50,000.00" in result

    def test_list_value_handling(self):
        """Test handling of list values (joined with newlines)"""
        result = self.translator.get("welcome.features", lang="ru")
        assert isinstance(result, str)
        assert "Портфель" in result
        assert "Обзор рынка" in result
        assert "\n" in result

    def test_missing_key_returns_fallback(self):
        """Test that missing keys return [key] format"""
        result = self.translator.get("nonexistent.key", lang="ru")
        assert result == "[nonexistent.key]"

    def test_invalid_language_defaults_to_russian(self):
        """Test that invalid language codes default to Russian"""
        result = self.translator.get("menu.portfolio", lang="invalid")
        assert result == "💼 Портфель"  # Russian

    def test_fallback_to_russian_for_missing_translation(self):
        """A key present in every language resolves in each of them, never to its bracketed name"""
        result_ru = self.translator.get("menu.portfolio", lang="ru")
        result_en = self.translator.get("menu.portfolio", lang="en")
        assert result_ru is not None and "[menu.portfolio]" not in result_ru
        assert result_en is not None
        assert "[menu.portfolio]" not in result_en

    def test_caching_works(self):
        """Test that translations are cached"""
        # First call loads from file
        result1 = self.translator.get("menu.portfolio", lang="ru")

        # Check cache is populated
        assert "ru" in self.translator._cache

        # Second call should use cache
        result2 = self.translator.get("menu.portfolio", lang="ru")

        assert result1 == result2

    def test_reload_clears_cache(self):
        """Test that reload() clears and reloads cache"""
        # Load some translations
        self.translator.get("menu.portfolio", lang="ru")
        assert len(self.translator._cache) > 0

        # Reload
        self.translator.reload()

        # Cache should be repopulated
        assert len(self.translator._cache) > 0

    def test_get_translator_singleton(self):
        """Test that get_translator returns singleton"""
        translator1 = get_translator()
        translator2 = get_translator()
        assert translator1 is translator2

    def test_convenience_function_t(self):
        """Test convenience function t()"""
        result = t("menu.portfolio", lang="ru")
        assert result == "💼 Портфель"

    def test_t_with_parameters(self):
        """Test convenience function with parameters"""
        result = t(
            "portfolio.coin_entry",
            lang="ru",
            coin="ETH",
            amount=2.5,
            price=format_price(3000)
        )
        assert "ETH" in result
        assert "2.500000" in result
        assert "$3,000.00" in result

    def test_all_common_keys_present(self):
        """Test that common translation keys are present"""
        common_keys = [
            "common.back",
            "common.main_menu",
            "common.loading",
            "common.error"
        ]

        for key in common_keys:
            result = self.translator.get(key, lang="ru")
            assert "[" not in result  # Not missing
            assert result != ""  # Not empty

    def test_all_menu_keys_present(self):
        """Test that all menu keys are present"""
        menu_keys = [
            "menu.portfolio",
            "menu.stats",
            "menu.analysis",
            "menu.movers",
            "menu.news",
            "menu.settings",
            "menu.my_tier"
        ]

        for key in menu_keys:
            result = self.translator.get(key, lang="ru")
            assert "[" not in result
            assert result != ""

    def test_parameter_formatting_precision(self):
        """Test different formatting precisions work"""
        # Test .2f (2 decimal places)
        result = self.translator.get(
            "stats.price",
            lang="ru",
            price=format_price(1234.56789)
        )
        assert "$1,234.57" in result

        # Cheap coins keep their significant digits instead of "$0.00"
        result = self.translator.get("stats.price", lang="ru", price=format_price(0.00001234))
        assert "$0.00001234" in result

        # Test percentage formatting
        result = self.translator.get(
            "stats.profit_loss",
            lang="ru",
            pl=15.789
        )
        assert "+15.79%" in result or "15.79" in result

    def test_parameter_substitution_missing_param(self):
        """Test behavior when parameter is missing"""
        # Should log warning but not crash
        result = self.translator.get(
            "portfolio.coin_entry",
            lang="ru",
            coin="BTC"
            # Missing amount and price parameters
        )
        # Should still return something
        assert result is not None
        assert "BTC" in result

    def test_unicode_handling(self):
        """Test that Unicode (Cyrillic) characters handled correctly"""
        result = self.translator.get("menu.portfolio", lang="ru")
        assert "Портфель" in result

        result = self.translator.get("analysis.analyzing_portfolio", lang="ru")
        assert "Анализирую" in result

    def test_emoji_preservation(self):
        """Test that emojis are preserved in translations"""
        result = self.translator.get("menu.portfolio", lang="ru")
        assert "💼" in result

        result = self.translator.get("welcome.title", lang="ru")
        assert "🤖" in result

    def test_formatting_markers_preserved(self):
        """Test that markdown-style formatting is preserved.

        Locale strings store **bold** markers; as_html() in handlers converts
        them to <b> at render time, so raw translations must keep the markers.
        """
        result = self.translator.get("portfolio.title", lang="ru")
        assert "**" in result  # Markdown bold

        result = self.translator.get("settings.title", lang="ru")
        assert "**" in result  # Markdown bold (converted to <b> by as_html)

    def test_multiline_strings(self):
        """Test that multiline strings work correctly"""
        result = self.translator.get("help.alerts_desc", lang="ru")
        assert "\n" in result  # Contains newlines
        assert "Проверка каждые 15 минут" in result


class TestTranslatorEdgeCases:
    """Test edge cases and error handling"""

    def setup_method(self):
        self.translator = Translator()

    def test_empty_key(self):
        """Test behavior with empty key"""
        result = self.translator.get("", lang="ru")
        assert result == "[]"

    def test_key_with_only_dots(self):
        """Test behavior with malformed key"""
        result = self.translator.get("...", lang="ru")
        assert "[" in result  # Returns fallback

    def test_deeply_nested_key(self):
        """Test accessing deeply nested keys"""
        result = self.translator.get("premium.button_contact", lang="ru")
        assert result is not None
        assert "[" not in result

    def test_parameter_with_special_characters(self):
        """Test parameters containing special characters"""
        result = self.translator.get(
            "analysis.no_market_data",
            lang="ru",
            coin="BTC/USDT"
        )
        assert "BTC/USDT" in result

    def test_very_long_parameter(self):
        """Test with very long parameter values"""
        long_string = "A" * 1000
        result = self.translator.get(
            "analysis.no_market_data",
            lang="ru",
            coin=long_string
        )
        assert long_string in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

