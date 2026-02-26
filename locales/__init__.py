"""
Locales package for multi-language support

Provides translation functionality for the entire bot.
"""

from .translator import Translator, get_translator, t

__all__ = ['Translator', 'get_translator', 't']
