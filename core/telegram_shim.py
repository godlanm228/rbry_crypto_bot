"""Lightweight shim to allow running without python-telegram-bot installed.
If the real library is available, re-export from it. Otherwise, provide minimal
stand-ins used by our handlers and analyzers for tests and local runs.
"""
from __future__ import annotations
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import ContextTypes
except Exception:
    class Update:  # very small stub
        def __init__(self):
            self.message = type('Msg', (), {'reply_text': lambda *a, **k: None})()
            self.callback_query = None

    class InlineKeyboardButton:
        def __init__(self, text, callback_data=None, url=None):
            self.text = text
            self.callback_data = callback_data
            self.url = url

    class InlineKeyboardMarkup:
        def __init__(self, keyboard):
            self.keyboard = keyboard

    class ContextTypes:
        DEFAULT_TYPE = object


# Optional Bot stub for environments without python-telegram-bot
try:
    from telegram import Bot  # type: ignore
except Exception:
    class Bot:
        def __init__(self, *args, **kwargs): pass
        async def send_message(self, *args, **kwargs): return None
        async def send_chat_action(self, *args, **kwargs): return None
