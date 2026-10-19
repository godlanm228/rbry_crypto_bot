"""News are shown in the user's language with only the source name linked."""

import modules.news as news


def _items():
    return [{"news": {"title": "Bitcoin ETF outflows reach $65M",
                      "summary": "<p>BlackRock fund saw outflows on Monday.</p>",
                      "link": "https://decrypt.co/1/etf-outflows"},
             "analysis": None}]


def test_translation_is_used_and_cached(monkeypatch):
    calls = []

    def fake_groq(system, user, **kwargs):
        calls.append(user)
        return {"items": [{"id": 0, "title": "Отток из Bitcoin ETF достиг $65 млн",
                           "summary": "Фонд BlackRock зафиксировал отток в понедельник."}]}

    monkeypatch.setattr(news, "groq_json", fake_groq)
    monkeypatch.setattr(news.config, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(news, "_translation_cache", {})

    items = news.localize_news(_items(), "ru")
    text = news._format_one(items[0])
    assert "<b>Отток из Bitcoin ETF достиг $65 млн</b>" in text
    assert "Фонд BlackRock" in text
    assert '<a href="https://decrypt.co/1/etf-outflows">Decrypt</a>' in text
    assert text.count("<a ") == 1  # only the source name is a link

    news.localize_news(_items(), "ru")
    assert len(calls) == 1  # second time from cache


def test_without_llm_the_original_is_shown(monkeypatch):
    monkeypatch.setattr(news, "groq_json", lambda *a, **k: None)
    monkeypatch.setattr(news.config, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(news, "_translation_cache", {})
    items = news.localize_news(_items(), "de")
    text = news._format_one(items[0])
    assert "Bitcoin ETF outflows reach $65M" in text
    assert "BlackRock fund saw outflows on Monday." in text


def test_english_users_get_no_translation_call(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not call the LLM for English")

    monkeypatch.setattr(news, "groq_json", boom)
    news.localize_news(_items(), "en")
