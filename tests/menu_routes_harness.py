"""Every button, text reply and command of the bot's menu on fake Telegram objects, in a process of its own.

    python tests/menu_routes_harness.py <repo root> <out.json>

tests/test_menu_routes.py runs it and checks that every route answers. It is also the golden master for a
refactor of the menu: run it on the code before and after, the two files must be equal (12.11.2026: the split of
modules/handlers.py, where it caught an import that no static check saw). Storage is a temporary folder, the
market is a fake snapshot, the network-bound calls are stubbed wherever a module holds their name.
"""
import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

ROOT, OUT = sys.argv[1], sys.argv[2]
work = tempfile.mkdtemp(prefix="rbry_routes_")
env = Path(work) / "dummy.env"
env.write_text("TELEGRAM_BOT_TOKEN=123456:DUMMYDUMMYDUMMYDUMMYDUMMYDUMMY\nTELEGRAM_OWNER_ID=1\n", encoding="utf-8")
os.environ["RBRY_ENV_FILE"] = str(env)
for name in ("ALLOWED_USER_IDS", "RBRY_PAPER_VIEW", "ALERTS_ENABLED", "AI_EXPLANATIONS_ENABLED", "GROQ_API_KEY",
             "HUGGINGFACE_API_KEY"):
    os.environ.pop(name, None)
os.chdir(work)
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)
sys.path.insert(0, ROOT)

import modules.handlers as handlers  # noqa: E402
import core.api.coingecko_client as cg  # noqa: E402
from core.portfolio_db import get_portfolio_manager  # noqa: E402
from modules import user_prefs  # noqa: E402

OWNER, USER = 1, 4242
ROWS = {
    "bitcoin": {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin", "current_price": 80000.0, "market_cap": 1.6e12,
                "price_change_percentage_1h_in_currency": 0.4, "price_change_percentage_24h_in_currency": -2.1,
                "price_change_percentage_7d_in_currency": 5.5, "total_volume": 3e10},
    "ethereum": {"id": "ethereum", "symbol": "eth", "name": "Ethereum", "current_price": 2500.0, "market_cap": 3e11,
                 "price_change_percentage_1h_in_currency": -0.2, "price_change_percentage_24h_in_currency": 12.3,
                 "price_change_percentage_7d_in_currency": -8.0, "total_volume": 1e10},
    "solana": {"id": "solana", "symbol": "sol", "name": "Solana", "current_price": 150.0, "market_cap": 7e10,
               "price_change_percentage_24h_in_currency": 1.0, "price_change_percentage_7d_in_currency": 2.0},
    "tether": {"id": "tether", "symbol": "usdt", "name": "Tether", "current_price": 1.0, "market_cap": 1.5e11},
    "the-open-network": {"id": "the-open-network", "symbol": "ton", "name": "Toncoin", "current_price": 2.0,
                         "market_cap": 5e9},
}


def patch_everywhere(name, value):
    for mod_name, mod in list(sys.modules.items()):
        if mod is not None and mod_name.startswith(("modules", "core.utils")) and hasattr(mod, name):
            setattr(mod, name, value)


async def fake_analysis(coin_id=None, current_price=None, portfolio_entry=None, indicators=None, context=None,
                        *args, user_id=None, show_explanations=None, news_context=None, **kwargs):
    return (f"ANALYSIS {coin_id} price={current_price} held={bool(portfolio_entry)} "
            f"expl={show_explanations} news={news_context}")


news_calls = []


async def fake_render_news(**kwargs):
    news_calls.append({k: (v if isinstance(v, (int, str, list)) else type(v).__name__) for k, v in kwargs.items()
                       if k != "reply_markup"})


async def no_network(*args, **kwargs):
    return {}


async def fake_top(limit=20):
    return list(ROWS.values())


patch_everywhere("analyze_coin_full", fake_analysis)
patch_everywhere("get_significant_news", lambda *a, **k: [])
patch_everywhere("render_news", fake_render_news)
patch_everywhere("get_markets_data", no_network)
patch_everywhere("_track_user_activity", lambda *a, **k: None)
cg.get_top_markets = fake_top
cg.get_markets_data = no_network

snapshot = SimpleNamespace(markets=ROWS, ts=1_790_000_000.0)
service = SimpleNamespace(get_usable=lambda max_age: snapshot, get_latest=lambda: snapshot)
registry = SimpleNamespace(resolve=lambda s: {"coingecko_id": "uniswap"} if s == "UNI" else None)
app = SimpleNamespace(bot_data={"market_snapshot": service, "coin_registry": registry})

pm = get_portfolio_manager()
pm.add_entry(USER, "BTC", 0.5, 50000.0)
pm.add_entry(USER, "ETH", 2.0, 2000.0)
user_prefs.set_timezone(USER, "Europe/Berlin")
user_prefs.set_language(USER, "ru")
user_prefs.set_language(OWNER, "en")           # the owner has no time zone yet: the first-run branch

TIME = re.compile(r"\b\d{2}:\d{2}(:\d{2})?\b")
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}\.\d{2}\.\d{4}\b")


def norm(x):
    if isinstance(x, str):
        return DATE.sub("DATE", TIME.sub("TIME", x))
    return x


def markup(m):
    rows = getattr(m, "inline_keyboard", None)
    if rows is None:
        return None if m is None else type(m).__name__
    return [[(b.text, b.callback_data) for b in row] for row in rows]


def calls(mock, name):
    out = []
    for c in mock.await_args_list:
        kwargs = {k: (markup(v) if k == "reply_markup" else norm(str(v)) if k in ("text", "parse_mode", "title",
                                                                                     "description", "payload") else
                      v if isinstance(v, (int, float, bool, str, type(None))) else type(v).__name__)
                  for k, v in c.kwargs.items()}
        out.append([name, [norm(a) if isinstance(a, str) else type(a).__name__ for a in c.args], kwargs])
    return out


def user(uid):
    return SimpleNamespace(id=uid, username=f"u{uid}", first_name=f"U{uid}", language_code="en")


def context_for():
    bot = SimpleNamespace(send_chat_action=AsyncMock(), send_invoice=AsyncMock(), send_message=AsyncMock())
    return SimpleNamespace(application=app, user_data={}, bot=bot, args=[])


def message_for(uid, text=None):
    return SimpleNamespace(text=text, chat=SimpleNamespace(id=uid), chat_id=uid, reply_text=AsyncMock(),
                           edit_text=AsyncMock(), reply_invoice=AsyncMock(), from_user=user(uid))


counter = [0]
record = []


def run(label, coro_fn, uid, mocks, ctx):
    try:
        asyncio.run(coro_fn())
        error = None
    except Exception as exc:                      # an exception is part of the behaviour to compare
        error = f"{type(exc).__name__}: {exc}"[:300]
    out = []
    for name, mock in mocks:
        out += calls(mock, name)
    for name in ("send_chat_action", "send_invoice", "send_message"):
        out += calls(getattr(ctx.bot, name), f"bot.{name}")
    record.append({"case": label, "user": uid, "error": error, "calls": out,
                   "user_data": json.loads(json.dumps(ctx.user_data, default=str))})


def press(data, uid, ctx=None):
    ctx = ctx or context_for()
    counter[0] += 1
    msg = message_for(uid)
    query = SimpleNamespace(id=f"q{counter[0]}", data=data, from_user=user(uid), message=msg, answer=AsyncMock(),
                            edit_message_text=AsyncMock(), edit_message_reply_markup=AsyncMock())
    update = SimpleNamespace(callback_query=query, effective_user=user(uid), effective_message=msg, message=None,
                             effective_chat=SimpleNamespace(id=uid))
    run(f"press {data}", lambda: handlers.menu_callback(update, ctx), uid,
        [("query.edit_message_text", query.edit_message_text), ("query.edit_message_reply_markup",
                                                                 query.edit_message_reply_markup),
         ("query.answer", query.answer), ("message.edit_text", msg.edit_text), ("message.reply_text", msg.reply_text)],
        ctx)
    return ctx


def say(text, uid, ctx=None):
    ctx = ctx or context_for()
    msg = message_for(uid, text)
    update = SimpleNamespace(effective_user=user(uid), message=msg, effective_message=msg,
                             effective_chat=SimpleNamespace(id=uid), callback_query=None)
    run(f"say {text!r}", lambda: handlers.handle_coin_input(update, ctx), uid,
        [("message.reply_text", msg.reply_text), ("message.edit_text", msg.edit_text)], ctx)
    return ctx


def command(name, uid, text=None):
    ctx = context_for()
    msg = message_for(uid, text or f"/{name}")
    update = SimpleNamespace(effective_user=user(uid), message=msg, effective_message=msg,
                             effective_chat=SimpleNamespace(id=uid), callback_query=None)
    fn = getattr(handlers, f"{name}_command")
    run(f"command /{name}", lambda: fn(update, ctx), uid,
        [("message.reply_text", msg.reply_text), ("message.edit_text", msg.edit_text)], ctx)


BUTTONS = ["portfolio", "paper:all", "alert_done:nope", "alert_remind:nope", "settings", "settings_tz", "settings_lang",
           "settings_explanations", "set_lang:de", "set_lang:ru", "set_tz:Europe/Kyiv", "analyze", "analyze_custom",
           "analyze_coin:BTC", "analyze_coin:SOL", "analyze_portfolio", "analyze_top10", "stats", "movers",
           "add_coin_prompt", "add_pick:BTC", "add_other", "sell_coin_prompt", "back_to_menu", "news", "help", "my_tier",
           "premium", "buy_plan:premium_1m", "buy_plan:beta_lifetime", "buy_plan:nope", "admin_panel", "admin_stats",
           "admin_grant_help", "admin_users_list", "something_unknown"]

for uid in (USER, OWNER):
    for name in ("start", "help", "settings", "analyze", "stats", "movers", "portfolio", "unknown"):
        command(name, uid)
    for data in BUTTONS:
        press(data, uid)
    # the add-coin dialogue: a button, then the reply
    ctx = press("add_pick:ETH", uid)
    say("one two three", uid, ctx)
    say("0,25", uid, ctx)
    ctx = press("add_other", uid)
    say("ton 10 1,4", uid, ctx)
    # typed coin after "choose a coin", a bare ticker, chatter
    ctx = press("analyze_custom", uid)
    say("btc", uid, ctx)
    ctx = press("analyze_custom", uid)
    say("!!", uid, ctx)
    say("uni", uid)
    say("привет как дела", uid)
    say("XYZQ", uid)
    for data in ("portfolio", "stats", "movers", "analyze_portfolio", "sell_coin_prompt", "admin_panel",
                 "admin_users_list"):
        press(data, uid)

Path(OUT).write_text(json.dumps({"cases": record, "news": news_calls}, ensure_ascii=False, indent=1), encoding="utf-8")
errors = [c for c in record if c["error"]]
print(f"{len(record)} cases, {sum(len(c['calls']) for c in record)} calls, {len(errors)} with an exception")
for c in errors:
    print("  ", c["user"], c["case"], "->", c["error"])
