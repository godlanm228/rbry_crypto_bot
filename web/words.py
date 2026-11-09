"""What the models' factors and the market's measures mean, in words the owner reads (09.11: "↑ atr_pct, age,
liquidity" said nothing). C1's factors as core/brain/evaluator.WORDS has them (the site does not import the
trading code), each with what its value means."""

from __future__ import annotations

# C1 (the daily brain): factor -> (what it is, how the value reads)
C1 = {
    "ret_1d": ("движение за день", "pct"), "ret_7d": ("движение за неделю", "pct"),
    "ret_30d": ("движение за месяц", "pct"), "ret_90d": ("движение за 3 месяца", "pct"),
    "ret_365d": ("движение за год", "pct"), "atr_pct": ("обычный дневной размах цены", "pct"),
    "vol_regime": ("размах сейчас против обычного", "x"), "dist_sma20": ("отрыв от 20-дневной средней, в дневных размахах", "num"),
    "dist_sma50": ("отрыв от 50-дневной средней, в дневных размахах", "num"), "dist_sma200": ("отрыв от 200-дневной средней, в дневных размахах", "num"),
    "sma50_slope": ("наклон 50-дневной средней", "pct"), "rsi14": ("RSI (перекупленность)", "num"),
    "range_pos_30": ("место в месячном диапазоне (0 — дно, 1 — верх)", "num"),
    "dd_90": ("падение от пика за 3 месяца", "pct"), "dd_365": ("падение от пика за год", "pct"),
    "volume_ratio": ("объём против обычного (0 — как обычно)", "num"), "support_dist": ("до поддержки, в дневных размахах", "num"),
    "resistance_dist": ("до сопротивления, в дневных размахах", "num"), "rs_30d": ("сила против BTC за месяц", "pct"),
    "day_range": ("размах свечи дня", "pct"), "close_pos": ("где закрылся день в своей свече", "num"),
    "liquidity": ("оборот в день", "usd"), "taker_buy_7d": ("доля покупок по рынку за неделю", "share"),
    "trades_ratio": ("число сделок против обычного", "x"), "age": ("возраст монеты на бирже", "days"),
    "funding_1d": ("фандинг за день", "pct"), "funding_7d": ("фандинг за неделю", "pct"),
    "perp_listed": ("есть фьючерс", "bool"), "delist_notice": ("объявлен делистинг", "bool"),
    "rs_rank": ("место по силе против BTC среди монет", "rank"),
    "ret_rank_7d": ("место по росту за неделю среди монет", "rank"),
    "liquidity_rank": ("место по обороту среди монет", "rank"), "place": ("место у мозга", "rank"),
}

EXIT = {"time": "срок вышел", "stop": "стоп", "target": "цель", "keep": "трейлинг-стоп", "liquidated": "ликвидация",
        "settled": "расчёт при делистинге", "open": "открыта", "ended": "данные кончились"}


def c1_factor(name: str) -> str:
    return C1.get(name, (name, ""))[0]


def exit_word(reason: str) -> str:
    if reason.startswith("event:"):
        return "выход по событию " + reason.split(":", 1)[1]
    return EXIT.get(reason, reason)


# the market's three measures (core/brain/regime.states_at), in words with their numbers
def breadth(share) -> str:
    if not isinstance(share, (int, float)):
        return "—"
    return f"{share * 100:.0f}% монет выше своей недельной средней — " + ("рынок широкий: растёт большинство"
                                                                         if share >= 0.5 else
                                                                         "рынок узкий: растёт меньшинство")


def alts(share) -> str:
    if not isinstance(share, (int, float)):
        return "—"
    return f"{share * 100:.0f}% монет обогнали BTC за 30 дней — " + ("сезон альтов" if share >= 0.5 else "сезон BTC")


def funding(level) -> str:
    if not isinstance(level, (int, float)):
        return "—"
    word = ("горячий: толпа платит за лонги" if level >= 0.0002 else
            "холодный: платят шортисты" if level < 0 else "обычный")
    return f"{level * 100:+.4f}% за выплату — {word}"


def c1_value(name: str, value) -> str:
    """A C1 factor's value as its kind reads."""
    if not isinstance(value, (int, float)) or value != value:
        return "—"
    kind = C1.get(name, ("", "num"))[1]
    if kind == "pct":
        return f"{value * 100:+.2f}%".replace("-", "−")
    if kind == "share":
        return f"{value * 100:.0f}%"
    if kind == "x":
        return f"{value:.2f}×"
    if kind == "usd":
        return f"${value:,.0f}".replace(",", " ")
    if kind == "days":
        return f"{value:.0f} дн."
    if kind == "bool":
        return "да" if value else "нет"
    return f"{value:.2f}"
