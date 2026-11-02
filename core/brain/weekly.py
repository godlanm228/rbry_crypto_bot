"""The brain's week and month: grade, refit, swap only when the new one passes.

Files in runtime/learning/brain/:
  champion.pkl       the model the bot uses
  champion-prev.pkl  the one before it, for a rollback
  record.pkl         every out-of-sample score with its outcome: the exam's
                     months first, then each week's champion on the coin-days
                     that finished since. The brain's report card, the source
                     of its calibration and the monthly audit's data.

Each week (``week``):
1. grade: the champion scores the days that finished since it was trained
   and that are not in the record yet; they join the record;
2. challenger: the same recipe refit on everything graded so far;
3. before the swap, two checks. The recipe must still work on recent data: a
   probe fit without the last PROBE_WEEKS graded weeks has to rank those
   weeks better than chance (best tenth ahead of the worst tenth). And the
   challenger must broadly agree with the champion about today's coins (rank
   correlation of at least MIN_AGREEMENT). Otherwise the champion stays and
   the owner is told: the market may have changed;
4. calibration: the last CALIBRATION_MONTHS of the record.

Each month (``audit``): the last finished month against the one before, from
the record, in plain words.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from core.brain.dataset import MIN_TURNOVER, RANKING_FEATURES
from core.brain.model import Brain, calibrations, day_place, train
from core.brain.walkforward import CALIBRATION_MONTHS, GAP_DAYS, KEYS, eligible
from core.sandbox.evaluate import t_cluster

logger = logging.getLogger(__name__)

PROBE_WEEKS = 8
MIN_AGREEMENT = 0.7
TOP = 0.9           # place above this: the day's best tenth
BOTTOM = 0.1        # place at or below this: the day's worst tenth
RECORD_COLUMNS = KEYS + ["place", "excess_14d", "hold_ret", "r_net", "model"]


def graded(rows: pd.DataFrame) -> np.ndarray:
    """Which rows can be learned from: tradable, and their 14 days are over."""
    return ((rows["turnover_30d"] >= MIN_TURNOVER) & rows["excess_14d"].notna()).to_numpy()


def spread(frame: pd.DataFrame) -> dict:
    """Best tenth of each day against its worst tenth: $ per $100 over 14 days, with t by week."""
    frame = frame[frame["excess_14d"].notna()]
    if frame.empty:
        return {"days": 0, "spread_usd": float("nan"), "t": float("nan")}
    top = frame[frame["place"] > TOP].groupby("day")["excess_14d"].mean()
    bottom = frame[frame["place"] <= BOTTOM].groupby("day")["excess_14d"].mean()
    diff = (top - bottom).dropna()
    weeks = diff.index.to_period("W").astype(str).to_numpy()
    return {"days": len(diff), "spread_usd": 100 * float(diff.mean()) if len(diff) else float("nan"),
            "t": t_cluster(diff.to_numpy(), weeks) if len(diff) > 2 else float("nan")}


def score_days(brain: Brain, rows: pd.DataFrame) -> pd.DataFrame:
    """Scores and places for whole days of eligible coins (a place is relative to the day)."""
    pool = eligible(rows)
    out = pool[KEYS + ["excess_14d", "hold_ret", "r_net"]].copy()
    out["score"] = brain.score(pool)
    out["place"] = day_place(out, out["score"]).to_numpy()
    return out


def grade_champion(champion: Brain, rows: pd.DataFrame, record: pd.DataFrame, model_id: str) -> pd.DataFrame:
    """New record rows: finished coin-days the champion never trained on and the record lacks."""
    after = pd.Timestamp(champion.trained_until)
    done = record["day"].max() if not record.empty else pd.Timestamp.min
    days = rows[(rows["day"] > max(after, done))]
    if days.empty:
        return record.iloc[0:0]
    scored = score_days(champion, days)
    scored = scored[scored["excess_14d"].notna()].assign(model=model_id)
    return scored[RECORD_COLUMNS]


def probe(rows: pd.DataFrame, features: tuple[str, ...] = RANKING_FEATURES, weeks: int = PROBE_WEEKS) -> dict:
    """Does the recipe still rank recent weeks it did not learn from?"""
    ok = graded(rows)
    days = rows["day"]
    newest = days[ok].max()
    start = newest - pd.Timedelta(weeks=weeks)
    brain = train(rows, features, ok & (days <= start - pd.Timedelta(days=GAP_DAYS)).to_numpy())
    test = rows[(rows["day"] > start) & (rows["day"] <= newest)]
    return {"from": str((start + pd.Timedelta(days=1)).date()), "to": str(newest.date()),
            **spread(score_days(brain, test))}


def agreement(a: Brain, b: Brain, rows: pd.DataFrame) -> float:
    """Rank correlation of two brains on the newest day's coins."""
    pool = eligible(rows)
    today = pool[pool["day"] == pool["day"].max()]
    if len(today) < 10:
        return float("nan")
    return float(pd.Series(a.score(today)).rank().corr(pd.Series(b.score(today)).rank()))


def calibrate_from(record: pd.DataFrame, brain: Brain, months: int = CALIBRATION_MONTHS) -> Brain:
    recent = record[record["day"] > record["day"].max() - pd.DateOffset(months=months)]
    recent = recent[recent["r_net"].notna()]
    brain.calibration = calibrations(recent["place"].to_numpy(), recent)
    brain.meta["calibrated_on"] = f"{recent['day'].min().date()} … {recent['day'].max().date()}, {len(recent):,} rows"
    return brain


@dataclass
class Week:
    champion: Brain
    record: pd.DataFrame
    promoted: bool
    added: int
    probe: dict
    agreement: float
    notes: list[str] = field(default_factory=list)


def week(rows: pd.DataFrame, champion: Optional[Brain], record: pd.DataFrame, champion_id: str,
         new_id: str, features: tuple[str, ...] = RANKING_FEATURES) -> Week:
    added = grade_champion(champion, rows, record, champion_id) if champion else record.iloc[0:0]
    if len(added):
        record = pd.concat([record, added], ignore_index=True) if len(record) else added.reset_index(drop=True)
    check = probe(rows, features)
    challenger = train(rows, features, graded(rows))
    agree = agreement(champion, challenger, rows) if champion else float("nan")
    notes = []
    if not (math.isfinite(check["spread_usd"]) and check["spread_usd"] > 0):
        notes.append(f"на последних {PROBE_WEEKS} неделях мозг не отличил лучшие монеты от худших")
    if champion and not (math.isfinite(agree) and agree >= MIN_AGREEMENT):
        notes.append(f"новая версия слишком расходится со старой (совпадение {agree:.2f})")
    promoted = not notes or champion is None
    winner = calibrate_from(record, challenger if promoted else champion)
    winner.meta["model_id"] = new_id if promoted else champion_id
    return Week(winner, record, promoted, len(added), check, agree, notes)


# the monthly audit

def month_numbers(record: pd.DataFrame, month: pd.Period) -> dict:
    m = record[record["day"].dt.to_period("M") == month]
    if m.empty:
        return {"rows": 0}
    top, bottom = m[m["place"] > TOP], m[m["place"] <= BOTTOM]
    return {
        "rows": len(m), "days": m["day"].nunique(),
        "top_beat_btc_of10": 10 * float((top["excess_14d"] > 0).mean()),
        "all_beat_btc_of10": 10 * float((m["excess_14d"] > 0).mean()),
        "top_hold_usd": 100 * float(top["hold_ret"].mean()), "all_hold_usd": 100 * float(m["hold_ret"].mean()),
        "bottom_hold_usd": 100 * float(bottom["hold_ret"].mean()),
        **{f"spread_{k}": v for k, v in spread(m).items()},
    }


def audit_text(record: pd.DataFrame, month: pd.Period, champion: Optional[Brain] = None) -> str:
    """The month in plain words: did the brain's best tenth do better, and better than last month?"""
    now, before = month_numbers(record, month), month_numbers(record, month - 1)
    lines = [f"🧠 Аудит мозга за {month.strftime('%m.%Y')}"]
    if not now["rows"]:
        return "\n".join(lines + ["Нет закрытых дней за этот месяц: оценка придёт позже."])

    def usd(x: float) -> str:
        return "—" if not math.isfinite(x) else f"{'+' if x >= 0 else '−'}${abs(x):.2f}"

    lines += [
        f"Покупка на $100 и 2 недели держать, в среднем по монете-дню ({now['days']} дн.):",
        f"• лучшие 10% по мозгу: {usd(now['top_hold_usd'])}, обогнали BTC {now['top_beat_btc_of10']:.1f} из 10",
        f"• все монеты: {usd(now['all_hold_usd'])}, обогнали BTC {now['all_beat_btc_of10']:.1f} из 10",
        f"• худшие 10%: {usd(now['bottom_hold_usd'])}",
        f"Лучшие против худших: {usd(now['spread_spread_usd'])} на $100"
        + (f" (месяц назад {usd(before['spread_spread_usd'])})" if before["rows"] else ""),
    ]
    if before["rows"] and math.isfinite(now["spread_spread_usd"]) and math.isfinite(before["spread_spread_usd"]):
        better = now["spread_spread_usd"] > before["spread_spread_usd"]
        lines.append("Мозг отличал монеты " + ("лучше" if better else "хуже") + ", чем месяцем раньше.")
    if now["days"] < 20 or not (math.isfinite(now["spread_t"]) and abs(now["spread_t"]) >= 2):
        lines.append("Один месяц — мало: разница может быть случайной.")
    if champion is not None:
        lines.append(f"Модель {champion.meta.get('model_id', '?')}, учена по {champion.trained_until}.")
    return "\n".join(lines)
