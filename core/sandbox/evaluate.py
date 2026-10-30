"""Does any rule, or any "similar situations" memory, have an edge?

Time is split once and for all:
- DEV     signals up to 2024-12-31 — rules and the memory learn here;
- VALID   2025 — first out-of-sample check, looked at freely;
- HOLDOUT 2026 — untouched; opened only through ``holdout_report`` which
  writes every opening to learning.db (the count goes into every report).

A rule is compared with its random-day twins: the same trade shape entered on
random days of the same coin and month (see replay.py). 2022–2026 crypto
drifted a lot, and a rule that "wins" in a bull market may only be riding it;
the twins ride the same market, so the difference is the rule's timing.
Standard errors are clustered by week: twenty coins on the same week are one
bet, not twenty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from core.sandbox.features import FEATURES

DEV_END = "2024-12-31"
VALID_END = "2025-12-31"
MIN_TRADES = 30
EDGE_T = 2.0


def period_of(days: pd.Series) -> pd.Series:
    return pd.Series(np.where(days <= DEV_END, "dev", np.where(days <= VALID_END, "valid", "holdout")),
                     index=days.index)


# --- statistics --------------------------------------------------------------

def t_cluster(values: np.ndarray, clusters: np.ndarray) -> float:
    """t of the mean with a cluster-robust (CR0) standard error."""
    n = len(values)
    if n < 2:
        return float("nan")
    e = values - values.mean()
    sums = pd.Series(e).groupby(np.asarray(clusters)).sum().to_numpy()
    g = len(sums)
    if g < 2:
        return float("nan")
    se = math.sqrt(float(np.sum(sums ** 2)) * g / (g - 1)) / n
    return float(values.mean() / se) if se > 0 else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def add_context(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()
    t["period"] = period_of(t["exit_day"])        # by when the result is known: no 2025 trade closed in 2026
    days = pd.to_datetime(t["signal_day"])
    t["week"] = days.dt.strftime("%G-W%V")
    t["month"] = days.dt.strftime("%Y-%m")
    t["year"] = days.dt.year
    # timing edge: the result minus its random-day twins (same coin, month and
    # trade shape); the control rules are random already and have none
    t["excess_r"] = t["r_net"] - t["twin_r"] if "twin_r" in t else np.nan
    return t


def describe(g: pd.DataFrame) -> dict:
    r = g["r_net"].to_numpy()
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    ex = g["excess_r"].dropna()
    return {
        "n": len(g), "coins": g["symbol"].nunique(),
        "win_rate": float((r > 0).mean()) if len(r) else float("nan"),
        "target_rate": float(g["reason"].isin(["target", "target_gap"]).mean()) if len(g) else float("nan"),
        "avg_r": float(r.mean()) if len(r) else float("nan"),
        "median_r": float(np.median(r)) if len(r) else float("nan"),
        "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
        "t": t_cluster(r, g["week"].to_numpy()),
        "avg_twin_r": float(g["twin_r"].mean()) if "twin_r" in g and g["twin_r"].notna().any() else float("nan"),
        "avg_excess_r": float(ex.mean()) if len(ex) else float("nan"),
        "t_excess": t_cluster(ex.to_numpy(), g.loc[ex.index, "week"].to_numpy()) if len(ex) else float("nan"),
        "avg_bars": float(g["bars"].mean()) if len(g) else float("nan"),
    }


def verdict(dev: dict, valid: dict) -> str:
    if dev["n"] < MIN_TRADES or valid["n"] < MIN_TRADES:
        return "мало сделок"
    excess_ok = all(math.isfinite(x["t_excess"]) and x["t_excess"] >= EDGE_T and x["avg_excess_r"] > 0
                    for x in (dev, valid))
    if excess_ok and dev["avg_r"] > 0 and valid["avg_r"] > 0:
        return "КАНДИДАТ: лучше случайного и в плюсе в обоих периодах"
    if excess_ok:
        return "лучше случайного, но после комиссий в минусе"
    if dev["avg_r"] > 0 and valid["avg_r"] > 0:
        return "в плюсе, но не лучше случайного входа (едет на рынке)"
    return "перевеса нет"


def strategy_table(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    clean = t[t["overlap"] == 0]
    for strategy, g in clean.groupby("strategy", sort=False):
        dev, valid = describe(g[g["period"] == "dev"]), describe(g[g["period"] == "valid"])
        years = g[g["period"] != "holdout"].groupby("year")["r_net"].mean().round(3).to_dict()
        rows.append({"strategy": strategy,
                     **{f"dev_{k}": v for k, v in dev.items()}, **{f"valid_{k}": v for k, v in valid.items()},
                     "by_year": years, "verdict": verdict(dev, valid)})
    return pd.DataFrame(rows)


# --- similar situations ------------------------------------------------------

@dataclass
class Scaler:
    centre: np.ndarray
    scale: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.centre) / self.scale
        z = np.clip(z, -4.0, 4.0)
        return np.nan_to_num(z, nan=0.0).astype(np.float32)


def fit_scaler(x: np.ndarray) -> Scaler:
    centre = np.nanmedian(x, axis=0)
    q75, q25 = np.nanpercentile(x, 75, axis=0), np.nanpercentile(x, 25, axis=0)
    scale = np.where((q75 - q25) > 1e-12, q75 - q25, 1.0)
    return Scaler(np.nan_to_num(centre), scale)


def knn_mean(memory: np.ndarray, values: np.ndarray, queries: np.ndarray, k: int,
             chunk: int = 256) -> np.ndarray:
    """Mean of ``values`` over the k nearest memory rows of every query."""
    k = min(k, len(memory))
    m_sq = (memory ** 2).sum(axis=1)
    out = np.empty(len(queries))
    for start in range(0, len(queries), chunk):
        q = queries[start:start + chunk]
        d = (q ** 2).sum(axis=1)[:, None] + m_sq[None, :] - 2.0 * q @ memory.T
        nearest = np.argpartition(d, k - 1, axis=1)[:, :k]
        out[start:start + chunk] = values[nearest].mean(axis=1)
    return out


def similar_situations_test(t: pd.DataFrame, situations: pd.DataFrame, k: int = 100) -> pd.DataFrame:
    """Memory = DEV trades; does its forecast rank VALID trades correctly?"""
    feats = list(FEATURES)
    sit = situations[["symbol", "day_str", *feats]].rename(columns={"day_str": "signal_day"})
    joined = t.merge(sit, on=["symbol", "signal_day"], how="inner")
    dev_sit = situations[situations["day_str"] <= DEV_END][feats].to_numpy(dtype=float)
    scaler = fit_scaler(dev_sit)
    rows = []
    for strategy, g in joined.groupby("strategy", sort=False):
        # target "r_net": the raw result, market drift included;
        # target "excess_r": the result over random-day twins (timing only)
        for target in ("r_net", "excess_r"):
            known = g[g[target].notna()]
            memory = known[(known["period"] == "dev") & (known["exit_day"] <= DEV_END)]
            query = known[(known["period"] == "valid") & (known["overlap"] == 0)]
            if len(memory) < 300 or len(query) < 100:
                continue
            kk = min(k, max(20, len(memory) // 20))
            pred = knn_mean(scaler(memory[feats].to_numpy(dtype=float)), memory[target].to_numpy(),
                            scaler(query[feats].to_numpy(dtype=float)), kk)
            real = query[target].to_numpy()
            quint = pd.qcut(pd.Series(pred).rank(method="first"), 5, labels=False).to_numpy()
            by_q = [float(real[quint == q].mean()) for q in range(5)]
            top = query[quint == 4]
            rho = spearman(pred, real)
            n = len(real)
            rows.append({
                "strategy": strategy, "target": target, "memory": len(memory), "k": kk, "n_valid": n,
                "spearman": rho,
                # naive t (independent trades); the clustered top-20% t is the stricter one
                "t_rho": rho * math.sqrt((n - 2) / max(1e-9, 1 - rho ** 2)) if math.isfinite(rho) else float("nan"),
                "avg_all": float(real.mean()), "avg_top20": by_q[4], "avg_bottom20": by_q[0],
                "quintiles": [round(x, 3) for x in by_q],
                "t_top20": t_cluster(top[target].to_numpy(), top["week"].to_numpy()),
                "top20_trades": len(top),
            })
    return pd.DataFrame(rows)


# --- situation types (clusters) ---------------------------------------------

def kmeans(x: np.ndarray, k: int = 8, iters: int = 60, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Plain k-means with k-means++ seeding; (centres, labels)."""
    rng = np.random.default_rng(seed)
    centres = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d = np.min(((x[:, None, :] - np.array(centres)[None]) ** 2).sum(axis=2), axis=1)
        centres.append(x[rng.choice(len(x), p=d / d.sum())])
    c = np.array(centres)
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        d = ((x[:, None, :] - c[None]) ** 2).sum(axis=2)
        new = d.argmin(axis=1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for j in range(k):
            if np.any(labels == j):
                c[j] = x[labels == j].mean(axis=0)
    return c, labels


def assign(x: np.ndarray, centres: np.ndarray) -> np.ndarray:
    return ((x[:, None, :] - centres[None]) ** 2).sum(axis=2).argmin(axis=1)


def cluster_table(t: pd.DataFrame, situations: pd.DataFrame, k: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = list(FEATURES)
    dev = situations[situations["day_str"] <= DEV_END]
    scaler = fit_scaler(dev[feats].to_numpy(dtype=float))
    sample = dev.sample(min(len(dev), 20000), random_state=7)
    centres, _ = kmeans(scaler(sample[feats].to_numpy(dtype=float)), k=k)
    labelled = situations[["symbol", "day_str"]].copy()
    labelled["cluster"] = assign(scaler(situations[feats].to_numpy(dtype=float)), centres)
    profile = pd.DataFrame(centres, columns=feats)
    profile["share_dev"] = pd.Series(assign(scaler(dev[feats].to_numpy(dtype=float)), centres)).value_counts(
        normalize=True).sort_index()
    joined = t[t["overlap"] == 0].merge(labelled.rename(columns={"day_str": "signal_day"}),
                                        on=["symbol", "signal_day"], how="inner")
    cells = (joined[joined["period"] != "holdout"]
             .groupby(["strategy", "cluster", "period"])["r_net"].agg(["count", "mean"]).unstack("period"))
    cells.columns = [f"{p}_{m}" for m, p in cells.columns]
    return profile, cells.reset_index()


def memory_filter(t: pd.DataFrame, situations: pd.DataFrame, strategy: str, target: str = "excess_r",
                  share: float = 0.2, k: int = 100) -> pd.DataFrame:
    """HOLDOUT trades of ``strategy`` the "similar situations" memory rates highest.

    Real-time honest: the cut-off is the top-``share`` prediction level seen in
    2025 (memory = up to 2024); 2026 trades are predicted from memory up to
    2025 and kept when they clear that fixed level.
    """
    feats = list(FEATURES)
    sit = situations[["symbol", "day_str", *feats]].rename(columns={"day_str": "signal_day"})
    g = t[(t["strategy"] == strategy) & t[target].notna()].merge(sit, on=["symbol", "signal_day"], how="inner")
    scaler = fit_scaler(situations[situations["day_str"] <= DEV_END][feats].to_numpy(dtype=float))

    def predict(memory, query):
        return knn_mean(scaler(memory[feats].to_numpy(dtype=float)), memory[target].to_numpy(),
                        scaler(query[feats].to_numpy(dtype=float)), min(k, len(memory)))

    valid = g[(g["period"] == "valid") & (g["overlap"] == 0)]
    level = float(np.quantile(predict(g[(g["period"] == "dev") & (g["exit_day"] <= DEV_END)], valid), 1 - share))
    hold = g[(g["period"] == "holdout") & (g["overlap"] == 0)].copy()
    hold["predicted"] = predict(g[g["exit_day"] <= VALID_END], hold)
    picked = hold[hold["predicted"] >= level].copy()
    picked["strategy"] = f"{strategy}+memory"
    return picked


def outcome_mix(g: pd.DataFrame) -> dict:
    """What ten such trades looked like: targets, stops, time exits, total R."""
    n = len(g)
    if not n:
        return {}
    return {"of10_target": 10 * g["reason"].isin(["target", "target_gap"]).mean(),
            "of10_stop": 10 * g["reason"].isin(["stop", "stop_gap"]).mean(),
            "of10_time": 10 * (g["reason"] == "time").mean(),
            "r_per_10": 10 * g["r_net"].mean()}


def holdout_report(t: pd.DataFrame, strategies: list[str], store, run_id: str,
                   extra: Optional[list[pd.DataFrame]] = None) -> pd.DataFrame:
    """The one look at 2026 for the listed rules; every call is logged."""
    names = strategies + [x["strategy"].iloc[0] for x in (extra or []) if not x.empty]
    opened = store.log_holdout(run_id, ",".join(names))
    groups = [t[(t["strategy"] == s) & (t["overlap"] == 0) & (t["period"] == "holdout")] for s in strategies]
    rows = []
    for name, g in zip(names, groups + [x for x in (extra or []) if not x.empty]):
        rows.append({"strategy": name, **describe(g), **outcome_mix(g), "holdout_opened_total": opened})
    return pd.DataFrame(rows)
