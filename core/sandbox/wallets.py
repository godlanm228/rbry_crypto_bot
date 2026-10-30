"""Our own eye on exchanges' Ethereum hot wallets (plan item 7, 30.10.2026): money leaving faster than ever.

Whale Alert and the sleuths' channels tell what they see; this reads the wallets itself through
Etherscan (ETHERSCAN_API_KEY of .env.test, the free plan): every few minutes the transfers of ETH,
USDT and USDC to or from each exchange's known hot wallets since the last look, in dollars. Money
moving between an exchange's own wallets does not count. An hour's net outflow bigger than anything
of the week before by half, and above a floor, is a signal for the exchange-risk situation
(core/sandbox/exchange_risk.py, as "wallets"); the first two days only learn what a week looks like.
"""

from __future__ import annotations

import time
from typing import Optional

HOUR_MS = 3_600_000
API = "https://api.etherscan.io/v2/api"
EVERY_S = 300
BLOCKS_BACK = 300                    # a first look starts an hour back (12-second blocks)
WINDOW_MS = HOUR_MS
KEEP_DAYS = 8
LEARN_HOURS = 48
FLOOR_USD = 100e6
OVER_WEEK = 1.5
PAGE = 1000
PAUSE_S = 0.25                       # the free plan takes 5 calls a second
# wallets as Etherscan labels them (Binance 14, 15, 16 hot, Binance 8 its big one); extend as exchanges matter
WALLETS = {"Binance": ("0x28c6c06298d514db089934071355e5743bf21d60", "0x21a31ee1afc51d94c2efccaa2092ad1028285549",
                       "0xdfd5293d8e347dfe59e90efd55b2956a1343963d", "0xf977814e90da44bfa03b6295a0616a897441acec")}
TOKENS = {"USDT": ("0xdac17f958d2ee523a2206206994597c13d831ec7", 6),
          "USDC": ("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", 6)}


def _call(client, key: str, params: dict) -> Optional[list]:
    try:
        resp = client.get(API, params={"chainid": 1, "apikey": key, **params}, timeout=20)
        data = resp.json()
    except Exception:                                  # a network or a JSON error: next round
        return None
    result = data.get("result")
    return result if isinstance(result, list) else None


def block_now(client, key: str) -> Optional[int]:
    try:
        resp = client.get(API, params={"chainid": 1, "module": "proxy", "action": "eth_blockNumber", "apikey": key},
                          timeout=20)
        return int(resp.json()["result"], 16)
    except Exception:
        return None


def flows(rows: list[dict], wallets: set[str], decimals: int, price: float) -> list[tuple[int, float, float]]:
    """(ms, out $, in $) of every transfer that leaves or enters the wallets from outside them."""
    out = []
    for r in rows:
        src, dst = r.get("from", "").lower(), r.get("to", "").lower()
        if (src in wallets) == (dst in wallets):         # between its own wallets, or not its at all
            continue
        usd = int(r.get("value", "0")) / 10 ** decimals * price
        ts = int(r.get("timeStamp", "0")) * 1000
        out.append((ts, usd, 0.0) if src in wallets else (ts, 0.0, usd))
    return out


def alarm(recent: list, hours: dict, now_ms: int, since_ms: int) -> Optional[tuple[float, float]]:
    """(net outflow of the last hour, the week's biggest hour) when it is a signal; else None. A quiet hour
    is an hour of nothing; the first LEARN_HOURS after the first look only learn."""
    if now_ms - since_ms < LEARN_HOURS * HOUR_MS:
        return None
    net = sum(o - i for ts, o, i in recent if ts >= now_ms - WINDOW_MS)
    this_hour = now_ms // HOUR_MS * HOUR_MS
    past = [o - i for h, (o, i) in hours.items() if int(h) < this_hour - HOUR_MS]
    biggest = max(past + [0.0])
    return (net, biggest) if net >= FLOOR_USD and net >= OVER_WEEK * biggest else None


class WalletWatch:
    """The wallets' flows kept in ``state`` (urgent_watch's state file): cursors, hours, the last two hours."""

    def __init__(self, key: str, state: dict):
        self.key, self.state = key, state
        state.setdefault("cursor", {})
        state.setdefault("hours", {})
        state.setdefault("recent", {})
        state.setdefault("at", 0.0)
        state.setdefault("since", 0)

    def round(self, client, now_ms: int, eth_price: float) -> dict:
        """{exchange: (since ms, what is seen)} when a signal; reads Etherscan at most every EVERY_S."""
        if time.time() - self.state["at"] < EVERY_S or not eth_price:
            return {}
        self.state["at"] = time.time()
        self.state["since"] = self.state["since"] or now_ms
        found, now_block = {}, None
        for exchange, wallets in WALLETS.items():
            own = set(wallets)
            new = []
            for wallet in wallets:
                for kind, (contract, decimals) in [("ETH", (None, 18)), *TOKENS.items()]:
                    cursor_key = f"{wallet}:{kind}"
                    start = self.state["cursor"].get(cursor_key)
                    if start is None:
                        now_block = now_block or block_now(client, self.key)
                        if now_block is None:
                            continue
                        start = now_block - BLOCKS_BACK
                    params = {"module": "account", "address": wallet, "startblock": start + 1, "sort": "asc",
                              "page": 1, "offset": PAGE}
                    params.update({"action": "txlist"} if contract is None
                                  else {"action": "tokentx", "contractaddress": contract})
                    time.sleep(PAUSE_S)
                    rows = _call(client, self.key, params)
                    if rows is None:
                        continue
                    if rows:
                        self.state["cursor"][cursor_key] = max(int(r["blockNumber"]) for r in rows)
                    else:
                        self.state["cursor"][cursor_key] = start
                    new += flows(rows, own, decimals, eth_price if contract is None else 1.0)
            hours = self.state["hours"].setdefault(exchange, {})
            for ts, o, i in new:
                h = str(ts // HOUR_MS * HOUR_MS)
                prev = hours.get(h, [0.0, 0.0])
                hours[h] = [prev[0] + o, prev[1] + i]
            cutoff = now_ms - KEEP_DAYS * 24 * HOUR_MS
            for h in [h for h in hours if int(h) < cutoff]:
                del hours[h]
            recent = [x for x in self.state["recent"].get(exchange, []) + [list(x) for x in new]
                      if x[0] >= now_ms - 2 * HOUR_MS]
            self.state["recent"][exchange] = recent
            hit = alarm(recent, hours, now_ms, self.state["since"])
            if hit:
                net, biggest = hit
                found[exchange] = (now_ms, f"с горячих кошельков {exchange} за час ушло ${net / 1e6:,.0f}M в ETH/USDT/"
                                           f"USDC — в {net / max(biggest, 1):.1f} раза больше худшего часа недели "
                                           f"(${biggest / 1e6:,.0f}M), Etherscan")
        return found
