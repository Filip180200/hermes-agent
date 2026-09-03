"""Dashboard portfolio card.

Live price fetch: Yahoo Finance chart API (stocks, free, no key — Stooq's
``/q/l/`` CSV endpoint was retired and its replacement sits behind a JS
proof-of-work challenge a server-side request can't solve, see the "Hermes
Dashboard" Obsidian note entry for 2026-09-03) + CoinGecko (crypto, free, no
key). Results are cached in-process for ``_CACHE_TTL_S`` so a page reload
doesn't hammer either API. Tracked positions come from
``dashboard_config/portfolio.json`` — see the "Dashboard — sekcje 1 i 3" note
in the Obsidian vault for how to edit that list (no restart needed, the
config is read fresh on cache expiry).
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter

_log = logging.getLogger("hermes_cli.web_server")

router = APIRouter()

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "dashboard_config" / "portfolio.json"
_CACHE_TTL_S = 60.0
_cache: Optional[tuple] = None  # (monotonic_ts, payload)


def _load_config() -> dict:
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        _log.exception("portfolio: failed to read %s", _CONFIG_PATH)
        return {"stocks": [], "crypto": []}


async def _fetch_stock_price(client, ticker: str) -> tuple:
    """(price, currency) for a Yahoo Finance symbol, e.g. ``CDR.WA``.
    ``(None, None)`` on any failure (unknown symbol, network error, missing
    price in the response)."""
    try:
        resp = await client.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        meta = resp.json()["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice")
        currency = meta.get("currency")
        if price is None:
            return None, None
        return float(price), currency
    except Exception:
        _log.warning("portfolio: yahoo finance fetch failed for %s", ticker, exc_info=True)
        return None, None


async def _fetch_crypto_prices(client, ids: List[str]) -> Dict[str, float]:
    if not ids:
        return {}
    try:
        resp = await client.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": ",".join(ids), "vs_currencies": "usd"},
        )
        resp.raise_for_status()
        data = resp.json()
        return {coin_id: entry["usd"] for coin_id, entry in data.items() if "usd" in entry}
    except Exception:
        _log.warning("portfolio: coingecko fetch failed", exc_info=True)
        return {}


async def _build_portfolio() -> dict:
    import httpx

    config = _load_config()
    stocks_cfg = config.get("stocks", [])
    crypto_cfg = config.get("crypto", [])

    async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
        stock_results = await asyncio.gather(
            *(_fetch_stock_price(client, s["ticker"]) for s in stocks_cfg)
        )
        crypto_prices = await _fetch_crypto_prices(client, [c["id"] for c in crypto_cfg])

    stocks = [
        {**entry, "price": price, "currency": currency}
        for entry, (price, currency) in zip(stocks_cfg, stock_results)
    ]
    crypto = [
        {
            **entry,
            "price": crypto_prices.get(entry["id"]),
            "currency": "USD" if entry["id"] in crypto_prices else None,
        }
        for entry in crypto_cfg
    ]
    return {"stocks": stocks, "crypto": crypto, "placeholder": False}


@router.get("/api/portfolio")
async def get_portfolio():
    global _cache
    now = time.monotonic()
    if _cache and now - _cache[0] < _CACHE_TTL_S:
        return _cache[1]
    payload = await _build_portfolio()
    _cache = (now, payload)
    return payload
