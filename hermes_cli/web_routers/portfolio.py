"""Dashboard portfolio card.

Live price fetch: Stooq (stocks, free, no key) + CoinGecko (crypto, free, no
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


async def _fetch_stock_price(client, ticker: str) -> Optional[float]:
    """Latest close for a Stooq symbol, e.g. ``CDR.WA``. None on any failure
    (unknown symbol, network error, Stooq returning "N/D")."""
    try:
        resp = await client.get(
            "https://stooq.com/q/l/",
            params={"s": ticker, "f": "sd2t2ohlcv", "h": "", "e": "csv"},
        )
        resp.raise_for_status()
        lines = resp.text.strip().splitlines()
        if len(lines) < 2:
            return None
        # header: Symbol,Date,Time,Open,High,Low,Close,Volume
        close = lines[1].split(",")[6]
        return float(close) if close not in ("", "N/D") else None
    except Exception:
        _log.warning("portfolio: stooq fetch failed for %s", ticker, exc_info=True)
        return None


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
        stock_prices = await asyncio.gather(
            *(_fetch_stock_price(client, s["ticker"]) for s in stocks_cfg)
        )
        crypto_prices = await _fetch_crypto_prices(client, [c["id"] for c in crypto_cfg])

    stocks = [
        {**entry, "price": price, "currency": "PLN" if price is not None else None}
        for entry, price in zip(stocks_cfg, stock_prices)
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
