"""Dashboard articles card — PhD reading list.

Queries OpenAlex (free, no key needed) for the keywords in
``dashboard_config/articles.json`` and stores results in
``dashboard_config/articles_store.json``, deduped by ``source-sourceId`` so
repeat fetches don't re-add what's already there.

Originally used Semantic Scholar's unauthenticated search endpoint, but that
tier shares a small global rate-limit pool that is essentially always
exhausted (persistent 429s even for a single request) — see the "Hermes
Dashboard" Obsidian note entry for 2026-09-03. OpenAlex has no such issue and
covers the same ground (works across all fields, including CS and
psychology).

This is a **local-file stand-in for the Firestore ``articles`` collection**
described in the "Dashboard — sekcje 1 i 3" Obsidian note — no Firestore
project/credentials are wired into hermes_cli yet. Swapping the store
functions below for Firestore reads/writes later should be a drop-in
replacement; the document shape already matches.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

_log = logging.getLogger("hermes_cli.web_server")

router = APIRouter()

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "dashboard_config" / "articles.json"
_STORE_PATH = Path(__file__).resolve().parent.parent / "dashboard_config" / "articles_store.json"

_OPENALEX_SEARCH_URL = "https://api.openalex.org/works"
_OPENALEX_SELECT = "id,title,abstract_inverted_index,authorships,primary_location,publication_date"
_RESULTS_PER_KEYWORD = 10


def _load_config() -> dict:
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        _log.exception("articles: failed to read %s", _CONFIG_PATH)
        return {"source": "semanticscholar", "keywords": []}


def _load_store() -> Dict[str, dict]:
    """id -> article dict, insertion order preserved (most-recently-fetched last)."""
    try:
        raw = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
        return {a["id"]: a for a in raw}
    except FileNotFoundError:
        return {}
    except Exception:
        _log.exception("articles: failed to read %s", _STORE_PATH)
        return {}


def _save_store(store: Dict[str, dict]) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(
        json.dumps(list(store.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def _search_openalex(client, keyword: str) -> List[dict]:
    try:
        resp = await client.get(
            _OPENALEX_SEARCH_URL,
            params={
                "search": keyword,
                "select": _OPENALEX_SELECT,
                "per-page": _RESULTS_PER_KEYWORD,
                "mailto": "lieberfilip@gmail.com",
            },
        )
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception:
        _log.warning("articles: openalex search failed for %r", keyword, exc_info=True)
        return []


def _reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """OpenAlex ships abstracts as a word -> [positions] inverted index
    (licensing reasons) instead of plain text — rebuild the sentence."""
    if not inverted_index:
        return ""
    positions = [(pos, word) for word, idxs in inverted_index.items() for pos in idxs]
    positions.sort()
    return " ".join(word for _, word in positions)


def _to_article_doc(paper: dict) -> dict:
    paper_id = (paper.get("id") or "").rsplit("/", 1)[-1] or None
    primary_location = paper.get("primary_location") or {}
    return {
        "id": f"openalex-{paper_id}",
        "source": "openalex",
        "sourceId": paper_id,
        "title": paper.get("title") or "(untitled)",
        "authors": [
            a.get("author", {}).get("display_name")
            for a in paper.get("authorships", [])
            if a.get("author", {}).get("display_name")
        ],
        "abstract": _reconstruct_abstract(paper.get("abstract_inverted_index")),
        "url": primary_location.get("landing_page_url") or paper.get("id"),
        "publishedDate": paper.get("publication_date"),
        "status": "new",
        "zoteroKey": None,
    }


@router.get("/api/articles")
async def list_articles(status: str = "new"):
    config = _load_config()
    store = _load_store()
    articles = list(store.values())
    if status != "all":
        articles = [a for a in articles if a.get("status") == status]
    articles.reverse()  # most recently fetched first
    return {
        "articles": articles,
        "keywords": config.get("keywords", []),
        "placeholder": False,
    }


@router.post("/api/articles/fetch")
async def fetch_articles():
    import httpx

    config = _load_config()
    keywords = config.get("keywords", [])
    store = _load_store()
    added = 0

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        for keyword in keywords:
            for paper in await _search_openalex(client, keyword):
                doc = _to_article_doc(paper)
                if doc["id"] not in store:
                    store[doc["id"]] = doc
                    added += 1

    _save_store(store)
    return {"placeholder": False, "fetched_keywords": keywords, "new_articles": added}


@router.patch("/api/articles/{article_id}/status")
async def update_article_status(article_id: str, body: Dict[str, Any]):
    status = body.get("status")
    if status not in ("new", "read", "saved", "dismissed"):
        raise HTTPException(status_code=400, detail="invalid status")
    store = _load_store()
    if article_id not in store:
        raise HTTPException(status_code=404, detail="article not found")
    store[article_id]["status"] = status
    _save_store(store)
    return store[article_id]
