"""food-app — AI receipt parsing.

``POST /api/food/parse-receipt``: food-app (separate repo/app, served under
``apps.lieberfilip.pl/food/``) uploads a photo of a grocery receipt here and
gets back a structured list of line items to review/edit before writing to
its own Firestore ``food_inventory`` collection. Same-origin as this backend
(root ``/`` on ``apps.lieberfilip.pl``), so the frontend calls this with a
plain relative ``fetch`` — no separate auth bridge needed, Cloudflare Access
already gates the whole domain.

Uses Hermes' own internal auxiliary-LLM client (``agent.auxiliary_client``,
the same one ``tools/vision_tools.py`` and ``agent/title_generator.py`` use
for one-off model calls) instead of a raw ``anthropic.Anthropic()`` client —
this process's env has no ``ANTHROPIC_API_KEY``, credentials live behind that
layer. Reuses whatever provider/model is already configured for the main
agent; food-app intentionally has no AI config of its own, see the
"food-app" Obsidian note.
"""

import base64
import binascii
import json
import logging
import re
from datetime import date
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

_log = logging.getLogger("hermes_cli.web_server")

router = APIRouter()

# Lazy-loaded, mirroring tools/vision_tools.py — agent.auxiliary_client pulls
# in credential_pool -> hermes_cli.auth -> httpx (~50ms cold), only needed
# once a food request actually comes in.
_async_call_llm = None
_extract_content_or_reasoning = None


def _load_auxiliary_client() -> None:
    global _async_call_llm, _extract_content_or_reasoning
    if _async_call_llm is None or _extract_content_or_reasoning is None:
        from agent.auxiliary_client import (
            async_call_llm as _acl,
            extract_content_or_reasoning as _ecr,
        )
        _async_call_llm = _acl
        _extract_content_or_reasoning = _ecr


# Generous headroom over a phone photo; vision models cap images well below this.
_MAX_IMAGE_BYTES = 8 * 1024 * 1024

_ALLOWED_UNITS = {"szt", "g", "kg", "ml", "l"}

_PROMPT_TEMPLATE = """Na zdjęciu jest paragon ze sklepu spożywczego. Wypisz z niego produkty spożywcze \
(pomiń pozycje niespożywcze, rabaty, sumy, dane sklepu).

Dzisiejsza data: {today}. Dla każdego produktu oszacuj realistyczną datę ważności na podstawie \
typowego okresu przydatności tego typu produktu (np. świeże pieczywo ~3 dni, nabiał ~1-2 tygodnie, \
mrożonki kilka miesięcy, konserwy/produkty suche wiele miesięcy) licząc od dzisiejszej daty — jeśli \
produkt w ogóle nie ma sensownej daty ważności (np. przyprawa, ocet), zwróć null.

Odpowiedz WYŁĄCZNIE czystym JSON-em (bez markdown, bez komentarzy) w formacie:
{{"items": [{{"name": "...", "quantity": 1, "unit": "szt|g|kg|ml|l", "expiryDate": "YYYY-MM-DD" | null}}]}}

Jeśli nie da się odczytać ilości, użyj quantity: 1, unit: "szt". Nazwa produktu po polsku, skrócona \
i czytelna (nie surowy skrót z paragonu)."""


class ParseReceiptRequest(BaseModel):
    data_url: str


class ParsedItem(BaseModel):
    name: str
    quantity: float = 1
    unit: str = "szt"
    expiryDate: Optional[str] = None


class ParseReceiptResponse(BaseModel):
    items: List[ParsedItem]


def _validate_data_url(data_url: str) -> None:
    if not data_url.startswith("data:") or "," not in data_url:
        raise HTTPException(status_code=400, detail="Invalid image payload")
    header, encoded = data_url.split(",", 1)
    if ";base64" not in header:
        raise HTTPException(status_code=400, detail="Image payload must be base64 encoded")
    media_type = header[5:].split(";", 1)[0] or "image/jpeg"
    if not media_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Payload must be an image")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Image payload is not valid base64")
    if not raw:
        raise HTTPException(status_code=400, detail="Image is empty")
    if len(raw) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large")


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
    return json.loads(text)


def _normalize_item(raw: dict) -> Optional[ParsedItem]:
    name = str(raw.get("name") or "").strip()
    if not name:
        return None
    try:
        quantity = float(raw.get("quantity", 1) or 1)
    except (TypeError, ValueError):
        quantity = 1
    if quantity <= 0:
        quantity = 1
    unit = str(raw.get("unit") or "szt").strip().lower()
    if unit not in _ALLOWED_UNITS:
        unit = "szt"
    expiry = raw.get("expiryDate")
    if expiry is not None and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(expiry)):
        expiry = None
    return ParsedItem(name=name, quantity=quantity, unit=unit, expiryDate=expiry)


@router.post("/api/food/parse-receipt", response_model=ParseReceiptResponse)
async def parse_receipt(payload: ParseReceiptRequest):
    _validate_data_url(payload.data_url)
    _load_auxiliary_client()

    prompt = _PROMPT_TEMPLATE.format(today=date.today().isoformat())

    try:
        response = await _async_call_llm(
            task="vision",
            max_tokens=2048,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": payload.data_url}},
                    ],
                }
            ],
        )
    except Exception:
        _log.exception("food: parse-receipt AI call failed")
        raise HTTPException(status_code=502, detail="Nie udało się przeanalizować paragonu")

    text = _extract_content_or_reasoning(response) or ""
    try:
        parsed = _extract_json(text)
        raw_items = parsed.get("items", [])
    except Exception:
        _log.warning("food: parse-receipt got non-JSON response: %s", text[:500])
        raise HTTPException(status_code=502, detail="Nie udało się zrozumieć odpowiedzi AI")

    items = [item for item in (_normalize_item(r) for r in raw_items) if item is not None]
    return ParseReceiptResponse(items=items)


class InventoryItemIn(BaseModel):
    name: str
    quantity: float = 1
    unit: str = "szt"
    expiryDate: Optional[str] = None


class RecipeOut(BaseModel):
    name: str
    kcal: int
    protein: int
    ingredients: List[str]
    steps: List[str]


class DayPlan(BaseModel):
    day: str
    recipe: RecipeOut


class GenerateWeekPlanRequest(BaseModel):
    items: List[InventoryItemIn]
    dailyKcalTarget: Optional[int] = None
    dailyProteinTarget: Optional[int] = None


class GenerateWeekPlanResponse(BaseModel):
    days: List[DayPlan]


class SuggestNowRequest(BaseModel):
    items: List[InventoryItemIn]


class SuggestNowResponse(BaseModel):
    suggestions: List[RecipeOut]


_DAY_NAMES = [
    "Poniedziałek",
    "Wtorek",
    "Środa",
    "Czwartek",
    "Piątek",
    "Sobota",
    "Niedziela",
]


def _format_inventory(items: List[InventoryItemIn]) -> str:
    if not items:
        return "(lodówka jest pusta)"
    lines = []
    for item in items:
        expiry = f", ważność: {item.expiryDate}" if item.expiryDate else ""
        lines.append(f"- {item.name}: {item.quantity} {item.unit}{expiry}")
    return "\n".join(lines)


def _normalize_recipe(raw: dict) -> Optional[RecipeOut]:
    name = str(raw.get("name") or "").strip()
    if not name:
        return None
    try:
        kcal = int(round(float(raw.get("kcal", 0) or 0)))
    except (TypeError, ValueError):
        kcal = 0
    try:
        protein = int(round(float(raw.get("protein", 0) or 0)))
    except (TypeError, ValueError):
        protein = 0
    ingredients = [str(i).strip() for i in (raw.get("ingredients") or []) if str(i).strip()]
    steps = [str(s).strip() for s in (raw.get("steps") or []) if str(s).strip()]
    return RecipeOut(name=name, kcal=kcal, protein=protein, ingredients=ingredients, steps=steps)


async def _call_claude_json(prompt: str, max_tokens: int) -> dict:
    _load_auxiliary_client()

    try:
        response = await _async_call_llm(
            max_tokens=max_tokens,
            temperature=0.4,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:
        _log.exception("food: AI call failed")
        raise HTTPException(status_code=502, detail="Nie udało się skontaktować z AI")

    text = _extract_content_or_reasoning(response) or ""
    try:
        return _extract_json(text)
    except Exception:
        _log.warning("food: got non-JSON response: %s", text[:500])
        raise HTTPException(status_code=502, detail="Nie udało się zrozumieć odpowiedzi AI")


@router.post("/api/food/generate-week-plan", response_model=GenerateWeekPlanResponse)
async def generate_week_plan(payload: GenerateWeekPlanRequest):
    target_line = ""
    if payload.dailyKcalTarget or payload.dailyProteinTarget:
        target_line = (
            f"\nDzienny cel: ~{payload.dailyKcalTarget or '?'} kcal, "
            f"~{payload.dailyProteinTarget or '?'} g białka (na cały dzień, ten przepis to jeden posiłek "
            "więc nie musi pokrywać całości celu)."
        )

    prompt = f"""Mam w lodówce/spiżarni:
{_format_inventory(payload.items)}
{target_line}

Zaproponuj plan posiłków na 7 dni (jeden przepis dziennie, obiadokolacja). Priorytetyzuj składniki \
z najbliższą datą ważności (żeby się nie zmarnowały) oraz przepisy wysokobiałkowe. Możesz zakładać \
podstawowe przyprawy/olej/sól, ale głównych składników używaj z listy powyżej gdzie to możliwe — \
jeśli czegoś brakuje, dopisz to jako dodatkowy składnik.

Odpowiedz WYŁĄCZNIE czystym JSON-em (bez markdown) w formacie:
{{"days": [{{"day": "Poniedziałek", "recipe": {{"name": "...", "kcal": 650, "protein": 40, \
"ingredients": ["..."], "steps": ["..."]}}}}]}} — dokładnie 7 dni, w kolejności \
{", ".join(_DAY_NAMES)}."""

    parsed = await _call_claude_json(prompt, max_tokens=8192)
    raw_days = parsed.get("days", [])

    days: List[DayPlan] = []
    for i, raw_day in enumerate(raw_days):
        recipe = _normalize_recipe(raw_day.get("recipe") or {})
        if recipe is None:
            continue
        day_name = str(raw_day.get("day") or "").strip() or (
            _DAY_NAMES[i] if i < len(_DAY_NAMES) else f"Dzień {i + 1}"
        )
        days.append(DayPlan(day=day_name, recipe=recipe))

    if not days:
        raise HTTPException(status_code=502, detail="AI nie zwróciło żadnego planu")

    return GenerateWeekPlanResponse(days=days)


@router.post("/api/food/suggest-now", response_model=SuggestNowResponse)
async def suggest_now(payload: SuggestNowRequest):
    prompt = f"""Mam w lodówce/spiżarni:
{_format_inventory(payload.items)}

Zaproponuj 1-3 szybkie posiłki, które mogę zrobić TERAZ z tego, co mam (priorytet: składniki z \
bliską datą ważności, wysoka zawartość białka). Możesz zakładać podstawowe przyprawy/olej/sól.

Odpowiedz WYŁĄCZNIE czystym JSON-em (bez markdown) w formacie:
{{"suggestions": [{{"name": "...", "kcal": 500, "protein": 30, "ingredients": ["..."], \
"steps": ["..."]}}]}}"""

    parsed = await _call_claude_json(prompt, max_tokens=2048)
    raw_suggestions = parsed.get("suggestions", [])

    suggestions = [
        r for r in (_normalize_recipe(s) for s in raw_suggestions) if r is not None
    ]
    if not suggestions:
        raise HTTPException(status_code=502, detail="AI nie zwróciło żadnych propozycji")

    return SuggestNowResponse(suggestions=suggestions)
