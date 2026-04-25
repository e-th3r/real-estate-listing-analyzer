from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from analyze_listing import _target_from_dataset, _target_from_scrape
from dataset_schema import _normalize_cian_url
from report_generator import (
    TargetListing,
    build_default_llm,
    build_report_prompt,
)
from vector_store import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_STORE_DIR,
    listing_id_from_url,
    load_vector_store,
    search_alternatives_with_scores,
)

logger = logging.getLogger("webapp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = ROOT / "data" / "structured" / "listings_clean.parquet"
PERSIST_DIR = ROOT / DEFAULT_VECTOR_STORE_DIR
STATIC_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Listing Analyzer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_store_lock = Lock()
_store = None


def _target_from_html_fallback(url: str) -> TargetListing:
    """Fallback when Cian API brute-force can't find the listing: fetch the
    public HTML page and parse __APP_INITIAL_STATE__ + DOM."""
    from cian_scraper import fetch_cian_listing, parse_cian_listing

    html = fetch_cian_listing(url)
    data = parse_cian_listing(html)
    structured = data.get("structured") or {}
    description = str(data.get("description") or "")
    try:
        listing_id = listing_id_from_url(url)
    except ValueError:
        listing_id = None
    return TargetListing(
        description=description,
        listing_id=listing_id,
        url=url,
        price_rub=structured.get("price_rub"),
        total_area_m2=structured.get("total_area_m2"),
        floor=structured.get("floor"),
        floors_total=structured.get("floors_total"),
    )


def _get_store():
    """Lazy-load Chroma — first call pays the embedding-model cost."""
    global _store
    with _store_lock:
        if _store is None:
            logger.info("Loading vector store from %s", PERSIST_DIR)
            _store = load_vector_store(PERSIST_DIR, model_name=DEFAULT_EMBEDDING_MODEL)
        return _store


class AnalyzeRequest(BaseModel):
    url: str = Field(..., description="Ссылка на объявление Cian")
    k: int = Field(5, ge=1, le=10)
    report: bool = Field(False, description="Запрашивать LLM-отчёт")
    no_scrape: bool = Field(False, description="Не лезть в сеть, если URL не в датасете")


def _alt_to_dict(doc, score: float) -> Dict[str, Any]:
    md = doc.metadata or {}
    snippet = " ".join((doc.page_content or "").split())
    if len(snippet) > 320:
        snippet = snippet[:320].rstrip() + "…"
    return {
        "url": md.get("url"),
        "listing_id": md.get("listing_id"),
        "price_rub": md.get("price_rub"),
        "total_area_m2": md.get("total_area_m2"),
        "price_per_m2": md.get("price_per_m2"),
        "floor": md.get("floor"),
        "floors_total": md.get("floors_total"),
        "latitude": md.get("latitude"),
        "longitude": md.get("longitude"),
        "distance": float(score),
        "snippet": snippet,
    }


def _target_to_dict(t: TargetListing, source: str) -> Dict[str, Any]:
    return {
        "url": t.url,
        "listing_id": t.listing_id,
        "price_rub": t.price_rub,
        "total_area_m2": t.total_area_m2,
        "price_per_m2": t.price_per_m2,
        "floor": t.floor,
        "floors_total": t.floors_total,
        "description": t.description,
        "source": source,
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    try:
        norm_url = _normalize_cian_url(req.url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Невалидный URL: {exc}") from exc

    target = _target_from_dataset(PARQUET_PATH, norm_url)
    source = "dataset"
    if target is None:
        if req.no_scrape:
            raise HTTPException(
                status_code=404,
                detail="URL не найден в датасете, а scrape отключён.",
            )
        api_err: Optional[Exception] = None
        try:
            target = _target_from_scrape(norm_url)
            source = "scrape_api"
        except Exception as exc:
            api_err = exc
            logger.warning("Cian API scrape failed, falling back to HTML: %s", exc)
            try:
                target = _target_from_html_fallback(norm_url)
                source = "scrape_html"
            except Exception as html_exc:
                logger.exception("HTML fallback also failed")
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"Не удалось получить данные с Cian. "
                        f"API: {api_err}; HTML: {html_exc}"
                    ),
                ) from html_exc

        if not target.description or len(target.description) < 20:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Получили пустое или слишком короткое описание со страницы — "
                    "семантический поиск не отработает. Попробуйте позже или другой URL."
                ),
            )

    try:
        store = _get_store()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    exclude = [target.listing_id] if target.listing_id else None
    pairs = search_alternatives_with_scores(
        store, query=target.description, k=req.k, exclude_ids=exclude
    )
    alternatives = [_alt_to_dict(doc, score) for doc, score in pairs]

    report_text: Optional[str] = None
    if req.report:
        try:
            llm = build_default_llm()
            messages = build_report_prompt(target, pairs)
            response = llm.invoke(messages)
            content = response.content
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p) for p in content
                )
            report_text = str(content)
        except Exception as exc:
            logger.exception("LLM report failed")
            report_text = f"(LLM-отчёт недоступен: {exc})"

    return {
        "target": _target_to_dict(target, source),
        "alternatives": alternatives,
        "report": report_text,
    }


@app.get("/api/stats")
def stats() -> Dict[str, Any]:
    if not PARQUET_PATH.exists():
        return {"total": 0, "clean": 0, "anomalies": 0, "median_price_per_m2": None}

    frame = pd.read_parquet(PARQUET_PATH)
    total = int(len(frame))
    median_ppm = None
    median_price = None
    if not frame.empty:
        ppm = (frame["price_rub"].astype(float) / frame["total_area_m2"].astype(float)).dropna()
        if not ppm.empty:
            median_ppm = float(ppm.median())
        median_price = float(frame["price_rub"].astype(float).median())

    anomalies_path = PARQUET_PATH.with_name("listings_anomalies.json")
    anomalies = 0
    if anomalies_path.exists():
        try:
            data = json.loads(anomalies_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                anomalies = len(data)
            elif isinstance(data, dict):
                anomalies = int(data.get("count", 0)) or len(data.get("rows", []))
        except Exception:
            anomalies = 0

    return {
        "total": total + anomalies,
        "clean": total,
        "anomalies": anomalies,
        "median_price_per_m2": median_ppm,
        "median_price_rub": median_price,
    }


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "parquet_exists": PARQUET_PATH.exists(),
        "vector_store_exists": PERSIST_DIR.exists(),
    }


# Serve UI
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("webapp.server:app", host="127.0.0.1", port=8000, reload=False)
