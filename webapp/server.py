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
from price_model import (
    DEFAULT_MODEL_PATH as DEFAULT_PRICE_MODEL_PATH,
    load_price_model,
    predict_fair_price,
)
from report_generator import (
    TargetListing,
    build_default_llm,
    generate_report,
)
from vector_store import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_STORE_DIR,
    listing_id_from_url,
    load_vector_store,
    search_alternatives_metadata_first,
)

logger = logging.getLogger("webapp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = ROOT / "data" / "structured" / "listings_clean.parquet"
PERSIST_DIR = ROOT / DEFAULT_VECTOR_STORE_DIR
PRICE_MODEL_PATH = ROOT / DEFAULT_PRICE_MODEL_PATH
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

_price_model_lock = Lock()
_price_model = None
_price_model_loaded = False


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
        latitude=structured.get("latitude"),
        longitude=structured.get("longitude"),
        images=list(data.get("images") or []),
    )


def _get_store():
    """Lazy-load Chroma — first call pays the embedding-model cost."""
    global _store
    with _store_lock:
        if _store is None:
            logger.info("Loading vector store from %s", PERSIST_DIR)
            _store = load_vector_store(PERSIST_DIR, model_name=DEFAULT_EMBEDDING_MODEL)
        return _store


def _get_price_model():
    """Lazy-load CatBoost-модели справедливой цены. Кэшируется навсегда; если
    файла нет, оставляем None — анализ продолжит работать без оценки."""
    global _price_model, _price_model_loaded
    with _price_model_lock:
        if not _price_model_loaded:
            if PRICE_MODEL_PATH.exists():
                logger.info("Loading price model from %s", PRICE_MODEL_PATH)
                try:
                    _price_model = load_price_model(PRICE_MODEL_PATH)
                except Exception as exc:
                    logger.warning("Failed to load price model: %s", exc)
                    _price_model = None
            else:
                logger.info("Price model not found at %s — пропускаем", PRICE_MODEL_PATH)
                _price_model = None
            _price_model_loaded = True
        return _price_model


def _compute_fair_price(target: TargetListing, *, skip: bool = False) -> Optional[float]:
    if skip:
        return None
    model = _get_price_model()
    if model is None:
        return None
    try:
        return predict_fair_price(
            model,
            total_area_m2=target.total_area_m2,
            floor=target.floor,
            floors_total=target.floors_total,
            latitude=target.latitude,
            longitude=target.longitude,
            description=target.description,
        )
    except Exception as exc:
        logger.warning("CatBoost predict failed: %s", exc)
        return None


class AnalyzeRequest(BaseModel):
    url: Optional[str] = Field(None, description="Ссылка на объявление Cian")
    text: Optional[str] = Field(
        None, description="Произвольное описание квартиры / запрос (аналог --text в CLI)"
    )
    k: int = Field(5, ge=1, le=10)
    report: bool = Field(False, description="Запрашивать LLM-отчёт")
    no_scrape: bool = Field(False, description="Не лезть в сеть, если URL не в датасете")
    no_price_model: bool = Field(
        False, description="Не подгружать CatBoost-оценку справедливой цены"
    )


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
        "image_url": md.get("image_url"),
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
        "images": list(t.images or []),
        "source": source,
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    if not req.url and not req.text:
        raise HTTPException(
            status_code=400,
            detail="Нужно задать url или text (аналог --url / --text в CLI).",
        )

    target: Optional[TargetListing] = None
    source = "text"
    if req.url:
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
    else:
        target = TargetListing(description=req.text or "")

    try:
        store = _get_store()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    exclude = [target.listing_id] if target.listing_id else None
    target_meta = {
        "price_per_m2": target.price_per_m2,
        "total_area_m2": target.total_area_m2,
        "floor": target.floor,
        "floors_total": target.floors_total,
    }
    pairs = search_alternatives_metadata_first(
        store,
        query=target.description,
        target_metadata=target_meta,
        k=req.k,
        exclude_ids=exclude,
    )
    alternatives = [_alt_to_dict(doc, score) for doc, score in pairs]

    fair_price = _compute_fair_price(target, skip=req.no_price_model)
    fair_price_block: Optional[Dict[str, Any]] = None
    if fair_price is not None:
        block: Dict[str, Any] = {"fair_price_rub": float(fair_price)}
        if target.price_rub:
            delta_abs = float(target.price_rub) - float(fair_price)
            block["delta_rub"] = float(delta_abs)
            block["delta_pct"] = float(delta_abs / float(fair_price) * 100.0)
        if target.total_area_m2:
            block["fair_price_per_m2"] = float(fair_price) / float(target.total_area_m2)
        fair_price_block = block

    report_text: Optional[str] = None
    if req.report:
        try:
            llm = build_default_llm()
            report_text = generate_report(
                target,
                store,
                llm,
                k=req.k,
                alternatives=pairs,
                fair_price_rub=fair_price,
            )
        except Exception as exc:
            logger.exception("LLM report failed")
            report_text = f"(LLM-отчёт недоступен: {exc})"

    return {
        "target": _target_to_dict(target, source),
        "alternatives": alternatives,
        "fair_price": fair_price_block,
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


@app.get("/api/images")
def images(url: str) -> Dict[str, Any]:
    """Lazy fetch of listing photos. Used by the UI to backfill the gallery
    when /api/analyze hit the dataset path (no images stored in parquet)."""
    try:
        norm_url = _normalize_cian_url(url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Невалидный URL: {exc}") from exc

    imgs: List[str] = []
    try:
        from cian_scraper import scrape_cian_listing_via_cianpython

        data = scrape_cian_listing_via_cianpython(norm_url)
        imgs = list(data.get("images") or [])
    except Exception as api_err:
        logger.info("Cian API images failed (%s); trying HTML", api_err)
        try:
            from cian_scraper import fetch_cian_listing, parse_cian_listing

            html = fetch_cian_listing(norm_url)
            data = parse_cian_listing(html)
            imgs = list(data.get("images") or [])
        except Exception as html_err:
            logger.warning("HTML images also failed: %s", html_err)
            imgs = []

    return {"url": norm_url, "images": imgs}


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "parquet_exists": PARQUET_PATH.exists(),
        "vector_store_exists": PERSIST_DIR.exists(),
        "price_model_exists": PRICE_MODEL_PATH.exists(),
    }


# Serve UI
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        str(STATIC_DIR / "index.html"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("webapp.server:app", host="127.0.0.1", port=8000, reload=False)
