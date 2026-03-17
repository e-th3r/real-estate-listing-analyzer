import json
import re
import html as html_lib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
}


@dataclass
class ListingStructuredData:
    price_rub: Optional[int]
    total_area_m2: Optional[float]
    floor: Optional[int]
    floors_total: Optional[int]
    latitude: Optional[float]
    longitude: Optional[float]


def fetch_domclick_listing(url: str, *, timeout: int = 15) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _unwrap_view_source_html(maybe_view_source_html: str) -> str:
    """
    Файлы вида `view-source_...html`, сохранённые из Chrome, содержат не исходный HTML,
    а оболочку "просмотр исходника": таблицу с `td.line-content` и экранированные теги
    как `&lt;div&gt;`.

    Этот хелпер пытается восстановить исходный HTML.
    """
    if "line-content" not in maybe_view_source_html or "&lt;" not in maybe_view_source_html:
        return maybe_view_source_html

    soup = BeautifulSoup(maybe_view_source_html, "html.parser")
    tds = soup.select("td.line-content")
    if not tds:
        return maybe_view_source_html

    # В line-content лежит текст строки исходника (включая &lt;...&gt;).
    joined = "\n".join(td.get_text("", strip=False) for td in tds)
    unescaped = html_lib.unescape(joined)

    # Простая эвристика: после unescape должен появиться doctype/html.
    if "<html" in unescaped.lower() and "<!doctype" in unescaped.lower():
        return unescaped

    return maybe_view_source_html


def _walk_json(obj: Any):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _walk_json(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_json(v)


def _extract_first_matching_number_from_json(
    data: Any,
    candidate_keys: List[str],
) -> Optional[float]:
    lowered = {k.lower() for k in candidate_keys}

    for key, value in _walk_json(data):
        if key.lower() not in lowered:
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"(\d+[.,]?\d*)", value.replace("\xa0", " "))
            if match:
                try:
                    return float(match.group(1).replace(",", "."))
                except ValueError:
                    continue
    return None


def _parse_structured_from_ldjson(html: str) -> Dict[str, Any]:
    """
    DomClick обычно кладёт Schema.org JSON-LD в <script type="application/ld+json">.
    Берём все такие блоки, объединяем и выдёргиваем интересующие поля.
    """
    html = _unwrap_view_source_html(html)
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script", type="application/ld+json")

    combined: List[Any] = []

    for script in scripts:
        if not script.string:
            continue
        text = script.string.strip()
        if not text:
            continue
        try:
            data = json.loads(text)
        except Exception:
            continue
        if isinstance(data, list):
            combined.extend(data)
        else:
            combined.append(data)

    result: Dict[str, Any] = {
        "price_rub": None,
        "total_area_m2": None,
        "floor": None,
        "floors_total": None,
        "latitude": None,
        "longitude": None,
    }

    if not combined:
        return result

    # На всякий случай оборачиваем в список общий корень
    root: Any = {"root": combined}

    price_val = _extract_first_matching_number_from_json(
        root,
        ["price", "priceRub", "priceRur", "totalPrice"],
    )
    if price_val is not None:
        result["price_rub"] = int(price_val)

    area_val = _extract_first_matching_number_from_json(
        root,
        ["area", "floorSize", "floorArea", "totalArea"],
    )
    if area_val is not None:
        result["total_area_m2"] = float(area_val)

    floor_val = _extract_first_matching_number_from_json(
        root,
        ["floor", "floorNumber", "floorLevel"],
    )
    if floor_val is not None:
        result["floor"] = int(floor_val)

    floors_total_val = _extract_first_matching_number_from_json(
        root,
        ["floorsCount", "floorCount", "buildingFloors", "buildingFloorsCount"],
    )
    if floors_total_val is not None:
        result["floors_total"] = int(floors_total_val)

    # Координаты – ищем словари с lat/lng или latitude/longitude или geo.lat/geo.lon
    lat = None
    lon = None
    for key, value in _walk_json(root):
        if not isinstance(value, dict):
            continue
        keys = {k.lower() for k in value.keys()}
        if {"lat", "lng"} <= keys or {"latitude", "longitude"} <= keys:
            try:
                lat_raw = value.get("lat") or value.get("latitude")
                lon_raw = value.get("lng") or value.get("longitude")
                lat = float(lat_raw)
                lon = float(lon_raw)
                break
            except Exception:
                continue
        # Вариант с geo: { "geo": { "latitude": ..., "longitude": ... } }
        if "geo" in value and isinstance(value["geo"], dict):
            g = value["geo"]
            try:
                if "latitude" in g and "longitude" in g:
                    lat = float(g["latitude"])
                    lon = float(g["longitude"])
                    break
            except Exception:
                continue

    result["latitude"] = lat
    result["longitude"] = lon

    return result


def parse_domclick_listing(html: str) -> Dict[str, Any]:
    structured_data = _parse_structured_from_ldjson(html)

    structured = ListingStructuredData(
        price_rub=structured_data["price_rub"],
        total_area_m2=structured_data["total_area_m2"],
        floor=structured_data["floor"],
        floors_total=structured_data["floors_total"],
        latitude=structured_data["latitude"],
        longitude=structured_data["longitude"],
    )

    # Описание и картинки можно добавить позже; сейчас главное — не null по ключевым полям.
    return {
        "structured": asdict(structured),
        "description": None,
        "images": [],
    }

