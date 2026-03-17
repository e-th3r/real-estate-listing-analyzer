import json
import re
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import requests
from bs4 import BeautifulSoup


@dataclass
class ListingStructuredData:
    price_rub: Optional[int]
    total_area_m2: Optional[float]
    floor: Optional[int]
    floors_total: Optional[int]
    latitude: Optional[float]
    longitude: Optional[float]


@dataclass
class ListingArtifacts:
    structured: ListingStructuredData
    description: Optional[str]
    images: List[str]


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
}


def fetch_cian_listing(url: str, *, timeout: int = 15) -> str:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    html = response.text

    # Если на странице нет ни JSON-состояния, ни ключевых слов,
    # сохраняем HTML для отладки, т.к. это может быть заглушка антибота.
    if "__APP_INITIAL_STATE__" not in html and "Цена" not in html and "Cian" not in html:
        debug_path = Path("debug_cian.html")
        try:
            debug_path.write_text(html, encoding="utf-8")
            print(
                f"[DEBUG] Страница Циана похожа на заглушку, HTML сохранён в {debug_path.resolve()}",
                file=sys.stderr,
            )
        except Exception:
            # Логирование не должно ломать основную логику
            pass

    return html


def _walk_json(obj: Any):
    """
    Глубокий обход JSON-структуры (dict/list/скаляры).
    """
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
    """
    Ищет первое числовое значение в JSON по списку ключей.
    """
    lowered = {k.lower() for k in candidate_keys}

    for key, value in _walk_json(data):
        if key.lower() not in lowered:
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            num = _extract_float(value)
            if num is not None:
                return num
    return None


def _parse_app_initial_state(html: str) -> Dict[str, Any]:
    """
    Пробует достать базовые поля (цена, площадь, этаж, этажность, координаты)
    из большого JSON `window.__APP_INITIAL_STATE__`, если он есть.
    """
    result: Dict[str, Any] = {
        "price_rub": None,
        "total_area_m2": None,
        "floor": None,
        "floors_total": None,
        "latitude": None,
        "longitude": None,
        "description": None,
        "images": [],
    }

    # Ищем <script>, внутри которого есть window.__APP_INITIAL_STATE__
    soup = BeautifulSoup(html, "html.parser")
    script_text: Optional[str] = None
    for script in soup.find_all("script"):
        text = script.string or ""
        if "window.__APP_INITIAL_STATE__" in text:
            script_text = text
            break

    if not script_text:
        # Может не быть __APP_INITIAL_STATE__, но в HTML всё равно лежат JSON-кусочки
        script_text = ""

    # Вырезаем JSON по балансу фигурных скобок
    idx = script_text.find("window.__APP_INITIAL_STATE__")
    if idx != -1:
        brace_start = script_text.find("{", idx)
    else:
        brace_start = -1

    data: Dict[str, Any] = {}

    if brace_start != -1:
        depth = 0
        in_str: Optional[str] = None
        escaped = False
        end_pos: Optional[int] = None

        for i, ch in enumerate(script_text[brace_start:], start=brace_start):
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == in_str:
                    in_str = None
                continue

            if ch in ('"', "'"):
                in_str = ch
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_pos = i
                    break

        if end_pos is not None:
            raw_json = script_text[brace_start : end_pos + 1]
            try:
                data = json.loads(raw_json)
            except Exception:  # noqa: BLE001
                data = {}

    # Если не смогли нормально распарсить JSON, дальше попытаемся вытащить числа
    # прямо из HTML по ключам

    if data:
        # Цена
        price_val = _extract_first_matching_number_from_json(
            data,
            ["totalPrice", "price", "priceRur", "priceRub"],
        )
        if price_val is not None:
            result["price_rub"] = int(price_val)

        # Общая площадь
        area_val = _extract_first_matching_number_from_json(
            data,
            ["totalArea", "areaTotal", "area"],
        )
        if area_val is not None:
            result["total_area_m2"] = float(area_val)

        # Этаж / этажность
        floor_val = _extract_first_matching_number_from_json(
            data,
            ["floorNumber", "floor"],
        )
        if floor_val is not None:
            result["floor"] = int(floor_val)

        floors_total_val = _extract_first_matching_number_from_json(
            data,
            ["floorsCount", "floorCount", "buildingFloorsCount"],
        )
        if floors_total_val is not None:
            result["floors_total"] = int(floors_total_val)

    # Координаты – ищем словари с lat/lng или latitude/longitude
    lat = None
    lon = None
    for key, value in _walk_json(data):
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
            except Exception:  # noqa: BLE001
                continue
    result["latitude"] = lat
    result["longitude"] = lon

    # --- ФОЛБЭК: прямой поиск по HTML, если ключи есть в тексте вида "price":75000 ---
    if result["price_rub"] is None:
        m_price = re.search(
            r'"(?:totalPrice|price|priceRub|priceRur)"\s*:\s*([0-9]+)',
            html,
        )
        if m_price:
            try:
                result["price_rub"] = int(m_price.group(1))
            except ValueError:
                pass

    if result["total_area_m2"] is None:
        m_area = re.search(
            r'"(?:totalArea|areaTotal)"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
            html,
        )
        if m_area:
            try:
                result["total_area_m2"] = float(m_area.group(1))
            except ValueError:
                pass

    if result["floor"] is None:
        m_floor = re.search(
            r'"(?:floorNumber|floor)"\s*:\s*([0-9]+)',
            html,
        )
        if m_floor:
            try:
                result["floor"] = int(m_floor.group(1))
            except ValueError:
                pass

    if result["floors_total"] is None:
        m_floors = re.search(
            r'"(?:floorsCount|buildingFloorsCount|floorCount)"\s*:\s*([0-9]+)',
            html,
        )
        if m_floors:
            try:
                result["floors_total"] = int(m_floors.group(1))
            except ValueError:
                pass

    # Описание – берем самое длинное текстовое поле с подходящим ключом
    best_desc = None
    for key, value in _walk_json(data):
        if not isinstance(value, str):
            continue
        if key.lower() not in {"description", "fullDescription", "offerText"}:
            # учитываем и ключи, которые просто содержат description
            if "description" not in key.lower():
                continue
        text = value.strip()
        if not text:
            continue
        if best_desc is None or len(text) > len(best_desc):
            best_desc = text
    result["description"] = best_desc

    # Картинки – собираем все URL, которые выглядят как ссылки на изображения
    image_urls: List[str] = []
    seen: set[str] = set()
    for key, value in _walk_json(data):
        if not isinstance(value, str):
            continue
        if not any(ext in value.lower() for ext in [".jpg", ".jpeg", ".png", ".webp"]):
            continue
        if value in seen:
            continue
        seen.add(value)
        image_urls.append(value)
    result["images"] = image_urls

    return result


def _extract_first_number(text: str) -> Optional[int]:
    match = re.search(r"(\d[\d\s]*)", text)
    if not match:
        return None
    digits = re.sub(r"\D", "", match.group(1))
    try:
        return int(digits)
    except ValueError:
        return None


def _extract_float(text: str) -> Optional[float]:
    match = re.search(r"(\d+[.,]?\d*)", text.replace("\xa0", " "))
    if not match:
        return None
    number = match.group(1).replace(",", ".")
    try:
        return float(number)
    except ValueError:
        return None


def _parse_coordinates_from_meta(soup: BeautifulSoup) -> Tuple[Optional[float], Optional[float]]:
    lat = soup.find("meta", attrs={"itemprop": "latitude"})
    lon = soup.find("meta", attrs={"itemprop": "longitude"})
    if lat and lon and lat.get("content") and lon.get("content"):
        try:
            return float(lat["content"]), float(lon["content"])
        except ValueError:
            return None, None

    # Fallback: look for JSON with "lat"/"lng"
    scripts = soup.find_all("script")
    coord_pattern = re.compile(r'"lat"\s*:\s*([0-9.]+).*?"lng"\s*:\s*([0-9.]+)', re.S)
    for script in scripts:
        if not script.string:
            continue
        match = coord_pattern.search(script.string)
        if match:
            try:
                return float(match.group(1)), float(match.group(2))
            except ValueError:
                continue

    return None, None


def _parse_price(soup: BeautifulSoup) -> Optional[int]:
    # Common Cian markup variants
    candidates = []

    # 1. Data attribute (new layout)
    candidates.extend(soup.select('[data-mark="MainPrice"], [data-mark="MainPriceContainer"]'))

    # 2. itemprop price
    candidates.extend(soup.select('[itemprop="price"], [itemprop="priceCurrency"]'))

    # 3. Fallback by text near "Цена"
    if not candidates:
        for node in soup.find_all(string=re.compile("Цена", re.I)):
            parent = node.parent
            if parent:
                siblings_text = parent.get_text(" ", strip=True)
                val = _extract_first_number(siblings_text)
                if val:
                    return val

    for el in candidates:
        text = el.get_text(" ", strip=True)
        value_attr = el.get("content") or el.get("data-value")
        if value_attr:
            try:
                return int(re.sub(r"\D", "", value_attr))
            except ValueError:
                pass
        val = _extract_first_number(text)
        if val:
            return val

    # 4. Very broad fallback: ищем первую сумму с «₽» по всей странице
    full_text = soup.get_text(" ", strip=True)
    money_match = re.search(r"([\d\s\u00A0]{4,})\s*₽", full_text)
    if money_match:
        digits = re.sub(r"\D", "", money_match.group(1))
        if digits:
            try:
                return int(digits)
            except ValueError:
                pass

    return None


def _parse_areas_and_floor(soup: BeautifulSoup) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    total_area = None
    floor = None
    floors_total = None

    info_blocks = soup.find_all(
        lambda tag: tag.name in {"li", "div", "span"}
        and tag.get_text(strip=True)
        and any(word in tag.get_text() for word in ["площадь", "этаж", "Этаж"])
    )

    for block in info_blocks:
        text = block.get_text(" ", strip=True)

        if total_area is None and re.search(r"общая площадь|площадь", text, re.I):
            total_area = _extract_float(text)

        if ("этаж" in text.lower() or "Этаж" in text) and floor is None:
            # Patterns: "Этаж: 5 из 17" or "5/17 этаж"
            match = re.search(r"(\d+)\s*(?:из|/)\s*(\d+)", text)
            if match:
                try:
                    floor = int(match.group(1))
                    floors_total = int(match.group(2))
                except ValueError:
                    pass
            else:
                value = _extract_first_number(text)
                if value is not None:
                    floor = value

    # Если ничего не нашли в структурированных блоках – пробуем по всему тексту страницы
    full_text = soup.get_text(" ", strip=True)

    if total_area is None:
        # Ищем числа перед "м²"
        area_match = re.search(r"(\d+[.,]?\d*)\s*м²", full_text)
        if area_match:
            val = _extract_float(area_match.group(0))
            if val is not None:
                total_area = val

    if floor is None or floors_total is None:
        # Паттерн "Этаж 5 из 17" или "5 из 17 этаж"
        m = re.search(r"[Ээ]таж[^\d]{0,5}(\d+)\s*(?:из|/)\s*(\d+)", full_text)
        if m:
            try:
                floor = int(m.group(1))
                floors_total = int(m.group(2))
            except ValueError:
                pass
        else:
            # Паттерн "5/17 этаж"
            m2 = re.search(r"(\d+)\s*/\s*(\d+)\s*[Ээ]таж", full_text)
            if m2:
                try:
                    floor = int(m2.group(1))
                    floors_total = int(m2.group(2))
                except ValueError:
                    pass

    return total_area, floor, floors_total


def _parse_description(soup: BeautifulSoup) -> Optional[str]:
    # 1. Cian-specific data-name
    desc = soup.select_one('[data-name="Description"], [data-testid="object-description"], [itemprop="description"]')
    if desc:
        text = desc.get_text("\n", strip=True)
        return text or None

    # 2. Fallback by class
    for cls in ["description", "OfferCard__description", "object_descrition_text"]:
        el = soup.find(class_=re.compile(cls, re.I))
        if el:
            text = el.get_text("\n", strip=True)
            if text:
                return text

    # 3. Fallback: long paragraph near "Описание"
    for node in soup.find_all(string=re.compile("Описание", re.I)):
        parent = node.parent
        if not parent:
            continue
        # Take following siblings
        texts: List[str] = []
        for sib in parent.next_siblings:
            if getattr(sib, "get_text", None):
                t = sib.get_text(" ", strip=True)
                if t:
                    texts.append(t)
        if texts:
            joined = "\n".join(texts).strip()
            if joined:
                return joined

    return None


def _parse_images(soup: BeautifulSoup) -> List[str]:
    urls: List[str] = []

    # Gallery-like structures
    gallery_selectors = [
        '[data-name="Gallery"] img',
        '[data-testid="gallery"] img',
        '.fotorama__stage__frame img',
        'img',
    ]

    seen = set()

    for selector in gallery_selectors:
        for img in soup.select(selector):
            src = (
                img.get("data-src")
                or img.get("data-original")
                or img.get("src")
            )
            if not src:
                continue
            src = src.strip()
            if not src or src in seen:
                continue
            seen.add(src)
            urls.append(src)

    return urls


def parse_cian_listing(html: str) -> Dict[str, Any]:
    # 1. Пытаемся вытащить максимум из window.__APP_INITIAL_STATE__
    json_data = _parse_app_initial_state(html)

    soup = BeautifulSoup(html, "html.parser")

    # 2. Поля из DOM используются как fallback, если JSON ничего не дал
    price = json_data.get("price_rub") or _parse_price(soup)
    total_area_dom, floor_dom, floors_total_dom = _parse_areas_and_floor(soup)
    total_area = json_data.get("total_area_m2") or total_area_dom
    floor = json_data.get("floor") or floor_dom
    floors_total = json_data.get("floors_total") or floors_total_dom

    lat_json = json_data.get("latitude")
    lon_json = json_data.get("longitude")
    if lat_json is not None and lon_json is not None:
        lat, lon = lat_json, lon_json
    else:
        lat, lon = _parse_coordinates_from_meta(soup)

    description = json_data.get("description") or _parse_description(soup)

    images_json = json_data.get("images") or []
    images_dom = _parse_images(soup)
    # Объединяем, убирая дубликаты, чтобы не потерять картинки ни из JSON, ни из DOM
    seen_imgs = set()
    merged_images: List[str] = []
    for src in list(images_json) + images_dom:
        if not src or src in seen_imgs:
            continue
        seen_imgs.add(src)
        merged_images.append(src)

    structured = ListingStructuredData(
        price_rub=price,
        total_area_m2=total_area,
        floor=floor,
        floors_total=floors_total,
        latitude=lat,
        longitude=lon,
    )

    artifacts = ListingArtifacts(
        structured=structured,
        description=description,
        images=merged_images,
    )

    # Return plain dicts so that the output is easily serializable
    return {
        "structured": asdict(artifacts.structured),
        "description": artifacts.description,
        "images": artifacts.images,
    }


if __name__ == "__main__":
    # Simple manual test helper:
    example_url = input("Введите URL объявления на Циане: ").strip()
    html = fetch_cian_listing(example_url)
    data = parse_cian_listing(html)
    print(json.dumps(data, ensure_ascii=False, indent=2))

