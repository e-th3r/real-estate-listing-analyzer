import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import sys

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


def fetch_yandex_listing(url: str, *, timeout: int = 15) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    html = resp.text

    # Сохраняем HTML для отладки, чтобы понимать, что реально отдаёт Яндекс
    try:
        debug_path = Path("debug_yandex.html")
        debug_path.write_text(html, encoding="utf-8")
        print(
            f"[DEBUG] HTML Яндекс.Недвижимости сохранён в {debug_path.resolve()}",
            file=sys.stderr,
        )
    except Exception:
        pass

    return html


def _parse_price(text: str) -> Optional[int]:
    """
    Пытаемся вытащить:
    - либо обычную цену вида "75 000 ₽"
    - либо цену в млн: "16,2 млн ₽"
    Берём первую попавшуюся.
    """
    # 1) Млн рублей
    m_mln = re.search(
        r"(\d+[,\.\s]?\d*)\s*млн\s*₽",
        text,
        flags=re.IGNORECASE,
    )
    if m_mln:
        raw = m_mln.group(1).replace(" ", "").replace(",", ".")
        try:
            return int(float(raw) * 1_000_000)
        except ValueError:
            pass

    # 2) Обычная рублёвая цена
    m_rub = re.search(
        r"(\d[\d\s\u00A0]{3,})\s*₽",
        text,
    )
    if m_rub:
        digits = re.sub(r"\D", "", m_rub.group(1))
        if digits:
            try:
                return int(digits)
            except ValueError:
                pass

    return None


def _parse_total_area(text: str) -> Optional[float]:
    """
    Для страниц ЖК на Яндексе часто есть блок:
    "от 25,9 до 185,4 м²"
    Берём первую площадь до "м²".
    """
    m = re.search(r"(\d+[,\.\s]?\d*)\s*м²", text)
    if not m:
        return None
    raw = m.group(1).replace(" ", "").replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_coordinates(text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    В тексте страницы есть ссылка на Яндекс.Карты вида:
    https://maps.yandex.ru/?pt=37.40264,55.830715&z=16
    Здесь pt = lon,lat.
    """
    m = re.search(r"maps\.yandex\.ru/\?pt=([0-9\.\-]+),([0-9\.\-]+)", text)
    if not m:
        return None, None
    try:
        lon = float(m.group(1))
        lat = float(m.group(2))
        return lat, lon
    except ValueError:
        return None, None


def parse_yandex_listing(html: str) -> Dict[str, Any]:
    """
    Универсальный парсер для страницы объекта/ЖК на Яндекс.Недвижимости.
    Опирается только на видимый текст страницы (без исполнения JS).
    """
    soup = BeautifulSoup(html, "html.parser")
    full_text = soup.get_text(" ", strip=True)

    price = _parse_price(full_text)
    total_area = _parse_total_area(full_text)
    lat, lon = _parse_coordinates(full_text)

    structured = ListingStructuredData(
        price_rub=price,
        total_area_m2=total_area,
        floor=None,
        floors_total=None,
        latitude=lat,
        longitude=lon,
    )

    # Для Яндекса описание в явном виде можно позже доработать;
    # пока берём большой кусок текста после заголовка "Об объекте".
    description: Optional[str] = None
    obj_header = soup.find(string=re.compile(r"Об объекте", re.I))
    if obj_header and obj_header.parent:
        parent = obj_header.parent
        texts = []
        for sib in parent.next_siblings:
            if getattr(sib, "get_text", None):
                t = sib.get_text(" ", strip=True)
                if t:
                    texts.append(t)
        if texts:
            description = " ".join(texts)

    return {
        "structured": asdict(structured),
        "description": description,
        "images": [],  # при необходимости можно отдельно реализовать
    }

