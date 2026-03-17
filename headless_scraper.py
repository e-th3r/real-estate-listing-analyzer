import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


@dataclass
class ListingStructuredData:
    price_rub: Optional[int]
    total_area_m2: Optional[float]
    floor: Optional[int]
    floors_total: Optional[int]
    latitude: Optional[float]
    longitude: Optional[float]


def _extract_float(text: str) -> Optional[float]:
    m = re.search(r"(\d+[.,]?\d*)", text.replace("\xa0", " "))
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "."))
    except ValueError:
        return None


def _extract_int(text: str) -> Optional[int]:
    m = re.search(r"(\d[\d\s\u00A0]*)", text)
    if not m:
        return None
    digits = re.sub(r"\D", "", m.group(1))
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _parse_coords_from_text(text: str) -> Tuple[Optional[float], Optional[float]]:
    # generic: "55.830715,37.40264" unlikely; keep minimal
    m = re.search(r"([0-9]{2}\.[0-9]+)[, ]+([0-9]{2}\.[0-9]+)", text)
    if not m:
        return None, None
    try:
        a = float(m.group(1))
        b = float(m.group(2))
    except ValueError:
        return None, None
    # heuristic: lat in [40..80], lon in [10..190]
    if 40 <= a <= 80 and 10 <= b <= 190:
        return a, b
    if 40 <= b <= 80 and 10 <= a <= 190:
        return b, a
    return None, None


def scrape_domclick_headless(
    url: str,
    *,
    timeout_ms: int = 25_000,
    headless: bool = True,
    user_data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Открывает страницу в Chromium (Playwright), ждёт загрузки и пытается извлечь
    цену/площадь/этаж/этажность/координаты из отрисованного DOM и/или JSON-LD.
    """
    with sync_playwright() as p:
        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
        ]

        if user_data_dir:
            context = p.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                headless=headless,
                locale="ru-RU",
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                ),
                viewport={"width": 1365, "height": 900},
                args=launch_args,
            )
            browser = context.browser
        else:
            browser = p.chromium.launch(headless=headless, args=launch_args)
            context = browser.new_context(
                locale="ru-RU",
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                ),
                viewport={"width": 1365, "height": 900},
            )
        # минимальная "stealth"-подстройка для антиботов
        context.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.chrome = window.chrome || { runtime: {} };
            Object.defineProperty(navigator, 'languages', { get: () => ['ru-RU','ru','en-US','en'] });
            Object.defineProperty(navigator, 'platform', { get: () => 'Win32' });
            """
        )
        page = context.new_page()

        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        try:
            page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except PlaywrightTimeoutError:
            # не критично: многие SPA бесконечно шумят в сети
            pass

        # Если попали на антибот-страницу (Qrator) — подождём, пока она пройдёт и страница перезагрузится
        try:
            page.wait_for_function(
                """
                () => {
                  const html = document.documentElement?.innerHTML || '';
                  const txt = document.body?.innerText || '';
                  const looksLikeQrator = html.includes('__qrator') || html.includes('qauth') || html.includes('Qrator');
                  if (looksLikeQrator) return false;
                  return txt.length > 200;
                }
                """,
                timeout=timeout_ms,
            )
        except PlaywrightTimeoutError:
            # попробуем один мягкий reload на случай, что куки проставились
            try:
                page.reload(wait_until="domcontentloaded", timeout=timeout_ms)
                page.wait_for_load_state("networkidle", timeout=timeout_ms)
            except Exception:
                pass

        # Debug artifacts: сохраняем отрисованный HTML и скриншот
        try:
            Path("debug_domclick_rendered.html").write_text(page.content(), encoding="utf-8")
        except Exception:
            pass
        try:
            page.screenshot(path="debug_domclick.png", full_page=True)
        except Exception:
            pass

        # Явная проверка на блокировку антиботом (по тексту на скриншоте)
        try:
            body_text = (page.locator("body").inner_text() or "").strip()
        except Exception:
            body_text = ""
        if "запрос выглядит необычно" in body_text.lower():
            raise RuntimeError(
                "DomClick anti-bot block detected ('zaprоs vyglyadit neobychno'). "
                "Try running with --headed and --user-data-dir (persistent profile) "
                "or from a different IP (no VPN/proxy)."
            )

        # 1) JSON-LD, если есть
        ld_json_blocks = page.locator('script[type="application/ld+json"]').all()
        price = area = floor = floors_total = None
        lat = lon = None

        for block in ld_json_blocks:
            txt = (block.text_content() or "").strip()
            if not txt:
                continue
            try:
                data = json.loads(txt)
            except Exception:
                continue
            # cheap recursive scan without importing existing helpers
            stack = [data]
            while stack:
                cur = stack.pop()
                if isinstance(cur, dict):
                    for k, v in cur.items():
                        lk = str(k).lower()
                        if price is None and lk in {"price", "pricerub", "pricerur", "totalprice"}:
                            if isinstance(v, (int, float)):
                                price = int(v)
                            elif isinstance(v, str):
                                iv = _extract_int(v)
                                if iv is not None:
                                    price = iv
                        if area is None and lk in {"area", "totalsize", "totalarea", "floorsize", "floorarea"}:
                            if isinstance(v, (int, float)):
                                area = float(v)
                            elif isinstance(v, str):
                                fv = _extract_float(v)
                                if fv is not None:
                                    area = fv
                        if floor is None and lk in {"floor", "floornumber", "floorlevel"}:
                            if isinstance(v, (int, float)):
                                floor = int(v)
                            elif isinstance(v, str):
                                iv = _extract_int(v)
                                if iv is not None:
                                    floor = iv
                        if floors_total is None and lk in {
                            "floorscount",
                            "floorcount",
                            "buildingfloors",
                            "buildingfloorscount",
                        }:
                            if isinstance(v, (int, float)):
                                floors_total = int(v)
                            elif isinstance(v, str):
                                iv = _extract_int(v)
                                if iv is not None:
                                    floors_total = iv

                        if isinstance(v, (dict, list)):
                            stack.append(v)
                elif isinstance(cur, list):
                    stack.extend(cur)

        # 2) DOM fallback: берём весь текст страницы и ищем паттерны
        full_text = page.locator("body").inner_text()

        if price is None:
            # "12 345 678 ₽"
            m = re.search(r"(\d[\d\s\u00A0]{3,})\s*₽", full_text)
            if m:
                price = _extract_int(m.group(0))

        if area is None:
            # "56,2 м²"
            m = re.search(r"(\d+[.,]?\d*)\s*м²", full_text)
            if m:
                area = _extract_float(m.group(0))

        if floor is None or floors_total is None:
            # "Этаж 5 из 17" / "5 из 17 этаж" / "5/17"
            m = re.search(r"(\d+)\s*(?:из|/)\s*(\d+)", full_text)
            if m:
                try:
                    floor = floor or int(m.group(1))
                    floors_total = floors_total or int(m.group(2))
                except ValueError:
                    pass

        if lat is None or lon is None:
            lat, lon = _parse_coords_from_text(full_text)

        context.close()
        if browser:
            browser.close()

    structured = ListingStructuredData(
        price_rub=price,
        total_area_m2=area,
        floor=floor,
        floors_total=floors_total,
        latitude=lat,
        longitude=lon,
    )

    return {
        "structured": asdict(structured),
        "description": None,
        "images": [],
    }

