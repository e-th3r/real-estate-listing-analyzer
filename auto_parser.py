import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cianparser

from cian_scraper import scrape_cian_listing_via_cianpython


def analyze_cian_listing(url: str) -> Dict[str, Any]:
    data = scrape_cian_listing_via_cianpython(url)
    data["url"] = url
    return data


def analyze_many(urls: Iterable[str], *, max_workers: int = 8) -> List[Tuple[str, Dict[str, Any], str]]:
    results: List[Tuple[str, Dict[str, Any], str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(analyze_cian_listing, url): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
                results.append((url, data, ""))
            except Exception as exc:  # noqa: BLE001
                results.append((url, {}, str(exc)))
    return results


def write_records_and_dvc_track(records: List[Dict[str, Any]], *, source: str) -> Path:
    root = Path(__file__).resolve().parent
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = raw_dir / f"cian_{source}_{ts}.ndjson"
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

    dvc_commands = [
        ["dvc", "add", "data"],
        [sys.executable, "-m", "dvc", "add", "data"],
        ["uv", "run", "python", "-m", "dvc", "add", "data"],
    ]
    last_result: subprocess.CompletedProcess[str] | None = None
    for cmd in dvc_commands:
        try:
            result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
        except FileNotFoundError:
            continue
        last_result = result
        if result.returncode == 0:
            break
    if last_result is None or last_result.returncode != 0:
        stdout = last_result.stdout.strip() if last_result else ""
        stderr = last_result.stderr.strip() if last_result else ""
        raise RuntimeError(
            "Не удалось обновить DVC-трекинг для data. Установи DVC и выполни `dvc add data`. "
            f"stdout: {stdout} "
            f"stderr: {stderr}"
        )

    return output_path


def _parse_rooms(raw: str) -> Any:
    raw = raw.strip().lower()
    if raw in {"all", "studio"}:
        return raw
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return "all"
    rooms: List[Any] = []
    for p in parts:
        if p == "studio":
            rooms.append("studio")
        else:
            rooms.append(int(p))
    return tuple(rooms)


def discover_urls(
    *,
    location: str,
    deal_type: str,
    rooms: Any,
    start_page: int,
    end_page: int,
    proxies: Optional[List[str]] = None,
) -> List[str]:
    parser = cianparser.CianParser(location=location, proxies=proxies)
    listings: List[Dict[str, Any]] = parser.get_flats(
        deal_type=deal_type,
        rooms=rooms,
        with_saving_csv=False,
        with_extra_data=False,
        additional_settings={"start_page": start_page, "end_page": end_page},
    )

    seen: set[str] = set()
    urls: List[str] = []
    for item in listings:
        url = item.get("url")
        if not isinstance(url, str) or not url:
            continue
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-parser РґР»СЏ Р¦РёР°РЅ: РѕР±РЅР°СЂСѓР¶РёРІР°РµС‚ РѕР±СЉСЏРІР»РµРЅРёСЏ С‡РµСЂРµР· cianparser "
            "(РїРѕРёСЃРє РїРѕ РіРѕСЂРѕРґСѓ/СЃРґРµР»РєРµ/РєРѕРјРЅР°С‚Р°Рј) Рё Р·Р°С‚РµРј РѕР±РѕРіР°С‰Р°РµС‚ РєР°Р¶РґРѕРµ С‡РµСЂРµР· "
            "Cian API, С‡С‚РѕР±С‹ РїРѕР»СѓС‡РёС‚СЊ РєРѕРѕСЂРґРёРЅР°С‚С‹ Рё РїСЂРѕР№С‚Рё Pandera-СЃС…РµРјСѓ."
        )
    )
    parser.add_argument(
        "--location",
        type=str,
        default="РњРѕСЃРєРІР°",
        help="Р“РѕСЂРѕРґ (РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ РІ СЃРїРёСЃРєРµ РїРѕРґРґРµСЂР¶РёРІР°РµРјС‹С… cianparser). РџРѕ СѓРјРѕР»С‡Р°РЅРёСЋ: РњРѕСЃРєРІР°",
    )
    parser.add_argument(
        "--deal-type",
        type=str,
        choices=["sale", "rent_long"],
        default="sale",
        help=(
            "РўРёРї СЃРґРµР»РєРё. РЎС…РµРјР° Pandera РѕР¶РёРґР°РµС‚ price_rub >= 500_000, РїРѕСЌС‚РѕРјСѓ "
            "rent_long РІР°Р»РёРґР°С†РёСЋ РЅРµ РїСЂРѕР№РґС‘С‚ Р±РµР· РїСЂР°РІРєРё СЃС…РµРјС‹."
        ),
    )
    parser.add_argument(
        "--rooms",
        type=str,
        default="1,2,3",
        help='РљРѕРјРЅР°С‚С‹: "all", "studio", Р»РёР±Рѕ СЃРїРёСЃРѕРє С‡РµСЂРµР· Р·Р°РїСЏС‚СѓСЋ ("1,2,3")',
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="РџРµСЂРІР°СЏ СЃС‚СЂР°РЅРёС†Р° РІС‹РґР°С‡Рё (РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ 1)",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=1,
        help="РџРѕСЃР»РµРґРЅСЏСЏ СЃС‚СЂР°РЅРёС†Р° РІС‹РґР°С‡Рё РІРєР»СЋС‡РёС‚РµР»СЊРЅРѕ (РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ 1)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="РџР°СЂР°Р»Р»РµР»СЊРЅРѕСЃС‚СЊ РїСЂРё РѕР±РѕРіР°С‰РµРЅРёРё С‡РµСЂРµР· Cian API (РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="РўРѕР»СЊРєРѕ СЃРѕР±СЂР°С‚СЊ URL Рё РЅР°РїРµС‡Р°С‚Р°С‚СЊ РёС…, Р±РµР· РѕР±РѕРіР°С‰РµРЅРёСЏ Рё DVC",
    )
    args = parser.parse_args()

    rooms = _parse_rooms(args.rooms)

    urls = discover_urls(
        location=args.location,
        deal_type=args.deal_type,
        rooms=rooms,
        start_page=args.start_page,
        end_page=args.end_page,
    )
    print(f"[auto_parser] РќР°Р№РґРµРЅРѕ {len(urls)} СѓРЅРёРєР°Р»СЊРЅС‹С… URL")

    if args.dry_run:
        for url in urls:
            print(url)
        return

    if not urls:
        print("[auto_parser] РќРµС‚ URL РґР»СЏ РѕР±РѕРіР°С‰РµРЅРёСЏ, РІС‹С…РѕР¶Сѓ.")
        return

    batch_results = analyze_many(urls, max_workers=args.max_workers)

    output: List[Dict[str, Any]] = []
    ok_count = 0
    err_count = 0
    for url, data, error in batch_results:
        record: Dict[str, Any] = {"url": url}
        if error:
            record["error"] = error
            err_count += 1
        else:
            record.update(data)
            ok_count += 1
        output.append(record)

    print(f"[auto_parser] РћР±РѕРіР°С‰РµРЅРѕ СѓСЃРїРµС€РЅРѕ: {ok_count}, СЃ РѕС€РёР±РєР°РјРё: {err_count}")
    path = write_records_and_dvc_track(output, source="auto")
    print(f"[auto_parser] NDJSON СЃРѕС…СЂР°РЅС‘РЅ: {path}")
    print(json.dumps({"location": args.location, "deal_type": args.deal_type, "rooms": str(rooms), "ok": ok_count, "err": err_count}, ensure_ascii=False))


if __name__ == "__main__":
    main()

