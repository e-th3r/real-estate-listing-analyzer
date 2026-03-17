import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from headless_scraper import scrape_domclick_headless


def analyze_cian_listing(url: str, *, headed: bool = False, user_data_dir: str | None = None) -> Dict[str, Any]:
    # Историческое имя функции сохранено, но теперь анализ идёт по ДомКлик
    # через headless-браузер (Playwright).
    data = scrape_domclick_headless(url, headless=not headed, user_data_dir=user_data_dir)
    data["url"] = url
    return data


def analyze_domclick_html_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    html = p.read_text(encoding="utf-8", errors="replace")
    # В режиме HTML-файла оставляем старый парсинг без браузера
    from domclick_scraper import parse_domclick_listing  # локальный импорт
    data = parse_domclick_listing(html)
    data["url"] = str(p.resolve())
    return data


def _read_urls_from_file(path: str) -> List[str]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in raw if line.strip() and not line.lstrip().startswith("#")]


def analyze_many(urls: Iterable[str], *, max_workers: int = 8) -> List[Tuple[str, Dict[str, Any], str]]:
    """
    Возвращает список (url, result_dict | None, error_message | "").
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real estate listing analyzer for Cian pages."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--url",
        type=str,
        help="URL одного объявления на Циане",
    )
    group.add_argument(
        "--html-file",
        type=str,
        help="Путь к сохранённому HTML (например, view-source_*.html)",
    )
    group.add_argument(
        "--urls-file",
        type=str,
        help="Путь к файлу со списком URL (по одному на строку)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Максимальное количество параллельных запросов (по умолчанию 8)",
    )
    parser.add_argument(
        "--ndjson",
        action="store_true",
        help="Выводить результат как NDJSON (по одному объявлению в строке)",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Запускать браузер НЕ headless (полезно для антибота)",
    )
    parser.add_argument(
        "--user-data-dir",
        type=str,
        default=None,
        help="Папка профиля Playwright/Chromium для постоянных cookie/сессий",
    )

    args = parser.parse_args()

    if args.url:
        result = analyze_cian_listing(args.url, headed=args.headed, user_data_dir=args.user_data_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.html_file:
        result = analyze_domclick_html_file(args.html_file)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    urls = _read_urls_from_file(args.urls_file)
    batch_results = analyze_many(urls, max_workers=args.max_workers)

    if args.ndjson:
        # По одной записи на строку
        for url, data, error in batch_results:
            record: Dict[str, Any] = {"url": url}
            if error:
                record["error"] = error
            else:
                record.update(data)
            print(json.dumps(record, ensure_ascii=False))
    else:
        # Один большой JSON
        output = []
        for url, data, error in batch_results:
            record: Dict[str, Any] = {"url": url}
            if error:
                record["error"] = error
            else:
                record.update(data)
            output.append(record)
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
