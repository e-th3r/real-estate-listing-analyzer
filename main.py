import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from cian_scraper import parse_cian_listing, scrape_cian_listing_via_cianpython
from dataset_schema import validate_scraped_records


def analyze_cian_listing(url: str) -> Dict[str, Any]:
    data = scrape_cian_listing_via_cianpython(url)
    data["url"] = url
    return data


def analyze_cian_html_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    html = p.read_text(encoding="utf-8", errors="replace")
    data = parse_cian_listing(html)
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


def write_records_and_dvc_track(records: List[Dict[str, Any]], *, source: str) -> Path:
    """
    Пишет NDJSON в data/raw и запускает `dvc add data`. Без Pandera-валидации —
    используется auto-режимом для массового инжеста (фильтрация строк происходит
    в build_clean_dataset.py).
    """
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
            result = subprocess.run(
                cmd,
                cwd=root,
                capture_output=True,
                text=True,
            )
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


def _serialize_to_dvc(records: List[Dict[str, Any]], *, source: str) -> Path:
    validate_scraped_records(records)
    return write_records_and_dvc_track(records, source=source)


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
    args = parser.parse_args()

    if args.url:
        result = analyze_cian_listing(args.url)
        _serialize_to_dvc([result], source="single")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.html_file:
        result = analyze_cian_html_file(args.html_file)
        _serialize_to_dvc([result], source="html")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    urls = _read_urls_from_file(args.urls_file)
    batch_results = analyze_many(urls, max_workers=args.max_workers)
    output: List[Dict[str, Any]] = []

    if args.ndjson:
        for url, data, error in batch_results:
            record: Dict[str, Any] = {"url": url}
            if error:
                record["error"] = error
            else:
                record.update(data)
            output.append(record)
            print(json.dumps(record, ensure_ascii=False))
    else:
        for url, data, error in batch_results:
            record: Dict[str, Any] = {"url": url}
            if error:
                record["error"] = error
            else:
                record.update(data)
            output.append(record)
        print(json.dumps(output, ensure_ascii=False, indent=2))
    _serialize_to_dvc(output, source="batch")


if __name__ == "__main__":
    main()
