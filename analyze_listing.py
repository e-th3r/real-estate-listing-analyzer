import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from dataset_schema import _normalize_cian_url
from env_loader import load_project_env

load_project_env()
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
    search_alternatives_with_scores,
)


def _target_from_dataset(parquet_path: Path, url: str) -> Optional[TargetListing]:
    if not parquet_path.exists():
        return None
    frame = pd.read_parquet(parquet_path)
    if frame.empty:
        return None
    norm = _normalize_cian_url(url)
    matches = frame[frame["url"] == norm]
    if matches.empty:
        return None
    row = matches.iloc[-1]
    return TargetListing(
        description=str(row["description"]),
        listing_id=listing_id_from_url(str(row["url"])),
        url=str(row["url"]),
        price_rub=float(row["price_rub"]),
        total_area_m2=float(row["total_area_m2"]),
        floor=int(row["floor"]),
        floors_total=int(row["floors_total"]),
    )


def _target_from_scrape(url: str) -> TargetListing:
    # Импорт по требованию: scrape тянет сетевые зависимости и cianparser.
    from cian_scraper import scrape_cian_listing_via_cianpython

    data = scrape_cian_listing_via_cianpython(url)
    description = str(data.get("description") or "")
    structured = data.get("structured") or {}
    norm_url = _normalize_cian_url(url)
    try:
        listing_id = listing_id_from_url(norm_url)
    except ValueError:
        listing_id = None
    return TargetListing(
        description=description,
        listing_id=listing_id,
        url=norm_url,
        price_rub=structured.get("price_rub"),
        total_area_m2=structured.get("total_area_m2"),
        floor=structured.get("floor"),
        floors_total=structured.get("floors_total"),
        images=list(data.get("images") or []),
    )


def _resolve_target(args: argparse.Namespace, parquet_path: Path) -> TargetListing:
    if args.url:
        from_dataset = _target_from_dataset(parquet_path, args.url)
        if from_dataset is not None:
            print(f"[i] Квартира найдена в датасете: {from_dataset.url}")
            return from_dataset
        if args.no_scrape:
            raise SystemExit(
                f"URL {args.url} не найден в parquet, а --no-scrape запрещает онлайн-запрос."
            )
        print("[i] Квартиры нет в датасете — подтягиваю через Cian API...")
        return _target_from_scrape(args.url)

    if args.text:
        return TargetListing(description=args.text)

    if args.text_file:
        path = Path(args.text_file)
        return TargetListing(description=path.read_text(encoding="utf-8"))

    raise SystemExit("Нужно задать один из: --url, --text, --text-file.")


def _fmt_int(value) -> str:
    return f"{int(value):,}".replace(",", " ")


def _print_alternatives(target: TargetListing, pairs) -> None:
    print()
    print(f"=== Топ-{len(pairs)} похожих объявлений ===")
    for rank, (doc, score) in enumerate(pairs, start=1):
        md = doc.metadata
        price_str = _fmt_int(md.get("price_rub", 0))
        area = md.get("total_area_m2")
        ppm_str = _fmt_int(md.get("price_per_m2", 0))
        snippet = doc.page_content.strip()
        if len(snippet) > 220:
            snippet = snippet[:220].rstrip() + "..."
        print(
            f"\n#{rank}  distance={score:.3f}\n"
            f"  URL:    {md.get('url')}\n"
            f"  Цена:   {price_str} ₽   Площадь: {area} м²   {ppm_str} ₽/м²\n"
            f"  Этаж:   {md.get('floor')}/{md.get('floors_total')}\n"
            f"  Текст:  {snippet}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Поиск похожих квартир в локальной векторной базе и генерация "
            "LLM-отчёта о выгодности конкретного объявления."
        ),
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", type=str, help="URL объявления Циан, которое проверяем.")
    src.add_argument(
        "--text",
        type=str,
        help="Произвольный текстовый запрос или описание квартиры.",
    )
    src.add_argument("--text-file", type=str, help="Путь к файлу с описанием квартиры / запросом.")

    parser.add_argument(
        "--parquet",
        type=str,
        default="data/structured/listings_clean.parquet",
        help="Где искать квартиру по URL (только для --url).",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=str(DEFAULT_VECTOR_STORE_DIR),
        help="Папка с Chroma persistent storage.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="HuggingFace модель эмбеддингов (должна совпадать с использованной при сборке).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Устройство для вычисления эмбеддингов запроса (cpu / cuda).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Сколько похожих объявлений возвращать.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Только показать похожие, без вызова LLM.",
    )
    parser.add_argument(
        "--no-scrape",
        action="store_true",
        help="Не лезть в сеть, если URL отсутствует в датасете.",
    )

    parser.add_argument("--llm-model", type=str, default=None, help="Модель LLM.")
    parser.add_argument("--llm-base-url", type=str, default=None, help="Base URL LLM.")
    parser.add_argument("--llm-api-key", type=str, default=None, help="API key LLM.")
    parser.add_argument("--llm-temperature", type=float, default=0.2, help="Температура LLM.")

    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    parquet_path = root / args.parquet if not Path(args.parquet).is_absolute() else Path(args.parquet)
    persist_dir = root / args.persist_dir if not Path(args.persist_dir).is_absolute() else Path(args.persist_dir)

    target = _resolve_target(args, parquet_path)

    print(f"[i] Загружаем векторное хранилище: {persist_dir}")
    store = load_vector_store(persist_dir, model_name=args.model, device=args.device)

    exclude = [target.listing_id] if target.listing_id else None
    pairs = search_alternatives_with_scores(
        store,
        query=target.description,
        k=args.k,
        exclude_ids=exclude,
    )
    _print_alternatives(target, pairs)

    if args.no_report:
        return

    print("\n[i] Запрашиваем LLM-отчёт...")
    llm = build_default_llm(
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        temperature=args.llm_temperature,
    )
    report = generate_report(target, store, llm, k=args.k)
    print("\n=== Отчёт LLM ===\n")
    print(report)


if __name__ == "__main__":
    main()
