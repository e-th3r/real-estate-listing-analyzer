import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from dataset_schema import coerce_dataset_types, filter_valid_rows, records_to_frame


def _read_ndjson_files(input_dir: Path) -> List[Dict[str, Any]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Папка с сырыми данными не найдена: {input_dir}")

    records: List[Dict[str, Any]] = []
    for file_path in sorted(input_dir.glob("*.ndjson")):
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_clean_dataset(input_dir: Path, output_path: Path, anomalies_path: Path | None) -> None:
    records = _read_ndjson_files(input_dir)
    raw_df = records_to_frame(records)
    typed_df = coerce_dataset_types(raw_df)
    clean_df, anomalies_df = filter_valid_rows(typed_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(output_path, index=False)

    if anomalies_path is not None:
        anomalies_path.parent.mkdir(parents=True, exist_ok=True)
        anomalies_df.to_json(anomalies_path, orient="records", force_ascii=False, indent=2)

    print(f"Всего сырых записей: {len(raw_df)}")
    print(f"Валидных записей: {len(clean_df)}")
    print(f"Аномалий отсеяно: {len(anomalies_df)}")
    print(f"Чистый датасет: {output_path}")
    if anomalies_path is not None:
        print(f"Лог аномалий: {anomalies_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сборка чистого датасета из NDJSON с валидацией Pandera.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw",
        help="Папка с сырыми NDJSON файлами.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/structured/listings_clean.parquet",
        help="Путь для сохранения чистого parquet-файла.",
    )
    parser.add_argument(
        "--anomalies-output",
        type=str,
        default="data/structured/listings_anomalies.json",
        help="Путь для сохранения списка отсеянных аномалий.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    anomalies_path = Path(args.anomalies_output) if args.anomalies_output else None

    build_clean_dataset(input_dir, output_path, anomalies_path)


if __name__ == "__main__":
    main()
