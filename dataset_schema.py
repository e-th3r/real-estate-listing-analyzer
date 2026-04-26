from __future__ import annotations

from typing import Any, Dict, List, Tuple
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

_DATASET_SCHEMA = pa.DataFrameSchema(
    {
        "url": pa.Column(
            str,
            nullable=False,
            checks=[
                pa.Check.str_startswith("http"),
                pa.Check.str_contains(r"cian\.ru"),
                pa.Check.str_matches(r".*/\d+/?$"),
            ],
        ),
        "price_rub": pa.Column(
            int,
            nullable=False,
            checks=[
                pa.Check.ge(500_000),
                pa.Check.le(5_000_000_000),
            ],
        ),
        "total_area_m2": pa.Column(
            float,
            nullable=False,
            checks=[
                pa.Check.ge(8.0),
                pa.Check.le(2_000.0),
            ],
        ),
        "floor": pa.Column(
            int,
            nullable=False,
            checks=[
                pa.Check.ge(1),
                pa.Check.le(250),
            ],
        ),
        "floors_total": pa.Column(
            int,
            nullable=False,
            checks=[
                pa.Check.ge(1),
                pa.Check.le(300),
            ],
        ),
        "latitude": pa.Column(
            float,
            nullable=False,
            checks=[
                pa.Check.ge(41.0),
                pa.Check.le(82.0),
            ],
        ),
        "longitude": pa.Column(
            float,
            nullable=False,
            checks=[
                pa.Check.ge(19.0),
                pa.Check.le(191.0),
            ],
        ),
        "description": pa.Column(
            str,
            nullable=False,
            checks=[
                pa.Check.str_length(min_value=20, max_value=50_000),
            ],
        ),
        "image_url": pa.Column(
            str,
            nullable=True,
            required=False,
        ),
    },
    checks=[
        pa.Check(
            lambda df: df["floor"] <= df["floors_total"],
            error="floor должен быть меньше или равен floors_total",
        ),
    ],
    strict=True,
    coerce=True,
)


DATASET_COLUMNS = [
    "url",
    "price_rub",
    "total_area_m2",
    "floor",
    "floors_total",
    "latitude",
    "longitude",
    "description",
    "image_url",
]


def records_to_frame(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for record in records:
        if "error" in record:
            continue
        structured = record.get("structured")
        if not isinstance(structured, dict):
            raise ValueError("В успешной записи отсутствует объект `structured`.")
        images = record.get("images") or []
        first_image = images[0] if isinstance(images, list) and images else None
        rows.append(
            {
                "url": record.get("url"),
                "price_rub": structured.get("price_rub"),
                "total_area_m2": structured.get("total_area_m2"),
                "floor": structured.get("floor"),
                "floors_total": structured.get("floors_total"),
                "latitude": structured.get("latitude"),
                "longitude": structured.get("longitude"),
                "description": record.get("description"),
                "image_url": first_image,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=DATASET_COLUMNS)
    return frame


def coerce_dataset_types(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    result = frame.copy()
    result["url"] = result["url"].astype("string")
    result["url"] = result["url"].map(_normalize_cian_url)
    result["description"] = (
        result["description"]
        .astype("string")
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    if "image_url" in result.columns:
        result["image_url"] = result["image_url"].astype("string")
    else:
        result["image_url"] = pd.Series([pd.NA] * len(result), dtype="string")

    numeric_columns = [
        "price_rub",
        "total_area_m2",
        "floor",
        "floors_total",
        "latitude",
        "longitude",
    ]
    for col in numeric_columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    return result


def _normalize_cian_url(url: Any) -> str:
    if url is None:
        return ""
    raw = str(url).strip()
    if not raw:
        return ""

    parts = urlsplit(raw)
    # Убираем query и fragment, чтобы схема валидировала канонический URL объявления.
    normalized_path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, normalized_path + "/", "", ""))


def filter_valid_rows(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), pd.DataFrame(columns=["row_index", "reason", *DATASET_COLUMNS])

    valid_rows: List[pd.DataFrame] = []
    anomalies: List[Dict[str, Any]] = []

    for idx, row in frame.iterrows():
        row_df = row.to_frame().T.reset_index(drop=True)
        try:
            validated = _DATASET_SCHEMA.validate(row_df, lazy=True)
            valid_rows.append(validated)
        except SchemaErrors as exc:
            reason = exc.failure_cases.head(5).to_string(index=False)
            row_payload = row.to_dict()
            anomalies.append({"row_index": int(idx), "reason": reason, **row_payload})

    clean_df = pd.concat(valid_rows, ignore_index=True) if valid_rows else pd.DataFrame(columns=DATASET_COLUMNS)
    anomalies_df = pd.DataFrame(anomalies)
    return clean_df, anomalies_df


