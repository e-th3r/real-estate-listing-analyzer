from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_MODEL_PATH = Path("data/models/price_catboost.cbm")
DEFAULT_METRICS_PATH = Path("data/models/price_catboost_metrics.json")

FEATURE_COLUMNS: List[str] = [
    "total_area_m2",
    "floor",
    "floors_total",
    "floor_ratio",
    "latitude",
    "longitude",
    "description_length",
]


def _build_features(frame: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=frame.index)
    feats["total_area_m2"] = frame["total_area_m2"].astype(float)
    feats["floor"] = frame["floor"].astype(float)
    feats["floors_total"] = frame["floors_total"].astype(float)
    feats["floor_ratio"] = feats["floor"] / feats["floors_total"]
    feats["latitude"] = frame["latitude"].astype(float)
    feats["longitude"] = frame["longitude"].astype(float)
    feats["description_length"] = (
        frame["description"].astype(str).str.len().astype(float)
    )
    return feats[FEATURE_COLUMNS]


def _spatial_groups(
    frame: pd.DataFrame,
    *,
    n_clusters: int,
    random_state: int,
) -> np.ndarray:
    """Кластеризует объявления по координатам — для честной CV по районам.

    На случайном split CatBoost тривиально запоминает координаты соседей и
    показывает оптимистичный MAPE; GroupKFold по этим кластерам не даёт ему
    подсматривать в тот же район.
    """
    from sklearn.cluster import KMeans

    coords = frame[["latitude", "longitude"]].astype(float).to_numpy()
    actual_clusters = max(2, min(n_clusters, len(coords) // 20))
    if len(coords) < actual_clusters * 2:
        return np.zeros(len(coords), dtype=int)
    km = KMeans(n_clusters=actual_clusters, random_state=random_state, n_init=10)
    return km.fit_predict(coords)


def train_price_model(
    parquet_path: Path,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    n_clusters: int = 10,
    n_splits: int = 5,
    iterations: int = 1500,
    learning_rate: float = 0.05,
    depth: int = 6,
    random_state: int = 42,
) -> Dict[str, Any]:
    from catboost import CatBoostRegressor
    from sklearn.model_selection import GroupKFold

    if not parquet_path.exists():
        raise FileNotFoundError(f"Чистый parquet не найден: {parquet_path}")

    frame = pd.read_parquet(parquet_path)
    if frame.empty:
        raise ValueError(f"Пустой parquet: {parquet_path}")

    X = _build_features(frame)
    y_log = np.log1p(frame["price_rub"].astype(float).to_numpy())
    groups = _spatial_groups(frame, n_clusters=n_clusters, random_state=random_state)

    n_groups = int(len(set(groups)))
    cv_splits = min(n_splits, n_groups) if n_groups > 1 else 0

    cv_metrics: List[Dict[str, float]] = []
    if cv_splits >= 2:
        gkf = GroupKFold(n_splits=cv_splits)
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y_log, groups)):
            fold_model = CatBoostRegressor(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                loss_function="RMSE",
                random_seed=random_state,
                verbose=0,
                early_stopping_rounds=100,
            )
            fold_model.fit(
                X.iloc[train_idx],
                y_log[train_idx],
                eval_set=(X.iloc[val_idx], y_log[val_idx]),
            )
            pred_log = fold_model.predict(X.iloc[val_idx])
            true_price = np.expm1(y_log[val_idx])
            pred_price = np.expm1(pred_log)
            mape = float(np.mean(np.abs(pred_price - true_price) / true_price))
            rmse_log = float(np.sqrt(np.mean((pred_log - y_log[val_idx]) ** 2)))
            cv_metrics.append({"fold": fold_idx, "mape": mape, "rmse_log": rmse_log})

    final = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="RMSE",
        random_seed=random_state,
        verbose=0,
    )
    final.fit(X, y_log)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(str(model_path))

    metrics: Dict[str, Any] = {
        "n_rows": int(len(frame)),
        "n_groups": n_groups,
        "features": FEATURE_COLUMNS,
        "cv": cv_metrics,
        "cv_mape_mean": (
            float(np.mean([m["mape"] for m in cv_metrics])) if cv_metrics else None
        ),
        "cv_rmse_log_mean": (
            float(np.mean([m["rmse_log"] for m in cv_metrics])) if cv_metrics else None
        ),
        "feature_importance": dict(
            zip(FEATURE_COLUMNS, final.feature_importances_.tolist())
        ),
        "model_path": str(model_path),
        "target": "log1p(price_rub)",
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def load_price_model(model_path: Path = DEFAULT_MODEL_PATH):
    if not model_path.exists():
        return None
    from catboost import CatBoostRegressor

    model = CatBoostRegressor()
    model.load_model(str(model_path))
    return model


def predict_fair_price(
    model,
    *,
    total_area_m2: Optional[float],
    floor: Optional[int],
    floors_total: Optional[int],
    latitude: Optional[float],
    longitude: Optional[float],
    description: str = "",
) -> Optional[float]:
    """Возвращает оценку справедливой цены (₽) или None, если фичей не хватает.

    Без координат модель деградирует кардинально (район — главный драйвер цены),
    поэтому в этом случае честно отказываемся.
    """
    if (
        total_area_m2 is None
        or floor is None
        or floors_total is None
        or latitude is None
        or longitude is None
    ):
        return None
    if float(floors_total) <= 0:
        return None
    features = pd.DataFrame(
        [
            {
                "total_area_m2": float(total_area_m2),
                "floor": float(floor),
                "floors_total": float(floors_total),
                "floor_ratio": float(floor) / float(floors_total),
                "latitude": float(latitude),
                "longitude": float(longitude),
                "description_length": float(len(description or "")),
            }
        ]
    )[FEATURE_COLUMNS]
    pred_log = float(model.predict(features)[0])
    return float(np.expm1(pred_log))
