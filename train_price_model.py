import argparse
from pathlib import Path

from price_model import (
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    train_price_model,
)


def _resolve(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Обучает CatBoostRegressor для оценки справедливой цены квартиры. "
            "Кросс-валидация — GroupKFold по пространственным кластерам, чтобы "
            "модель не подсматривала цены соседей того же района."
        ),
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default="data/structured/listings_clean.parquet",
        help="Чистый датасет, на котором обучаемся.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Куда сохранить .cbm модель.",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=str(DEFAULT_METRICS_PATH),
        help="Куда сохранить JSON с метриками и feature importance.",
    )
    parser.add_argument("--clusters", type=int, default=10, help="Сколько KMeans-кластеров для группировки CV.")
    parser.add_argument("--folds", type=int, default=5, help="Число фолдов GroupKFold.")
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    parquet_path = _resolve(root, args.parquet)
    model_path = _resolve(root, args.model_path)
    metrics_path = _resolve(root, args.metrics_path)

    print(f"[i] Тренируем на {parquet_path}")
    metrics = train_price_model(
        parquet_path,
        model_path=model_path,
        metrics_path=metrics_path,
        n_clusters=args.clusters,
        n_splits=args.folds,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        random_state=args.random_state,
    )

    print(f"[+] Модель сохранена: {model_path}")
    print(f"[+] Метрики:          {metrics_path}")
    print(f"    Объявлений:        {metrics['n_rows']}")
    print(f"    Гео-кластеров:     {metrics['n_groups']}")
    if metrics["cv_mape_mean"] is not None:
        print(f"    CV MAPE по районам: {metrics['cv_mape_mean'] * 100:.2f}%")
        print(f"    CV RMSE (log):      {metrics['cv_rmse_log_mean']:.4f}")
    else:
        print("    CV пропущена — мало гео-кластеров для GroupKFold.")
    print("    Feature importance:")
    importance = sorted(
        metrics["feature_importance"].items(), key=lambda x: -x[1]
    )
    for name, imp in importance:
        print(f"      {name:>22}: {imp:6.2f}")


if __name__ == "__main__":
    main()
