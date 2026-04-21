import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from vector_store import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_STORE_DIR,
    build_vector_store,
)


def _run_dvc_add(cwd: Path) -> None:
    """Повторяет логику main.write_records_and_dvc_track: пробует dvc через
    несколько лаунчеров, потому что в Windows-окружении уверенно работает не
    каждый из них.
    """
    candidates = [
        ["dvc", "add", "data"],
        [sys.executable, "-m", "dvc", "add", "data"],
        ["uv", "run", "python", "-m", "dvc", "add", "data"],
    ]
    last_result: subprocess.CompletedProcess[str] | None = None
    for cmd in candidates:
        try:
            result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        except FileNotFoundError:
            continue
        last_result = result
        if result.returncode == 0:
            return
    stdout = last_result.stdout.strip() if last_result else ""
    stderr = last_result.stderr.strip() if last_result else ""
    raise RuntimeError(
        "Не удалось выполнить `dvc add data`. Запусти вручную. "
        f"stdout: {stdout} stderr: {stderr}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Построить локальное векторное хранилище ChromaDB из чистого parquet-датасета "
            "и зафиксировать его под DVC."
        ),
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default="data/structured/listings_clean.parquet",
        help="Путь к чистому parquet-датасету.",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=str(DEFAULT_VECTOR_STORE_DIR),
        help="Папка для Chroma persistent storage. По умолчанию data/vector_store.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="HuggingFace sentence-transformers модель для эмбеддингов.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Устройство для вычисления эмбеддингов (cpu / cuda).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Размер батча при добавлении документов в Chroma.",
    )
    parser.add_argument(
        "--no-dvc",
        action="store_true",
        help="Не запускать `dvc add data` после сборки (только построить локально).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    parquet_path = (root / args.parquet).resolve() if not Path(args.parquet).is_absolute() else Path(args.parquet)
    persist_dir = (root / args.persist_dir).resolve() if not Path(args.persist_dir).is_absolute() else Path(args.persist_dir)

    # Полностью пересоздаём хранилище, чтобы избежать конфликтов схемы Chroma
    # между моделями эмбеддингов с разной размерностью.
    if persist_dir.exists():
        shutil.rmtree(persist_dir)

    print(f"Источник: {parquet_path}")
    print(f"Модель эмбеддингов: {args.model} (device={args.device})")
    print(f"Папка хранилища: {persist_dir}")

    store = build_vector_store(
        parquet_path=parquet_path,
        persist_dir=persist_dir,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )
    count = store._collection.count()
    print(f"Документов в коллекции: {count}")

    if args.no_dvc:
        print("Пропускаем `dvc add data` (--no-dvc).")
        return

    print("Запускаем `dvc add data`...")
    _run_dvc_add(root)
    print("DVC-трекинг обновлён. Не забудь `git add data.dvc` и `dvc push`.")


if __name__ == "__main__":
    main()
