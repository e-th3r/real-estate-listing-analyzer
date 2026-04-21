from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


# Многоязычная модель sentence-transformers, нормально работает с русским описанием
# и не требует префиксов "query:"/"passage:". Достаточно компактна для CPU.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_VECTOR_STORE_DIR = Path("data/vector_store")
COLLECTION_NAME = "cian_listings"

_LISTING_ID_RE = re.compile(r"/(\d+)/?$")


def get_embeddings(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    *,
    device: str = "cpu",
) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def listing_id_from_url(url: str) -> str:
    match = _LISTING_ID_RE.search(str(url).rstrip("/") + "/")
    if not match:
        raise ValueError(f"Не удалось извлечь listing_id из URL: {url!r}")
    return match.group(1)


def frame_to_documents(frame: pd.DataFrame) -> List[Document]:
    if frame.empty:
        return []

    # Дедупликация на случай, если одна и та же квартира попала в parquet из
    # нескольких NDJSON-инжестов.
    deduped = frame.drop_duplicates(subset=["url"], keep="last").reset_index(drop=True)

    documents: List[Document] = []
    for _, row in deduped.iterrows():
        listing_id = listing_id_from_url(row["url"])
        metadata = {
            "listing_id": listing_id,
            "url": str(row["url"]),
            "price_rub": int(row["price_rub"]),
            "total_area_m2": float(row["total_area_m2"]),
            "floor": int(row["floor"]),
            "floors_total": int(row["floors_total"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "price_per_m2": float(row["price_rub"]) / float(row["total_area_m2"]),
        }
        documents.append(
            Document(
                page_content=str(row["description"]),
                metadata=metadata,
                id=listing_id,
            )
        )
    return documents


def build_vector_store(
    parquet_path: Path,
    persist_dir: Path,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
    batch_size: int = 64,
) -> Chroma:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Чистый parquet не найден: {parquet_path}")

    frame = pd.read_parquet(parquet_path)
    documents = frame_to_documents(frame)
    if not documents:
        raise ValueError("В parquet нет валидных строк — нечего эмбеддить.")

    persist_dir.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings(model_name, device=device)
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    # Полный пересбор: вычищаем коллекцию (если уже была) и заполняем заново.
    reset = getattr(store, "reset_collection", None)
    if callable(reset):
        reset()
    else:
        store.delete_collection()
        store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )

    for start in range(0, len(documents), batch_size):
        chunk = documents[start : start + batch_size]
        store.add_documents(chunk, ids=[d.id for d in chunk])

    return store


def load_vector_store(
    persist_dir: Path,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
) -> Chroma:
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Векторное хранилище не найдено: {persist_dir}. "
            "Сначала запусти `python build_vector_store.py`."
        )
    embeddings = get_embeddings(model_name, device=device)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def _result_signature(doc: Document) -> tuple:
    md = doc.metadata
    content = " ".join(doc.page_content.split())
    if len(content) > 400:
        content = content[:400]
    return (
        content,
        md.get("price_rub"),
        md.get("total_area_m2"),
        md.get("floor"),
        md.get("floors_total"),
    )


def search_alternatives(
    store: Chroma,
    query: str,
    *,
    k: int = 5,
    exclude_ids: Optional[Iterable[str]] = None,
) -> List[Document]:
    """Возвращает топ-k похожих квартир по описанию, исключая указанные listing_id."""
    excluded = set(exclude_ids or [])
    # Берём заметный запас, потому что в сырых объявлениях много дублей
    # одного и того же лота под разными URL.
    fetch_k = max(25, k * 10 + len(excluded))
    candidates = store.similarity_search(query, k=fetch_k)
    if excluded:
        candidates = [d for d in candidates if d.metadata.get("listing_id") not in excluded]

    deduped: List[Document] = []
    seen_signatures: set[tuple] = set()
    for candidate in candidates:
        signature = _result_signature(candidate)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped.append(candidate)
        if len(deduped) >= k:
            break
    return deduped[:k]


def search_alternatives_with_scores(
    store: Chroma,
    query: str,
    *,
    k: int = 5,
    exclude_ids: Optional[Iterable[str]] = None,
) -> List[tuple[Document, float]]:
    """Аналог search_alternatives, но с оценкой расстояния (меньше — ближе)."""
    excluded = set(exclude_ids or [])
    fetch_k = max(25, k * 10 + len(excluded))
    pairs = store.similarity_search_with_score(query, k=fetch_k)
    if excluded:
        pairs = [(d, s) for d, s in pairs if d.metadata.get("listing_id") not in excluded]

    deduped: List[tuple[Document, float]] = []
    seen_signatures: set[tuple] = set()
    for doc, score in pairs:
        signature = _result_signature(doc)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped.append((doc, score))
        if len(deduped) >= k:
            break
    return deduped[:k]
