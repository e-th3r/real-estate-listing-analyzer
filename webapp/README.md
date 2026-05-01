# Webapp — Listing Analyzer UI

Тонкая FastAPI-обёртка над пайплайном `analyze_listing.py`.

## Запуск

```bash
# из корня проекта
pip install -r requirements.txt          # подтянет fastapi + uvicorn
python -m uvicorn webapp.server:app --host 127.0.0.1 --port 8000
# или просто
python webapp/server.py
```

Открыть: <http://127.0.0.1:8000/>

## Эндпоинты

- `GET /` — статический `index.html`.
- `GET /api/stats` — счётчики по `data/structured/listings_clean.parquet` и `listings_anomalies.json` (медиана ₽/м², медиана цены, всего/чистых/аномалий).
- `POST /api/analyze` — тело `{ "url": "...", "k": 5, "report": false, "no_scrape": false }`.
  - Сначала ищет URL в parquet (`_target_from_dataset`), при отсутствии — скрейпит через `_target_from_scrape` (Cian API).
  - Затем тянет топ-K похожих из векторной базы Chroma.
  - Если `report: true` — собирает LLM-отчёт через `build_default_llm` (читает `LISTING_LLM_*` env).
- `GET /api/health` — есть ли parquet и vector store на диске.

## Предполагаемое состояние диска

- `data/structured/listings_clean.parquet` — собран через `python build_clean_dataset.py`.
- `data/vector_store/` — собран через `python build_vector_store.py`.

Без parquet `/api/stats` вернёт нули; без vector store `/api/analyze` отдаст 503.

## LLM-отчёт

Конфигурируется переменными окружения, читает `report_generator.build_default_llm`:

```bash
export LISTING_LLM_BASE_URL="http://localhost:1234/v1"   # LM Studio / Ollama
export LISTING_LLM_MODEL="qwen2.5-7b-instruct"
export LISTING_LLM_API_KEY="local"
```

Или для OpenAI — `OPENAI_API_KEY` + `LISTING_LLM_MODEL=gpt-4o-mini`.
