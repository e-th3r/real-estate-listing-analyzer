# Real Estate Listing Analyzer

Проект для сбора объявлений Циан, валидации данных через Pandera, очистки
аномалий, индексации в локальной векторной базе (Chroma + multilingual
sentence-transformers) и генерации LLM-отчёта о выгодности конкретного
объявления на фоне похожих. Чистый датасет хранится в Parquet и трекается
через DVC, а сверху лежит тонкий FastAPI-веб-интерфейс.

Для хранения и синхронизации DVC-данных используется Synology NAS DS218 по SSH.

## Что умеет проект

- Авто-сбор по городу через `cianparser` + обогащение через Cian API
- Автосохранение сырых записей в `data/raw/*.ndjson`
- Очистка датасета от аномалий через Pandera и экспорт в `.parquet`
- Построение векторного индекса Chroma поверх чистого parquet
- Семантический поиск похожих объявлений
- LLM-отчёт «выгодная / средняя / переоценённая сделка» с разбором плюсов и минусов
- FastAPI-веб-приложение со статистикой и анализом по URL
- Трекинг данных в DVC (`dvc add data`)

## Актуальная структура

```text
real-estate-listing-analyzer/
├── cian_scraper.py            # Cian API + HTML парсер
├── auto_parser.py             # авто-ингест по городу
├── dataset_schema.py          # Pandera-схема и правила качества
├── build_clean_dataset.py     # NDJSON → чистый parquet + аномалии
├── vector_store.py            # Chroma + HuggingFaceEmbeddings
├── build_vector_store.py      # parquet → Chroma persistent store + dvc add
├── analyze_listing.py         # CLI: похожие квартиры + LLM-отчёт
├── report_generator.py        # промпты и LangChain pipeline для отчёта
├── webapp/
│   ├── server.py              # FastAPI: /api/stats, /api/analyze, /api/health
│   ├── index.html             # одностраничный UI
│   └── README.md
├── data.dvc
├── data/
│   ├── raw/                   # NDJSON-инжесты
│   ├── structured/            # listings_clean.parquet + listings_anomalies.json
│   └── vector_store/          # Chroma persistent storage
├── requirements.txt
└── pyproject.toml
```

## ML-пайплайн

Сквозной поток состоит из четырёх стадий; выход каждой служит входом следующей,
а всё содержимое `data/` трекается одним DVC-output'ом.

```text
[1] Ingest        →  [2] Clean         →  [3] Index         →  [4] Analyze
auto_parser.py       build_clean_       build_vector_         analyze_listing.py
cian_scraper.py      dataset.py         store.py              webapp/server.py
                     dataset_schema.py  vector_store.py       report_generator.py
                                                              ↓
data/raw/*.ndjson →  listings_clean.    data/vector_store/ →  top-K соседи
                     parquet            (Chroma collection    + LLM-отчёт
                     + anomalies.json    `cian_listings`)     (ChatOpenAI)
```

### 1. Ingest — сбор сырых данных

`auto_parser.py` итерирует поисковые страницы Циана через `cianparser`
(`location + deal_type + rooms + start_page..end_page`) и дообогащает каждое
объявление через Cian API (`scrape_cian_listing_via_cianpython`) — координаты
и часть полей `cianparser` не возвращает.

- Pandera-валидация на этом шаге **намеренно пропущена**: массовый сбор
  содержит шум, фильтрация делегирована шагу 2.
- HTML-парсер (`parse_cian_listing`) остаётся как fallback для webapp.
- Выход: `data/raw/cian_auto_{UTC_timestamp}.ndjson` + `dvc add data`.

### 2. Clean — валидация и фильтрация аномалий

`build_clean_dataset.py` читает все `data/raw/*.ndjson`, расплющивает вложенный
`structured` через `records_to_frame` и валидирует **построчно** через
`filter_valid_rows`. Невалидные строки не валят пайплайн, а уезжают в файл
аномалий.

- Чистые: `data/structured/listings_clean.parquet`
- Аномалии: `data/structured/listings_anomalies.json`
- Pandera-схема (`dataset_schema.py`, `strict=True, coerce=True`):
  - URL нормализуется и обязан матчить `.*/\d+/?$`
  - `price_rub` ∈ [500K, 5B] ₽, `total_area_m2` ∈ [8, 2000]
  - `floor` ∈ [1, 250], `floors_total` ∈ [1, 300], `floor ≤ floors_total`
  - Координаты под bbox России: lat ∈ [41, 82], lon ∈ [19, 191]
  - `description` ∈ [20, 50000] символов после схлопывания пробелов

### 3. Index — векторизация описаний

`build_vector_store.py` + `vector_store.py` строят семантический индекс поверх
чистого parquet.

- Эмбеддинги: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  по умолчанию (multilingual, CPU-friendly, без `query:`/`passage:`-префиксов).
  Можно переопределить через `--model` / `--device`.
- Хранилище: Chroma persistent collection `cian_listings` в `data/vector_store/`.
- Дедупликация по `url` (берётся последняя версия) в `frame_to_documents`.
- `listing_id` извлекается из URL регуляркой `/(\d+)/?$` и используется на
  шаге 4, чтобы исключить таргет из собственного списка соседей.
- **Полный `rmtree(persist_dir)` перед каждой сборкой** — Chroma несовместим
  между разными размерностями эмбеддингов.

#### Что лежит в `data/vector_store/`

Chroma persistent storage с одной коллекцией `cian_listings`. Каждый документ
коллекции — одно объявление; идентификатор документа = `listing_id` (числовой
ID из URL Циана), что гарантирует идемпотентный апсерт при пересборке.

Поля документа:

| Поле | Что хранит |
|---|---|
| `id` | `listing_id` — числовой ID объявления, извлечённый из URL регуляркой `/(\d+)/?$` |
| `page_content` | Текст описания (`description`) — именно он эмбеддится моделью |
| `embedding` | Нормализованный (L2) вектор размерности модели; для дефолтной MiniLM это 384 измерения |

Метаданные (`metadata`) — плоские числовые/строковые поля, по которым удобно
фильтровать или показывать соседей в UI без обращения к parquet:

| Ключ | Тип | Источник |
|---|---|---|
| `listing_id` | `str` | дублирует `id` для удобства фильтрации |
| `url` | `str` | нормализованный URL объявления |
| `price_rub` | `int` | цена в рублях |
| `total_area_m2` | `float` | общая площадь |
| `floor` | `int` | этаж |
| `floors_total` | `int` | этажность дома |
| `latitude` / `longitude` | `float` | координаты (нужны для гео-сравнения) |
| `price_per_m2` | `float` | производное `price_rub / total_area_m2`, считается на лету в `frame_to_documents` |
| `image_url` | `str` *(опц.)* | первая фотография; пишется только если поле непустое |

На диске Chroma раскладывает это в SQLite + бинарные файлы индекса HNSW.
Размерность эмбеддингов закрепляется при создании коллекции, поэтому
`build_vector_store.py` всегда делает `rmtree(persist_dir)` перед сборкой —
сменить модель без пересборки нельзя.

Поиск (`search_alternatives_with_scores`) фетчит с запасом
(`max(25, k*10 + len(excluded))`), отбрасывает таргет по `listing_id` и
дедуплицирует результаты по сигнатуре `(содержимое[:400], price, area, floor,
floors_total)` — на Циане один и тот же лот часто публикуется под разными
URL, и без этого top-K вырождается в копии одного объявления.

### 4. Analyze — поиск похожих + LLM-отчёт

Один и тот же код используется CLI и webapp.

- **Резолв таргета** (`TargetListing`): из `--url` сначала parquet, при
  отсутствии — скрейп (Cian API → HTML fallback в webapp); либо из `--text` /
  `--text-file`.
- **Поиск соседей**: `search_alternatives_with_scores` возвращает top-K пар
  `(Document, distance)` по косинусной дистанции (меньше = ближе).
- **LLM-отчёт** (опционально): `report_generator.build_default_llm` собирает
  `ChatOpenAI` (LM Studio / Ollama / OpenAI-совместимый эндпоинт) и через
  LangChain pipeline формирует структурированный вывод «выгодная / средняя /
  переоценённая» с разбором плюсов и минусов на фоне комп-сета.

### Контракт данных между стадиями

Скрейпер кладёт численные поля в вложенный `structured` dict (`price_rub`,
`total_area_m2`, `floor`, `floors_total`, `latitude`, `longitude`); `description`
и `url` — на верхнем уровне. `records_to_frame` (`dataset_schema.py`) плющит
это в плоские колонки, те же колонки становятся метаданными Chroma в
`frame_to_documents` (`vector_store.py`). **Новое поле нужно добавлять в обоих
местах** — иначе оно молча выпадет либо на валидации, либо при индексации.

## Требования

- Python 3.11+
- Git
- DVC (`dvc[ssh]`)
- Synology NAS DS218 c SSH-доступом
- Доступ к LLM-эндпоинту (LM Studio / Ollama / OpenAI-совместимый) — только для отчётов

## Установка (Windows PowerShell)

```powershell
cd C:\Users\maxim\OneDrive\Documents\GitHub\real-estate-listing-analyzer

python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "dvc[ssh]"
```

Альтернатива через `uv`:

```powershell
uv sync
```

## 1. Авто-сбор сырых данных

Скрипт `auto_parser.py` автоматически находит объявления через
[NurjahonErgashevMe/cianparser](https://github.com/NurjahonErgashevMe/cianparser)
(итерация по страницам поиска Циана), а затем обогащает каждое объявление
через Cian API (координаты, описание, этаж и т.п.).

```powershell
# Москва, продажа, 1-2-3 комнаты, первые 3 страницы поиска
python auto_parser.py --location "Москва" --deal-type sale --rooms "1,2,3" --start-page 1 --end-page 3

# Только собрать и напечатать URL, без обогащения и DVC
python auto_parser.py --location "Санкт-Петербург" --rooms all --end-page 5 --dry-run
```

После запуска `auto_parser.py`:

- Сохраняет сырые данные в `data/raw/*.ndjson`
- Выполняет `dvc add data`

Pandera-валидация **не применяется на входе**: массовый сбор неизбежно
содержит аномалии, и их фильтрация делегирована `build_clean_dataset.py`.
`cianparser` не возвращает координаты, поэтому обогащение через Cian API
обязательно — без координат строки всё равно будут отсеяны на шаге 2.

## 2. Очистка и сбор финального датасета

Скрипт: `build_clean_dataset.py`

```powershell
python build_clean_dataset.py
```

По умолчанию:

- вход: `data/raw/*.ndjson`
- чистый датасет: `data/structured/listings_clean.parquet`
- аномалии: `data/structured/listings_anomalies.json`

Кастомные пути:

```powershell
python build_clean_dataset.py `
  --input-dir data/raw `
  --output data/structured/listings_clean.parquet `
  --anomalies-output data/structured/listings_anomalies.json
```

## 3. Построение векторного индекса

Скрипт: `build_vector_store.py`. Берёт чистый parquet, считает эмбеддинги
описаний моделью `paraphrase-multilingual-MiniLM-L12-v2` (по умолчанию,
работает на CPU) и кладёт коллекцию `cian_listings` в Chroma persistent
storage.

```powershell
# Полная пересборка из data/structured/listings_clean.parquet в data/vector_store
python build_vector_store.py

# Без вызова `dvc add data` (например, для локальных экспериментов)
python build_vector_store.py --no-dvc

# Кастомная модель / GPU
python build_vector_store.py --model intfloat/multilingual-e5-base --device cuda
```

Хранилище **всегда пересоздаётся**: при разной размерности эмбеддингов схема
Chroma несовместима, поэтому скрипт делает `rmtree(persist_dir)` перед
сборкой.

## 4. Анализ конкретного объявления (CLI)

Скрипт: `analyze_listing.py`. Тянет проверяемое объявление либо из parquet
по URL, либо скрейпит онлайн, ищет топ-K похожих в Chroma и (опционально)
просит LLM сгенерировать структурированный отчёт.

```powershell
# По URL: сначала проверяем, есть ли квартира в датасете; если нет — скрейпим
python analyze_listing.py --url "https://www.cian.ru/sale/flat/328442756/"

# Только похожие, без LLM
python analyze_listing.py --url "https://www.cian.ru/sale/flat/328442756/" --no-report

# Запрос свободным текстом (без скрейпинга)
python analyze_listing.py --text "1-к квартира 38 м² у метро Сокол, 7/12, 14 млн"
python analyze_listing.py --text-file "C:\path\to\listing.txt"
```

Конфигурация LLM — через флаги (`--llm-model`, `--llm-base-url`,
`--llm-api-key`, `--llm-temperature`) или переменные окружения
`LISTING_LLM_*` / `OPENAI_API_KEY` (см. секцию ниже).

## 5. Веб-интерфейс

FastAPI-приложение в `webapp/`. Поднимает три эндпоинта поверх тех же
функций, что и CLI (`_target_from_dataset`, `_target_from_scrape`,
`build_default_llm`).

```powershell
# из корня проекта
python -m uvicorn webapp.server:app --host 127.0.0.1 --port 8000
# или напрямую
python webapp/server.py
```

Открыть: <http://127.0.0.1:8000/>

- `GET /` — статический `index.html`.
- `GET /api/stats` — счётчики по `listings_clean.parquet` и
  `listings_anomalies.json` (медиана ₽/м² и цены, всего/чистых/аномалий).
- `POST /api/analyze` — `{ "url", "k", "report", "no_scrape" }`. Сначала
  ищет URL в parquet, при отсутствии скрейпит через Cian API
  (с HTML-fallback). Если `report: true` — собирает LLM-отчёт.
- `GET /api/health` — есть ли parquet и vector store на диске.

Без parquet `/api/stats` отдаёт нули; без vector store `/api/analyze` отвечает 503.

### LLM-конфигурация

`report_generator.build_default_llm` читает переменные окружения:

```bash
export LISTING_LLM_BASE_URL="http://localhost:1234/v1"   # LM Studio / Ollama
export LISTING_LLM_MODEL="qwen2.5-7b-instruct"
export LISTING_LLM_API_KEY="local"
```

Для облачного OpenAI достаточно `OPENAI_API_KEY` + `LISTING_LLM_MODEL=gpt-4o-mini`.

## Схема Pandera и правила качества

Схема описана в `dataset_schema.py`.

Проверяются поля:

- `url`
- `price_rub`
- `total_area_m2`
- `floor`
- `floors_total`
- `latitude`
- `longitude`
- `description`

Жесткие правила:

- URL объявления Циан, нормализуется (убираются query/fragment)
- Диапазоны цены/площади/этажей/координат
- `floor <= floors_total`
- Минимальная/максимальная длина описания

Поведение:

- В `build_clean_dataset.py` строки, не прошедшие схему, отбрасываются в файл аномалий

## Настройка DVC remote (Synology DS218 по SSH)

```powershell
dvc remote add -d storage ssh://<NAS_USER>@<NAS_HOST>/volume1/dvc/real-estate-listing-analyzer
dvc remote modify storage keyfile "$env:USERPROFILE\.ssh\id_rsa"
dvc remote list --verbose
```

Проверка статуса и push:

```powershell
dvc status -c
dvc push
```

## Типовой workflow

```powershell
# 1) Собрать сырые данные в авто-режиме по городу
python auto_parser.py --location "Москва" --end-page 5

# 2) Собрать чистый parquet
python build_clean_dataset.py

# 3) Перестроить векторный индекс
python build_vector_store.py

# 4) Поднять веб-интерфейс или запустить CLI-анализ
python -m uvicorn webapp.server:app --port 8000
# либо
python analyze_listing.py --url "https://www.cian.ru/sale/flat/328442756/"

# 5) Зафиксировать и отправить
git add data.dvc
git commit -m "Update raw, cleaned and vector datasets"
dvc push
```

## Легальность

Cкрейпинг в рамках правил площадки и действующего законодательства.
