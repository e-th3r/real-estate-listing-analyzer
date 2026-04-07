# Real Estate Listing Analyzer

Проект для сбора объявлений Циан, валидации данных через Pandera, очистки аномалий и сохранения чистого датасета в Parquet с трекингом через DVC.

Для хранения и синхронизации DVC-данных используется Synology NAS DS218 по SSH.

## Что умеет проект

- Скрейпинг объявления Циан по URL
- Разбор локального HTML объявления
- Автосохранение сырых записей в `data/raw/*.ndjson`
- Валидация схемы и правил качества через Pandera
- Очистка датасета от аномалий и экспорт в `.parquet`
- Трекинг данных в DVC (`dvc add data`)

## Актуальная структура

```text
real-estate-listing-analyzer/
├── cian_scraper.py
├── main.py
├── dataset_schema.py
├── build_clean_dataset.py
├── data.dvc
├── data/
│   ├── raw/
│   └── structured/
├── requirements.txt
└── pyproject.toml
```

## Требования

- Python 3.11+
- Git
- DVC (`dvc[ssh]`)
- Synology NAS DS218 c SSH-доступом

## Установка (Windows PowerShell)

```powershell
cd C:\Users\maxim\OneDrive\Documents\GitHub\real-estate-listing-analyzer

python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "dvc[ssh]"
```

## 1. Сбор сырых данных

Точка входа: `main.py`

```powershell
python main.py --help
```

Режимы запуска:

```powershell
# Один URL
python main.py --url "https://www.cian.ru/sale/flat/328442756/"

# Batch из файла URL
python main.py --urls-file "C:\path\to\urls.txt"
python main.py --urls-file "C:\path\to\urls.txt" --ndjson

# Разбор локального HTML
python main.py --html-file "C:\path\to\listing.html"
```

После запуска `main.py`:

- Проверяет данные по Pandera-схеме (`dataset_schema.py`)
- Сохраняет сырые данные в `data/raw/*.ndjson`
- Выполняет `dvc add data`

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

- В `main.py` нарушение схемы прерывает сохранение в DVC
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
# 1) Собрать сырые данные
python main.py --urls-file "C:\path\to\urls.txt" --ndjson

# 2) Собрать чистый parquet
python build_clean_dataset.py

# 3) Зафиксировать и отправить
git add data.dvc
git commit -m "Update raw and cleaned datasets"
dvc push
```

## Легальность

Используй скрейпинг в рамках правил площадки и действующего законодательства.
