# Real Estate Listing Analyzer

Скрипт для анализа объявлений Циан с автоматической сериализацией результатов в `data/raw` и трекингом данных через DVC.
Для хранения и синхронизации DVC-данных используется Synology NAS DS218 по SSH.

## Что делает проект

- Получает данные объявления по URL Циан через API-скрейпер
- Поддерживает разбор сохраненного HTML-файла
- Возвращает данные в JSON-формате
- Автоматически сохраняет результаты в `data/raw/*.ndjson`
- Автоматически выполняет `dvc add data` после каждого запуска

## Актуальная структура

```text
real-estate-listing-analyzer/
├── cian_scraper.py
├── main.py
├── data.dvc
├── data/
│   └── raw/
├── requirements.txt
└── pyproject.toml
```

## Требования

- Python 3.11+
- Git
- DVC (для хранения и пуша данных в Synology NAS DS218 по SSH)

## Быстрый старт (Windows PowerShell)

```powershell
cd C:\Users\maxim\OneDrive\Documents\GitHub\real-estate-listing-analyzer

python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "dvc[ssh]"
```

## Точка входа

- Основная точка входа: `main.py`
- Запуск:

```powershell
python main.py --help
```

## Режимы запуска

### 1) Один URL

```powershell
python main.py --url "https://www.cian.ru/sale/flat/292125772/"
```

### 2) Пакет URL-ов

```powershell
python main.py --urls-file "C:\path\to\urls.txt"
```

С NDJSON-выводом в консоль:

```powershell
python main.py --urls-file "C:\path\to\urls.txt" --ndjson
```

### 3) Локальный HTML

```powershell
python main.py --html-file "C:\path\to\listing.html"
```

## Что сохраняется в data

После каждого запуска создается файл в `data/raw`:

- `cian_single_<UTC_TIMESTAMP>.ndjson`
- `cian_batch_<UTC_TIMESTAMP>.ndjson`
- `cian_html_<UTC_TIMESTAMP>.ndjson`

Каждая строка NDJSON — отдельный сериализованный объект объявления.

## Интеграция с DVC

Встроенный пайплайн в `main.py`:

1. Скрапинг данных
2. Сериализация в `data/raw`
3. `dvc add data`

Если DVC не установлен или не найден, скрипт вернет понятную ошибку.
Удаленное DVC-хранилище настроено на Synology NAS DS218 по SSH.

## Настройка DVC remote (SSH)

```powershell
dvc remote add -d storage ssh://<USER>@<HOST>/<ABSOLUTE_PATH_ON_SERVER>
dvc remote modify storage keyfile "$env:USERPROFILE\.ssh\id_rsa"
dvc remote list --verbose
```

Пример для Synology NAS DS218:

```powershell
dvc remote add -d storage ssh://<NAS_USER>@<NAS_HOST>/volume1/dvc/real-estate-listing-analyzer
dvc remote modify storage keyfile "$env:USERPROFILE\.ssh\id_rsa"
```

Проверка синхронизации:

```powershell
dvc status -c
dvc push
```

## Типовой рабочий цикл

```powershell
python main.py --urls-file "C:\path\to\urls.txt" --ndjson
git add data.dvc
git commit -m "Update scraped data snapshot"
dvc push
```

## Легальность

Скрейпинг выполняй с учетом правил площадки и действующего законодательства.
