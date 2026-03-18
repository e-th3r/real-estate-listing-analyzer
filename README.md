# Real Estate Listing Analyzer 🏠📈

## 📖 О проекте

Анализатор объявлений о недвижимости с **Циан.ру** с версионированием данных через **DVC** и хранением на **Synology NAS**.

**Три типа артефактов:**

- 🏷️ **Структурированные данные**: цена, площадь, этаж, координаты
- 📝 **Текстовые описания**: свободный текст от авторов объявлений
- 🖼️ **Изображения**: фото интерьеров и планировок квартир


## 🏗️ Архитектура проекта

```
real-estate-listing-analyzer/
├── src/                   # Скрипты скрейпинга и обработки
│   ├── scraper.py         # BeautifulSoup парсер Циан
│   ├── process.py         # Парсинг → pandas DataFrame
│   └── download_images.py # Скачивание фото
├── data/                  # DVC управляемые данные
│   ├── raw/              # Сырые JSON
│   ├── structured/       # listings.parquet
│   └── images/           # Фото квартир
├── dvc.yaml              # ML пайплайн
├── .dvc/                 # DVC метафайлы (git)
└── params.yaml           # Настройки
```


## 🚀 Быстрый старт

### 1. Клонирование и окружение

```powershell
git clone https://github.com/maxim-perchikov/real-estate-listing-analyzer.git
cd real-estate-listing-analyzer

# Создай виртуальное окружение
python -m venv .venv
.venv\Scripts\Activate.ps1

# Установи зависимости
pip install -r requirements.txt
```


### 2. Инициализация DVC + NAS

```powershell
# Инициализация DVC
dvc init

# Подключение Synology NAS (SSH)
dvc remote add -d nas-ssh ssh://Maxim@91.77.167.XXX/home/Maxim/dvc-data
dvc remote modify nas-ssh keyfile "$env:USERPROFILE\.ssh\dvc_synology"
dvc push
```


### 3. Запуск пайплайна

```powershell
# Полный пайплайн (скрейпинг → обработка → фото)
dvc repro

# Синхронизация с NAS
dvc push -j 1
```


## 📊 Результат пайплайна

| Артефакт | Путь | Описание |
| :-- | :-- | :-- |
| `listings.parquet` | `data/structured/` | 1000+ объявлений с ценой, площадью, этажом |
| `descriptions.json` | `data/raw/` | Полные тексты объявлений |
| `images/` | `data/images/` | Фото интерьеров (jpg/png) |

## 🔧 Настройка с нуля

### Требования:

```
Python 3.10+
Git 2.40+
DVC 3.5+
Synology NAS (DSM 7+ с SSH/SFTP)
```


### Зависимости (`requirements.txt`):

```txt
requests==2.31.0
beautifulsoup4==4.12.2
lxml==4.9.3
pandas==2.1.4
pyarrow==14.0.1
tqdm==4.66.1
```


## 🌐 Доступ с других устройств

1. **DDNS**: `maxim-nas.synology.me`
2. **Скопируй приватный ключ**: `C:\Users\maxim\.ssh\dvc_synology`
3. **Подключись**: `dvc remote add nas-ssh ssh://Maxim@maxim-nas.synology.me/...`

## 📈 Пример данных

```bash
# Структурированные данные
$ dvc desc data/structured/listings.parquet
size: 2.4MB, 1250 строк

# Схема колонок:
listing_id, price_rub, total_area, floor, district, scraped_at
12345678, 12500000, 56.2, 5/12, Хамовники, 2026-03-18
```


## 🔄 Полезные команды DVC

```powershell
dvc pull          # Скачать данные с NAS
dvc push -j 1     # Загрузить на NAS
dvc dag           # Показать пайплайн
dvc metrics show  # Метрики качества
dvc status        # Статус изменений
```


## 🛡️ Лицензия и легальность

⚠️ **Скрейпинг Циан.ru только для образовательных целей!**

- Соблюдай `robots.txt`
- Добавляй задержки между запросами
- Не перегружай сервер

**Лицензия проекта**: MIT

***

**Скрейпер → DVC → Synology NAS → Анализ недвижимости!** 🏠→📊→☁️

*Автор: Maxim Perchikov | Data Science Student | Moscow, 2026*

