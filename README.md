# Paper Helper

Телеграм-бот, помогающий анализировать статьи с arXiv.

Команда:

- Шумилов Тимофей Николаевич
- Артамонов Никита Николаевич
- Нилов Лев Евгеньевич

---

## Структура проекта

```
ModernLLM/
├── bot/
│   ├── __init__.py
│   ├── handlers.py          # Telegram ConversationHandler, state machine
│   └── keyboards.py         # Inline-клавиатуры и callback-константы
├── core/
│   ├── __init__.py
│   ├── arxiv_loader.py      # Загрузка метаданных и PDF с arXiv
│   ├── pdf_parser.py        # Извлечение текста (PyMuPDF)
│   ├── chunker.py           # Word-based sliding window chunker
│   ├── embedder.py          # Эмбеддинги через llama-cpp (GGUF)
│   ├── vector_store.py      # Qdrant: upsert / search / delete
│   ├── db.py                # PostgreSQL (asyncpg): сессии и история
│   └── rag_pipeline.py      # Оркестратор: embed → retrieve → LLM
├── config.py                # Настройки (pydantic-settings + .env)
├── main.py                  # Точка входа + APScheduler
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Как запустить

### Требования

- Docker и Docker Compose
- Токен Telegram-бота ([BotFather](https://t.me/BotFather))
- API-ключ [OpenRouter](https://openrouter.ai/)

### 1. Настройка окружения

```bash
cp .env.example .env
```

Откройте `.env` и заполните:

```env
TELEGRAM_BOT_TOKEN=<ваш токен>
POSTGRES_PASSWORD=<пароль>
OPENROUTER_API_KEY=<ваш ключ>
```

При локальном запуске (вне Docker) также укажите:

```env
POSTGRES_HOST=localhost
QDRANT_HOST=localhost
MODELS_CACHE_DIR=./data/models
PDF_DOWNLOAD_DIR=./data/pdfs
```

### 2. Запуск через Docker Compose

```bash
docker compose up --build
```

Сервисы:
- **postgres** — `localhost:5432`
- **qdrant** — `localhost:6333` (веб-интерфейс: [http://localhost:6333/dashboard](http://localhost:6333/dashboard))
- **bot** — Telegram-бот

> При первом запуске бот автоматически скачает GGUF-файл модели (~400 МБ) с HuggingFace.

### 3. Локальный запуск (без Docker)

```bash
# Поднимите только инфраструктуру
docker compose up postgres qdrant -d

# Установите зависимости
pip install -r requirements.txt

# Запустите бота
python -m main
```

### 4. GPU-поддержка (CUDA)

Для использования RTX 3070 пересоберите образ с аргументом:

```bash
docker compose build --build-arg CMAKE_ARGS="-DGGML_CUDA=ON"
docker compose up
```

### 5. Команды бота в Telegram

Зарегистрируйте команды через BotFather:

```
start - Начать работу / показать приветствие
summarize - Краткое изложение текущей статьи
endsession - Завершить текущую сессию
```

---

## Ключевые фичи

| Фича | Описание |
|---|---|
| Загрузка по ID | Принимает ID (`2312.12456`) или ссылку на arXiv |
| PDF-парсинг | Извлечение текстового слоя через PyMuPDF |
| Chunking | Скользящее окно 512 слов / 64 слова перекрытие |
| Векторный поиск | Qdrant, метрика Cosine, фильтрация по `session_id` |
| Q&A | RAG: top-5 релевантных фрагментов + LLM-ответ |
| Суммаризация | top-15 фрагментов, структурированный промпт |
| История диалогов | PostgreSQL, изолированные сессии |
| Сжатие контекста | При достижении 95% лимита LLM — автосжатие истории |
| TTL сессий | Автоматическое удаление через 24 часа |
| Оценка (👍/👎) | Логирование фидбека для анализа качества |
| Возврат к сессии | Список активных сессий последних 24 часов |

---

## Отчёт по результатам MVP-фазы

Полный список изменений реализации MVP относительно исходной архитектуры из `design_doc.md` — см. [`report.md`](report.md).

### Что реализовано

Реализован полный функционал, описанный в чеклисте design_doc.md:
- Бот принимает ID статьи и успешно её загружает
- Данные векторизуются и сохраняются в Qdrant
- Бот возвращает осмысленные ответы на основе статьи
- Работают все inline-кнопки (Суммаризировать, Завершить сессию, Оценка, Возврат к сессии)

### Технические решения и их обоснование

#### Embedding: llama-cpp-python + GGUF вместо sentence-transformers

Модель `enacimie/Qwen3-Embedding-0.6B-Q4_K_M-GGUF` — квантизованная версия Qwen3-Embedding-0.6B в формате GGUF (Q4_K_M). Выбор обусловлен:
- **Точное соответствие design_doc** — используется именно указанная модель
- **Экономия памяти** — Q4_K_M даёт ~4x снижение размера (~400 МБ вместо ~1.2 ГБ FP16) при минимальной потере качества
- **llama-cpp-python** — стандартный инструмент для инференса GGUF на CPU/GPU

#### asyncpg вместо SQLAlchemy

Минимум слоёв абстракции — для простой схемы из двух таблиц прямое использование asyncpg оптимально по производительности и читаемости. SQLAlchemy добавил бы излишнюю сложность.

#### Единая Qdrant-коллекция с payload-фильтрами

Вместо создания отдельной коллекции для каждой сессии используется одна коллекция `arxiv_chunks` с payload-индексом по `session_id`. Преимущества: проще управление, нет overhead создания/удаления коллекций, Qdrant эффективно обрабатывает фильтрацию по keyword-индексам.

#### Word-based chunking

Токен-based chunking требует запуска токенизатора при каждой индексации, что добавляет зависимость и время. Word-based (512 слов ≈ 394 токена) достаточен для MVP: Qwen3-Embedding поддерживает до 8192 токенов, что оставляет большой запас.

#### python-telegram-bot v21 ConversationHandler

PTB — наиболее зрелая async-библиотека для Telegram-ботов на Python. ConversationHandler из коробки реализует state machine (IDLE / READY / SESSION_ENDED), что соответствует описанным в design_doc сценариям взаимодействия.

#### OpenRouter вместо прямого API модели

Qwen3 VL 235B недоступен для локального запуска на RTX 3070 (требует ~200+ ГБ VRAM). OpenRouter предоставляет единый совместимый с OpenAI API доступ к модели, включая бесплатный tier для разработки.

#### APScheduler внутри процесса бота

Для MVP с ~10 одновременными сессиями отдельный контейнер-планировщик избыточен. APScheduler запускает hourly-джоб очистки истёкших сессий внутри основного процесса. При масштабировании легко заменяется Celery beat.

#### Sync LLM call (stream=False)

Стриминг ответа усложняет handler-код и требует дополнительного управления сообщениями в Telegram. Для MVP синхронный вызов с индикатором "Typing..." — разумный компромисс. Стриминг запланирован на Phase 2.

---

## Требования к окружению

| Компонент | Минимум |
|---|---|
| RAM | 8 ГБ (16 ГБ рекомендуется) |
| VRAM | 0 (CPU-режим) / 2 ГБ+ для GPU |
| Диск | ~2 ГБ (модель + данные) |
| Python | 3.11+ |
| Docker | 24+ |
