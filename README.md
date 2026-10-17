# OTUS. Advanced RAG

Небольшой проект для улучшенного RAG с поддержкой метаданных обработанных документов на
LlamaIndex/Qdrant/LangChain/Gradio

Модели:

- embedding - qwen3-embedding:0.6b (локальная через ollama)
- llm для чата - qwen3:4b (локальная через ollama)

Парсинг:

- чтение pdf через pymupdf, html через BeautifulSoup, epub через EbookLib
- потом нарезка через MarkdownNodeParser по заголовкам
- далее ограничение больших chunk через SentenceSplitter

## Что внутри

- `src/main.py` — основной модуль (логика загрузки документов, сохранение в store, pipeline LlangChain и ui)
- `src/parset/MultiTagHTMLReader.py` — парсер html документов с поддержкой нескольких тэгов
- `data/` — кэш файлов для nltk

## Подготовка

1. Установите зависимости:

```bash
uv sync
```

Запустить Qdrant в docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Активируйте виртуальное окружение:

```bash
source .venv/bin/activate
```

## Запуск

```bash
python src/main.py
```

## Результаты

В консоле отображаются:

- время парсинга/загрузки документа в vector store
- chunk полученные из LlamaIndex по запросу пользователя
- сообщения передаваемые в LLM по запросу пользователя
