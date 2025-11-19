## FastAPI сервер
1. Поднимите OpenSearch: `docker-compose up -d`
2. Установите зависимости: `pip install -r requirements.txt`
3. (опционально) Укажите переменные окружения:
   - `DATA_PATH` — путь к директории с JSON-файлами (по умолчанию берётся `./data` из корня репозитория)
   - `NEWS_INDEX_NAME` — имя индекса OpenSearch
   - `FORCE_REBUILD_NEWS_INDEX=1` — чтобы принудительно переиндексировать данные при старте сервера
   - `OPENSEARCH_RETRY_ATTEMPTS` / `OPENSEARCH_RETRY_DELAY` — сколько раз и с каким интервалом API ждёт доступности OpenSearch
   - `MISTRAL_API_KEY` — ключ доступа к Mistral API (см. пример в `api_testing.ipynb`)
4. Запустите API: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`

> Все переменные можно задать в `.env` (см. пример в корне репозитория) — сервер автоматически подхватывает этот файл при старте.
> Логи сохраняются в `build/rag_db.log`, а каждый ответ API записывается в директорию `outputs/`.

Эндпоинты:
- `GET /health` — проверка статуса и кол-ва документов в индексе
- `GET /news/latest?limit=5` — последние новости по дате
- `GET /news/{id}` — получить конкретное сообщение
- `POST /news/query` — поиск по коллекции (LLM решает стратегию, опционально можно задать `force_mode`=`vector|text`)

### Пример использования

После запуска `uvicorn` можно выполнить запрос на поиск:

```bash
curl -s -X POST http://localhost:8000/news/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Последние новости России", "limit": 3, "force_mode": "vector"}' \
  > /dev/null && python scripts/show_summary.py
```

Для удобства есть скрипт. Сначала указываете текст, потом количество новостей:
```bash
./scripts/query_news.sh "Последние новости России" 5
```


Пример ответа (усечённый):

```json
{
  "query": "санкции против отрасли",
  "limit": 3,
  "count": 3,
  "results": [
    {
      "id": "2145",
      "date": "2024-05-12T09:47:00",
      "text": "…сообщение из Telegram…",
      "views": 1234,
      "forwards": 56,
      "has_media": false,
      "channel_title": "MyChannel",
      "channel_username": "@mychannel",
      "score": 18.42
    }
  ]
}
```

`scripts/show_summary.py` берёт самый свежий файл `outputs/answer_YYYYMMDD-HHMMSS.json` (или путь, переданный аргументом) и выводит аккуратно отформатированную сводку + топ новостей на терминал.

### Как устроена обработка запроса

1. **Принимаем запрос.** Пользователь вызывает `POST /news/query` с текстом вопроса.
2. **LLM формирует стратегию.** Агент на базе `mistral-large-latest` решает, использовать ли SQL, векторный или текстовый поиск и возвращает JSON-план.
3. **Ищем по БД.** Согласно плану выполняются SQL/векторные/BM25 запросы к OpenSearch (в индекс предварительно загружены все JSON из `DATA_PATH`, по умолчанию `./data`).
4. **LLM извлекает факты и суммаризирует.** Первый прогон агента вытаскивает сырое содержание из найденных сообщений, второй строит итоговую сводку. Оба текста (`extracted_information` и `summary`) возвращаются вместе с исходными документами.