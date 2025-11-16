# RAG Elasticsearch module


Запуск:


1. `docker-compose up -d`
2. Установить зависимости: `pip install -r requirements.txt`
3. Прописать и положить свои документы в example.py или используйте свой парсер для Telegram
4. Запустить `python -m rag_db.example`


Функции модуля:
- создание индекса с dense_vector
- индексирование документов батчами
- семантический (векторный) поиск
- SQL-like запросы через Elasticsearch SQL endpoint
- Ctrl+F (фразовый / полнотекстовый) поиск