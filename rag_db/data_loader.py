# rag_db/data_loader.py

import os
import json
from typing import List, Dict, Optional
from .indexer import Indexer
from .utils import Embedder


class DataLoader:
    """
    Загружает JSON-файлы из директории data,
    создаёт эмбеддинги и отправляет документы в OpenSearch.
    """

    def __init__(self,
                 data_dir: str = "./data",
                 index_name: str = "news",
                 embedder: Optional[Embedder] = None,
                 indexer: Optional[Indexer] = None):

        self.data_dir = data_dir
        self.embedder = embedder or Embedder()
        self.indexer = indexer or Indexer(index_name)


    # ---------------------------
    #  Загрузка файлов
    # ---------------------------
    def load_files(self) -> List[Dict]:
        """
        Читает все .json файлы из data/
        Возвращает список raw документов (словарей).
        """
        docs = []
        for filename in os.listdir(self.data_dir):
            if not filename.endswith(".json"):
                continue

            full_path = os.path.join(self.data_dir, filename)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    if isinstance(content, dict):
                        docs.append(content)
                    elif isinstance(content, list):
                        docs.extend(content)
            except Exception as e:
                print(f"[DataLoader] Ошибка чтения '{filename}': {e}")

        print(f"[DataLoader] Загружено документов: {len(docs)}")
        return docs

    # ---------------------------
    #  Генерация эмбеддингов
    # ---------------------------
    @staticmethod
    def extract_text(doc: Dict) -> str:
        """
        Попытка достать текст. Можно адаптировать под структуру своей БД.
        """
        if "content" in doc:
            return str(doc["content"])
        if "text" in doc:
            return str(doc["text"])

        # fallback: конкатенировать все строковые значения
        collected = []
        for v in doc.values():
            if isinstance(v, str):
                collected.append(v)
        return "\n".join(collected)

    def embed_documents(self, docs: List[Dict]) -> List[Dict]:
        """
        Добавляет embedding в каждый документ.
        Возвращает новый список документов.
        """
        texts = [self.extract_text(doc) for doc in docs]
        vectors = self.embedder.embed(texts)

        enriched_docs = []
        for doc, emb in zip(docs, vectors):
            new_doc = dict(doc)
            new_doc["embedding"] = emb
            enriched_docs.append(new_doc)

        return enriched_docs

    # ---------------------------
    #  Главная функция —
    #  загрузка → эмбеддинги → индексирование
    # ---------------------------
    def load_into_index(self, force_recreate: bool = False):
        """
        Полный цикл загрузки данных:
        - создаём/пересоздаём индекс
        - грузим файлы
        - делаем эмбеддинги
        - отправляем в OpenSearch
        """
        print(f"[DataLoader] Создание индекса '{self.indexer.index}'")
        self.indexer.create_index(force=force_recreate)

        docs = self.load_files()
        if not docs:
            print("[DataLoader] Нет данных для загрузки")
            return

        print("[DataLoader] Создаём эмбеддинги...")
        docs_with_emb = self.embed_documents(docs)

        # если в документе нет id — генерируем
        indexed_docs = []
        for i, d in enumerate(docs_with_emb):
            if "_id" not in d:
                d["_id"] = str(i)
            indexed_docs.append(d)

        print("[DataLoader] Индексируем документы...")
        self.indexer.index_documents(indexed_docs, id_key="_id", batch_size=64)

        print("[DataLoader] Загрузка завершена. Всего документов в индексе:",
              self.indexer.count())
