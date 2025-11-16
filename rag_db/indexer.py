# rag_db/indexer.py
from typing import List, Dict, Optional, Iterable
from opensearchpy import OpenSearch, helpers  # pyright: ignore[reportMissingImports]
from opensearchpy.exceptions import NotFoundError  # pyright: ignore[reportMissingImports]
from .es_client import get_client
import math
import time

# Настройки индекса — убедись, что dimension совпадает с Embedder
DEFAULT_INDEX = "news"
VECTOR_DIMS = 384  # <- совпадает с all-MiniLM-L6-v2

INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        # Для KNN: включаем индексирование (обязательный флаг для opensearch-knn)
        "index": {"knn": True}
    },
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "content": {"type": "text"},
            # knn_vector — требует, чтобы в OpenSearch был установлен opensearch-knn
            "embedding": {"type": "knn_vector", "dimension": VECTOR_DIMS}
        }
    }
}


class Indexer:
    """
    Индексатор для OpenSearch с поддержкой векторного поля (knn_vector).
    Предполагает: OpenSearch с установленным opensearch-knn (локально).
    """

    def __init__(
        self,
        index: str,
        host: str = "localhost",
        port: int = 9200,
        user: str = "admin",
        password: str = "Korobochka54!"
    ):
        self.index = index
        self.client = get_client(
            host=host,
            port=port,
            user=user,
            password=password
        )


    # -------------------------
    # Управление индексом
    # -------------------------
    def create_index(self, force: bool = False, mapping: Dict = None):
        """
        Создаёт индекс с mapping (или DEFAULT).
        Если force=True и индекс существует — удалит его.
        """
        mapping = mapping or INDEX_MAPPING
        try:
            if force and self.client.indices.exists(index=self.index):
                print(f"[Indexer] deleting index '{self.index}' (force=True)")
                self.client.indices.delete(index=self.index)
                # дать ES/OS пару секунд на очистку
                time.sleep(0.5)

            if not self.client.indices.exists(index=self.index):
                self.client.indices.create(index=self.index, body=mapping)
                print(f"[Indexer] index '{self.index}' created")
            else:
                print(f"[Indexer] index '{self.index}' already exists")
        except Exception as exc:
            # полезная подсказка при ошибке mapper_parsing_exception
            print("[Indexer] Error creating index:", exc)
            raise

    def delete_index(self):
        if self.client.indices.exists(index=self.index):
            self.client.indices.delete(index=self.index)

    def refresh(self):
        self.client.indices.refresh(index=self.index)

    def count(self) -> int:
        res = self.client.count(index=self.index)
        return res.get("count", 0)

    def exists(self, doc_id: str) -> bool:
        try:
            return self.client.exists(index=self.index, id=doc_id)
        except NotFoundError:
            return False

    # -------------------------
    # Индексация
    # -------------------------
    def index_document(self, doc_id: str, body: Dict):
        """
        Индексирует 1 документ.
        body должен содержать ключи: 'content' (строка), опционально 'title', 'embedding' (list[float]).
        """
        # простая валидация
        if "embedding" in body:
            emb = body["embedding"]
            if emb is None:
                raise ValueError("embedding value is None")
            if len(emb) != VECTOR_DIMS:
                raise ValueError(f"embedding has wrong dimension {len(emb)} != {VECTOR_DIMS}")

        return self.client.index(index=self.index, id=doc_id, body=body)

    def index_documents(self, docs: Iterable[Dict],
                        id_key: Optional[str] = None,
                        batch_size: int = 128,
                        progress: bool = False):
        """
        Batch индексирование.
        docs: iterable of dict (each dict is doc body or contains id_key)
        id_key: если указан, из каждого doc берётся уникальный id = doc[id_key]
        """
        actions = []
        n = 0
        for item in docs:
            n += 1
            if id_key and id_key in item:
                doc_id = str(item[id_key])
                body = {k: v for k, v in item.items() if k != id_key}
            else:
                # генерируем авто id
                doc_id = None
                body = item

            # validate embedding if present
            if "embedding" in body and body["embedding"] is not None:
                if len(body["embedding"]) != VECTOR_DIMS:
                    raise ValueError(f"Document embedding has wrong dimension: {len(body['embedding'])} != {VECTOR_DIMS}")

            action = {
                "_index": self.index,
                "_id": doc_id,
                "_source": body
            }
            actions.append(action)

            if len(actions) >= batch_size:
                helpers.bulk(self.client, actions)
                actions = []

        if actions:
            helpers.bulk(self.client, actions)

        # refresh so documents are visible
        self.refresh()

    # -------------------------
    # Поиск
    # -------------------------
    def search_text(self, query: str, top_k: int = 5):
        """
        Простое полнотекстовое совпадение (match).
        """
        body = {
            "size": top_k,
            "query": {
                "match": {
                    "content": query
                }
            },
            "_source": ["date", "text", "views", "channel_title"]
        }
        return self.client.search(index=self.index, body=body)

    def search_phrase(self, phrase: str, top_k: int = 20):
        """
        Фразовая проверка (match_phrase) — Ctrl+F аналог.
        """
        body = {
            "size": top_k,
            "query": {
                "match_phrase": {
                    "content": {
                        "query": phrase
                    }
                }
            },
            "_source": ["date", "text", "views", "channel_title"]
        }
        return self.client.search(index=self.index, body=body)

    def vector_search(self, query_vector: List[float], top_k: int = 5):
        """
        Поиск по вектору — использует knn query в OpenSearch.
        Требует opensearch-knn plugin и поле типа knn_vector.
        """
        if len(query_vector) != VECTOR_DIMS:
            raise ValueError(f"query_vector dimension {len(query_vector)} does not match INDEX dimension {VECTOR_DIMS}")

        body = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": top_k
                    }
                }
            },
            "_source": ["date", "text", "views", "channel_title"]
        }

        return self.client.search(index=self.index, body=body)

    def sql_query(self, sql: str):
        """
        Выполняет SQL-запрос к OpenSearch через SQL-плагин.
        Возвращает список словарей.
        """
        response = self.client.transport.perform_request(
            "POST",
            "/_plugins/_sql",
            body={"query": sql}
        )

        # поддержка разных форматов ответа
        if 'columns' in response and 'rows' in response:
            columns = [col['name'] for col in response['columns']]
            rows = response['rows']
        elif 'schema' in response and 'datarows' in response:
            columns = [col['name'] for col in response['schema']]
            rows = response['datarows']
        else:
            raise ValueError(f"Неизвестный формат ответа SQL: {response}")

        return [dict(zip(columns, row)) for row in rows]
