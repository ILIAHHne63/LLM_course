# rag_db/es_client.py
from opensearchpy import OpenSearch, RequestsHttpConnection  # pyright: ignore[reportMissingImports]
import os

# Настройки подключения через переменные окружения (удобно для Docker / prod)
ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = int(os.getenv("ES_PORT", "9200"))
ES_USER = os.getenv("ES_USER", "admin")
ES_PASSWORD = os.getenv("ES_PASSWORD", "Korobochka54!")
ES_USE_SSL = os.getenv("ES_USE_SSL", "true").lower() in ("1", "true", "yes")

def get_client(host: str = ES_HOST,
               port: int = ES_PORT,
               user: str = ES_USER,
               password: str = ES_PASSWORD,
               use_ssl: bool = ES_USE_SSL) -> OpenSearch:
    """
    Возвращает экземпляр OpenSearch клиента.
    Не выполняет никаких действий при импорте (ленивая инициализация).
    """
    scheme_host = host
    # OpenSearch client accepts hosts list as dicts (host, port) or url string.
    client = OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        use_ssl=False,
        verify_certs=False,
    )
    return client
