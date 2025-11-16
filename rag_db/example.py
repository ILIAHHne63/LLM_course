# rag_db/example.py

from rag_db.data_loader import DataLoader
from rag_db.indexer import Indexer
from rag_db.utils import Embedder
from rag_db.logger import get_logger

log = get_logger(__name__)


def run_example():
    log.info("=== ИНИЦИАЛИЗАЦИЯ ===")
    loader = DataLoader("./data", "news")
    loader.load_into_index(force_recreate=True)

    indexer = Indexer("news")
    embedder = Embedder()

    log.info("=== ТЕКСТОВЫЙ ПОИСК ===")
    query = "град снегопад погода"
    text_res = indexer.search_text(query, top_k=3)

    for h in text_res.get("hits", {}).get("hits", []):
        log.info("TEXT HIT: " + str(h["_source"]))

    log.info("=== ВЕКТОРНЫЙ ПОИСК ===")
    vec = embedder.embed(query)
    vec_res = indexer.vector_search(vec, top_k=3)

    for h in vec_res.get("hits", {}).get("hits", []):
        log.info("VECTOR HIT: " + str(h["_source"]))


    log.info("=== SQL ПОИСК ===")
    sql = "SELECT channel_title, date, text FROM news WHERE text LIKE '%Путин%'"
    res_sql = indexer.sql_query(sql)
    print(res_sql)


if __name__ == "__main__":
    run_example()
