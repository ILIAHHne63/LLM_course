from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, TYPE_CHECKING

from opensearchpy.exceptions import NotFoundError
from opensearchpy.exceptions import ConnectionError as OSConectionError

from rag_db.data_loader import DataLoader
from rag_db.indexer import Indexer
from rag_db.logger import get_logger
from rag_db.utils import Embedder

if TYPE_CHECKING:
    from .reasoner import RetrievalPlan

log = get_logger(__name__)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes"}


def resolve_dataset_path() -> Path:
    env_path = os.getenv("DATA_PATH")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parents[1] / "data"

@dataclass
class NewsRecord:
    id: str
    date: Optional[str]
    text: str
    views: Optional[int]
    forwards: Optional[int]
    channel_title: Optional[str]
    channel_username: Optional[str]
    has_media: bool
    score: Optional[float] = None


class NewsStore:
    """Thin wrapper that reuses rag_db's loader/indexer for serving API traffic."""

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        index_name: Optional[str] = None,
    ):
        self.dataset_path = dataset_path or resolve_dataset_path()
        self.index_name = index_name or os.getenv("NEWS_INDEX_NAME", "news_mychannel")
        self.embedder = Embedder()
        self.indexer = Indexer(self.index_name)
        self.loader = DataLoader(
            data_dir=str(self.dataset_path),
            index_name=self.index_name,
            embedder=self.embedder,
            indexer=self.indexer,
        )
        self._wait_for_opensearch()
        self._ensure_index()

    def _wait_for_opensearch(self) -> None:
        delay = float(os.getenv("OPENSEARCH_RETRY_DELAY", "3"))
        attempts = int(os.getenv("OPENSEARCH_RETRY_ATTEMPTS", "10"))
        for attempt in range(1, attempts + 1):
            try:
                if self.indexer.client.ping():
                    return
            except OSConectionError as exc:
                log.warning(
                    "OpenSearch ping failed (attempt %s/%s): %s",
                    attempt,
                    attempts,
                    exc,
                )
            except Exception as exc:  # pragma: no cover
                log.warning(
                    "Unexpected error when pinging OpenSearch (attempt %s/%s): %s",
                    attempt,
                    attempts,
                    exc,
                )
            time.sleep(delay)
        raise RuntimeError(
            "OpenSearch is not reachable; ensure `docker-compose up -d` has completed."
        )

    def _ensure_index(self) -> None:
        force = _env_flag("FORCE_REBUILD_NEWS_INDEX")
        needs_data = force
        if not needs_data:
            try:
                needs_data = self.indexer.count() == 0
            except Exception:
                needs_data = True
        if needs_data:
            log.info(
                "Building OpenSearch index '%s' from %s",
                self.index_name,
                self.dataset_path,
            )
            self.loader.load_into_index(force_recreate=True)

    @staticmethod
    def _hit_to_record(hit: dict) -> NewsRecord:
        source = hit.get("_source", {})
        text = source.get("text") or source.get("content") or ""
        return NewsRecord(
            id=str(hit.get("_id")),
            date=source.get("date"),
            text=text,
            views=source.get("views"),
            forwards=source.get("forwards"),
            channel_title=source.get("channel_title"),
            channel_username=source.get("channel_username"),
            has_media=bool(source.get("has_media")),
            score=hit.get("_score"),
        )

    def _hits_to_records(self, hits: Sequence[dict]) -> List[NewsRecord]:
        return [self._hit_to_record(hit) for hit in hits]

    def latest(self, limit: int) -> List[NewsRecord]:
        body = {
            "size": limit,
            "sort": [{"date": {"order": "desc"}}],
            "query": {"match_all": {}},
        }
        res = self.indexer.client.search(index=self.index_name, body=body)
        return self._hits_to_records(res.get("hits", {}).get("hits", []))

    def get(self, news_id: str) -> Optional[NewsRecord]:
        try:
            res = self.indexer.client.get(index=self.index_name, id=str(news_id))
        except NotFoundError:
            return None
        return self._hit_to_record(res)

    def search(
        self,
        query: str,
        limit: int,
        mode: str = "vector",
        fallback_to_latest: bool = True,
    ) -> List[NewsRecord]:
        mode = (mode or "vector").lower()
        if not query:
            return self.latest(limit)
        if mode == "phrase":
            res = self.indexer.search_phrase(query, top_k=limit)
        elif mode == "text":
            res = self.indexer.search_text(query, top_k=limit)
        else:
            vector = self.embedder.embed(query)
            res = self.indexer.vector_search(vector, top_k=limit)

        hits = res.get("hits", {}).get("hits", [])
        if not hits and fallback_to_latest:
            return self.latest(limit)
        return self._hits_to_records(hits)

    def retrieve_with_plan(self, plan: "RetrievalPlan", limit: int) -> List[NewsRecord]:
        collected: List[NewsRecord] = []
        seen: set[str] = set()

        def add_records(records: Sequence[NewsRecord]):
            for record in records:
                if record.id in seen:
                    continue
                seen.add(record.id)
                collected.append(record)

        if plan.use_sql and plan.sql_query:
            sql_query = plan.sql_query
            if self.index_name != "news":
                sql_query = re.sub(r"\\bnews\\b", self.index_name, sql_query)
                plan.sql_query = sql_query
            try:
                rows = self.indexer.sql_query(sql_query)
                sql_records = [self._row_to_record(row) for row in rows]
                add_records([r for r in sql_records if r])
            except Exception as exc:
                log.warning("SQL plan failed: %s", exc)

        if plan.use_vector_search and plan.vector_search_query:
            vector_records = self.search(
                plan.vector_search_query, limit, mode="vector", fallback_to_latest=False
            )
            add_records(vector_records)
        elif plan.use_text_search and plan.text_search_query:
            text_mode = "phrase" if '"' in plan.text_search_query else "text"
            text_records = self.search(
                plan.text_search_query,
                limit,
                mode=text_mode,
                fallback_to_latest=False,
            )
            add_records(text_records)

        if not collected:
            return self.latest(limit)
        return collected[:limit]

    def _row_to_record(self, row: dict) -> Optional[NewsRecord]:
        if not row:
            return None
        doc_id = row.get("_id") or row.get("id") or row.get("doc_id")
        if doc_id is None:
            doc_id = str(abs(hash(row.get("text", ""))))
        text = row.get("text") or row.get("content") or ""
        return NewsRecord(
            id=str(doc_id),
            date=row.get("date"),
            text=text,
            views=row.get("views"),
            forwards=row.get("forwards"),
            channel_title=row.get("channel_title"),
            channel_username=row.get("channel_username"),
            has_media=bool(row.get("has_media")),
            score=row.get("_score"),
        )
