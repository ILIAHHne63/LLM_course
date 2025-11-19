import json
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import load_env

load_env()

from .llm_client import LLMClient
from .news_store import NewsRecord, NewsStore, resolve_dataset_path
from .reasoner import RagReasoner
from .schemas import NewsItem, NewsQuery, SearchResponse, SearchStrategy

STORE = NewsStore(resolve_dataset_path())
LLM = LLMClient()
REASONER = RagReasoner(STORE, LLM)
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"

app = FastAPI(
    title="Telegram News Search API",
    description=(
        "Exposes Telegram channel exports stored under ./data via rag_db + OpenSearch. "
        "Use /news/query to perform dense, BM25, or phrase search."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _as_schema(record: NewsRecord, score):
    return NewsItem(
        id=record.id,
        date=record.date,
        text=record.text,
        views=record.views,
        forwards=record.forwards,
        has_media=record.has_media,
        channel_title=record.channel_title,
        channel_username=record.channel_username,
        score=score,
    )


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok", "items_indexed": STORE.indexer.count()}


@app.get("/news/latest", response_model=List[NewsItem])
def latest_news(limit: int = Query(5, ge=1, le=50)) -> List[NewsItem]:
    return [_as_schema(record, record.score) for record in STORE.latest(limit)]


@app.get("/news/{news_id}", response_model=NewsItem)
def get_news(news_id: str) -> NewsItem:
    record = STORE.get(news_id)
    if not record:
        raise HTTPException(status_code=404, detail="News item not found")
    return _as_schema(record, record.score)


@app.post("/news/query", response_model=SearchResponse)
def query_news(request: NewsQuery) -> SearchResponse:
    plan, matches, extracted, summary = REASONER.answer(
        request.query, request.limit, request.force_mode
    )
    strategy = SearchStrategy(
        use_sql=plan.use_sql,
        sql_query=plan.sql_query,
        use_vector_search=plan.use_vector_search,
        vector_search_query=plan.vector_search_query,
        use_text_search=plan.use_text_search,
        text_search_query=plan.text_search_query,
        raw_response=plan.raw_response,
    )
    response = SearchResponse(
        query=request.query,
        limit=request.limit,
        count=len(matches),
        strategy=strategy,
        extracted_information=extracted,
        summary=summary,
        results=[_as_schema(record, record.score) for record in matches],
    )

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    path = OUTPUT_DIR / f"answer_{timestamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = response.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return response
