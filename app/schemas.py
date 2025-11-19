from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class NewsItem(BaseModel):
    """Public representation of a single news entry."""

    id: str
    date: Optional[datetime]
    text: str = Field(..., description="Raw text of the news item")
    views: Optional[int] = Field(None, ge=0)
    forwards: Optional[int] = Field(None, ge=0)
    has_media: bool = False
    channel_title: Optional[str] = None
    channel_username: Optional[str] = None
    score: Optional[float] = Field(
        None,
        description="Score reported by OpenSearch. Can be None for match_all queries.",
    )


class NewsQuery(BaseModel):
    query: str = Field(..., min_length=1, description="Free-form question or query")
    limit: int = Field(
        5, ge=1, le=50, description="Maximum number of news items to return"
    )
    force_mode: Optional[Literal["vector", "text"]] = Field(
        None,
        description="Optional override to skip planning and force a retrieval mode.",
    )


class SearchStrategy(BaseModel):
    use_sql: bool = False
    sql_query: Optional[str] = None
    use_vector_search: bool = True
    vector_search_query: Optional[str] = None
    use_text_search: bool = False
    text_search_query: Optional[str] = None
    raw_response: Optional[str] = Field(
        None, description="Raw JSON returned by the LLM planner."
    )


class SearchResponse(BaseModel):
    query: str
    limit: int
    count: int = Field(..., description="How many items satisfied the request")
    strategy: SearchStrategy
    extracted_information: Optional[str] = Field(
        None, description="Facts extracted from the retrieved news items before summarization."
    )
    summary: Optional[str] = Field(
        None,
        description="LLM-generated synthesis of the retrieved snippets.",
    )
    metrics: Optional[dict] = Field(
        None,
        description="Metrics based on database search perfomance and objectivity/subjectivity prediction by GroNLP/mdebertav3-subjectivity-multilingual.",
    )
    results: List[NewsItem]
