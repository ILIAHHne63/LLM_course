from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Sequence

from langchain_core.messages import BaseMessage

from .llm_client import LLMClient
from .metrics import evaluate_subjectivity_filtering_extended
from .news_store import NewsRecord, NewsStore
from .prompts import (
    analyst_system_prompt,
    extraction_prompt,
    search_prompt,
    search_system_prompt,
    summarization_prompt,
    summarizer_system_prompt,
)


@dataclass
class RetrievalPlan:
    use_sql: bool
    sql_query: Optional[str]
    use_vector_search: bool
    vector_search_query: Optional[str]
    use_text_search: bool
    text_search_query: Optional[str]
    raw_response: Optional[str] = None


class RetrievalPlanner:
    """LLM-driven planner that mirrors the logic from api_testing.ipynb."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(self, user_query: str) -> RetrievalPlan:
        messages: List[BaseMessage] = [search_system_prompt, search_prompt(user_query)]
        raw = self.llm.generate(messages)
        payload = self._parse_plan(raw)
        if not payload:
            payload = {
                "use_sql": False,
                "use_vector_search": True,
                "vector_search_query": user_query,
                "use_text_search": False,
            }
        return RetrievalPlan(
            use_sql=bool(payload.get("use_sql")),
            sql_query=payload.get("sql_query"),
            use_vector_search=bool(payload.get("use_vector_search")),
            vector_search_query=payload.get("vector_search_query"),
            use_text_search=bool(payload.get("use_text_search")),
            text_search_query=payload.get("text_search_query"),
            raw_response=raw,
        )

    def _parse_plan(self, raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}


class Analyst:
    """Runs the two-phase extraction + summarization workflow."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def extract(self, user_query: str, records: Sequence[NewsRecord]) -> str:
        posts = [
            {"date": record.date, "text": record.text}
            for record in records
            if record.text
        ]
        if not posts:
            return ""
        messages = [analyst_system_prompt, extraction_prompt(user_query, posts)]
        return self.llm.generate(messages), messages[1]

    def summarize(self, user_query: str, extracted_info: str) -> str:
        if not extracted_info:
            return "Не удалось извлечь подтверждённую информацию из найденных сообщений."
        messages = [summarizer_system_prompt, summarization_prompt(user_query, extracted_info)]
        return self.llm.generate(messages)


class RagReasoner:
    """High-level orchestrator for the 4-step pipeline."""

    def __init__(self, store: NewsStore, llm: Optional[LLMClient] = None):
        self.store = store
        self.llm = llm or LLMClient()
        self.planner = RetrievalPlanner(self.llm)
        self.analyst = Analyst(self.llm)

    def answer(
        self,
        user_query: str,
        limit: int,
        force_mode: Optional[str] = None,
    ) -> tuple[RetrievalPlan, List[NewsRecord], str, str]:
        if force_mode:
            plan = self._plan_from_override(user_query, force_mode)
        else:
            plan = self.planner.plan(user_query)

        records = self.store.retrieve_with_plan(plan, limit)
        extracted, query_llm = self.analyst.extract(user_query, records)
        metrics = evaluate_subjectivity_filtering_extended(query_llm.content, extracted)
        summary = self.analyst.summarize(user_query, extracted)
        return plan, records, extracted, summary, metrics

    def _plan_from_override(self, query: str, mode: str) -> RetrievalPlan:
        mode = mode.lower()
        return RetrievalPlan(
            use_sql=False,
            sql_query=None,
            use_vector_search=mode == "vector",
            vector_search_query=query if mode == "vector" else None,
            use_text_search=mode == "text",
            text_search_query=query if mode == "text" else None,
            raw_response=None,
        )
