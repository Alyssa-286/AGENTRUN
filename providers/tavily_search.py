"""
providers/tavily_search.py — Tavily-specific implementation of SearchProvider.
"""

from __future__ import annotations

import os

import requests

from providers.search_provider import SearchProvider, SearchResult


class TavilySearchProvider(SearchProvider):
    API_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set TAVILY_API_KEY environment variable first.")

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        try:
            response = requests.post(
                self.API_URL,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            print(f"[TavilySearchProvider] search failed: {exc}")
            return []

        results: list[SearchResult] = []
        for item in data.get("results", []):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", "")[:500],
                    source="tavily",
                )
            )
        return results
