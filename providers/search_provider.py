"""
providers/search_provider.py — provider-agnostic web search interface.

The agent and tool layers only depend on this interface, not on any specific
search API. That keeps the runtime interchangeable and makes later evaluation
work provider-independent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


class SearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Return structured search results for the query."""
