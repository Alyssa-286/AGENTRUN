"""
providers/weather_provider.py — provider-agnostic weather interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass


@dataclass
class WeatherResult:
    city: str
    temperature_c: float
    condition: str
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


class WeatherProvider(ABC):
    @abstractmethod
    def get_weather(self, city: str) -> WeatherResult | None:
        """Return structured weather data, or None on failure."""
