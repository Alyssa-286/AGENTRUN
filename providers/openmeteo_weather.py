"""
providers/openmeteo_weather.py — Open-Meteo implementation of WeatherProvider.
"""

from __future__ import annotations

import requests

from providers.weather_provider import WeatherProvider, WeatherResult

WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "rain showers",
    95: "thunderstorm",
}


class OpenMeteoWeatherProvider(WeatherProvider):
    GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def get_weather(self, city: str) -> WeatherResult | None:
        coords = self._geocode(city)
        if coords is None:
            return None
        latitude, longitude = coords

        try:
            response = requests.get(
                self.FORECAST_URL,
                params={"latitude": latitude, "longitude": longitude, "current_weather": "true"},
                timeout=10,
            )
            response.raise_for_status()
            current = response.json()["current_weather"]
        except Exception as exc:
            print(f"[OpenMeteoWeatherProvider] forecast lookup failed: {exc}")
            return None

        code = current.get("weathercode")
        return WeatherResult(
            city=city,
            temperature_c=current.get("temperature"),
            condition=WEATHER_CODES.get(code, f"code {code}"),
            source="open-meteo",
        )

    def _geocode(self, city: str) -> tuple[float, float] | None:
        try:
            response = requests.get(self.GEOCODE_URL, params={"name": city, "count": 1}, timeout=10)
            response.raise_for_status()
            results = response.json().get("results")
        except Exception as exc:
            print(f"[OpenMeteoWeatherProvider] geocoding failed: {exc}")
            return None

        if not results:
            return None

        top = results[0]
        return top["latitude"], top["longitude"]
