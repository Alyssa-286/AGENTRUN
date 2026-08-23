"""
mcp_server.py — a standalone MCP server exposing our tools.

This is the only file that constructs Tool objects and provider instances
for live use. The agent no longer imports tools.py or providers/ directly.
"""

import warnings

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

try:
    from pydantic_settings.exceptions import IncompleteFieldDefinitionWarning
except ImportError:
    IncompleteFieldDefinitionWarning = None  # older pydantic_settings versions

from providers.openmeteo_weather import OpenMeteoWeatherProvider
from providers.tavily_search import TavilySearchProvider
from tools import CALCULATOR_TOOL, WORD_COUNT_TOOL, make_weather_tool, make_web_search_tool


if IncompleteFieldDefinitionWarning is not None:
    warnings.filterwarnings("ignore", category=IncompleteFieldDefinitionWarning)

load_dotenv()

_search_provider = TavilySearchProvider()
_weather_provider = OpenMeteoWeatherProvider()

_LOCAL_TOOLS = {
    tool.name: tool
    for tool in [
        CALCULATOR_TOOL,
        WORD_COUNT_TOOL,
        make_web_search_tool(_search_provider),
        make_weather_tool(_weather_provider),
    ]
}

mcp = FastMCP("agent-from-scratch-tools")


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4)'."""

    return _LOCAL_TOOLS["calculator"].run(expression=expression)


@mcp.tool()
def word_count(text: str) -> str:
    """Count the number of words in a piece of text."""

    return _LOCAL_TOOLS["word_count"].run(text=text)


@mcp.tool()
def web_search(query: str) -> str:
    """Search the live web for current information and return structured JSON."""

    return _LOCAL_TOOLS["web_search"].run(query=query)


@mcp.tool()
def get_weather(city: str) -> str:
    """Get current real-time weather for a city."""

    return _LOCAL_TOOLS["get_weather"].run(city=city)


if __name__ == "__main__":
    mcp.run(transport="stdio")