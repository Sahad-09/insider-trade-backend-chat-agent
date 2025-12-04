"""LangGraph tools for enhanced language model capabilities.

This package contains custom tools that can be used with LangGraph to extend
the capabilities of language models. Includes tools for:
- Web search (DuckDuckGo)
- SEC Form 4 insider trading analysis (Financial Modeling Prep API)
"""

from langchain_core.tools import BaseTool

from .duckduckgo_search import duckduckgo_search_tool
from .insider import (
    get_company_profile,
    get_insider_roster,
    get_insider_statistics,
    get_recent_market_insider_activity,
    get_stock_price,
    search_insider_trades_by_symbol,
)

tools: list[BaseTool] = [
    duckduckgo_search_tool,
    # Insider trading tools
    search_insider_trades_by_symbol,
    get_recent_market_insider_activity,
    get_stock_price,
    get_company_profile,
    get_insider_roster,
    get_insider_statistics,
]
