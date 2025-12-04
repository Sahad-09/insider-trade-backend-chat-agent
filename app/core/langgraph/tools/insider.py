"""Tools for insider trading analysis using Financial Modeling Prep API.

Documentation: https://site.financialmodelingprep.com/developer/docs
"""

import json

from langchain_core.tools import tool

from app.services.insider_trade import insider_trade_service


@tool
async def search_insider_trades_by_symbol(symbol: str, limit: int = 50) -> str:
    """Search for recent SEC Form 4 insider trading activity for a specific stock symbol.
    
    Use this tool when the user asks for insider trades, purchases, or sales for a specific company.
    Returns filings including: insider name, transaction type, shares, price, and filing date.
    
    Args:
        symbol: The stock ticker symbol (e.g., 'AAPL', 'MSFT', 'NVDA').
        limit: Number of records to return (default 50, max 100).
        
    Returns:
        JSON string of insider trading records from SEC Form 4 filings.
    """
    try:
        trades = await insider_trade_service.get_insider_trades(symbol, limit=limit)
        if not trades:
            return f"No recent insider trading activity found for {symbol}."
        return json.dumps(trades[:limit], indent=2)
    except Exception as e:
        return f"Error fetching insider trades for {symbol}: {str(e)}"


@tool
async def get_recent_market_insider_activity(limit: int = 50) -> str:
    """Get the most recent insider trading activity across the entire market.
    
    Use this tool to find the latest insider trades market-wide, identify trending stocks,
    or discover unusual insider activity patterns.
    
    Args:
        limit: Number of records to return (default 50, max 100).
        
    Returns:
        JSON string of recent market-wide insider trading records.
    """
    try:
        trades = await insider_trade_service.get_recent_insider_activity(limit=limit)
        if not trades:
            return "No recent market insider activity found."
        return json.dumps(trades[:limit], indent=2)
    except Exception as e:
        return f"Error fetching recent market activity: {str(e)}"


@tool
async def get_stock_price(symbol: str) -> str:
    """Get the current real-time stock price and quote data.
    
    Use this tool to get current price, daily change, volume, and market cap for context.
    
    Args:
        symbol: The stock ticker symbol (e.g., 'AAPL').
        
    Returns:
        JSON string with price, change, volume, market cap, etc.
    """
    try:
        quote = await insider_trade_service.get_stock_quote(symbol)
        if not quote:
            return f"No quote found for {symbol}."
        return json.dumps(quote, indent=2)
    except Exception as e:
        return f"Error fetching quote for {symbol}: {str(e)}"


@tool
async def get_company_profile(symbol: str) -> str:
    """Get detailed company profile including sector, industry, and description.
    
    Use this tool to get context about a company before analyzing insider trades.
    
    Args:
        symbol: The stock ticker symbol (e.g., 'AAPL').
        
    Returns:
        JSON string with company name, sector, industry, market cap, CEO, description.
    """
    try:
        profile = await insider_trade_service.get_company_profile(symbol)
        if not profile:
            return f"No profile found for {symbol}."
        return json.dumps(profile, indent=2)
    except Exception as e:
        return f"Error fetching profile for {symbol}: {str(e)}"


@tool
async def get_insider_roster(symbol: str) -> str:
    """Get list of all insiders (executives and directors) for a company.
    
    Use this tool to identify who the key insiders are before analyzing their trades.
    
    Args:
        symbol: The stock ticker symbol (e.g., 'AAPL').
        
    Returns:
        JSON string with list of insiders and their positions.
    """
    try:
        roster = await insider_trade_service.get_insider_roster(symbol)
        if not roster:
            return f"No insider roster found for {symbol}."
        return json.dumps(roster, indent=2)
    except Exception as e:
        return f"Error fetching insider roster for {symbol}: {str(e)}"


@tool
async def get_insider_statistics(symbol: str) -> str:
    """Get insider trading statistics summary for a company.
    
    Use this tool to get aggregated stats on insider buying vs selling activity.
    
    Args:
        symbol: The stock ticker symbol (e.g., 'AAPL').
        
    Returns:
        JSON string with buy/sell counts, volumes, and trends.
    """
    try:
        stats = await insider_trade_service.get_insider_statistics(symbol)
        if not stats:
            return f"No insider statistics found for {symbol}."
        return json.dumps(stats, indent=2)
    except Exception as e:
        return f"Error fetching insider statistics for {symbol}: {str(e)}"

