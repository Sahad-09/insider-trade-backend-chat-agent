"""Service for interacting with Financial Modeling Prep API for insider trading data.

Uses the new stable API format: https://financialmodelingprep.com/stable/{endpoint}
Documentation: https://site.financialmodelingprep.com/developer/docs
"""

from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import logger


class InsiderTradeService:
    """Service to fetch insider trading data from Financial Modeling Prep.
    
    Uses the new stable API endpoints (not legacy v3/v4).
    """

    # New stable API base URL
    BASE_URL = "https://financialmodelingprep.com/stable/"

    def __init__(self):
        """Initialize the service with API key."""
        self.api_key = settings.FINANCIAL_MODELING_PREP_API_KEY
        if not self.api_key:
            logger.warning("financial_modeling_prep_api_key_missing")

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for requests."""
        return {"Content-Type": "application/json"}

    def _get_params(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get query parameters including API key."""
        p = params or {}
        p["apikey"] = self.api_key
        return p

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Any:
        """Make an async request to the stable API."""
        if not self.api_key:
            raise ValueError("Financial Modeling Prep API key is not set")

        url = f"{self.BASE_URL}{endpoint}"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    params=self._get_params(params),
                    headers=self._get_headers(),
                    timeout=15.0
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(
                    "fmp_api_error",
                    status_code=e.response.status_code,
                    endpoint=endpoint,
                    url=url,
                    error=str(e),
                    response_text=e.response.text[:500] if e.response.text else None,
                )
                raise
            except Exception as e:
                logger.error("fmp_request_failed", endpoint=endpoint, error=str(e))
                raise

    async def get_insider_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent insider trades for a specific company.
        
        Endpoint: /stable/insider-trading?symbol=AAPL
        
        Args:
            symbol: The stock ticker symbol (e.g., AAPL)
            limit: Number of records to return
            
        Returns:
            List of insider trading records with fields:
            - symbol, filingDate, transactionDate, reportingName, transactionType,
            - securitiesOwned, securitiesTransacted, price, link
        """
        return await self._make_request(
            "insider-trading",
            params={"symbol": symbol.upper(), "limit": limit}
        )

    async def get_recent_insider_activity(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent insider trading activity across the market (RSS feed).
        
        Endpoint: /stable/insider-trading-rss-feed
        
        Args:
            limit: Number of records to return
            
        Returns:
            List of recent insider trading records market-wide
        """
        return await self._make_request(
            "insider-trading-rss-feed",
            params={"limit": limit}
        )

    async def get_stock_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time stock quote.
        
        Endpoint: /stable/quote?symbol=AAPL
        
        Args:
            symbol: The stock ticker symbol
            
        Returns:
            Stock quote data with price, change, volume, etc.
        """
        data = await self._make_request("quote", params={"symbol": symbol.upper()})
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return data if isinstance(data, dict) else {}

    async def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """Get company profile data.
        
        Endpoint: /stable/profile?symbol=AAPL
        
        Args:
            symbol: The stock ticker symbol
            
        Returns:
            Company profile with name, sector, industry, market cap, etc.
        """
        data = await self._make_request("profile", params={"symbol": symbol.upper()})
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return data if isinstance(data, dict) else {}

    async def get_sec_filings(self, symbol: str, filing_type: str = "4", limit: int = 50) -> List[Dict[str, Any]]:
        """Get SEC filings for a company.
        
        Endpoint: /stable/sec-filings?symbol=AAPL&type=4
        
        Args:
            symbol: The stock ticker symbol
            filing_type: SEC form type (e.g., "4" for Form 4 insider trades)
            limit: Number of records to return
            
        Returns:
            List of SEC filings
        """
        return await self._make_request(
            "sec-filings",
            params={"symbol": symbol.upper(), "type": filing_type, "limit": limit}
        )

    async def get_insider_roster(self, symbol: str) -> List[Dict[str, Any]]:
        """Get list of insiders for a company.
        
        Endpoint: /stable/insider-roaster?symbol=AAPL
        
        Args:
            symbol: The stock ticker symbol
            
        Returns:
            List of company insiders with their positions
        """
        return await self._make_request(
            "insider-roaster",
            params={"symbol": symbol.upper()}
        )

    async def get_insider_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get insider trading statistics for a company.
        
        Endpoint: /stable/insider-roaster-statistics?symbol=AAPL
        
        Args:
            symbol: The stock ticker symbol
            
        Returns:
            Statistics on insider buying/selling activity
        """
        data = await self._make_request(
            "insider-roaster-statistics",
            params={"symbol": symbol.upper()}
        )
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return data if isinstance(data, dict) else {}


insider_trade_service = InsiderTradeService()

