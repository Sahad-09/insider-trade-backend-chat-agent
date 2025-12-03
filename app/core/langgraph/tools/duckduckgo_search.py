"""DuckDuckGo search tool for LangGraph.

This module provides a DuckDuckGo search tool that can be used with LangGraph
to perform web searches. It returns up to 10 search results and handles errors
gracefully.
"""

from typing import Any

from langchain_community.tools import DuckDuckGoSearchResults

from app.core.logging import logger


class LoggedDuckDuckGoSearchResults(DuckDuckGoSearchResults):
    """DuckDuckGo search tool with debug logging."""

    def _run(self, query: str, run_manager: Any = None) -> str:
        """Execute the search synchronously with logging."""
        # Yellow background, black text for DuckDuckGo tool
        logger.info(
            "\033[43m\033[30m[TOOL: DUCKDUCKGO_SEARCH]\033[0m duckduckgo_search_called",
            query=query,
            tool_name=self.name,
        )
        try:
            result = super()._run(query, run_manager)
            logger.debug(
                "duckduckgo_search_completed",
                query=query,
                result_length=len(str(result)),
                tool_name=self.name,
            )
            logger.info(
                "\033[43m\033[30m[TOOL: DUCKDUCKGO_SEARCH]\033[0m duckduckgo_search_success",
                query=query,
                result_preview=str(result)[:200] + "..." if len(str(result)) > 200 else str(result),
            )
            return result
        except Exception as e:
            logger.error(
                "\033[43m\033[30m[TOOL: DUCKDUCKGO_SEARCH]\033[0m duckduckgo_search_failed",
                query=query,
                error=str(e),
                tool_name=self.name,
                exc_info=True,
            )
            raise

    async def _arun(self, query: str, run_manager: Any = None) -> str:
        """Execute the search asynchronously with logging."""
        # Yellow background, black text for DuckDuckGo tool
        logger.info(
            "\033[43m\033[30m[TOOL: DUCKDUCKGO_SEARCH]\033[0m duckduckgo_search_called_async",
            query=query,
            tool_name=self.name,
        )
        try:
            result = await super()._arun(query, run_manager)
            logger.debug(
                "duckduckgo_search_completed_async",
                query=query,
                result_length=len(str(result)),
                tool_name=self.name,
            )
            logger.info(
                "\033[43m\033[30m[TOOL: DUCKDUCKGO_SEARCH]\033[0m duckduckgo_search_success_async",
                query=query,
                result_preview=str(result)[:200] + "..." if len(str(result)) > 200 else str(result),
            )
            return result
        except Exception as e:
            logger.error(
                "\033[43m\033[30m[TOOL: DUCKDUCKGO_SEARCH]\033[0m duckduckgo_search_failed_async",
                query=query,
                error=str(e),
                tool_name=self.name,
                exc_info=True,
            )
            raise


duckduckgo_search_tool = LoggedDuckDuckGoSearchResults(num_results=10, handle_tool_error=True)
