"""This file contains the graph schema for the application."""

from typing import Annotated, Literal, Optional

from langgraph.graph.message import add_messages
from pydantic import (
    BaseModel,
    Field,
)


class GraphState(BaseModel):
    """State definition for the LangGraph Agent/Workflow."""

    messages: Annotated[list, add_messages] = Field(
        default_factory=list, description="The messages in the conversation"
    )
    long_term_memory: str = Field(default="", description="The long term memory of the conversation")
    deep_analysis: bool = Field(default=False, description="Whether to use deep analysis mode (insider analyst)")
    route: Optional[Literal["chat", "insider_analyst"]] = Field(
        default=None, description="The determined route for this query"
    )
