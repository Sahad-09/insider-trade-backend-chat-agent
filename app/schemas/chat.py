"""This file contains the chat schema for the application."""

import re
from typing import (
    List,
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


class Message(BaseModel):
    """Message model for chat endpoint.

    Attributes:
        role: The role of the message sender (user or assistant).
        content: The content of the message.
    """

    model_config = {"extra": "ignore"}

    role: Literal["user", "assistant", "system"] = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message", min_length=1, max_length=3000)

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate the message content.

        Args:
            v: The content to validate

        Returns:
            str: The validated content

        Raises:
            ValueError: If the content contains disallowed patterns
        """
        # Check for potentially harmful content
        if re.search(r"<script.*?>.*?</script>", v, re.IGNORECASE | re.DOTALL):
            raise ValueError("Content contains potentially harmful script tags")

        # Check for null bytes
        if "\0" in v:
            raise ValueError("Content contains null bytes")

        return v


class ChatRequest(BaseModel):
    """Request model for chat endpoint.

    Attributes:
        messages: List of messages in the conversation.
        use_reasoning: Whether to use o3-mini reasoning model (optional).
        deep_analysis: Whether to use deep analysis mode with SEC analyst.
    """

    messages: List[Message] = Field(
        ...,
        description="List of messages in the conversation",
        min_length=1,
    )
    use_reasoning: bool = Field(
        default=False,
        description="Use o3-mini reasoning model for more complex reasoning tasks"
    )
    reasoning_effort: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Reasoning effort level for o3-mini (only used when use_reasoning=True)"
    )
    deep_analysis: bool = Field(
        default=False,
        description="Use deep analysis mode with specialized SEC insider trading analyst"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint.

    Attributes:
        messages: List of messages in the conversation.
    """

    messages: List[Message] = Field(..., description="List of messages in the conversation")


class StreamResponse(BaseModel):
    """Response model for streaming chat endpoint.

    Attributes:
        content: The content of the current chunk.
        done: Whether the stream is complete.
    """

    content: str = Field(default="", description="The content of the current chunk")
    done: bool = Field(default=False, description="Whether the stream is complete")


class StreamThinkingResponse(BaseModel):
    """Response model for streaming chat endpoint with thinking states.

    Attributes:
        thinking_title: The title/description of the current thinking state.
        response: The response content (empty during thinking, filled when done).
        status: The status of the stream ("thinking", "reasoning", or "done").
        reasoning: List of reasoning steps from o3-mini model (optional).
        model: The model used to generate this response (optional).
    """

    thinking_title: str = Field(default="", description="The title/description of the current thinking state")
    response: str = Field(default="", description="The response content")
    status: Literal["thinking", "reasoning", "done"] = Field(..., description="The status of the stream")
    reasoning: List[str] = Field(default_factory=list, description="Reasoning steps from o3-mini model")
    model: str = Field(default="", description="The model used to generate this response")