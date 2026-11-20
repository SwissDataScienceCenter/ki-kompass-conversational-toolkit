"""LLM interface and backend-agnostic message models."""

from abc import ABC, abstractmethod
from enum import StrEnum
from collections.abc import AsyncGenerator

from pydantic import BaseModel

from conversational_toolkit.tools.base import Tool


class Roles(StrEnum):
    """Conversation roles as used by the OpenAI chat completions API."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DEVELOPER = "developer"


class Function(BaseModel):
    """The function name and JSON-encoded arguments inside a tool call."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """A single tool invocation requested by the LLM."""

    id: str
    function: Function
    type: str


class MessageContent(BaseModel):
    type: str
    text: str | None = None
    image_url: str | None = None


class LLMMessage(BaseModel):
    """A chat message. 'tool_call_id' and 'name' are set on TOOL messages that answer a tool call."""

    content: list[MessageContent]
    role: Roles = Roles.ASSISTANT
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class LLM(ABC):
    """Language model backend. 'tools' are exposed to the model and executed by 'ToolAgent'."""

    def __init__(self) -> None:
        self.tools: list[Tool] | None = []

    @abstractmethod
    async def generate(self, conversation: list[LLMMessage]) -> LLMMessage:
        """Return the complete response to 'conversation'."""

    @abstractmethod
    def generate_stream(self, conversation: list[LLMMessage]) -> AsyncGenerator[LLMMessage, None]:
        """Yield response chunks as they arrive."""
