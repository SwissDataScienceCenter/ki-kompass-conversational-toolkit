from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Sequence
from typing import Any

from pydantic import BaseModel, Field

from conversational_toolkit.chunking.base import Chunk
from conversational_toolkit.llms.base import LLM, LLMMessage, MessageContent, Roles


class QueryWithContext(BaseModel):
    query: str
    history: list[LLMMessage]


class AgentAnswer(LLMMessage):
    """An 'LLMMessage' plus the sources and follow-up questions behind it."""

    step_by_step_thinking: str = ""
    sources: Sequence[Chunk] = Field(default_factory=list)
    follow_up_questions: Sequence[str] = Field(default_factory=list)


class Agent(ABC):
    """Turns a 'QueryWithContext' into a stream of 'AgentAnswer's.

    Attributes:
        description: What the agent is for; 'Router' uses it to pick an agent.
        max_steps: Maximum tool-calling iterations ('ToolAgent' only).
    """

    def __init__(self, system_prompt: str, llm: LLM, description: str = "", max_steps: int = 20):
        self.description = description
        self.system_prompt = system_prompt
        self.llm = llm
        self.max_steps = max_steps

    def build_tool_answer(self, tool_call_id: str, function_name: str, function_response: dict[str, Any]) -> LLMMessage:
        return LLMMessage(
            tool_call_id=tool_call_id,
            role=Roles.TOOL,
            name=function_name,
            content=[MessageContent(type="text", text=str(function_response))],
        )

    async def answer(self, query_with_context: QueryWithContext) -> AgentAnswer:
        """Consume 'answer_stream' and return its last answer."""
        response_message = None
        async for message in self.answer_stream(query_with_context):
            response_message = message
        if response_message is None:
            raise ValueError("No response received from the agent.")
        return response_message

    @abstractmethod
    def answer_stream(self, query_with_context: QueryWithContext) -> AsyncGenerator[AgentAnswer, None]:
        """Yield growing partial answers; the last one is complete and carries the sources."""

    async def _answer_post_processing(self, answer: AgentAnswer) -> AgentAnswer:
        """Hook applied to every streamed answer; identity by default."""
        return answer
