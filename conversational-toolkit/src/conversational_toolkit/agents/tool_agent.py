import json
from collections.abc import AsyncGenerator

from loguru import logger

from conversational_toolkit.agents.base import Agent, AgentAnswer, QueryWithContext
from conversational_toolkit.chunking.base import Chunk
from conversational_toolkit.llms.base import LLMMessage, MessageContent, Roles, ToolCall


class ToolAgent(Agent):
    """ReAct loop over the tools on 'self.llm': call the LLM, run the requested tools, repeat until no tool calls.

    Tools receive '_query' and '_history' besides the LLM's arguments. A tool that raises returns '{"error": ...}' to
    the LLM instead. Keys starting with '_' are hidden from the LLM; chunks under '_sources' are attached to the answer.
    """

    async def answer_stream(self, query_with_context: QueryWithContext) -> AsyncGenerator[AgentAnswer, None]:
        sources: list[Chunk] = []
        messages = [
            LLMMessage(role=Roles.SYSTEM, content=[MessageContent(type="text", text=self.system_prompt)]),
            *query_with_context.history,
            LLMMessage(role=Roles.USER, content=[MessageContent(type="text", text=query_with_context.query)]),
        ]
        tools = {tool.name: tool for tool in self.llm.tools or []}

        for _ in range(self.max_steps):
            tool_calls: list[ToolCall] = []
            content = ""
            async for response_chunk in self.llm.generate_stream(messages):
                if response_chunk.content:
                    content += response_chunk.content[0].text or ""
                    yield await self._answer_post_processing(
                        AgentAnswer(
                            content=[MessageContent(type="text", text=content)],
                            role=Roles.ASSISTANT,
                            sources=sources.copy(),
                        )
                    )
                if response_chunk.tool_calls:
                    tool_calls += response_chunk.tool_calls

            messages.append(
                LLMMessage(
                    role=Roles.ASSISTANT, content=[MessageContent(type="text", text=content)], tool_calls=tool_calls
                )
            )
            if not tool_calls:
                return

            for tool_call in tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                    result = await tools[name].call(
                        {"_query": query_with_context.query, "_history": query_with_context.history, **args}
                    )
                except Exception as e:
                    logger.warning(f"Tool {name} failed: {e}")
                    result = {"error": str(e)}
                sources += [Chunk.model_validate(chunk) for chunk in result.get("_sources", [])]
                logger.debug(f"Tool {name} returned: {result}")
                visible = {k: v for k, v in result.items() if not k.startswith("_")}
                messages.append(self.build_tool_answer(tool_call.id, name, visible))

        yield AgentAnswer(
            content=[MessageContent(type="text", text="Request is too complex to execute")], role=Roles.ASSISTANT
        )
