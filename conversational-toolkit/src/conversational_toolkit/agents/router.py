import json
from collections.abc import AsyncGenerator

from conversational_toolkit.agents.base import Agent, AgentAnswer, QueryWithContext
from conversational_toolkit.llms.base import LLM, LLMMessage, MessageContent, Roles


class Router(Agent):
    """Lets the LLM pick the sub-agent whose 'description' best fits the query, then delegates to it.

    The LLM must answer with JSON (e.g. 'response_format={"type": "json_object"}').
    """

    def __init__(self, llm: LLM, agents: list[Agent], description: str = "") -> None:
        self.agents = agents
        super().__init__(self.construct_system_prompt(), llm, description)

    def construct_system_prompt(self) -> str:
        categories = "\n".join(f"Category {i}: {agent.description}" for i, agent in enumerate(self.agents, start=1))
        return (
            "Classify the user's query, given the conversation history, into the category of the agent best suited "
            f"to answer it.\n\n{categories}\n\n"
            'Answer with JSON: {"step_by_step_thinking": "<your reasoning>", "category": <category number>}'
        )

    async def _get_agent(self, query_with_context: QueryWithContext) -> Agent:
        routing = await self.llm.generate(
            [
                LLMMessage(role=Roles.SYSTEM, content=[MessageContent(type="text", text=self.system_prompt)]),
                *query_with_context.history,
                LLMMessage(role=Roles.USER, content=[MessageContent(type="text", text=query_with_context.query)]),
            ]
        )
        category = int(json.loads(routing.content[0].text or "")["category"])
        if not 1 <= category <= len(self.agents):
            raise ValueError(f"Router LLM returned unknown category {category}")
        return self.agents[category - 1]

    async def answer_stream(self, query_with_context: QueryWithContext) -> AsyncGenerator[AgentAnswer, None]:
        agent = await self._get_agent(query_with_context)
        async for response in agent.answer_stream(query_with_context):
            yield response
