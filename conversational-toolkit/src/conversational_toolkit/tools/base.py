from abc import ABC, abstractmethod
from typing import Any, Literal, TypedDict


class FunctionDescription(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolDescription(TypedDict):
    type: Literal["function"]
    function: FunctionDescription


class Tool(ABC):
    """A function the LLM can call; 'parameters' is the JSON schema of its arguments."""

    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    async def call(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run the tool. 'ToolAgent' adds '_query' and '_history' to 'args'.

        Keys starting with '_' are not shown to the LLM; chunks under '_sources' are attached to the answer.
        """

    def json_schema(self) -> ToolDescription:
        """Tool descriptor in OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }
