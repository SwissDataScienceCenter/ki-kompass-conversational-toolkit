from collections.abc import Sequence
from typing import Any

from conversational_toolkit.agents.base import Agent, QueryWithContext
from conversational_toolkit.chunking.base import Chunk
from conversational_toolkit.retriever.base import Retriever
from conversational_toolkit.tools.base import Tool


def _schema(**properties: tuple[str, str]) -> dict[str, Any]:
    """Object schema with all properties required, given as name=(type, description)."""
    return {
        "type": "object",
        "properties": {
            name: {"type": type_, "description": description}
            for name, (type_, description) in properties.items()
        },
        "required": list(properties),
        "additionalProperties": False,
    }


def chunks_to_text(chunks: Sequence[Chunk]) -> str:
    """Format chunks as markdown sections separated by a rule."""
    return "\n\n---\n\n".join(
        f"## {c.metadata.get('source_file', '?')}: {c.title}\n```\n{c.content}\n```"
        for c in chunks
    )


def _source_dicts(chunks: Sequence[Chunk]) -> list[dict[str, Any]]:
    return [
        c.model_dump(include={"title", "content", "mime_type", "metadata"})
        for c in chunks
    ]


class SumTwoNumbers(Tool):
    name = "sum_two_numbers"
    description = "Sum two numbers."
    parameters = _schema(
        number_1=("number", "The first number."),
        number_2=("number", "The second number."),
    )

    async def call(self, args: dict[str, Any]) -> dict[str, Any]:
        a, b = args.get("number_1"), args.get("number_2")
        if not isinstance(a, int | float) or not isinstance(b, int | float):
            raise ValueError("number_1 and number_2 must be numbers.")
        return {"result": a + b}


class HalfBoldText(Tool):
    name = "half_bold_text"
    description = "Bold every other word of a text, starting with the first."
    parameters = _schema(text=("string", "The text to modify."))

    async def call(self, args: dict[str, Any]) -> dict[str, Any]:
        text = args.get("text")
        if not isinstance(text, str):
            raise ValueError("text must be a string.")
        words = text.split()
        return {
            "result": " ".join(
                f"**{w}**" if i % 2 == 0 else w for i, w in enumerate(words)
            )
        }


class RetrieveRelevantChunks(Tool):
    name = "retrieve_relevant_chunks"
    description = "Retrieve the document chunks most relevant to a query."
    parameters = _schema(query=("string", "The query to retrieve relevant chunks for."))

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    async def call(self, args: dict[str, Any]) -> dict[str, Any]:
        chunks = await self.retriever.retrieve(args["query"])
        return {"result": chunks_to_text(chunks), "_sources": _source_dicts(chunks)}


class RAGAgentAsTool(Tool):
    name = "rag_agent"
    description = (
        "Answer a question about PrimePack products or documents with a RAG agent."
    )
    parameters = _schema(
        query=(
            "string",
            "A self-contained question about PrimePack products or documents.",
        )
    )

    def __init__(self, agent: Agent) -> None:
        self.agent = agent

    async def call(self, args: dict[str, Any]) -> dict[str, Any]:
        answer = await self.agent.answer(
            QueryWithContext(query=args["query"], history=[])
        )
        return {
            "result": answer.content[0].text if answer.content else "",
            "_sources": _source_dicts(answer.sources),
        }
