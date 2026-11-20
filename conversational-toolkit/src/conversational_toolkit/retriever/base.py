from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from conversational_toolkit.chunking.base import Chunk

T_co = TypeVar("T_co", bound=Chunk, covariant=True)


class Retriever(ABC, Generic[T_co]):
    """Returns the 'top_k' chunks most relevant to a query."""

    def __init__(self, top_k: int):
        self.top_k = top_k

    @abstractmethod
    async def retrieve(self, query: str) -> list[T_co]:
        """Return up to 'top_k' chunks, best first."""
