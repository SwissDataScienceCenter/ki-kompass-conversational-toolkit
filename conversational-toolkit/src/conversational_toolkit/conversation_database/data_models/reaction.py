from abc import ABC, abstractmethod

from pydantic import BaseModel


class Reaction(BaseModel):
    """User feedback on an assistant message; a new reaction replaces the user's previous one."""

    id: str
    user_id: str
    message_id: str
    content: str
    note: str | None = None


class ReactionDatabase(ABC):
    @abstractmethod
    async def create_reaction(self, reaction: Reaction) -> Reaction:
        pass

    @abstractmethod
    async def get_reactions_by_message_id(self, message_id: str) -> list[Reaction]:
        pass

    @abstractmethod
    async def delete_reactions(self, reaction_ids: list[str]) -> bool:
        pass
