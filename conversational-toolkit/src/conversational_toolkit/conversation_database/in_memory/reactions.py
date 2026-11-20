from conversational_toolkit.conversation_database.data_models.reaction import Reaction, ReactionDatabase
from conversational_toolkit.conversation_database.in_memory.base import JSONFileStore


class InMemoryReactionDatabase(JSONFileStore[Reaction], ReactionDatabase):
    model = Reaction

    async def create_reaction(self, reaction: Reaction) -> Reaction:
        return self._put(reaction)

    async def get_reactions_by_message_id(self, message_id: str) -> list[Reaction]:
        return self._filter(message_id=message_id)

    async def delete_reactions(self, reaction_ids: list[str]) -> bool:
        self._delete(reaction_ids)
        return True
