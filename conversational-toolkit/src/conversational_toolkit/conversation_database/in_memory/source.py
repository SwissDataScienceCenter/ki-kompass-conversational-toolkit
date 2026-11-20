from conversational_toolkit.conversation_database.data_models.source import Source, SourceDatabase
from conversational_toolkit.conversation_database.in_memory.base import JSONFileStore


class InMemorySourceDatabase(JSONFileStore[Source], SourceDatabase):
    model = Source

    async def create_source(self, source: Source) -> Source:
        return self._put(source)

    async def get_sources_by_message_id(self, message_id: str) -> list[Source]:
        return self._filter(message_id=message_id)

    async def delete_sources(self, source_ids: list[str]) -> bool:
        self._delete(source_ids)
        return True
