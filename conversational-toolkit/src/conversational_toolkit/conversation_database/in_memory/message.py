from conversational_toolkit.conversation_database.data_models.message import Message, MessageDatabase
from conversational_toolkit.conversation_database.in_memory.base import JSONFileStore


class InMemoryMessageDatabase(JSONFileStore[Message], MessageDatabase):
    model = Message

    async def create_message(self, message: Message) -> Message:
        return self._put(message)

    async def get_messages_by_conversation_id(self, conversation_id: str) -> list[Message]:
        return self._filter(conversation_id=conversation_id)

    async def get_message_by_id(self, message_id: str) -> Message:
        return self._get(message_id)

    async def delete_message(self, message_id: str) -> bool:
        return self._delete([message_id])
