from conversational_toolkit.conversation_database.data_models.conversation import Conversation, ConversationDatabase
from conversational_toolkit.conversation_database.in_memory.base import JSONFileStore


class InMemoryConversationDatabase(JSONFileStore[Conversation], ConversationDatabase):
    model = Conversation

    async def create_conversation(self, conversation: Conversation) -> Conversation:
        return self._put(conversation)

    async def get_conversations_by_user_id(self, user_id: str) -> list[Conversation]:
        return self._filter(user_id=user_id)

    async def get_conversation_by_id(self, conversation_id: str) -> Conversation:
        return self._get(conversation_id)

    async def update_conversation(self, conversation: Conversation) -> Conversation:
        self._get(conversation.id)
        return self._put(conversation)

    async def delete_conversation(self, conversation_id: str) -> bool:
        return self._delete([conversation_id])
