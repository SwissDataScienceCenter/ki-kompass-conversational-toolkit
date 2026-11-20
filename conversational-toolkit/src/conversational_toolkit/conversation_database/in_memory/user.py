from conversational_toolkit.conversation_database.data_models.user import User, UserDatabase
from conversational_toolkit.conversation_database.in_memory.base import JSONFileStore


class InMemoryUserDatabase(JSONFileStore[User], UserDatabase):
    model = User

    async def create_user(self, user: User) -> User:
        return self._put(user)

    async def get_user_by_id(self, user_id: str) -> User | None:
        return self.records.get(user_id)
