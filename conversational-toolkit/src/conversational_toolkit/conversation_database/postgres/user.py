from sqlalchemy.orm import Mapped, mapped_column, relationship

from conversational_toolkit.conversation_database.data_models.user import User, UserDatabase
from conversational_toolkit.conversation_database.postgres.index import Base, PostgreSQLDatabase


class UserTable(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(primary_key=True)
    conversations = relationship("ConversationTable", order_by="ConversationTable.id", back_populates="user")
    messages = relationship("MessageTable", order_by="MessageTable.id", back_populates="user")


class PostgreSQLUserDatabase(PostgreSQLDatabase, UserDatabase):
    async def create_user(self, user: User) -> User:
        async with self.make_session() as session, session.begin():
            session.add(UserTable(id=user.id))
        return user

    async def get_user_by_id(self, user_id: str) -> User | None:
        async with self.make_session() as session:
            user = await session.get(UserTable, user_id)
        return User(id=str(user.id)) if user else None
