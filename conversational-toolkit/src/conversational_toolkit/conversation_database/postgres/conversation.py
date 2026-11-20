from sqlalchemy import BigInteger, Column, ForeignKey, String, select
from sqlalchemy.orm import relationship

from conversational_toolkit.conversation_database.data_models.conversation import Conversation, ConversationDatabase
from conversational_toolkit.conversation_database.postgres.index import Base, PostgreSQLDatabase


class ConversationTable(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))
    create_timestamp = Column(BigInteger)
    update_timestamp = Column(BigInteger)
    title = Column(String)
    user = relationship("UserTable", back_populates="conversations")
    messages = relationship("MessageTable", order_by="MessageTable.id", back_populates="conversation")

    def to_model(self) -> Conversation:
        return Conversation(
            id=str(self.id),
            user_id=str(self.user_id),
            create_timestamp=int(self.create_timestamp),  # type: ignore[arg-type]
            update_timestamp=int(self.update_timestamp),  # type: ignore[arg-type]
            title=str(self.title),
        )


class PostgreSQLConversationDatabase(PostgreSQLDatabase, ConversationDatabase):
    async def create_conversation(self, conversation: Conversation) -> Conversation:
        async with self.make_session() as session, session.begin():
            session.add(ConversationTable(**conversation.model_dump()))
        return conversation

    async def get_conversations_by_user_id(self, user_id: str) -> list[Conversation]:
        async with self.make_session() as session:
            result = await session.execute(select(ConversationTable).filter_by(user_id=user_id))
        return [conversation.to_model() for conversation in result.scalars()]

    async def get_conversation_by_id(self, conversation_id: str) -> Conversation:
        async with self.make_session() as session:
            conversation = await session.get(ConversationTable, conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation with id {conversation_id} not found")
        return conversation.to_model()

    async def update_conversation(self, conversation: Conversation) -> Conversation:
        async with self.make_session() as session, session.begin():
            row = await session.get(ConversationTable, conversation.id)
            if row is None:
                raise ValueError(f"Conversation with id {conversation.id} not found")
            for key, value in conversation.model_dump(exclude={"id"}).items():
                setattr(row, key, value)
        return conversation

    async def delete_conversation(self, conversation_id: str) -> bool:
        async with self.make_session() as session, session.begin():
            row = await session.get(ConversationTable, conversation_id)
            if row is None:
                return False
            await session.delete(row)
        return True
