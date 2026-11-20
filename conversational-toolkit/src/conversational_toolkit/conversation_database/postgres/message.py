from sqlalchemy import BigInteger, Column, ForeignKey, String, Text, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from conversational_toolkit.conversation_database.data_models.message import Message, MessageDatabase
from conversational_toolkit.conversation_database.postgres.index import Base, PostgreSQLDatabase
from conversational_toolkit.llms.base import Roles


class MessageTable(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))
    conversation_id = Column(String, ForeignKey("conversations.id"))
    content = Column(Text)
    role = Column(String)
    create_timestamp = Column(BigInteger)
    parent_id = Column(String, ForeignKey("messages.id"))
    metadata_ = Column("metadata", JSONB)
    conversation = relationship("ConversationTable", back_populates="messages")
    user = relationship("UserTable", back_populates="messages")
    sources = relationship("SourceTable", order_by="SourceTable.id", back_populates="message")
    reactions = relationship("ReactionTable", order_by="ReactionTable.id", back_populates="message")

    def to_model(self) -> Message:
        return Message(
            id=str(self.id),
            user_id=self.user_id,  # type: ignore[arg-type]
            conversation_id=str(self.conversation_id),
            content=str(self.content),
            role=Roles(str(self.role)),
            create_timestamp=int(self.create_timestamp),  # type: ignore[arg-type]
            parent_id=self.parent_id,  # type: ignore[arg-type]
            metadata=self.metadata_,  # type: ignore[arg-type]
        )


class PostgreSQLMessageDatabase(PostgreSQLDatabase, MessageDatabase):
    async def create_message(self, message: Message) -> Message:
        async with self.make_session() as session, session.begin():
            session.add(MessageTable(**message.model_dump(exclude={"metadata"}), metadata_=message.metadata))
        return message

    async def get_messages_by_conversation_id(self, conversation_id: str) -> list[Message]:
        async with self.make_session() as session:
            result = await session.execute(select(MessageTable).filter_by(conversation_id=conversation_id))
        return [message.to_model() for message in result.scalars()]

    async def get_message_by_id(self, message_id: str) -> Message:
        async with self.make_session() as session:
            message = await session.get(MessageTable, message_id)
        if message is None:
            raise ValueError(f"Message with id {message_id} not found")
        return message.to_model()

    async def delete_message(self, message_id: str) -> bool:
        async with self.make_session() as session, session.begin():
            row = await session.get(MessageTable, message_id)
            if row is None:
                return False
            await session.delete(row)
        return True
