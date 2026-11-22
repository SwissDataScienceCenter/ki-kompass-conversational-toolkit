from typing import Any

from sqlalchemy import BigInteger, ForeignKey, Text, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from conversational_toolkit.conversation_database.data_models.message import Message, MessageDatabase
from conversational_toolkit.conversation_database.postgres.index import Base, PostgreSQLDatabase
from conversational_toolkit.llms.base import Roles


class MessageTable(Base):
    __tablename__ = "messages"
    id: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"))
    content: Mapped[str] = mapped_column(Text)
    role: Mapped[str]
    create_timestamp: Mapped[int] = mapped_column(BigInteger)
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("messages.id"))
    metadata_: Mapped[list[dict[str, Any]] | None] = mapped_column("metadata", JSONB)
    conversation = relationship("ConversationTable", back_populates="messages")
    user = relationship("UserTable", back_populates="messages")
    sources = relationship("SourceTable", order_by="SourceTable.id", back_populates="message")
    reactions = relationship("ReactionTable", order_by="ReactionTable.id", back_populates="message")

    def to_model(self) -> Message:
        return Message(
            id=self.id,
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            content=self.content,
            role=Roles(self.role),
            create_timestamp=self.create_timestamp,
            parent_id=self.parent_id,
            metadata=self.metadata_,
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
