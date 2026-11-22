from typing import Any

from sqlalchemy import ForeignKey, Text, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from conversational_toolkit.conversation_database.data_models.source import Source, SourceDatabase
from conversational_toolkit.conversation_database.postgres.index import Base, PostgreSQLDatabase


class SourceTable(Base):
    __tablename__ = "sources"
    id: Mapped[str] = mapped_column(primary_key=True)
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id"))
    content: Mapped[str] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB)
    message = relationship("MessageTable", back_populates="sources")

    def to_model(self) -> Source:
        return Source(id=self.id, message_id=self.message_id, content=self.content, metadata=self.metadata_)


class PostgreSQLSourceDatabase(PostgreSQLDatabase, SourceDatabase):
    async def create_source(self, source: Source) -> Source:
        async with self.make_session() as session, session.begin():
            session.add(SourceTable(**source.model_dump(exclude={"metadata"}), metadata_=source.metadata))
        return source

    async def get_sources_by_message_id(self, message_id: str) -> list[Source]:
        async with self.make_session() as session:
            result = await session.execute(select(SourceTable).filter_by(message_id=message_id))
        return [source.to_model() for source in result.scalars()]

    async def delete_sources(self, source_ids: list[str]) -> bool:
        async with self.make_session() as session, session.begin():
            for row in (await session.execute(select(SourceTable).where(SourceTable.id.in_(source_ids)))).scalars():
                await session.delete(row)
        return True
