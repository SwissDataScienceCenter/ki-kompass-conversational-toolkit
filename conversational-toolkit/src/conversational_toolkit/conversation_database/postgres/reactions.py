from sqlalchemy import ForeignKey, Text, select
from sqlalchemy.orm import Mapped, mapped_column, relationship

from conversational_toolkit.conversation_database.data_models.reaction import Reaction, ReactionDatabase
from conversational_toolkit.conversation_database.postgres.index import Base, PostgreSQLDatabase


class ReactionTable(Base):
    __tablename__ = "reactions"
    id: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id"))
    content: Mapped[str] = mapped_column(Text)
    note: Mapped[str | None] = mapped_column(Text)
    message = relationship("MessageTable", back_populates="reactions")

    def to_model(self) -> Reaction:
        return Reaction(
            id=self.id, user_id=self.user_id, message_id=self.message_id, content=self.content, note=self.note
        )


class PostgreSQLReactionDatabase(PostgreSQLDatabase, ReactionDatabase):
    async def create_reaction(self, reaction: Reaction) -> Reaction:
        async with self.make_session() as session, session.begin():
            session.add(ReactionTable(**reaction.model_dump()))
        return reaction

    async def get_reactions_by_message_id(self, message_id: str) -> list[Reaction]:
        async with self.make_session() as session:
            result = await session.execute(select(ReactionTable).filter_by(message_id=message_id))
        return [reaction.to_model() for reaction in result.scalars()]

    async def delete_reactions(self, reaction_ids: list[str]) -> bool:
        async with self.make_session() as session, session.begin():
            for row in (
                await session.execute(select(ReactionTable).where(ReactionTable.id.in_(reaction_ids)))
            ).scalars():
                await session.delete(row)
        return True
