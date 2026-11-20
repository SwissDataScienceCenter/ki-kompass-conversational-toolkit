from sqlalchemy import Column, ForeignKey, String, Text, select
from sqlalchemy.orm import relationship

from conversational_toolkit.conversation_database.data_models.reaction import Reaction, ReactionDatabase
from conversational_toolkit.conversation_database.postgres.index import Base, PostgreSQLDatabase


class ReactionTable(Base):
    __tablename__ = "reactions"
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))
    message_id = Column(String, ForeignKey("messages.id"))
    content = Column(Text)
    note = Column(Text, nullable=True)
    message = relationship("MessageTable", back_populates="reactions")

    def to_model(self) -> Reaction:
        return Reaction(
            id=str(self.id),
            user_id=str(self.user_id),
            message_id=str(self.message_id),
            content=str(self.content),
            note=self.note,  # type: ignore[arg-type]
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
