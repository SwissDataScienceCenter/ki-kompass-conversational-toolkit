from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class PostgreSQLDatabase:
    """Shared engine/session setup. Requires the 'postgres' extra; call 'create_table' once before use."""

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine
        self.make_session = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)

    async def create_table(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
