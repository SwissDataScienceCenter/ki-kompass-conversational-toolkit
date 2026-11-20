import numpy as np
from loguru import logger
from numpy.typing import NDArray
from openai import AsyncOpenAI, omit

from conversational_toolkit.embeddings.base import EmbeddingsModel


class OpenAIEmbeddings(EmbeddingsModel):
    """OpenAI embeddings. 'dimensions' truncates the output (text-embedding-3 models only)."""

    def __init__(self, model_name: str, dimensions: int | None = None):
        self.client = AsyncOpenAI()
        self.model_name = model_name
        self.dimensions = dimensions
        logger.debug(f"OpenAI embeddings model loaded: {model_name}")

    async def get_embeddings(
        self, texts: str | list[str], batch_size: int = 100, max_chars: int = 30_000
    ) -> NDArray[np.float64]:
        """Embed texts in batches of 'batch_size', truncating each to 'max_chars' to stay under the token limit."""
        if isinstance(texts, str):
            texts = [texts]
        texts = [t[:max_chars] for t in texts]
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            response = await self.client.embeddings.create(
                input=texts[i : i + batch_size],
                model=self.model_name,
                dimensions=self.dimensions if self.dimensions is not None else omit,
            )
            embeddings.extend(d.embedding for d in response.data)
        return np.asarray(embeddings)
