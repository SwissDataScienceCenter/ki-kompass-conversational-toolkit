import asyncio
from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from conversational_toolkit.embeddings.base import EmbeddingsModel


class SentenceTransformerEmbeddings(EmbeddingsModel):
    def __init__(self, model_name: str, **kwargs: Any):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, **kwargs)
        logger.debug(f"Sentence Transformer embeddings model loaded: {model_name} with kwargs: {kwargs}")

    async def get_embeddings(self, texts: str | list[str], **kwargs_encode: Any) -> NDArray[np.float64]:
        """Encode texts in a worker thread; 'kwargs_encode' is passed to 'SentenceTransformer.encode'."""
        embeddings = await asyncio.to_thread(lambda: self.model.encode(texts, **kwargs_encode))
        return np.atleast_2d(embeddings)
