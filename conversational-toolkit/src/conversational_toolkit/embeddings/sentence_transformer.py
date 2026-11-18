from typing import Union, Any
from loguru import logger
from numpy._typing import NDArray
from sentence_transformers import SentenceTransformer
from conversational_toolkit.embeddings.base import EmbeddingsModel
import numpy as np


class SentenceTransformerEmbeddings(EmbeddingsModel):
    def __init__(self, model_name: str, **kwargs: Any):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, **kwargs)
        self.model.eval()
        logger.debug(f"Sentence Transformer embeddings model loaded: {model_name} with kwargs: {kwargs}")

    async def get_embeddings(self, texts: Union[str, list[str]], **kwargs_encode: Any) -> NDArray[np.float64]:
        """Encode a string or a list of strings into embeddings using the model."""
        # Encode the texts using the model
        embedded_chunk = self.model.encode(texts, **kwargs_encode)

        # If a single string is given, convert the output to a numpy array of numpy arrays
        if isinstance(texts, str):
            embedded_chunk = np.array([embedded_chunk])

        logger.debug(f"{self.model_name} embeddings size: {embedded_chunk.shape}")
        return embedded_chunk  # type: ignore
