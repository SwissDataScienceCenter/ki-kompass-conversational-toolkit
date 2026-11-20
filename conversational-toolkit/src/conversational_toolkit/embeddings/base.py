from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class EmbeddingsModel(ABC):
    """Text embedding model."""

    @abstractmethod
    async def get_embeddings(self, texts: str | list[str]) -> NDArray[np.float64]:
        """Embed one or more texts into an array of shape '(n, embedding_size)'."""
