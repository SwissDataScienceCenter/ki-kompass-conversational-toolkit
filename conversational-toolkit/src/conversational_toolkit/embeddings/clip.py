import asyncio
import base64
import io

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling

from conversational_toolkit.chunking.base import Chunk

MODEL_NAME = "openai/clip-vit-base-patch32"


def _pooled(output: tuple | BaseModelOutputWithPooling) -> torch.Tensor:
    """Projected embedding of a CLIP get_*_features call (a model output since transformers 5)."""
    if not isinstance(output, BaseModelOutputWithPooling) or output.pooler_output is None:
        raise TypeError(f"Expected BaseModelOutputWithPooling with pooler_output, got {type(output).__name__}")
    return output.pooler_output


class CLIPEmbeddings:
    """CLIP text and image embeddings in a shared space."""

    def __init__(self) -> None:
        self.clip_model = CLIPModel.from_pretrained(MODEL_NAME)
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        self.tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

    @torch.no_grad()
    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        return _pooled(self.clip_model.get_text_features(**inputs)).numpy()

    @torch.no_grad()
    def _encode_images(self, images: list[str]) -> np.ndarray:
        decoded = [Image.open(io.BytesIO(base64.b64decode(img))) for img in images]
        inputs = self.processor(images=decoded, return_tensors="pt", padding=True)  # type: ignore[call-arg]
        return _pooled(self.clip_model.get_image_features(**inputs)).numpy()

    async def get_text_embeddings(self, texts: str | list[str]) -> np.ndarray:
        return await asyncio.to_thread(self._encode_texts, [texts] if isinstance(texts, str) else texts)

    async def get_image_embeddings(self, images: str | list[str]) -> np.ndarray:
        return await asyncio.to_thread(self._encode_images, [images] if isinstance(images, str) else images)

    async def get_embeddings(self, chunks: str | Chunk | list[Chunk]) -> np.ndarray:
        """Embed text and image chunks, batching each modality; rows keep the input order."""
        if isinstance(chunks, str):
            chunks = [Chunk(content=chunks, mime_type="text/plain", title="Query")]
        elif isinstance(chunks, Chunk):
            chunks = [chunks]

        rows: list[np.ndarray] = [np.empty(0)] * len(chunks)
        for prefix, encode in (("text", self.get_text_embeddings), ("image", self.get_image_embeddings)):
            idx = [i for i, c in enumerate(chunks) if c.mime_type.startswith(prefix)]
            if idx:
                for i, row in zip(idx, await encode([chunks[i].content for i in idx]), strict=True):
                    rows[i] = row
        if any(row.size == 0 for row in rows):
            raise ValueError("Chunks must have a text/* or image/* mime type")
        return np.stack(rows)
