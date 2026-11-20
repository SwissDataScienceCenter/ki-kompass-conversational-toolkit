import asyncio
import base64
import io
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from qwen_vl_utils.vision_process import process_vision_info  # type: ignore[import-untyped]
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

from dataclasses import dataclass
from transformers.utils.generic import ModelOutput

from conversational_toolkit.chunking.base import Chunk
from conversational_toolkit.embeddings.base import EmbeddingsModel


@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    attention_mask: torch.Tensor | None = None


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    _checkpoint_conversion_mapping: dict[str, str] = {}  # noqa: RUF012
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> Qwen3VLForEmbeddingOutput:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


class _Qwen3VLEmbedder:
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-VL-Embedding-2B",
        *,
        instruction: str = "Represent the user's input.",
        max_length: int = 8192,
        normalize: bool = True,
        output_dim: int | None = None,
        dtype: torch.dtype | None = None,
        attn_implementation: str | None = None,
        device: str | None = None,
        min_pixels: int = 4 * (16 * 2) * (16 * 2),
        max_pixels: int = 1800 * (16 * 2) * (16 * 2),
    ):
        self.max_length = max_length
        self.instruction = instruction
        self.normalize = normalize
        self.output_dim = output_dim
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)

        model_kwargs: dict[str, Any] = {}
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = Qwen3VLForEmbedding.from_pretrained(model_name_or_path, device_map=self.device, **model_kwargs)
        self.model.eval()

        self.processor = Qwen3VLProcessor.from_pretrained(model_name_or_path, padding_side="right")

    @staticmethod
    def _pool_last_token(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        flipped = attention_mask.flip(dims=[1])
        last_one_from_end = flipped.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_from_end - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    def _format_conversation(
        self,
        *,
        text: str | None = None,
        image: Image.Image | None = None,
        instruction: str | None = None,
    ) -> list[dict]:
        inst = (instruction or self.instruction).strip()
        if inst and inst[-1] not in ".!?;:":
            inst += "."

        content = []
        if image is not None:
            content.append(
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            )
        content.append({"type": "text", "text": text or ""})

        return [
            {"role": "system", "content": [{"type": "text", "text": inst}]},
            {"role": "user", "content": content},
        ]

    def _preprocess(self, conversations: list[list[dict]]) -> dict[str, torch.Tensor]:
        processor: Any = self.processor
        text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)

        images, video_inputs, raw_video_kwargs = process_vision_info(
            conversations,
            image_patch_size=16,
            return_video_metadata=True,
            return_video_kwargs=True,
        )
        video_kwargs: dict[str, Any] = raw_video_kwargs or {}

        videos_list: list[Any] | None
        video_metadata_list: list[Any] | None
        if video_inputs is not None:
            _videos, _video_metadata = zip(*video_inputs)
            videos_list = list(_videos)
            video_metadata_list = list(_video_metadata)
        else:
            videos_list, video_metadata_list = None, None

        inputs = processor(
            text=text,
            images=images,
            videos=videos_list,
            video_metadata=video_metadata_list,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    @torch.no_grad()
    def encode(
        self,
        items: list[dict],
        *,
        normalize: bool | None = None,
        output_dim: int | None = None,
    ) -> torch.Tensor:
        conversations = [
            self._format_conversation(
                text=it.get("text"),
                image=it.get("image"),
                instruction=it.get("instruction"),
            )
            for it in items
        ]

        model_inputs = self._preprocess(conversations)
        outputs = self.model(**model_inputs)

        emb = self._pool_last_token(outputs.last_hidden_state, model_inputs["attention_mask"])

        od = output_dim if output_dim is not None else self.output_dim
        if od is not None:
            emb = emb[:, :od]

        do_norm = self.normalize if normalize is None else normalize
        if do_norm:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb


def _strip_data_uri(s: str) -> str:
    """Strip the 'data:<mime>;base64,' prefix from a data URI, returning raw base64."""
    if s.startswith("data:"):
        _, _, payload = s.partition(",")
        return payload
    return s


def _b64decode(s: str) -> bytes:
    """Decode base64, stripping data URI prefix and adding missing padding."""
    raw = _strip_data_uri(s)
    raw += "=" * (-len(raw) % 4)
    return base64.b64decode(raw)


def _sniff_mime_type(s: str) -> str:
    try:
        data = _b64decode(s)
    except Exception:
        return "text/plain"
    if data[:4] == b"\x89PNG":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    return "text/plain"


def _image_item(b64: str) -> dict[str, Any]:
    return {"image": Image.open(io.BytesIO(_b64decode(b64))).convert("RGB"), "text": ""}


class Qwen3VLEmbeddings(EmbeddingsModel):
    """Qwen3-VL text and image embeddings in a shared space."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-VL-Embedding-2B",
        *,
        instruction: str = "Represent the user's input.",
        output_dim: int | None = None,
        normalize: bool = True,
        dtype: torch.dtype | None = None,
        attn_implementation: str | None = None,
        device: str | None = None,
    ):
        self.embedder = _Qwen3VLEmbedder(
            model_name_or_path=model_name_or_path,
            instruction=instruction,
            output_dim=output_dim,
            normalize=normalize,
            dtype=dtype,
            attn_implementation=attn_implementation,
            device=device,
        )

    async def _encode(self, items: list[dict]) -> np.ndarray:
        emb = await asyncio.to_thread(self.embedder.encode, items)
        return emb.cpu().numpy()

    async def get_text_embeddings(self, texts: str | list[str]) -> np.ndarray:
        return await self._encode([{"text": t} for t in ([texts] if isinstance(texts, str) else texts)])

    async def get_image_embeddings(self, images: str | list[str]) -> np.ndarray:
        return await self._encode([_image_item(img) for img in ([images] if isinstance(images, str) else images)])

    async def get_embeddings(self, chunks: str | list[str] | Chunk | list[Chunk]) -> np.ndarray:  # type: ignore[override]
        chunk_list: list[Chunk]
        if isinstance(chunks, str):
            chunk_list = [Chunk(content=chunks, mime_type="text/plain", title="Query")]
        elif isinstance(chunks, Chunk):
            chunk_list = [chunks]
        elif all(isinstance(c, str) for c in chunks):
            chunk_list = [
                Chunk(content=c, mime_type=_sniff_mime_type(c), title=f"Chunk {i}")
                for i, c in enumerate(cast(list[str], chunks))
            ]
        else:
            chunk_list = cast(list[Chunk], chunks)

        items: list[dict] = []
        for chunk in chunk_list:
            if chunk.mime_type.startswith("text"):
                items.append({"text": chunk.content})
            elif chunk.mime_type.startswith("image"):
                items.append(_image_item(chunk.content))
            else:
                raise ValueError(f"Unknown Data Type: {chunk.mime_type}")
        return await self._encode(items)
