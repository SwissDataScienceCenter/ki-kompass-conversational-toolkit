"""Chunk model and chunker interface."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A piece of a document; 'content' is text, or base64 for images (see 'mime_type')."""

    title: str
    content: str
    mime_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunker(ABC):
    """Splits a file into chunks; subclasses add format-specific 'make_chunks' arguments."""

    @abstractmethod
    def make_chunks(self, *args: Any, **kwargs: Any) -> list[Chunk]:
        """Parse the source file into chunks."""
