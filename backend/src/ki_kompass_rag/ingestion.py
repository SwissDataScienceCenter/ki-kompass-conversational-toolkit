"""Chunking strategies (header-based, fixed-size, paragraph-aware) and chunk-size statistics."""

from dataclasses import dataclass
from functools import lru_cache
from statistics import median

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from conversational_toolkit.chunking.base import Chunk
from conversational_toolkit.chunking.pdf_chunker import (
    PDFChunker,
)

EMBEDDING_TOKEN_LIMIT = 8191  # OpenAI text-embedding-3-*

_PDF_CHUNKER = PDFChunker(do_ocr=False)


@dataclass
class ChunkStats:
    strategy: str
    total_chunks: int
    avg_chars: float
    min_chars: int
    max_chars: int
    median_chars: float
    over_token_limit: int

    def __str__(self) -> str:
        return (
            f"strategy={self.strategy}, chunks={self.total_chunks}, avg={self.avg_chars:.0f}, "
            f"median={self.median_chars:.0f}, min={self.min_chars}, max={self.max_chars}, "
            f">{EMBEDDING_TOKEN_LIMIT} tokens={self.over_token_limit}"
        )


def estimate_tokens(text: str) -> int:
    """Rough token count: ~4 characters per token."""
    return len(text) // 4


def analyze_chunks(chunks: list[Chunk], strategy_name: str) -> ChunkStats:
    """Size statistics of chunks, in characters."""
    lengths = [len(c.content) for c in chunks] or [0]
    return ChunkStats(
        strategy=strategy_name,
        total_chunks=len(chunks),
        avg_chars=sum(lengths) / len(lengths),
        min_chars=min(lengths),
        max_chars=max(lengths),
        median_chars=median(lengths),
        over_token_limit=sum(
            estimate_tokens(c.content) > EMBEDDING_TOKEN_LIMIT for c in chunks
        ),
    )


def char_histogram(
    chunks: list[Chunk], bins: int = 10, title: str = "Chunk character lengths"
) -> Figure:
    """Histogram of chunk character lengths (displayed by Jupyter as the cell output)."""
    fig, ax = plt.subplots()
    ax.hist([len(c.content) for c in chunks], bins=bins, edgecolor="black")
    ax.set(xlabel="Characters", ylabel="Count", title=title)
    plt.close(fig)
    return fig


@lru_cache(maxsize=8)
def _markdown(file_path: str) -> str:
    return _PDF_CHUNKER.to_markdown(file_path)


def header_based_chunks(file_path: str) -> list[Chunk]:
    """One chunk per Markdown heading section (PDFChunker default)."""
    return _PDF_CHUNKER.make_chunks(file_path, write_images=False)


def fixed_size_chunks(
    file_path: str, chunk_size: int = 800, overlap: int = 100
) -> list[Chunk]:
    """Fixed character windows with overlap; may cut mid-sentence."""
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")
    markdown = _markdown(file_path)
    return [
        Chunk(
            title=f"chunk_{i:04d}",
            content=markdown[start : start + chunk_size],
            mime_type="text/markdown",
            metadata={
                "chunk_index": i,
                "start_char": start,
                "end_char": min(start + chunk_size, len(markdown)),
            },
        )
        for i, start in enumerate(range(0, len(markdown), chunk_size - overlap))
    ]


def paragraph_aware_chunks(file_path: str, target_chars: int = 600) -> list[Chunk]:
    """Split on blank lines and merge paragraphs up to target_chars (a longer paragraph stays whole)."""
    groups: list[list[str]] = []
    size = 0
    for para in filter(None, (p.strip() for p in _markdown(file_path).split("\n\n"))):
        if groups and size + len(para) <= target_chars:
            groups[-1].append(para)
            size += len(para)
        else:
            groups.append([para])
            size = len(para)
    return [
        Chunk(
            title=f"para_chunk_{i:04d}",
            content="\n\n".join(group),
            mime_type="text/markdown",
            metadata={"chunk_index": i},
        )
        for i, group in enumerate(groups)
    ]


def compare_strategies(file_path: str) -> dict[str, ChunkStats]:
    """Stats of the three strategies on one file, keyed by strategy name."""
    chunkings = {
        "header_based": header_based_chunks(file_path),
        "fixed_size_800": fixed_size_chunks(file_path, chunk_size=800, overlap=100),
        "paragraph_500": paragraph_aware_chunks(file_path, target_chars=500),
    }
    return {name: analyze_chunks(chunks, name) for name, chunks in chunkings.items()}
