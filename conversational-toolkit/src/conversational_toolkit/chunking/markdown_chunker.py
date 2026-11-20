from pathlib import Path

from PIL import Image

from conversational_toolkit.chunking.pdf_chunker import MarkdownConverterEngine, PDFChunker


class MarkdownChunker(PDFChunker):
    """Chunks .md and .txt files like 'PDFChunker', reading them as UTF-8 instead of converting."""

    def _convert(
        self, file_path: str, engine: MarkdownConverterEngine, with_images: bool
    ) -> tuple[str, list[tuple[str, Image.Image]]]:
        return Path(file_path).read_text(encoding="utf-8"), []
