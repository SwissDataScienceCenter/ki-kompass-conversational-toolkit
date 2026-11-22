import base64
import io
import re
from enum import StrEnum
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, TableStructureOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.items.picture.picture import PictureItem
from markitdown import MarkItDown  # type: ignore[import-untyped]
from PIL import Image

from conversational_toolkit.chunking.base import Chunk, Chunker

_HEADER = re.compile(r"^(#{1,6}\s.*)$", re.MULTILINE)
_STRUCTURED_LINE = re.compile(r"^\s*(\||[-*+]\s|\d+[.)]\s|```|>|#{1,6}\s)", re.MULTILINE)


class MarkdownConverterEngine(StrEnum):
    MARKITDOWN = "markitdown"
    DOCLING = "docling"


def _join_wrapped_lines(text: str) -> str:
    """Undo hard line wraps in prose paragraphs; tables, lists, code and headers keep their line breaks."""
    return "\n\n".join(
        para if _STRUCTURED_LINE.search(para) else para.replace("\n", " ") for para in text.split("\n\n")
    )


def _png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class PDFChunker(Chunker):
    """Convert PDFs to markdown with docling (or MarkItDown) and split them into section chunks.

    Args:
        do_ocr: Run OCR on bitmap regions. Needed for scanned PDFs; for PDFs with a text layer it mostly adds time
            and "RapidOCR returned empty result" warnings.
        table_mode: TableFormer model for table structure. ACCURATE (docling's default) handles complex tables
            better; FAST makes more mistakes on merged cells.
        table_cell_matching: Map the PDF's text cells onto the recognised table grid (docling's default). Set to
            False to use the cells the table model predicts; try this if values go missing from tables.
    """

    def __init__(
        self,
        do_ocr: bool = True,
        table_mode: TableFormerMode = TableFormerMode.ACCURATE,
        table_cell_matching: bool = True,
    ) -> None:
        self.do_ocr = do_ocr
        self.table_mode = table_mode
        self.table_cell_matching = table_cell_matching
        # Creating a DocumentConverter reloads docling's models, so keep one per image setting.
        self._converters: dict[bool, DocumentConverter] = {}

    def _docling_converter(self, with_images: bool) -> DocumentConverter:
        if with_images not in self._converters:
            pipeline_options = PdfPipelineOptions(
                do_ocr=self.do_ocr,
                generate_picture_images=with_images,
                table_structure_options=TableStructureOptions(
                    mode=self.table_mode, do_cell_matching=self.table_cell_matching
                ),
            )
            self._converters[with_images] = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )
        return self._converters[with_images]

    def _convert(
        self, file_path: str, engine: MarkdownConverterEngine, with_images: bool
    ) -> tuple[str, list[tuple[str, Image.Image]]]:
        """Return the document as markdown and its named pictures."""
        if engine == MarkdownConverterEngine.MARKITDOWN:
            return str(MarkItDown().convert(file_path).text_content), []

        document = self._docling_converter(with_images).convert(file_path).document
        pictures = [
            element.image.pil_image
            for element, _ in document.iterate_items()
            if with_images and isinstance(element, PictureItem) and element.image and element.image.pil_image
        ]
        stem = Path(file_path).stem
        return document.export_to_markdown(compact_tables=True), [
            (f"{stem}-picture-{i}.png", p) for i, p in enumerate(pictures, 1)
        ]

    def to_markdown(self, file_path: str, engine: MarkdownConverterEngine = MarkdownConverterEngine.DOCLING) -> str:
        """Convert a document to markdown without extracting pictures."""
        return self._convert(file_path, engine, with_images=False)[0]

    def make_chunks(
        self,
        file_path: str,
        engine: MarkdownConverterEngine = MarkdownConverterEngine.DOCLING,
        write_images: bool = True,
        image_path: str | None = None,
    ) -> list[Chunk]:
        """Split a document into one chunk per markdown section, plus one base64 PNG chunk per picture.

        Text chunks get 'source_file' and 'chunk_index' metadata. Pictures are also saved to 'image_path' if given.
        MarkItDown extracts no pictures.
        """
        markdown, images = self._convert(file_path, engine, write_images)
        chunks = self._text_chunks(markdown)
        for i, chunk in enumerate(chunks):
            chunk.metadata |= {"source_file": Path(file_path).name, "chunk_index": i}

        if image_path is not None:
            Path(image_path).mkdir(parents=True, exist_ok=True)
        for name, image in images:
            if image_path is not None:
                image.save(Path(image_path) / name)
            chunks.append(
                Chunk(title=name, content=_png_base64(image), mime_type="image/png", metadata={"chapters": []})
            )
        return chunks

    def _text_chunks(self, markdown: str) -> list[Chunk]:
        matches = list(_HEADER.finditer(markdown))
        preamble = markdown[: matches[0].start()] if matches else markdown
        chunks = []
        if preamble.strip():
            chunks.append(
                Chunk(
                    title="",
                    content=_join_wrapped_lines(preamble),
                    mime_type="text/markdown",
                    metadata={"chapters": []},
                )
            )

        chapters: list[str] = []
        for i, match in enumerate(matches):
            header = match.group(1).strip()
            level = len(header) - len(header.lstrip("#"))
            chapters = [*chapters[: level - 1], header]
            end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
            chunks.append(
                Chunk(
                    title=header,
                    content=_join_wrapped_lines(markdown[match.start() : end]),
                    mime_type="text/markdown",
                    metadata={"chapters": chapters.copy()},
                )
            )
        return chunks
