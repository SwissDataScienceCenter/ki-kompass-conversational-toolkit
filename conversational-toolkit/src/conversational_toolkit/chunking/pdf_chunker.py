import os
import tempfile
import io
import base64
import re
from enum import StrEnum

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from docling_core.types.doc.items.picture.picture import PictureItem
from pathlib import Path
from markitdown import MarkItDown  # type: ignore[import-untyped]
from PIL import Image  # type: ignore[import-untyped]

from conversational_toolkit.chunking.base import Chunk, Chunker


class MarkdownConverterEngine(StrEnum):
    MARKITDOWN = "markitdown"
    DOCLING = "docling"


class PDFChunker(Chunker):
    """Convert PDFs to markdown with docling (or MarkItDown) and split them into section chunks.

    Args:
        do_ocr: Run OCR on bitmap regions (scanned pages, text inside images). Needed for scanned
            PDFs; for PDFs that already have a text layer it mostly adds time and "RapidOCR returned
            empty result" warnings for images without text (logos, photos).
    """

    # TODO: Improve by not creating temporary files for images and support more image formats
    # TODO: Currently resizing, maybe not desired.

    def __init__(self, do_ocr: bool = True) -> None:
        self.do_ocr = do_ocr
        # One converter per image setting, reused across files: creating a DocumentConverter
        # reloads docling's layout and table models every time.
        self._converters: dict[bool, DocumentConverter] = {}

    def _docling_converter(self, with_images: bool) -> DocumentConverter:
        if with_images not in self._converters:
            pipeline_options = PdfPipelineOptions(do_ocr=self.do_ocr, generate_picture_images=with_images)
            self._converters[with_images] = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )
        return self._converters[with_images]

    def _pdf2markdown(
        self,
        file_path: str,
        engine: MarkdownConverterEngine = MarkdownConverterEngine.DOCLING,
        write_images: bool = False,
        image_path: str | None = None,
    ) -> str:
        if engine == MarkdownConverterEngine.MARKITDOWN:
            if write_images:
                raise NotImplementedError("Image extraction is not supported with MarkItDown engine.")
            result = MarkItDown().convert(file_path)
            return str(result.text_content)
        elif engine == MarkdownConverterEngine.DOCLING:
            with_images = bool(write_images and image_path)
            conv_result = self._docling_converter(with_images).convert(file_path)

            if with_images:
                # Manually save images from PictureItem elements
                doc_filename = Path(file_path).stem
                picture_counter = 0
                for element, _level in conv_result.document.iterate_items():
                    if isinstance(element, PictureItem) and element.image and element.image.pil_image:
                        picture_counter += 1
                        image_filename = Path(str(image_path)) / f"{doc_filename}-picture-{picture_counter}.png"
                        with open(image_filename, "wb") as fp:
                            element.image.pil_image.save(fp, format="PNG")

            return conv_result.document.export_to_markdown()  # type: ignore[no-any-return]
        else:
            raise NotImplementedError(f"Engine '{engine}' is not supported.")

    def _normalize_newlines(self, text: str) -> str:
        paragraphs = text.split("\n\n")
        processed_paragraphs = [para.replace("\n", " ") for para in paragraphs]
        return "\n\n".join(processed_paragraphs)

    def make_chunks(
        self,
        file_path: str,
        engine: MarkdownConverterEngine = MarkdownConverterEngine.DOCLING,
        write_images: bool = True,
        image_path: str | None = None,
    ) -> list[Chunk]:
        """Split a PDF into one chunk per markdown section, plus one chunk per extracted image.

        Images are extracted into a temporary directory that is removed afterwards. Pass
        `image_path` to keep the extracted images in that directory instead.
        """
        if not write_images:
            return self._text_chunks(self._pdf2markdown(file_path, engine))
        if image_path is not None:
            os.makedirs(image_path, exist_ok=True)
            return self._chunks_with_images(file_path, engine, image_path)
        with tempfile.TemporaryDirectory(prefix="pdf_chunker_") as tmp_dir:
            return self._chunks_with_images(file_path, engine, tmp_dir)

    def _chunks_with_images(self, file_path: str, engine: MarkdownConverterEngine, image_path: str) -> list[Chunk]:
        image_names_before = set(os.listdir(image_path))
        markdown = self._pdf2markdown(file_path, engine, write_images=True, image_path=image_path)
        chunks = self._text_chunks(markdown)

        # Only pick up the images written for this file, not files already in the directory.
        for file_name in sorted(set(os.listdir(image_path)) - image_names_before):
            extension = os.path.splitext(file_name)[1].lower()
            if extension in [".png", ".jpg", ".jpeg", ".gif"]:
                image_file_path = os.path.join(image_path, file_name)
                with Image.open(image_file_path) as img:
                    buffered = io.BytesIO()
                    img.save(buffered, format=img.format)
                    base64_encoding = base64.b64encode(buffered.getvalue()).decode("utf-8")

                chunks.append(
                    Chunk(
                        title=file_name,
                        content=base64_encoding,
                        mime_type=f"image/{extension[1:]}",
                        metadata={"chapters": []},
                    )
                )
        return chunks

    def _text_chunks(self, markdown: str) -> list[Chunk]:
        header_pattern = re.compile(r"^(#{1,6}\s.*)$", re.MULTILINE)
        matches = list(header_pattern.finditer(markdown))

        if not matches:
            return [
                Chunk(
                    title="",
                    content=self._normalize_newlines(markdown),
                    mime_type="text/markdown",
                    metadata={"chapters": []},
                )
            ]

        chunks: list[Chunk] = []
        current_chapters: list[str] = []
        for i, match in enumerate(matches):
            header_line = match.group(1).strip()
            header_level = header_line.count("#", 0, header_line.find(" "))

            if len(current_chapters) < header_level:
                current_chapters.append(header_line)
            else:
                current_chapters = [*current_chapters[: header_level - 1], header_line]

            end_idx = matches[i + 1].start() if i < len(matches) - 1 else len(markdown)
            chunk_text = markdown[match.start() : end_idx]

            chunks.append(
                Chunk(
                    title=header_line,
                    content=self._normalize_newlines(chunk_text),
                    mime_type="text/markdown",
                    metadata={"chapters": current_chapters.copy()},
                )
            )
        return chunks
