import os
from pathlib import Path

from PIL import Image

from conversational_toolkit.chunking.pdf_chunker import PDFChunker

MARKDOWN = "# Title\n\nIntro text.\n\n## Section\n\nMore text."


class FakePDFChunker(PDFChunker):
    """Skips docling: returns fixed markdown and writes one image when asked to."""

    def _pdf2markdown(
        self, file_path, engine=None, write_images=False, image_path=None
    ):
        if write_images and image_path:
            Image.new("RGB", (4, 4)).save(Path(image_path) / "doc-picture-1.png")
        return MARKDOWN


def test_default_uses_temp_dir_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    chunks = FakePDFChunker().make_chunks("doc.pdf")

    assert [c.mime_type for c in chunks] == [
        "text/markdown",
        "text/markdown",
        "image/png",
    ]
    assert os.listdir(tmp_path) == []  # nothing like ./tmp is left behind


def test_explicit_image_path_keeps_images_and_ignores_existing_files(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (4, 4)).save(image_dir / "other.png")

    chunks = FakePDFChunker().make_chunks("doc.pdf", image_path=str(image_dir))

    assert [c.title for c in chunks if c.mime_type.startswith("image/")] == [
        "doc-picture-1.png"
    ]
    assert sorted(os.listdir(image_dir)) == ["doc-picture-1.png", "other.png"]


def test_without_images_returns_only_text(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    chunks = FakePDFChunker().make_chunks("doc.pdf", write_images=False)

    assert all(c.mime_type == "text/markdown" for c in chunks)
    assert os.listdir(tmp_path) == []


def test_docling_converter_is_created_once_per_image_setting(monkeypatch):
    import conversational_toolkit.chunking.pdf_chunker as module

    created = []

    class FakeConverter:
        def __init__(self, format_options):
            created.append(format_options)

    monkeypatch.setattr(module, "DocumentConverter", FakeConverter)
    chunker = PDFChunker(do_ocr=False)

    first = chunker._docling_converter(with_images=False)
    assert chunker._docling_converter(with_images=False) is first
    chunker._docling_converter(with_images=True)

    assert len(created) == 2
    options = [opts[module.InputFormat.PDF].pipeline_options for opts in created]
    assert [o.do_ocr for o in options] == [False, False]
    assert [o.generate_picture_images for o in options] == [False, True]


def test_table_options_reach_docling(monkeypatch):
    import conversational_toolkit.chunking.pdf_chunker as module

    created = []
    monkeypatch.setattr(
        module,
        "DocumentConverter",
        lambda format_options: created.append(format_options),
    )
    PDFChunker(
        table_mode=module.TableFormerMode.FAST, table_cell_matching=False
    )._docling_converter(with_images=False)

    table_options = created[0][
        module.InputFormat.PDF
    ].pipeline_options.table_structure_options
    assert table_options.mode == module.TableFormerMode.FAST
    assert table_options.do_cell_matching is False
