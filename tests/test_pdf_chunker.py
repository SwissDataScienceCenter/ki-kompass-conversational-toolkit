from pathlib import Path

from PIL import Image

from conversational_toolkit.chunking.pdf_chunker import PDFChunker

MARKDOWN = """Preamble text
wrapped over two lines.

# Title

Intro text.

## Section

| a | b |
|---|---|
| 1 | 2 |

- first
- second
"""


class FakePDFChunker(PDFChunker):
    """Skips docling: returns fixed markdown and one picture when asked for images."""

    def _convert(self, file_path, engine, with_images):
        pictures = [("doc-picture-1.png", Image.new("RGB", (4, 4)))]
        return MARKDOWN, pictures if with_images else []


def test_sections_preamble_and_images():
    chunks = FakePDFChunker().make_chunks("dir/doc.pdf")

    assert [c.title for c in chunks] == [
        "",
        "# Title",
        "## Section",
        "doc-picture-1.png",
    ]
    assert [c.mime_type for c in chunks] == ["text/markdown"] * 3 + ["image/png"]
    assert chunks[0].content.strip() == "Preamble text wrapped over two lines."
    assert chunks[2].metadata["chapters"] == ["# Title", "## Section"]
    assert [c.metadata.get("chunk_index") for c in chunks] == [0, 1, 2, None]
    assert {c.metadata.get("source_file") for c in chunks[:3]} == {"doc.pdf"}


def test_tables_and_lists_keep_line_breaks():
    section = FakePDFChunker().make_chunks("doc.pdf", write_images=False)[-1].content

    assert "| a | b |\n|---|---|\n| 1 | 2 |" in section
    assert "- first\n- second" in section


def test_image_path_saves_pictures(tmp_path):
    image_dir = tmp_path / "images"
    FakePDFChunker().make_chunks("doc.pdf", image_path=str(image_dir))

    assert [p.name for p in image_dir.iterdir()] == ["doc-picture-1.png"]


def test_without_images_returns_only_text(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    chunks = FakePDFChunker().make_chunks("doc.pdf", write_images=False)

    assert all(c.mime_type == "text/markdown" for c in chunks)
    assert list(Path(tmp_path).iterdir()) == []


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
