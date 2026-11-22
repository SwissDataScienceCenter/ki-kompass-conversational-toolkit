import asyncio

import numpy as np

import pytest

from conversational_toolkit.chunking.base import Chunk
from ki_kompass_rag import baseline_rag, ingestion
from ki_kompass_rag.tool_agents import (
    HalfBoldText,
    RetrieveRelevantChunks,
    chunks_to_text,
)
from ki_kompass_rag.utils.json import parse_llm_json_stream


def chunk(content: str, source: str = "a.pdf") -> Chunk:
    return Chunk(
        title="t",
        content=content,
        mime_type="text/markdown",
        metadata={"source_file": source},
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            '```json\n{"answer": "hi", "used_sources_id": ["1"]}\n```',
            {"answer": "hi", "used_sources_id": ["1"]},
        ),
        ('{"answer": "trunc', {"answer": "trunc"}),
        ('Here you go: {"answer": "x"}', {"answer": "x"}),
        ("plain text", {"answer": "plain text"}),
        ("{]", None),
    ],
)
def test_parse_llm_json_stream(text, expected):
    assert parse_llm_json_stream(text) == expected


def test_chunks_to_text():
    text = chunks_to_text([chunk("x", "a.pdf"), chunk("y", "b.pdf")])
    assert text == "## a.pdf: t\n```\nx\n```\n\n---\n\n## b.pdf: t\n```\ny\n```"
    assert chunks_to_text([]) == ""


def test_retrieve_tool_returns_sources():
    class FakeRetriever:
        async def retrieve(self, query):
            return [chunk(query)]

    result = asyncio.run(RetrieveRelevantChunks(FakeRetriever()).call({"query": "q"}))  # type: ignore[arg-type]
    assert result["_sources"] == [
        {
            "title": "t",
            "content": "q",
            "mime_type": "text/markdown",
            "metadata": {"source_file": "a.pdf"},
        }
    ]


def test_half_bold_text():
    result = asyncio.run(HalfBoldText().call({"text": "a b c"}))
    assert result == {"result": "**a** b **c**"}


def test_load_chunks_filters_then_limits(tmp_path, monkeypatch):
    for name in ["EVALUATION_gt.md", "a.png", "b.md", "c.txt", "d.md"]:
        (tmp_path / name).write_text("# Heading\n\nbody")
    monkeypatch.setattr(baseline_rag, "DATA_DIR", tmp_path)

    chunks = baseline_rag.load_chunks(max_files=2)

    assert {c.metadata["source_file"] for c in chunks} == {"b.md", "c.txt"}


def test_build_llm_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported backend"):
        baseline_rag.build_llm("bogus")


def test_build_llm_ollama_needs_a_model_name():
    with pytest.raises(ValueError, match="model name"):
        baseline_rag.build_llm("ollama")


def test_fixed_size_chunks_overlap(monkeypatch):
    monkeypatch.setattr(ingestion, "_markdown", lambda _: "abcdefghij")
    chunks = ingestion.fixed_size_chunks("x.pdf", chunk_size=4, overlap=1)
    assert [c.content for c in chunks] == ["abcd", "defg", "ghij", "j"]


def test_paragraph_aware_chunks_merges_up_to_target(monkeypatch):
    monkeypatch.setattr(ingestion, "_markdown", lambda _: "aa\n\nbb\n\ncccccc\n\nd")
    chunks = ingestion.paragraph_aware_chunks("x.pdf", target_chars=5)
    assert [c.content for c in chunks] == ["aa\n\nbb", "cccccc", "d"]


def test_analyze_chunks():
    stats = ingestion.analyze_chunks([chunk("a" * n) for n in (1, 3, 10)], "s")
    assert (
        stats.total_chunks,
        stats.min_chars,
        stats.max_chars,
        stats.median_chars,
    ) == (3, 1, 10, 3)
    assert ingestion.analyze_chunks([], "empty").total_chunks == 0


def test_build_embeddings_picks_openai_by_name(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    assert isinstance(
        baseline_rag.build_embeddings("text-embedding-3-small"),
        baseline_rag.OpenAIEmbeddings,
    )


def test_vector_store_records_its_embedding_model(tmp_path):
    class FakeEmbeddings:
        model_name = "fake-model"

        async def get_embeddings(self, texts):
            return np.ones((len(texts), 2))

    old = baseline_rag.ChromaDBVectorStore(str(tmp_path / "old"))
    assert baseline_rag.store_embedding_model(old) == "text-embedding-3-small"

    store = asyncio.run(
        baseline_rag.build_vector_store(
            [chunk("x")],
            FakeEmbeddings(),  # type: ignore[arg-type]
            db_path=tmp_path / "new",
            reset=True,
        )
    )
    reopened = baseline_rag.ChromaDBVectorStore(str(tmp_path / "new"))
    assert baseline_rag.store_embedding_model(reopened) == "fake-model"
    assert reopened.space == "cosine"
    assert store.collection.count() == 1


def test_split_text_packs_paragraphs_and_repeats_table_header():
    paragraphs = "\n\n".join(["a" * 40] * 5)
    assert baseline_rag.split_text(paragraphs, max_chars=100) == [
        f"{'a' * 40}\n\n{'a' * 40}",
        f"{'a' * 40}\n\n{'a' * 40}",
        "a" * 40,
    ]

    table = "\n".join(["| h |", "| - |", *[f"| {i:02} |" for i in range(6)]])
    pieces = baseline_rag.split_text(table, max_chars=30)
    assert all(p.startswith("| h |\n| - |\n") for p in pieces)
    assert sum(p.count("\n") - 1 for p in pieces) == 6
    assert baseline_rag.split_text("short") == ["short"]
