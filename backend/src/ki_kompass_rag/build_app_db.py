"""Build the vector stores used by the chat app (`ki_kompass_rag.main`).

The notebooks do not need this: they build their own store on first run.

Creates two ChromaDB collections under DB_DIR (default: backend/db):
    vs_text   text chunks embedded with OpenAI embeddings
    vs_image  image chunks embedded with Qwen3-VL (downloads a ~2B parameter model; slow on CPU)

Usage:
    uv run python -m ki_kompass_rag.build_app_db
    uv run python -m ki_kompass_rag.build_app_db --skip-images   # text store only
"""

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from conversational_toolkit.embeddings.openai import OpenAIEmbeddings
from ki_kompass_rag.baseline_rag import EMBEDDING_MODEL, build_vector_store, load_chunks

_ROOT = Path(__file__).parents[3]
load_dotenv(_ROOT / ".env.local")

DB_DIR = Path(os.getenv("DB_DIR", str(_ROOT / "backend" / "db")))
TEXT_VS_PATH = DB_DIR / "vs_text"
IMAGE_VS_PATH = DB_DIR / "vs_image"

# The OpenAI embedding model accepts at most ~8k tokens per input.
MAX_TEXT_CHUNK_CHARS = 8 * 1024


async def build(skip_images: bool = False) -> None:
    logger.info(f"Loading and chunking all documents (stores go to {DB_DIR})")
    all_chunks = load_chunks(max_files=None)

    text_chunks = [
        c
        for c in all_chunks
        if c.mime_type.startswith("text/") and len(c.content) < MAX_TEXT_CHUNK_CHARS
    ]
    await build_vector_store(
        text_chunks,
        OpenAIEmbeddings(model_name=EMBEDDING_MODEL),
        db_path=TEXT_VS_PATH,
        reset=True,
    )

    if skip_images:
        logger.info("Skipping the image store")
        return

    # Imported here so --skip-images works without loading the Qwen3-VL dependencies.
    from conversational_toolkit.embeddings.qwen_vl import Qwen3VLEmbeddings

    image_chunks = [c for c in all_chunks if c.mime_type.startswith("image/")]
    await build_vector_store(
        image_chunks,
        Qwen3VLEmbeddings(device="cpu"),
        db_path=IMAGE_VS_PATH,
        reset=True,
        batch_size=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--skip-images", action="store_true", help="only build the text store"
    )
    args = parser.parse_args()
    asyncio.run(build(skip_images=args.skip_images))


if __name__ == "__main__":
    main()
