"""Build the chat app's vector stores: DB_DIR/vs_text (EMBEDDING_MODEL) and DB_DIR/vs_image (Qwen3-VL, slow on CPU).

Usage: python -m ki_kompass_rag.build_app_db [--skip-images]
"""

import argparse
import asyncio

from loguru import logger

from ki_kompass_rag.baseline_rag import (
    DB_DIR,
    build_embeddings,
    build_vector_store,
    load_chunks,
)

TEXT_VS_PATH = DB_DIR / "vs_text"
IMAGE_VS_PATH = DB_DIR / "vs_image"


async def build(skip_images: bool = False) -> None:
    logger.info(f"Building app stores in {DB_DIR}")
    chunks = load_chunks(with_images=not skip_images)
    await build_vector_store(
        [c for c in chunks if c.mime_type.startswith("text/")],
        build_embeddings(),
        db_path=TEXT_VS_PATH,
        reset=True,
    )
    if skip_images:
        return

    # Deferred: loads torch and the Qwen3-VL stack.
    from conversational_toolkit.embeddings.qwen_vl import Qwen3VLEmbeddings

    await build_vector_store(
        [c for c in chunks if c.mime_type.startswith("image/")],
        Qwen3VLEmbeddings(device="cpu"),
        db_path=IMAGE_VS_PATH,
        reset=True,
        batch_size=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-images", action="store_true", help="only build the text store"
    )
    asyncio.run(build(skip_images=parser.parse_args().skip_images))


if __name__ == "__main__":
    main()
