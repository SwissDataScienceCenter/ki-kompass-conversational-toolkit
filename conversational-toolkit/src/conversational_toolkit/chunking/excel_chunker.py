from loguru import logger
import openpyxl  # type: ignore[import-untyped]

from conversational_toolkit.chunking.base import Chunk, Chunker


class ExcelChunker(Chunker):
    """One markdown-table chunk per non-empty sheet; the first non-empty row is the header."""

    def make_chunks(self, file_path: str) -> list[Chunk]:
        try:
            wb = openpyxl.load_workbook(file_path, data_only=True)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return []

        chunks: list[Chunk] = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = [
                [cell.value for cell in row] for row in ws.iter_rows() if any(cell.value is not None for cell in row)
            ]

            if not rows:
                continue

            header = [str(cell) if cell is not None else "" for cell in rows[0]]
            separator = ["---"] * len(header)

            lines = [
                "| " + " | ".join(header) + " |",
                "| " + " | ".join(separator) + " |",
            ]
            for row in rows[1:]:
                cells = [str(cell) if cell is not None else "" for cell in row]
                cells += [""] * (len(header) - len(cells))
                lines.append("| " + " | ".join(cells) + " |")

            chunks.append(
                Chunk(
                    title=sheet_name,
                    content="\n".join(lines),
                    mime_type="text/markdown",
                    metadata={
                        "sheet": sheet_name,
                        "rows": len(rows) - 1,
                        "columns": len(header),
                    },
                )
            )

        return chunks
