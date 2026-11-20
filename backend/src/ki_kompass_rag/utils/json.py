import re
from typing import Any

from partial_json_parser import loads as partial_json_loads  # type: ignore[import-untyped]


def parse_llm_json_stream(text: str) -> dict[str, Any] | None:
    """Parse a possibly fenced or truncated JSON object; plain text becomes {"answer": text}."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start = cleaned.find("{")
    if start == -1:
        return {"answer": text}
    try:
        obj = partial_json_loads(cleaned[start:])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None
