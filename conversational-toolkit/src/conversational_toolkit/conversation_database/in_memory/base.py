import json
from pathlib import Path

from pydantic import BaseModel


class JSONFileStore[T: BaseModel]:
    """Keeps records in a dict keyed by id and mirrors it to a JSON file after every write."""

    model: type[T]

    def __init__(self, json_file_path: str):
        self.path = Path(json_file_path)
        self.records: dict[str, T] = {}
        if self.path.exists():
            self.records = {k: self.model.model_validate(v) for k, v in json.loads(self.path.read_text()).items()}

    def _save(self) -> None:
        self.path.write_text(json.dumps({k: v.model_dump(mode="json") for k, v in self.records.items()}, indent=4))

    def _put(self, record: T) -> T:
        self.records[record.id] = record  # type: ignore[attr-defined]
        self._save()
        return record

    def _delete(self, ids: list[str]) -> bool:
        removed = [self.records.pop(i) for i in ids if i in self.records]
        if removed:
            self._save()
        return bool(removed)

    def _get(self, record_id: str) -> T:
        if record_id not in self.records:
            raise ValueError(f"{self.model.__name__} with id {record_id} not found")
        return self.records[record_id]

    def _filter(self, **fields: str) -> list[T]:
        return [r for r in self.records.values() if all(getattr(r, k) == v for k, v in fields.items())]
