from collections.abc import Iterable

from tsann.types import Record


class TemporalDataset:
    def __init__(self, records: Iterable[Record] = ()):
        self.records: list[Record] = []
        self.id_to_record: dict[int, Record] = {}
        for record in records:
            self.insert(record)

    def insert(self, record: Record) -> None:
        if record.id in self.id_to_record:
            raise ValueError(f"Duplicate record id {record.id}")
        self.records.append(record)
        self.id_to_record[record.id] = record

    def __len__(self) -> int:
        return len(self.records)
