from abc import ABC, abstractmethod

from tsann.types import Query, Record, SearchResult


class BaseTemporalSubsetIndex(ABC):
    @abstractmethod
    def build(self, records: list[Record]) -> None:
        raise NotImplementedError

    @abstractmethod
    def insert(self, record: Record) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, record_id: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def expire(self, before_time: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: Query) -> SearchResult:
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> dict:
        raise NotImplementedError
