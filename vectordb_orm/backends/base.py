from abc import ABC, abstractmethod
from typing import Type
from vectordb_orm.base import VectorSchemaBase
import numpy as np
from vectordb_orm.attributes import AttributeCompare
from vectordb_orm.results import QueryResult

class BackendBase(ABC):
    @abstractmethod
    def create_collection(self, schema: Type[VectorSchemaBase]):
        pass

    @abstractmethod
    def delete_collection(self, schema: Type[VectorSchemaBase]):
        pass

    @abstractmethod
    def insert(self, entity: VectorSchemaBase) -> int:
        pass

    @abstractmethod
    def delete(self, entity: VectorSchemaBase):
        pass

    @abstractmethod
    def search(
        self,
        cls: Type[VectorSchemaBase],
        output_fields: list[str],
        filters: list[AttributeCompare] | None,
        search_embedding: np.ndarray | None,
        search_embedding_key: str | None,
        limit: int,
        offset: int,
    ) -> list[QueryResult]:
        pass

    @abstractmethod
    def flush(self, schema: Type[VectorSchemaBase]):
        pass

    @abstractmethod
    def load(self, schema: Type[VectorSchemaBase]):
        pass
