from abc import ABC, abstractmethod
from typing import Type

import numpy as np

from vectordb_orm.attributes import AttributeCompare
from vectordb_orm.base import VectorSchemaBase
from vectordb_orm.fields import PrimaryKeyField
from vectordb_orm.results import QueryResult


class BackendBase(ABC):
    max_fetch_size: int

    @abstractmethod
    def create_collection(self, schema: Type[VectorSchemaBase]):
        pass

    @abstractmethod
    def clear_collection(self, schema: Type[VectorSchemaBase]):
        pass

    @abstractmethod
    def delete_collection(self, schema: Type[VectorSchemaBase]):
        pass

    @abstractmethod
    def insert(self, entity: VectorSchemaBase) -> int:
        pass

    @abstractmethod
    def insert_batch(self, entities: list[VectorSchemaBase], show_progress: bool) -> list[int]:
        pass

    @abstractmethod
    def delete(self, entity: VectorSchemaBase):
        pass

    @abstractmethod
    def search(
        self,
        schema: Type[VectorSchemaBase],
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

    def _get_primary(self, schema: Type[VectorSchemaBase]):
        """
        If the class has a primary key, return it, otherwise return None
        """
        for attribute_name in schema.__annotations__.keys():
            if isinstance(schema._type_configuration.get(attribute_name), PrimaryKeyField):
                return attribute_name
        return None

    def _assert_has_primary(self, schema: Type[VectorSchemaBase]):
        """
        Ensure we have a primary key, this is the only field that's fully required
        """
        if self._get_primary(schema) is None:
            raise ValueError(f"Class {schema.__name__} does not have a primary key, specify `PrimaryKeyField` on the class definition.")
