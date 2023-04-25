from typing import Type

from pymilvus import Milvus

from vectordb_orm.backends.base import BackendBase
from vectordb_orm.base import VectorSchemaBase
from vectordb_orm.query import VectorQueryBuilder


class VectorSession:
    """
    Core session object used to interact with a vector database backend.

    """
    def __init__(self, backend: BackendBase):
        self.backend = backend

    def query(self, *query_objects: VectorSchemaBase | VectorQueryBuilder):
        return VectorQueryBuilder(self.backend, *query_objects)

    def create_collection(self, schema: Type[VectorSchemaBase]):
        return self.backend.create_collection(schema)

    def clear_collection(self, schema: Type[VectorSchemaBase]):
        return self.backend.clear_collection(schema)

    def delete_collection(self, schema: Type[VectorSchemaBase]):
        return self.backend.delete_collection(schema)

    def insert(self, obj: VectorSchemaBase) -> VectorSchemaBase:
        new_id = self.backend.insert(obj)
        obj.id = new_id
        return obj

    def insert_batch(self, objs: list[VectorSchemaBase], show_progress: bool = False) -> list[VectorSchemaBase]:
        new_ids = self.backend.insert_batch(objs, show_progress=show_progress)
        for new_id, obj in zip(new_ids, objs):
            obj.id = new_id
        return objs

    def delete(self, obj: VectorSchemaBase) -> None:
        if not obj.id:
            raise ValueError("Cannot delete object that hasn't been inserted into the database")

        self.backend.delete(obj)
        obj.id = None

    def flush(self, schema: Type[VectorSchemaBase]):
        self.backend.flush(schema)

    def load(self, schema: Type[VectorSchemaBase]):
        self.backend.load(schema)
