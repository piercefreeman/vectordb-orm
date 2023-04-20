from vectordb_orm.base import VectorSchemaBase
from vectordb_orm.query import VectorQueryBuilder
from vectordb_orm.backends.base import BackendBase
from pymilvus import Milvus
from typing import Type

class VectorSession:
    """
    Core session object used to interact with a vector database backend.

    """
    def __init__(self, backend: BackendBase):
        self.backend = backend

    def query(self, *query_objects: VectorSchemaBase | VectorQueryBuilder):
        return VectorQueryBuilder(self.backend, *query_objects)

    def create_collection(self, schema: Type[VectorSchemaBase]):
        """
        Create a Milvus collection using the given Milvus client.

        This function is called internally during the creation of a MilvusBase instance. It translates the typehinted
        attributes of the class into a Milvus collection schema.

        :param milvus_client: An instance of the Milvus client.
        :returns: The created Milvus collection.
        :raises ValueError: If the class does not have a primary key defined.
        """
        schema._assert_has_primary()
        return self.backend.create_collection(schema)

    def delete_collection(self, schema: Type[VectorSchemaBase]):
        """
        Delete a Milvus collection using the given Milvus client.

        :param milvus_client: An instance of the Milvus client.
        """
        return self.backend.delete_collection(schema)

    def insert(self, obj: VectorSchemaBase) -> int:
        """
        Insert the current MilvusBase object into the database using the provided Milvus client.

        :param milvus_client: An instance of the Milvus client.
        :returns: The primary key of the inserted object.
        """
        new_id = self.backend.insert(obj)
        obj.id = new_id
        return obj

    def delete(self, obj: VectorSchemaBase) -> None:
        """
        Delete the current MilvusBase object from the database using the provided Milvus client.

        :param milvus_client: An instance of the Milvus client.
        :raises ValueError: If the object has not been inserted into the database before deletion.
        """
        if not obj.id:
            raise ValueError("Cannot delete object that hasn't been inserted into the database")

        self.backend.delete(obj)
        obj.id = None

    def flush(self, schema: Type[VectorSchemaBase]):
        """
        Flush the collection for the given schema.

        :param milvus_client: An instance of the Milvus client.
        """
        self.backend.flush(schema)

    def load(self, schema: Type[VectorSchemaBase]):
        """
        Load the collection for the given schema.

        :param milvus_client: An instance of the Milvus client.
        """
        self.backend.load(schema)
