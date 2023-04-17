from typing import List
from pymilvus import Milvus, IndexType, Collection
from pymilvus.client.types import DataType
from pymilvus.client.abstract import ChunkedQueryResult
from pymilvus.orm.schema import CollectionSchema, FieldSchema
from vectordb_orm.attributes import AttributeCompare
from dataclasses import dataclass


class MilvusBaseMeta(type):
    def __getattr__(cls, name):
        if name in ['text', 'id', 'embedding']:
            return AttributeCompare(name)
        raise AttributeError(f"'{cls.__name__}' object has no attribute '{name}'")


class MilvusBase(metaclass=MilvusBaseMeta):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()

    @classmethod
    def collection_name(cls) -> str:
        return cls.__name__.lower()

    @classmethod
    def embedding_dim(cls) -> int:
        return cls.__embedding_dim__

    @classmethod
    def _create_collection(cls, milvus_client: Milvus):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=cls.embedding_dim())
        ]

        schema = CollectionSchema(fields=fields, description="MilvusBase collection")
        #index_params = {'index_type': IndexType.IVF_SQ8, 'metric_type': MetricType.L2}
        #milvus_client.create_collection(collection_name=cls.collection_name(), schema=schema, index_params=index_params)
        collection = Collection(cls.collection_name(), schema)

        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }
        collection.create_index("embedding", index)
        return collection

    def insert(self, milvus_client: Milvus) -> int:
        vector = self.embedding.values
        entities = [
            {
                "name": "embedding",
                "type": DataType.FLOAT_VECTOR,
                "values": [vector]
            },
            {
                "name": "text",
                "type": DataType.VARCHAR,
                "values": [self.text]
            }
        ]
        mutation_result = milvus_client.insert(collection_name=self.collection_name(), entities=entities)
        self.id = mutation_result.primary_keys[0]
        return self.id

    @classmethod
    def _from_result(cls, id: int, embedding: List[float]) -> 'MilvusBase':
        # Convert the embedding list to an Embedding object
        embedding = Embedding(embedding)

        # Create a new instance of the MilvusBase subclass and set its attributes
        obj = cls()
        obj.embedding = embedding
        obj.id = id
