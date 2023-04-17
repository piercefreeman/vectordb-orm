from typing import List
from pymilvus import Milvus, IndexType, Collection
from pymilvus.client.types import DataType
from pymilvus.client.abstract import ChunkedQueryResult
from pymilvus.orm.schema import CollectionSchema, FieldSchema
from pymilvus.client.types import MetricType
from enum import Enum
from dataclasses import dataclass

class Embedding:
    def __init__(self, values: List[float]):
        self.values = values

class MilvusSession:
    def __init__(self, milvus_client: Milvus):
        self.milvus_client = milvus_client

    def query(self, cls: "MilvusBase"):
        # We must load this collection into memory before executing the search
        #self.milvus_client.load_collection(cls.collection_name())

        return MilvusQueryBuilder(self.milvus_client, cls)


class OperationType(Enum):
    """
    Values of the operation type should correspond to the operators used in the Milvus query language

    """
    EQUALS = '=='
    GREATER_THAN = '>'
    LESS_THAN = '<'
    LESS_THAN_EQUAL = '<='
    GREATER_THAN_EQUAL = '>='
    NOT_EQUAL = '!='

@dataclass
class QueryResult:
    result: "MilvusBase"
    score: float
    distance: float

class AttributeCompare:
    def __init__(self, attr, value=None, op=None):
        self.attr = attr
        self.value = value
        self.op = op

    def __eq__(self, other):
        return AttributeCompare(self.attr, other, OperationType.EQUALS)

    def __gt__(self, other):
        return AttributeCompare(self.attr, other, OperationType.GREATER_THAN)

    def __lt__(self, other):
        return AttributeCompare(self.attr, other, OperationType.LESS_THAN)

    def __le__(self, other):
        return AttributeCompare(self.attr, other, OperationType.LESS_THAN_EQUAL)

    def __ge__(self, other):
        return AttributeCompare(self.attr, other, OperationType.GREATER_THAN_EQUAL)

    def to_expression(self):
        value = self.value
        if isinstance(value, str):
            value = f"\"{self.value}\""

        return f"{self.attr} {self.op.value} {value}"


class MilvusQueryBuilder:
    def __init__(self, milvus_client: Milvus, cls):
        self.milvus_client = milvus_client
        self.cls = cls
        self.filters = []

    def filter(self, *filters: AttributeCompare):
        for f in filters:
            self.filters.append(f.to_expression())
        return self

    def similarity_ranking(self, embedding_attr, query_embedding, top_k=10):
        filters = " and ".join(self.filters)
        print("FILTERS", filters)

        search_param = {
            "collection_name": self.cls.collection_name(),
            "query_records": [query_embedding.values],
            "top_k": top_k,
            "params": {"nprobe": 16}
        }
        if filters:
            search_param["expr"] = filters
        print("params", search_param)

        search_result = self.milvus_client.search(
            data=search_param["query_records"],
            anns_field=embedding_attr.attr,
            param=search_param["params"],
            limit=search_param["top_k"],
            collection_name=search_param["collection_name"],
            expression=search_param["expr"] if "expr" in search_param else None,
            # TODO: Exclude embeddings by default
            output_fields=["id", "text"]
        )
        return self._result_to_objects(search_result)

    def all(self):
        filters = ", ".join(self.filters)

        search_param = {
            "collection_name": self.cls.collection_name(),
            "top_k": 10,
        }
        if filters:
            search_param["expr"] = filters

        search_result = self.milvus_client.search(
            data=[],
            param={},
            limit=search_param["top_k"],
            collection_name=search_param["collection_name"],
            output_fields=["id", "text"]
        )
        return self._result_to_objects(search_result)

    def _result_to_objects(self, search_result: ChunkedQueryResult):
        query_results : list[QueryResult] = []

        for hit in search_result:
            for result in hit:
                obj = self.cls(text=result.entity.get("text"), embedding=None)
                obj.id = result.id
                query_results.append(QueryResult(obj, result.score, result.distance))

        return query_results

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
