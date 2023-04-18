from vectordb_orm.base import MilvusBase
from vectordb_orm.query import MilvusQueryBuilder
from pymilvus import Milvus

class MilvusSession:
    def __init__(self, milvus_client: Milvus):
        self.milvus_client = milvus_client

    def query(self, *query_objects: MilvusBase | MilvusQueryBuilder):
        return MilvusQueryBuilder(self.milvus_client, *query_objects)
