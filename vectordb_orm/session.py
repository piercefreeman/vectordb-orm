from vectordb_orm.base import MilvusBase
from vectordb_orm.query import MilvusQueryBuilder
from pymilvus import Milvus

class MilvusSession:
    def __init__(self, milvus_client: Milvus):
        self.milvus_client = milvus_client

    def query(self, cls: MilvusBase):
        # We must load this collection into memory before executing the search
        #self.milvus_client.load_collection(cls.collection_name())

        return MilvusQueryBuilder(self.milvus_client, cls)
