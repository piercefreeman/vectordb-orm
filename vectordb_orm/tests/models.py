from vectordb_orm import MilvusBase, Embedding
from pymilvus import Milvus

class MyObject(MilvusBase, milvus_client=Milvus()):
    __embedding_dim__ = 128
    text: str
    embedding: Embedding
    id: int

    def __init__(self, text: str, embedding: Embedding):
        super().__init__()
        self.text = text
        self.embedding = embedding
        self.id = None

    def __repr__(self):
        return f"MyObject(id={self.id}, text='{self.text}')"

    @classmethod
    def collection_name(cls) -> str:
        return 'my_collection'
