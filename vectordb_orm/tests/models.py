from vectordb_orm import MilvusBase, EmbeddingField, VarCharField, PrimaryKeyField
from pymilvus import Milvus
from vectordb_orm.indexes import IVF_FLAT
import numpy as np

class MyObject(MilvusBase, milvus_client=Milvus()):
    __collection_name__ = 'my_collection'

    id: int = PrimaryKeyField()
    text: str = VarCharField(max_length=128)
    embedding: np.ndarray = EmbeddingField(dim=128, index=IVF_FLAT(cluster_units=128))

    def __init__(self, text: str, embedding: np.array):
        super().__init__()
        self.text = text
        self.embedding = embedding
        self.id = None

    def __repr__(self):
        return f"MyObject(id={self.id}, text='{self.text}')"
