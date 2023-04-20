from vectordb_orm import MilvusBase, EmbeddingField, VarCharField, PrimaryKeyField, ConsistencyType
from vectordb_orm.indexes import IVF_FLAT, BIN_FLAT
import numpy as np

class MyObject(MilvusBase):
    __collection_name__ = 'my_collection'
    __consistency_type__ = ConsistencyType.STRONG

    id: int = PrimaryKeyField()
    text: str = VarCharField(max_length=128)
    embedding: np.ndarray = EmbeddingField(dim=128, index=IVF_FLAT(cluster_units=128))


class BinaryEmbeddingObject(MilvusBase):
    __collection_name__ = 'binary_collection'
    __consistency_type__ = ConsistencyType.STRONG

    id: int = PrimaryKeyField()
    embedding: np.ndarray[np.bool_] = EmbeddingField(dim=128, index=BIN_FLAT())
