import numpy as np

from vectordb_orm import (ConsistencyType, EmbeddingField, PrimaryKeyField,
                          VarCharField, VectorSchemaBase)
from vectordb_orm.backends.milvus.indexes import (Milvus_BIN_FLAT,
                                                  Milvus_IVF_FLAT)


class MyObject(VectorSchemaBase):
    __collection_name__ = 'my_collection'
    __consistency_type__ = ConsistencyType.STRONG

    id: int = PrimaryKeyField()
    text: str = VarCharField(max_length=128)
    embedding: np.ndarray = EmbeddingField(dim=128, index=Milvus_IVF_FLAT(cluster_units=128))


class BinaryEmbeddingObject(VectorSchemaBase):
    __collection_name__ = 'binary_collection'
    __consistency_type__ = ConsistencyType.STRONG

    id: int = PrimaryKeyField()
    embedding: np.ndarray[np.bool_] = EmbeddingField(dim=128, index=Milvus_BIN_FLAT())
