import numpy as np

from vectordb_orm import (ConsistencyType, EmbeddingField, Milvus_BIN_FLAT,
                          Milvus_IVF_FLAT, PineconeIndex,
                          PineconeSimilarityMetric, PrimaryKeyField,
                          VarCharField, VectorSchemaBase)


class MilvusMyObject(VectorSchemaBase):
    __collection_name__ = 'my_collection'
    __consistency_type__ = ConsistencyType.STRONG

    id: int = PrimaryKeyField()
    text: str = VarCharField(max_length=128)
    embedding: np.ndarray = EmbeddingField(dim=128, index=Milvus_IVF_FLAT(cluster_units=128))


class MilvusBinaryEmbeddingObject(VectorSchemaBase):
    __collection_name__ = 'binary_collection'
    __consistency_type__ = ConsistencyType.STRONG

    id: int = PrimaryKeyField()
    embedding: np.ndarray[np.bool_] = EmbeddingField(dim=128, index=Milvus_BIN_FLAT())


class PineconeMyObject(VectorSchemaBase):
    __collection_name__ = 'my_collection'

    id: int = PrimaryKeyField()
    text: str = VarCharField(max_length=128)
    embedding: np.ndarray = EmbeddingField(dim=128, index=PineconeIndex(metric_type=PineconeSimilarityMetric.COSINE))
